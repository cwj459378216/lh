"""pc_backtest_loop.py

按日期循环跑 pc.py 的“策略->当日入选股票”，并基于本地通达信日线（CSV）做逐笔回测。

核心思路：
- 第一步：逐日执行 pc.py，生成“当日可买股票”的 CSV 汇总。
- 第二步：读取这些 CSV，并基于本地通达信日线回测。

输出：
- trades CSV：每笔交易明细（买入/卖出/收益/原因等）
- summary CSV：整体、按策略、按股票的胜率与均值

注意：
- 会访问 10jqka 接口，跑 3 个月 + 110 个策略请求量较大，建议先用 --max-days/--max-strategies 试跑；
- 默认开启磁盘缓存（pc/cache），重复运行会显著加速。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

# 确保仓库根目录在 sys.path，便于直接运行该脚本
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

# 复用 pc.py 的网络抓取/策略解析
import pc as pcmod  # type: ignore




@dataclass
class LoopConfig:
	strategies_csv: str = str(Path(__file__).parent / "csv" / "1.31_dedup70.csv")
	data_dir: str = str(Path(__file__).parent.parent / "通达信" / "data" / "pytdx" / "daily_raw")
	pc_csv_dir: str = str(Path.cwd() / "csv")
	pc_csv_prefix: str = "pc_daily_"
	skip_pc_generate: bool = False

	# 回测区间（YYYYMMDD / YYYY-MM-DD 均可）
	start_date: str | None = None
	end_date: str | None = None
	months: int = 3

	# 交易细节
	buy_price_mode: str = "next_open"  # 目前只实现 next_open
	sell_price_mode: str = "close"  # 目前只实现 close
	stop_loss_mode: str = "open_low"  # open_low: 低于止损价触发(开盘优先)；close: 按收盘价触发
	min_hold_days: int = 1
	initial_capital: float = 1_000_000.0
	buy_fixed_amount: float = 10_000.0
	commission_rate: float = 0.000085
	commission_min: float = 0.1
	stamp_tax_rate_sell: float = 0.0005
	lot_size: int = 100

	# 是否在区间末尾强制平仓（统计口径）
	force_close_at_end: bool = True

	# 读取策略时的过滤阈值（与 pc.py 保持一致）
	mwr0_min: float = 0.0
	mwr1_max: float = 99999.0

	# 限流
	max_days: int | None = None
	max_strategies: int | None = None

	# 缓存
	cache_dir: str = str(Path(__file__).parent / "cache")
	cache_enabled: bool = True


def _parse_date_any(x: str | None) -> pd.Timestamp | None:
	if x is None:
		return None
	s = str(x).strip()
	if not s:
		return None
	# 纯数字 YYYYMMDD
	if s.isdigit() and len(s) == 8:
		return pd.to_datetime(s, format="%Y%m%d", errors="raise").normalize()
	return pd.to_datetime(s, errors="raise").normalize()


def _code_to_symbol(code6: str) -> str:
	c = str(code6).strip()
	if len(c) != 6 or not c.isdigit():
		return c
	market = "SH" if c.startswith("6") else "SZ"
	return f"{c}.{market}"


def _load_csv_path(fp: str) -> pd.DataFrame:
	"""读取通达信日线 CSV。

	期望列：trade_date, open, high, low, close, volume, amount
	trade_date 支持 YYYYMMDD 或 YYYY-MM-DD。
	"""
	try:
		df = pd.read_csv(fp)
	except Exception:
		return pd.DataFrame()

	exp = ["trade_date", "open", "high", "low", "close", "volume", "amount"]
	if any(c not in df.columns for c in exp):
		return pd.DataFrame()

	td_raw = df["trade_date"]
	td_str = td_raw.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
	td = pd.to_datetime(td_str, format="%Y%m%d", errors="coerce")
	mask = td.isna() & td_str.ne("")
	if bool(mask.any()):
		td.loc[mask] = pd.to_datetime(td_str.loc[mask], errors="coerce")
	df["trade_date"] = td
	df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

	for c in ["open", "high", "low", "close", "volume", "amount"]:
		df[c] = pd.to_numeric(df[c], errors="coerce")
	return df


def _load_symbol_df(symbol: str, data_dir: str) -> pd.DataFrame:
	fn = f"{symbol}.csv"
	fp = os.path.join(data_dir, fn)
	return _load_csv_path(fp)


def _get_next_open(symbol_df: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
	if symbol_df is None or symbol_df.empty or "open" not in symbol_df.columns:
		return None, None
	df = symbol_df.sort_values("trade_date")
	idx = df.index[df["trade_date"] == date]
	if len(idx) == 0:
		return None, None
	pos = df.index.get_loc(idx[0])
	if isinstance(pos, slice):
		pos = pos.start
	npos = pos + 1
	if npos >= len(df):
		return None, None
	row = df.iloc[npos]
	try:
		px = float(row.get("open") or 0)
	except Exception:
		px = 0
	if px <= 0:
		return None, None
	return pd.to_datetime(row.get("trade_date")).normalize(), px


def _get_close(symbol_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
	if symbol_df is None or symbol_df.empty:
		return None
	row = symbol_df[symbol_df["trade_date"] == date]
	if row.empty:
		return None
	try:
		px = float(row["close"].iloc[0])
	except Exception:
		return None
	return px if px > 0 else None


def _disk_cache_path(cache_dir: str, key: str) -> str:
	h = hashlib.sha1(key.encode("utf-8")).hexdigest()
	sub = os.path.join(cache_dir, h[:2])
	os.makedirs(sub, exist_ok=True)
	return os.path.join(sub, f"{h}.json")


def _cache_get(cache_dir: str, key: str) -> Any | None:
	fp = _disk_cache_path(cache_dir, key)
	if not os.path.exists(fp):
		return None
	try:
		with open(fp, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return None


def _cache_set(cache_dir: str, key: str, obj: Any) -> None:
	fp = _disk_cache_path(cache_dir, key)
	try:
		with open(fp, "w", encoding="utf-8") as f:
			json.dump(obj, f, ensure_ascii=False)
	except Exception:
		pass


class CachedFetcher:
	def __init__(self, session: requests.Session, *, cache_dir: str, enabled: bool = True):
		self.session = session
		self.cache_dir = cache_dir
		self.enabled = enabled
		self._mem: dict[str, Any] = {}

	def get_json(self, url: str) -> Any:
		if url in self._mem:
			return self._mem[url]
		if self.enabled:
			if (v := _cache_get(self.cache_dir, f"url:{url}")) is not None:
				self._mem[url] = v
				return v
		v = pcmod.fetch_json(self.session, url)
		self._mem[url] = v
		if self.enabled:
			_cache_set(self.cache_dir, f"url:{url}", v)
		return v

	def fetch_detail(self, strategy_id: int) -> Any:
		url = f"https://backtest.10jqka.com.cn/strategysquare/detail?strategyId={strategy_id}"
		return self.get_json(url)

	def fetch_backtestresult(self, strategy_id: int) -> Any:
		url = f"https://backtest.10jqka.com.cn/strategysquare/backtestresult?strategyId={strategy_id}"
		return self.get_json(url)

	def fetch_historypick(self, query: str, hold_num: str | int, trade_date: str) -> Any:
		q = quote(str(query), safe="")
		url = (
			"https://backtest.10jqka.com.cn/tradebacktest/historypick"
			f"?query={q}&hold_num={hold_num}&trade_date={trade_date}"
		)
		return self.get_json(url)


def _read_strategy_ids(csv_path: str) -> tuple[list[int], dict[int, dict[str, Any]]]:
	"""返回 strategy_ids，以及 extra_info（含 daySaleStrategy/winRate/fetched_query 等）。"""
	df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
	ids: list[int] = []
	extra: dict[int, dict[str, Any]] = {}
	for _, r in df.iterrows():
		sid_s = str(r.get("strategy_id") or "").strip()
		if not sid_s.isdigit():
			continue
		sid = int(sid_s)
		ids.append(sid)
		extra[sid] = {k: (r.get(k) if k in r else None) for k in df.columns}
	return ids, extra


def _available_trading_days(data_dir: str) -> list[pd.Timestamp]:
	# 复用 backtest_select_stocks_local 的逻辑会更快？这里简单扫一个文件即可：
	files = sorted([fn for fn in os.listdir(data_dir) if fn.lower().endswith(".csv")])
	for fn in files[:20]:
		df = _load_csv_path(os.path.join(data_dir, fn))
		if df is not None and not df.empty and "trade_date" in df.columns:
			days = pd.to_datetime(df["trade_date"], errors="coerce").dropna().dt.normalize().tolist()
			if days:
				return sorted(list(set(days)))
	# 兜底：全量扫（慢）
	days: set[pd.Timestamp] = set()
	for fn in files:
		df = _load_csv_path(os.path.join(data_dir, fn))
		if df is None or df.empty or "trade_date" not in df.columns:
			continue
		for d in pd.to_datetime(df["trade_date"], errors="coerce").dropna().dt.normalize().tolist():
			days.add(pd.to_datetime(d).normalize())
	return sorted(days)


def _iter_test_days(cfg: LoopConfig) -> tuple[list[pd.Timestamp], pd.Timestamp, pd.Timestamp]:
	all_days = _available_trading_days(cfg.data_dir)
	if not all_days:
		raise RuntimeError(f"data_dir 无交易日数据: {cfg.data_dir}")

	end = _parse_date_any(cfg.end_date) if cfg.end_date else max(all_days)
	start = _parse_date_any(cfg.start_date) if cfg.start_date else None
	if start is None:
		start = (pd.to_datetime(end) - pd.DateOffset(months=int(cfg.months))).normalize()

	test_days = [d for d in all_days if start <= d <= end]
	if cfg.max_days is not None:
		test_days = test_days[: int(cfg.max_days)]
	return test_days, start, end


def _yyyymmdd(d: pd.Timestamp) -> str:
	return pd.to_datetime(d).strftime("%Y%m%d")


def _pc_csv_path(cfg: LoopConfig, d: pd.Timestamp) -> Path:
	name = f"{cfg.pc_csv_prefix}{_yyyymmdd(d)}.csv"
	return Path(cfg.pc_csv_dir) / name


def _ensure_pc_csv_for_day(cfg: LoopConfig, d: pd.Timestamp, *, python_exec: str) -> Path:
	out_path = _pc_csv_path(cfg, d)
	if out_path.exists() or cfg.skip_pc_generate:
		return out_path

	env = os.environ.copy()
	env["PC_TRADE_DATE"] = pd.to_datetime(d).strftime("%Y-%m-%d")
	env["PC_OUTPUT_FILENAME"] = out_path.name

	cmd = [python_exec, str(Path(__file__).parent / "pc.py")]
	res = subprocess.run(cmd, cwd=str(_ROOT), env=env)
	if res.returncode != 0:
		raise RuntimeError(f"pc.py 运行失败: trade_date={env['PC_TRADE_DATE']}, code={res.returncode}")
	return out_path


def _extract_stock_map_from_row(row: dict[str, Any]) -> dict[str, str]:
	stocks_json = row.get("stocks")
	if isinstance(stocks_json, str) and stocks_json.strip():
		try:
			obj = json.loads(stocks_json)
			if isinstance(obj, dict):
				return {str(k): str(v) for k, v in obj.items()}
		except Exception:
			pass

	stock_map: dict[str, str] = {}
	for k, v in row.items():
		if not (isinstance(k, str) and k.endswith("_code")):
			continue
		code = str(v) if v is not None else ""
		if not code:
			continue
		name = str(row.get(k.replace("_code", "_name")) or "")
		stock_map[code] = name
	return stock_map


def _read_pc_daily_csv(path: Path) -> pd.DataFrame:
	if not path.exists():
		return pd.DataFrame()
	try:
		return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
	except Exception:
		try:
			return pd.read_csv(path, dtype=str, keep_default_na=False)
		except Exception:
			return pd.DataFrame()


def _to_float_pct(v: Any) -> float | None:
	"""把 upperIncome/fallIncome/lowerIncome 解析成“百分数数值”(例如 5 表示 5%)."""
	if v is None:
		return None
	if isinstance(v, (int, float)):
		return float(v)
	s = str(v).strip()
	if not s:
		return None
	s = s.replace("%", "")
	try:
		return float(s)
	except Exception:
		return None


def _to_float_rate_pct(v: Any) -> float | None:
	"""把胜率解析为百分数值（如 78.5 或 0.785）。输出统一为百分数。"""
	if v is None:
		return None
	if isinstance(v, (int, float)):
		num = float(v)
		# 兼容 0~1 比例
		if num <= 1.0:
			return num * 100.0
		return num
	if isinstance(v, str):
		s = v.strip().replace("%", "")
		if not s:
			return None
		try:
			num = float(s)
		except Exception:
			return None
		if num <= 1.0:
			return num * 100.0
		return num
	return None


def _simulate_trade(
	symbol_df: pd.DataFrame,
	*,
	signal_date: pd.Timestamp,
	upper_income_pct: float | None,
	fall_income_pct: float | None,
	lower_income_pct: float | None,
	max_hold_days: int | None,
	min_hold_days: int,
	stop_loss_mode: str,
	end_date: pd.Timestamp,
	force_close_at_end: bool,
) -> dict[str, Any] | None:
	"""逐笔交易模拟。

	买入：signal_date 下一交易日开盘
	卖出：
	- 止损：收益 <= -lower
	- 止盈回撤：若累计最高收益 >= upper，则当 (peak_ret - cur_ret) >= fall 时卖出
	- 最大持仓天数：到达后卖出

	价格口径：卖出使用当日收盘价。
	"""

	buy_date, buy_price = _get_next_open(symbol_df, signal_date)
	if buy_date is None or buy_price is None:
		return None

	df = symbol_df.sort_values("trade_date")
	df = df[(df["trade_date"] >= buy_date) & (df["trade_date"] <= end_date)].copy()
	if df.empty:
		return None

	peak_close = None
	peak_ret = None

	# 交易日计数（从买入成交日算第1个交易日）
	hold_td = 0
	# 禁止买入当日卖出：至少持有到下一交易日
	effective_min_hold = max(2, int(min_hold_days))
	prev_close: float | None = None
	prev_high: float | None = None
	reached_upper = False

	for _, row in df.iterrows():
		d = pd.to_datetime(row.get("trade_date")).normalize()
		c = row.get("close")
		hi = row.get("high")
		lo = row.get("low")
		op = row.get("open")
		try:
			close_px = float(c) if c is not None else 0.0
		except Exception:
			close_px = 0.0
		try:
			high_px = float(hi) if hi is not None else 0.0
		except Exception:
			high_px = 0.0
		try:
			low_px = float(lo) if lo is not None else 0.0
		except Exception:
			low_px = 0.0
		try:
			open_px = float(op) if op is not None else 0.0
		except Exception:
			open_px = 0.0
		if close_px <= 0:
			continue

		hold_td += 1

		if peak_close is None or close_px > float(peak_close):
			peak_close = close_px
			peak_ret = (float(peak_close) - float(buy_price)) / float(buy_price) * 100.0

		cur_ret = (float(close_px) - float(buy_price)) / float(buy_price) * 100.0

		reasons: list[str] = []

		# 达到止盈阈值后，启用“前一日最高价回落”止盈
		if upper_income_pct is not None and cur_ret >= float(upper_income_pct):
			reached_upper = True

		if reached_upper and hold_td >= effective_min_hold and prev_high is not None and prev_high > 0:
			if fall_income_pct is not None:
				tp_price = float(prev_high) * (1.0 - float(fall_income_pct) / 100.0)
				if low_px > 0 and low_px <= tp_price:
					sell_price = float(tp_price)
					if open_px > 0 and open_px < tp_price:
						sell_price = float(open_px)
					return {
						"buy_date": buy_date,
						"buy_price": float(buy_price),
						"sell_date": d,
						"sell_price": float(sell_price),
						"hold_td": hold_td,
						"pnl_pct": (float(sell_price) - float(buy_price)) / float(buy_price) * 100.0,
						"reason": "take_profit_trail",
						"peak_ret_pct": None if peak_ret is None else float(peak_ret),
					}

		# 最少持仓天数：避免买入当天就卖出
		if hold_td >= effective_min_hold:
			# 止损
			if lower_income_pct is not None:
				if str(stop_loss_mode).lower() == "close":
					if cur_ret <= -float(lower_income_pct):
						reasons.append("stop_loss")
				else:
					# open_low：若当日最低价 <= 止损价则止损；若开盘价 <= 止损价则按开盘价成交
					sl_price = float(buy_price) * (1.0 - float(lower_income_pct) / 100.0)
					if low_px > 0 and low_px <= sl_price:
						sell_price = float(sl_price)
						if open_px > 0 and open_px <= sl_price:
							sell_price = float(open_px)
						return {
							"buy_date": buy_date,
							"buy_price": float(buy_price),
							"sell_date": d,
							"sell_price": float(sell_price),
							"hold_td": hold_td,
							"pnl_pct": (float(sell_price) - float(buy_price)) / float(buy_price) * 100.0,
							"reason": "stop_loss",
							"peak_ret_pct": None if peak_ret is None else float(peak_ret),
						}

		# 最大持仓天数（同样遵守“买入当日不可卖出”）；若已达止盈阈值则不以持股周期卖出
		if not reached_upper and max_hold_days is not None and hold_td >= max(int(max_hold_days), effective_min_hold):
			reasons.append("max_hold_days")

		if reasons:
			return {
				"buy_date": buy_date,
				"buy_price": float(buy_price),
				"sell_date": d,
				"sell_price": float(close_px),
				"hold_td": hold_td,
				"pnl_pct": float(cur_ret),
				"reason": "+".join(sorted(set(reasons))),
				"peak_ret_pct": None if peak_ret is None else float(peak_ret),
			}

		prev_close = float(close_px)
		prev_high = float(high_px) if high_px > 0 else prev_high

	# 走到区间末尾仍未触发
	if not force_close_at_end:
		return None

	last = df.tail(1).iloc[0]
	last_date = pd.to_datetime(last.get("trade_date")).normalize()
	if last_date <= buy_date:
		return None
	last_close = float(last.get("close") or 0)
	if last_close <= 0:
		return None
	last_ret = (float(last_close) - float(buy_price)) / float(buy_price) * 100.0

	return {
		"buy_date": buy_date,
		"buy_price": float(buy_price),
		"sell_date": last_date,
		"sell_price": float(last_close),
		"hold_td": hold_td,
		"pnl_pct": float(last_ret),
		"reason": "force_close_end",
		"peak_ret_pct": None if peak_ret is None else float(peak_ret),
	}


def _collect_daily_picks(
	fetcher: CachedFetcher,
	*,
	strategy_id: int,
	trade_date: str,
	extra_info: dict[int, dict[str, Any]],
	mwr0_min: float,
	mwr1_max: float,
) -> dict[str, Any] | None:
	"""拉取单个策略在 trade_date 的入选股票 + 该策略的止盈止损参数。"""

	# 1) 过滤：回测胜率/持股周期（来自 backtestresult.reportData.maxWinRate）
	bt = fetcher.fetch_backtestresult(strategy_id)
	max_win_rate = pcmod._get_max_win_rate(bt)
	mwr0, mwr0_num, mwr1, mwr1_num = pcmod._parse_max_win_rate(max_win_rate)

	if (mwr0_num is None and mwr0_min > 0) or (mwr0_num is not None and float(mwr0_num) < float(mwr0_min)):
		return None
	if (mwr1_num is None and mwr1_max < 9000) or (mwr1_num is not None and float(mwr1_num) > float(mwr1_max)):
		return None

	# 2) detail：拿 query/dayBuyStockNum + 止盈止损参数
	detail = fetcher.fetch_detail(strategy_id)
	result = detail.get("result", {}) if isinstance(detail, dict) else {}
	qs = result.get("queryString", {}) if isinstance(result, dict) else {}

	query = qs.get("query")
	day_buy_stock_num = qs.get("dayBuyStockNum")
	if not query or not day_buy_stock_num:
		return None

	upper = _to_float_pct(result.get("upperIncome"))
	fall = _to_float_pct(result.get("fallIncome"))
	lower = _to_float_pct(result.get("lowerIncome"))

	# 3) historypick：拿 stocks
	hp = fetcher.fetch_historypick(query=str(query), hold_num=day_buy_stock_num, trade_date=trade_date)
	stocks = pcmod._get_historypick_stocks(hp, context="loop", sid=strategy_id, non_dict_verb="提示")
	if not stocks:
		return None

	stock_map = pcmod._extract_mainboard_stock_map(stocks)
	if not stock_map:
		return None

	extra = extra_info.get(strategy_id, {})

	return {
		"strategy_id": int(strategy_id),
		"query": query,
		"trade_date": trade_date,
		"dayBuyStockNum": day_buy_stock_num,
		"upperIncome": upper,
		"fallIncome": fall,
		"lowerIncome": lower,
		"stocks": stock_map,
		"extra": extra,
		"mwr0": mwr0,
		"mwr1": mwr1,
	}


def _summarize(trades: pd.DataFrame) -> pd.DataFrame:
	def _first_non_null(s: pd.Series) -> float | None:
		for v in s.tolist():
			if v is None:
				continue
			try:
				if pd.isna(v):
					continue
			except Exception:
				pass
			return float(v)
		return None

	by_strategy = (
		trades.groupby(["strategy_id"], dropna=False, as_index=False)
		.agg(
			strategy_win_rate=("strategy_win_rate", _first_non_null),
			trades=("pnl_pct", "count"),
			win_rate=("win", "mean"),
			avg_pnl_pct=("pnl_pct", "mean"),
			sum_pnl_pct=("pnl_pct", "sum"),
			avg_pnl_amount=("pnl_amount", "mean"),
			sum_pnl_amount=("pnl_amount", "sum"),
		)
		.sort_values(["trades", "win_rate"], ascending=[False, False])
	)

	return by_strategy


def run_loop(cfg: LoopConfig) -> tuple[str, str]:
	test_days, start, end = _iter_test_days(cfg)
	Path(cfg.pc_csv_dir).mkdir(parents=True, exist_ok=True)

	out_dir = Path.cwd() / "csv"
	out_dir.mkdir(parents=True, exist_ok=True)
	ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
	trades_path = str(out_dir / f"pc_loop_trades_{ts}.csv")
	summary_path = str(out_dir / f"pc_loop_summary_{ts}.csv")

	# 行情缓存：按 symbol 缓存 DataFrame，避免重复读文件
	symbol_df_cache: dict[str, pd.DataFrame] = {}

	# Phase 1: 生成 pc.py 每日 CSV
	pc_csv_paths: list[Path] = []
	python_exec = sys.executable
	for i, d in enumerate(test_days, start=1):
		out_path = _ensure_pc_csv_for_day(cfg, d, python_exec=python_exec)
		pc_csv_paths.append(out_path)
		if cfg.skip_pc_generate:
			print(f"[pc] {i}/{len(test_days)} 已有CSV: {out_path.name}")
		else:
			print(f"[pc] {i}/{len(test_days)} 生成CSV: {out_path.name}")

	# Phase 2: 读取 pc.csv 并回测
	trade_rows: list[dict[str, Any]] = []
	for i, (d, pc_path) in enumerate(zip(test_days, pc_csv_paths), start=1):
		td_str = pd.to_datetime(d).strftime("%Y-%m-%d")
		df = _read_pc_daily_csv(pc_path)
		print(f"[bt] {i}/{len(test_days)} trade_date={td_str} rows={len(df)}")
		if df is None or df.empty:
			continue

		if cfg.max_strategies is not None:
			df = df.head(int(cfg.max_strategies))

		for _, row in df.iterrows():
			rowd: dict[str, Any] = row.to_dict()
			stocks = _extract_stock_map_from_row(rowd)
			if not stocks:
				continue

			upper = _to_float_pct(rowd.get("upperIncome"))
			fall = _to_float_pct(rowd.get("fallIncome"))
			lower = _to_float_pct(rowd.get("lowerIncome"))
			strategy_win_rate = _to_float_rate_pct(rowd.get("winRate"))

			max_hold_days = None
			ds = rowd.get("daySaleStrategy")
			if ds is not None and str(ds).strip().isdigit():
				max_hold_days = int(str(ds).strip())

			strategy_id_raw = rowd.get("property_id") or rowd.get("strategy_id") or rowd.get("strategyId")
			try:
				strategy_id = int(str(strategy_id_raw).strip())
			except Exception:
				strategy_id = None

			for code6, name in stocks.items():
				symbol = _code_to_symbol(code6)
				if symbol not in symbol_df_cache:
					try:
						symbol_df_cache[symbol] = _load_symbol_df(symbol, cfg.data_dir)
					except Exception:
						symbol_df_cache[symbol] = pd.DataFrame()
				sdf = symbol_df_cache[symbol]
				if sdf is None or sdf.empty:
					continue

				sim = _simulate_trade(
					sdf,
					signal_date=pd.to_datetime(d).normalize(),
					upper_income_pct=upper,
					fall_income_pct=fall,
					lower_income_pct=lower,
					max_hold_days=max_hold_days,
					min_hold_days=int(cfg.min_hold_days),
					stop_loss_mode=str(cfg.stop_loss_mode),
					end_date=pd.to_datetime(end).normalize(),
					force_close_at_end=bool(cfg.force_close_at_end),
				)
				if not sim:
					continue

				buy_price = float(sim.get("buy_price") or 0)
				sell_price = float(sim.get("sell_price") or 0)
				if buy_price <= 0 or sell_price <= 0:
					continue

				# 固定金额买入，按 100 股一手
				shares = int(cfg.buy_fixed_amount / buy_price / cfg.lot_size) * cfg.lot_size
				if shares <= 0:
					continue

				buy_value = shares * buy_price
				buy_comm = max(buy_value * cfg.commission_rate, cfg.commission_min)
				sell_value = shares * sell_price
				sell_comm = max(sell_value * cfg.commission_rate, cfg.commission_min) + sell_value * cfg.stamp_tax_rate_sell
				pnl_amount = sell_value - sell_comm - buy_value - buy_comm
				pnl_pct = (pnl_amount / buy_value) * 100.0 if buy_value > 0 else 0.0

				trade_rows.append(
					{
						"signal_date": td_str,
						"strategy_id": strategy_id,
						"symbol": symbol,
						"stock_name": name,
						"strategy_win_rate": strategy_win_rate,
						"upperIncome": upper,
						"fallIncome": fall,
						"lowerIncome": lower,
						"max_hold_days": max_hold_days,
						"buy_date": pd.to_datetime(sim.get("buy_date")).strftime("%Y-%m-%d")
						if sim.get("buy_date") is not None
						else "",
						"buy_price": buy_price,
						"sell_date": pd.to_datetime(sim.get("sell_date")).strftime("%Y-%m-%d")
						if sim.get("sell_date") is not None
						else "",
						"sell_price": sell_price,
						"hold_td": sim.get("hold_td"),
						"shares": shares,
						"buy_amount": buy_value,
						"buy_commission": buy_comm,
						"sell_amount": sell_value,
						"sell_commission": sell_comm,
						"pnl_amount": pnl_amount,
						"pnl_pct": pnl_pct,
						"win": 1 if pnl_amount > 0 else 0,
						"reason": sim.get("reason"),
						"query": rowd.get("query"),
						"dayBuyStockNum": rowd.get("dayBuyStockNum"),
					}
				)

	trades = pd.DataFrame(trade_rows)
	if trades.empty:
		# 仍写空文件，方便批处理
		trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
		pd.DataFrame(
			[
				{
					"start": start.strftime("%Y-%m-%d"),
					"end": end.strftime("%Y-%m-%d"),
					"trades": 0,
					"note": "no trades",
				}
			]
		).to_csv(summary_path, index=False, encoding="utf-8-sig")
		return trades_path, summary_path

	trades = trades.sort_values(["signal_date", "strategy_id", "symbol"]).reset_index(drop=True)

	# 中文表头（交易明细）
	trades_cn = trades.rename(
		columns={
			"signal_date": "信号日",
			"strategy_id": "策略ID",
			"symbol": "股票代码",
			"stock_name": "股票名称",
			"strategy_win_rate": "原始策略胜率(%)",
			"upperIncome": "止盈_收益率(%)",
			"fallIncome": "止盈回撤(%)",
			"lowerIncome": "止损_收益率(%)",
			"max_hold_days": "最大持仓天数",
			"buy_date": "买入日",
			"buy_price": "买入价",
			"sell_date": "卖出日",
			"sell_price": "卖出价",
			"hold_td": "持仓交易日数",
			"shares": "买入股数",
			"buy_amount": "买入金额",
			"buy_commission": "买入手续费",
			"sell_amount": "卖出金额",
			"sell_commission": "卖出手续费",
			"pnl_amount": "盈亏(元)",
			"pnl_pct": "盈亏(%)",
			"win": "是否盈利",
			"reason": "卖出原因",
			"query": "策略条件",
			"dayBuyStockNum": "单日买入数",
		}
	)
	trades_cn.to_csv(trades_path, index=False, encoding="utf-8-sig")

	summary_all = _summarize(trades)

	# 中文表头（汇总，仅按策略）
	summary_cn = summary_all.rename(
		columns={
			"strategy_id": "策略ID",
			"strategy_win_rate": "原始策略胜率(%)",
			"trades": "交易数",
			"win_rate": "胜率",
			"avg_pnl_pct": "平均盈亏(%)",
			"sum_pnl_pct": "累计盈亏(%)",
			"avg_pnl_amount": "平均盈亏(元)",
			"sum_pnl_amount": "累计盈亏(元)",
		}
	)
	summary_cn.to_csv(summary_path, index=False, encoding="utf-8-sig")

	print(f"[out] trades: {trades_path}")
	print(f"[out] summary: {summary_path}")

	return trades_path, summary_path


def _build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="循环回测：pc 策略每日入选股票 -> 本地行情止盈止损回测")
	p.add_argument("--strategies-csv", type=str, default=str(Path(__file__).parent / "csv" / "1.31_dedup70.csv"))
	p.add_argument(
		"--data-dir",
		type=str,
		default=str(Path(__file__).parent.parent / "通达信" / "data" / "pytdx" / "daily_raw"),
	)
	p.add_argument("--pc-csv-dir", type=str, default=str(Path.cwd() / "csv"))
	p.add_argument("--pc-csv-prefix", type=str, default="pc_daily_")
	p.add_argument("--skip-pc-generate", action="store_true", help="跳过 pc.py 生成阶段，仅读取已有 CSV")

	p.add_argument("--start-date", type=str, default=None, help="YYYYMMDD 或 YYYY-MM-DD")
	p.add_argument("--end-date", type=str, default=None, help="YYYYMMDD 或 YYYY-MM-DD")
	p.add_argument("--months", type=int, default=3, help="当未指定 start-date 时，用 end-date 往前推 N 个月")

	p.add_argument("--min-hold-days", type=int, default=1, help="最少持仓交易日")
	p.add_argument(
		"--stop-loss-mode",
		type=str,
		default="open_low",
		help="止损模式：open_low(最低价触发/开盘优先成交) 或 close(按收盘价触发)",
	)
	p.add_argument("--no-force-close", action="store_true", help="区间结束时不强制平仓（未触发规则的交易不计入）")

	p.add_argument("--mwr0-min", type=float, default=0.0, help="策略历史 maxWinRate0 最低阈值(0~1)，0 表示不筛")
	p.add_argument("--mwr1-max", type=float, default=99999.0, help="策略历史 maxWinRate1 最大阈值")

	p.add_argument("--max-days", type=int, default=None, help="最多回测多少个交易日（调试用）")
	p.add_argument("--max-strategies", type=int, default=None, help="最多跑多少个策略（调试用）")

	p.add_argument("--cache-dir", type=str, default=str(Path(__file__).parent / "cache"))
	p.add_argument("--no-cache", action="store_true")

	return p


def main() -> None:
	ap = _build_argparser()
	args = ap.parse_args()

	cfg = LoopConfig(
		strategies_csv=str(args.strategies_csv),
		data_dir=str(args.data_dir),
		pc_csv_dir=str(args.pc_csv_dir),
		pc_csv_prefix=str(args.pc_csv_prefix),
		skip_pc_generate=bool(args.skip_pc_generate),
		start_date=args.start_date,
		end_date=args.end_date,
		months=int(args.months),
		min_hold_days=int(args.min_hold_days),
		stop_loss_mode=str(args.stop_loss_mode),
		force_close_at_end=(not bool(args.no_force_close)),
		mwr0_min=float(args.mwr0_min),
		mwr1_max=float(args.mwr1_max),
		max_days=args.max_days,
		max_strategies=args.max_strategies,
		cache_dir=str(args.cache_dir),
		cache_enabled=(not bool(args.no_cache)),
	)

	run_loop(cfg)


if __name__ == "__main__":
	main()