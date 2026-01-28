import json
import random
import time
from typing import Any, Dict
from urllib.parse import quote

from pathlib import Path

import requests


def _preview_value(v: Any, limit: int = 120) -> str:
    s = str(v)
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _split_first_two(v: Any) -> tuple[Any, Any]:
    if isinstance(v, list):
        v0 = v[0] if len(v) > 0 else None
        v1 = v[1] if len(v) > 1 else None
        return v0, v1
    return None, None


def _parse_max_win_rate(max_win_rate: Any) -> tuple[Any, float | None, Any, float | None]:
    mwr0, mwr1 = _split_first_two(max_win_rate)
    mwr0_num = _parse_win_rate_ratio(mwr0)
    mwr1_num = _safe_float(mwr1)
    return mwr0, mwr0_num, mwr1, mwr1_num


def _get_historypick_stocks(
    historypick: Any,
    *,
    context: str,
    sid: Any,
    non_dict_verb: str,
) -> Any:
    stocks = None
    if isinstance(historypick, dict):
        hp_result = historypick.get("result")
        if isinstance(hp_result, dict):
            stocks = hp_result.get("stocks")
        else:
            preview = _preview_value(hp_result)
            print(
                f"{non_dict_verb}[{context}]：historypick.result 非 dict，sid={sid}，type={type(hp_result).__name__}，value={preview}"
            )
    return stocks


def _derive_csv_filename(base: str, suffix: str) -> str:
    if base.lower().endswith(".csv"):
        return base[:-4] + suffix + ".csv"
    return base + suffix + ".csv"


def _write_dict_csv(path: str, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    import csv

    if not rows:
        return
    fns = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        if fieldnames is None:
            w.writerows(rows)
        else:
            for r in rows:
                w.writerow({k: r.get(k) for k in fns})


def _get_csv_output_dir() -> Path:
    """返回当前工作目录下的 csv 输出目录，并确保其存在。"""
    out_dir = Path.cwd() / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_headers() -> Dict[str, str]:
    # 尽量模拟常见浏览器请求头（不保证能绕过风控，仅用于降低“过于像脚本”的特征）
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://backtest.10jqka.com.cn/",
        "Origin": "https://backtest.10jqka.com.cn",
    }


def _human_sleep(min_s: float = 0.8, max_s: float = 2.2) -> None:
    # 加一点“人工抖动”
    time.sleep(random.uniform(min_s, max_s))


def fetch_json(session: requests.Session, url: str, *, retries: int = 3) -> Any:
    # 简单重试 + 退避（遇到限流/临时网络抖动时更稳）
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        _human_sleep(0.6, 1.8)
        try:
            resp = session.get(url, timeout=30)

            # 常见限流：429 Too Many Requests；以及 5xx
            if resp.status_code in (429, 500, 502, 503, 504):
                backoff = min(10.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.6))
                time.sleep(backoff)
                continue

            resp.raise_for_status()

            try:
                return resp.json()
            except ValueError:
                return json.loads(resp.text)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            backoff = min(10.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.6))
            time.sleep(backoff)

    raise last_exc if last_exc else RuntimeError("Request failed")


def fetch_list(session: requests.Session, page: int = 1, page_num: int = 10, sort_type: str = "hot", keyword: str = "") -> Any:
    url = (
        "https://backtest.10jqka.com.cn/strategysquare/list"
        f"?order=desc&page={page}&pageNum={page_num}&sortType={sort_type}&keyword={keyword}"
    )
    return fetch_json(session, url)


def fetch_detail(session: requests.Session, strategy_id: int) -> Any:
    url = f"https://backtest.10jqka.com.cn/strategysquare/detail?strategyId={strategy_id}"
    return fetch_json(session, url)


def fetch_historypick(session: requests.Session, query: str, hold_num: str | int, trade_date: str) -> Any:
    q = quote(str(query), safe="")
    url = (
        "https://backtest.10jqka.com.cn/tradebacktest/historypick"
        f"?query={q}&hold_num={hold_num}&trade_date={trade_date}"
    )
    return fetch_json(session, url)


def fetch_backtestresult(session: requests.Session, strategy_id: int) -> Any:
    url = f"https://backtest.10jqka.com.cn/strategysquare/backtestresult?strategyId={strategy_id}"
    return fetch_json(session, url)


def _get_report_data(backtestresult: Any) -> Dict[str, Any]:
    if not isinstance(backtestresult, dict):
        return {}
    return (
        backtestresult.get("result", {})
        .get("statistics", {})
        .get("reportData", {})
    )


def _get_max_annual_yield(backtestresult: Any) -> Any:
    return _get_report_data(backtestresult).get("maxAnnualYield")


def _get_max_win_rate(backtestresult: Any) -> Any:
    return _get_report_data(backtestresult).get("maxWinRate")


def _extract_mainboard_stock_map(stocks: Any) -> dict[str, str]:
    stock_map: dict[str, str] = {}
    if not isinstance(stocks, list):
        return stock_map

    for s in stocks:
        if not isinstance(s, dict):
            continue
        code = s.get("stock_code") or s.get("stockCode") or s.get("code")
        name = s.get("stock_name") or s.get("stockName") or s.get("name")
        if not (code and name):
            continue

        code_str = str(code)
        # 沪深主板：60xxxx（沪A）/00xxxx（深A）
        if code_str.startswith("60") or code_str.startswith("00"):
            stock_map[code_str] = str(name)

    return stock_map


def _parse_win_rate_ratio(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        num = float(v)
        # 兼容百分数（80 表示 80%）
        if num > 1.0:
            return num / 100.0
        return num
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.endswith("%"):
            try:
                return float(s[:-1].strip()) / 100.0
            except Exception:
                return None
        try:
            num = float(s)
        except Exception:
            return None
        if num > 1.0:
            return num / 100.0
        return num
    return None


def _read_scanned_strategy_csv(csv_path: str) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not isinstance(r, dict):
                continue
            rows.append(r)
    return rows


def _load_config(config_path: str | None = None) -> dict[str, Any]:
    """加载配置文件（默认同目录 pc_config.toml）。

    支持：.toml（优先，支持注释） / .json
    """
    default_path = Path(__file__).with_name("pc_config.toml")
    path = Path(config_path) if config_path else default_path

    if not path.exists():
        return {}

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}

    # 默认按 toml 读取
    try:
        import tomllib  # py3.11+
    except Exception:
        return {}

    try:
        with open(path, "rb") as f:
            obj = tomllib.load(f)
    except Exception as e:
        # 配置文件写错时不要直接崩溃：提示后回退到默认配置
        print(f"[config] 解析失败，将忽略配置文件：{path} ({type(e).__name__}: {e})")
        return {}

    return obj if isinstance(obj, dict) else {}


def _cfg_get(cfg: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur.get(k)
    return cur


def _replay_from_scanned_records(
    session: requests.Session,
    *,
    scanned_records: list[dict[str, Any]],
    trade_date: str,
    mwr0_min: float,
    mwr1_max: float,
    skip_seen_strategy_ids: bool = True,
    enable_list_winrate_prefilter: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """根据 scanned_strategy_ids.csv 的记录，重新跑一遍过滤逻辑，并输出用于对比的明细。"""
    final_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []

    seen_strategy_ids: set[int] = set()

    for r in scanned_records:
        sid_raw = r.get("strategy_id")
        try:
            sid = int(str(sid_raw).strip())
        except Exception:
            continue

        if skip_seen_strategy_ids:
            if sid in seen_strategy_ids:
                continue
            seen_strategy_ids.add(sid)

        sort_type = r.get("sort_type")
        page = r.get("page")
        list_wr = r.get("list_winRate")
        old_prefilter_pass = r.get("prefilter_pass")

        list_wr_num = _parse_win_rate_ratio(list_wr)
        new_prefilter_pass = not (list_wr_num is None or list_wr_num < mwr0_min)

        # scan 模式才有 list_winRate；manual 模式记录里为空，此时不做预过滤
        if enable_list_winrate_prefilter and list_wr is not None and not new_prefilter_pass:
            compare_rows.append(
                {
                    "strategy_id": sid,
                    "sort_type": sort_type,
                    "page": page,
                    "list_winRate": list_wr,
                    "old_prefilter_pass": old_prefilter_pass,
                    "new_prefilter_pass": new_prefilter_pass,
                    "replay_mwr0": None,
                    "replay_mwr0_num": None,
                    "replay_mwr1": None,
                    "replay_mwr1_num": None,
                    "replay_pass": False,
                    "replay_reason": "prefilter_list_winRate",
                }
            )
            continue

        # 重新请求回测结果，按同样阈值过滤
        backtestresult = fetch_backtestresult(session, sid)
        max_win_rate = _get_max_win_rate(backtestresult)

        mwr0, mwr0_num, mwr1, mwr1_num = _parse_max_win_rate(max_win_rate)
        if mwr0_num is None or mwr0_num < mwr0_min:
            compare_rows.append(
                {
                    "strategy_id": sid,
                    "sort_type": sort_type,
                    "page": page,
                    "list_winRate": list_wr,
                    "old_prefilter_pass": old_prefilter_pass,
                    "new_prefilter_pass": new_prefilter_pass,
                    "replay_mwr0": mwr0,
                    "replay_mwr0_num": mwr0_num,
                    "replay_mwr1": None,
                    "replay_mwr1_num": None,
                    "replay_pass": False,
                    "replay_reason": "mwr0_min",
                }
            )
            continue

        if mwr1_num is None or mwr1_num > mwr1_max:
            compare_rows.append(
                {
                    "strategy_id": sid,
                    "sort_type": sort_type,
                    "page": page,
                    "list_winRate": list_wr,
                    "old_prefilter_pass": old_prefilter_pass,
                    "new_prefilter_pass": new_prefilter_pass,
                    "replay_mwr0": mwr0,
                    "replay_mwr0_num": mwr0_num,
                    "replay_mwr1": mwr1,
                    "replay_mwr1_num": mwr1_num,
                    "replay_pass": False,
                    "replay_reason": "mwr1_max",
                }
            )
            continue

        # 通过过滤：再补齐 detail/historypick 信息用于输出结果
        detail = fetch_detail(session, sid)
        result = detail.get("result", {}) if isinstance(detail, dict) else {}
        qs = result.get("queryString", {}) if isinstance(result, dict) else {}
        query = qs.get("query")
        day_buy_stock_num = qs.get("dayBuyStockNum")

        stock_map: dict[str, str] = {}
        if query and day_buy_stock_num:
            historypick = fetch_historypick(session, query=query, hold_num=day_buy_stock_num, trade_date=trade_date)
            stocks = _get_historypick_stocks(historypick, context="replay", sid=sid, non_dict_verb="提示")
            if stocks:
                stock_map = _extract_mainboard_stock_map(stocks)

        max_annual_yield = _get_max_annual_yield(backtestresult)

        final_rows.append(
            {
                "sort_type": "replay",
                "query": query,
                "property_id": sid,
                "trade_date": trade_date,
                "dayBuyStockNum": day_buy_stock_num,
                "fallIncome": result.get("fallIncome"),
                "upperIncome": result.get("upperIncome"),
                "lowerIncome": result.get("lowerIncome"),
                "maxAnnualYield": max_annual_yield,
                "maxWinRate": max_win_rate,
                "mainboardStockMap": stock_map,
            }
        )

        compare_rows.append(
            {
                "strategy_id": sid,
                "sort_type": sort_type,
                "page": page,
                "list_winRate": list_wr,
                "old_prefilter_pass": old_prefilter_pass,
                "new_prefilter_pass": new_prefilter_pass,
                "replay_mwr0": mwr0,
                "replay_mwr0_num": mwr0_num,
                "replay_mwr1": mwr1,
                "replay_mwr1_num": mwr1_num,
                "replay_pass": True,
                "replay_reason": "pass",
            }
        )

    return final_rows, compare_rows


def _collect_rows_for_strategy_ids(
    session: requests.Session,
    *,
    strategy_ids: list[int],
    trade_date: str,
    mwr0_min: float,
    mwr1_max: float,
    scanned_strategy_records: list[dict[str, Any]] | None = None,
    seen_strategy_ids: set[int] | None = None,
    skip_seen_strategy_ids: bool = True,
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []

    if seen_strategy_ids is None:
        seen_strategy_ids = set()

    for idx, sid in enumerate(strategy_ids, start=1):
        try:
            sid_int = int(sid)
        except Exception:
            continue

        if skip_seen_strategy_ids:
            if sid_int in seen_strategy_ids:
                continue
            seen_strategy_ids.add(sid_int)

        if idx % 3 == 0:
            print(f"进度[manual]: {idx}/{len(strategy_ids)} sid={sid_int}")

        if scanned_strategy_records is not None:
            scanned_strategy_records.append(
                {
                    "mode": "manual",
                    "sort_type": "manual",
                    "page": None,
                    "strategy_id": sid_int,
                    "list_winRate": None,
                    "prefilter_pass": None,
                }
            )

        detail = fetch_detail(session, sid_int)
        result = detail.get("result", {}) if isinstance(detail, dict) else {}
        qs = result.get("queryString", {}) if isinstance(result, dict) else {}

        query = qs.get("query")
        day_buy_stock_num = qs.get("dayBuyStockNum")

        stock_map: dict[str, str] = {}
        if query and day_buy_stock_num:
            historypick = fetch_historypick(session, query=query, hold_num=day_buy_stock_num, trade_date=trade_date)
            stocks = _get_historypick_stocks(historypick, context="manual", sid=sid, non_dict_verb="提示")
            if stocks:
                stock_map = _extract_mainboard_stock_map(stocks)

        backtestresult = fetch_backtestresult(session, sid_int)
        max_annual_yield = _get_max_annual_yield(backtestresult)
        max_win_rate = _get_max_win_rate(backtestresult)

        mwr0, mwr0_num, mwr1, mwr1_num = _parse_max_win_rate(max_win_rate)
        if mwr0_num is None or mwr0_num < mwr0_min:
            print(f"跳过[manual]：胜率不达标 sid={sid_int} mwr0={mwr0} min={mwr0_min:g}")
            continue

        if mwr1_num is None or mwr1_num > mwr1_max:
            print(f"跳过[manual]：持股周期超限 sid={sid_int} mwr1={mwr1} max={mwr1_max:g}")
            continue

        merged: dict[str, Any] = {
            "sort_type": "manual",
            "query": query,
            "property_id": sid_int,
            "trade_date": trade_date,
            "dayBuyStockNum": day_buy_stock_num,
            "fallIncome": result.get("fallIncome"),
            "upperIncome": result.get("upperIncome"),
            "lowerIncome": result.get("lowerIncome"),
            "maxAnnualYield": max_annual_yield,
            "maxWinRate": max_win_rate,
            "mainboardStockMap": stock_map,
        }
        final_rows.append(merged)

    return final_rows


def _collect_rows_for_sort(
    session: requests.Session,
    *,
    sort_type: str,
    keyword: str,
    trade_date: str,
    min_output_count: int,
    max_pages: int,
    page_num: int,
    mwr0_min: float,
    mwr1_max: float,
    scanned_strategy_records: list[dict[str, Any]] | None = None,
    seen_strategy_ids: set[int] | None = None,
    skip_seen_strategy_ids: bool = True,
    enable_list_winrate_prefilter: bool = True,
) -> list[dict[str, Any]]:
    scanned_strategies = 0
    matched_filters = 0
    filtered_out_mainboard = 0
    filtered_out_mwr0 = 0
    filtered_out_list_winrate = 0

    final_rows: list[dict[str, Any]] = []

    if seen_strategy_ids is None:
        seen_strategy_ids = set()

    page = 1
    while page <= max_pages and len(final_rows) < min_output_count:
        lst = fetch_list(session, page=page, page_num=page_num, sort_type=sort_type, keyword=keyword)
        items = lst.get("result", {}).get("list", []) if isinstance(lst, dict) else []
        if not items:
            break

        for it in items:
            if len(final_rows) >= min_output_count:
                break

            prop = it.get("property", {}) if isinstance(it, dict) else {}
            sid = prop.get("id")
            if sid is None:
                continue

            try:
                sid_int = int(sid)
            except Exception:
                continue

            # 去重：同一个策略如果之前查询过（跨页/跨 sort_type），直接跳过
            if skip_seen_strategy_ids:
                if sid_int in seen_strategy_ids:
                    continue
                seen_strategy_ids.add(sid_int)

            scanned_strategies += 1

            # 预过滤：先用列表页自带的 winRate 做一次过滤，减少后续 detail/backtestresult/historypick 请求
            list_wr = prop.get("winRate")
            list_wr_num = _parse_win_rate_ratio(list_wr)

            prefilter_pass = not (list_wr_num is None or list_wr_num < mwr0_min)
            if scanned_strategy_records is not None:
                scanned_strategy_records.append(
                    {
                        "mode": "scan",
                        "sort_type": sort_type,
                        "page": page,
                        "strategy_id": sid_int,
                        "list_winRate": list_wr,
                        "prefilter_pass": prefilter_pass,
                    }
                )

            if enable_list_winrate_prefilter:
                if not prefilter_pass:
                    filtered_out_list_winrate += 1
                    continue

            if scanned_strategies % 5 == 0:
                print(
                    f"进度[{sort_type}]: page={page}/{max_pages}, 已扫描策略={scanned_strategies}, "
                    f"mwr0>={mwr0_min:g} 且 mwr1<={mwr1_max:g} 命中={matched_filters}, "
                    f"已收集输出={len(final_rows)}/{min_output_count}"
                )

            detail = fetch_detail(session, sid_int)
            result = detail.get("result", {}) if isinstance(detail, dict) else {}
            qs = result.get("queryString", {}) if isinstance(result, dict) else {}

            query = qs.get("query")
            day_buy_stock_num = qs.get("dayBuyStockNum")
            if not query or not day_buy_stock_num:
                continue

            historypick = fetch_historypick(session, query=query, hold_num=day_buy_stock_num, trade_date=trade_date)

            stocks = _get_historypick_stocks(historypick, context=str(sort_type), sid=sid, non_dict_verb="跳过")

            if not stocks:
                continue

            stock_map = _extract_mainboard_stock_map(stocks)
            if not stock_map:
                filtered_out_mainboard += 1
                continue

            backtestresult = fetch_backtestresult(session, sid_int)
            max_annual_yield = _get_max_annual_yield(backtestresult)
            max_win_rate = _get_max_win_rate(backtestresult)

            mwr0, mwr0_num, mwr1, mwr1_num = _parse_max_win_rate(max_win_rate)
            if mwr0_num is None or mwr0_num < mwr0_min:
                filtered_out_mwr0 += 1
                continue

            if mwr1_num is None or mwr1_num > mwr1_max:
                continue

            matched_filters += 1

            merged = dict(historypick)
            merged["sort_type"] = sort_type
            merged["query"] = query
            merged["property_id"] = sid_int
            merged["trade_date"] = trade_date
            merged["dayBuyStockNum"] = day_buy_stock_num
            merged["fallIncome"] = result.get("fallIncome")
            merged["upperIncome"] = result.get("upperIncome")
            merged["lowerIncome"] = result.get("lowerIncome")
            merged["maxAnnualYield"] = max_annual_yield
            merged["maxWinRate"] = max_win_rate
            merged["mainboardStockMap"] = stock_map

            final_rows.append(merged)

        print(
            f"进度[{sort_type}]: page={page}/{max_pages} 完成, 本页策略={len(items)}, 已扫描策略={scanned_strategies}, "
            f"mwr0>={mwr0_min:g} 且 mwr1<={mwr1_max:g} 命中={matched_filters}, "
            f"已收集输出={len(final_rows)}/{min_output_count}"
        )

        page += 1

    print(
        f"汇总[{sort_type}]: 已扫描策略={scanned_strategies}, mwr0>={mwr0_min:g} 且 mwr1<={mwr1_max:g} 命中={matched_filters}, "
        f"列表胜率预过滤剔除={filtered_out_list_winrate}, 胜率过滤剔除={filtered_out_mwr0}, 主板过滤剔除={filtered_out_mainboard}, "
        f"最终输出={len(final_rows)} (目标 min_count={min_output_count})"
    )

    if len(final_rows) < min_output_count:
        print(f"提示[{sort_type}]: 未达到 min_count，已收集 {len(final_rows)}/{min_output_count}")

    return final_rows


def main(
    *,
    trade_date: str | None = None,
    # trade_date: str | None = "2026-01-27",
    min_count: int = 5,
    max_pages: int = 100,
    page_num: int = 10,
    # gain(本月), winRate(成功率), profit(年化收益), hot(热度)
    sort_types: list[str] | None = None,
    # sort_types: list[str] | None = ["winRate"],
    keyword: str = "",
) -> None:
    # =============================
    # 配置优先：读取 pc_config.toml（同目录）
    # =============================
    cfg = _load_config()

    cfg_trade_date = _cfg_get(cfg, ["run", "trade_date"], trade_date)
    if not cfg_trade_date:
        cfg_trade_date = None
    trade_date = cfg_trade_date

    cfg_min_count = _cfg_get(cfg, ["run", "min_count"], min_count)
    cfg_max_pages = _cfg_get(cfg, ["run", "max_pages"], max_pages)
    cfg_page_num = _cfg_get(cfg, ["run", "page_num"], page_num)

    sort_types_cfg = _cfg_get(cfg, ["run", "sort_types"], sort_types)
    keyword_cfg = _cfg_get(cfg, ["run", "keyword"], keyword)

    mwr0_min_cfg = _cfg_get(cfg, ["filters", "mwr0_min"], 0.8)
    mwr1_max_cfg = _cfg_get(cfg, ["filters", "mwr1_max"], 5.0)
    enable_list_winrate_prefilter = bool(_cfg_get(cfg, ["filters", "enable_list_winrate_prefilter"], True))
    skip_seen_strategy_ids = bool(_cfg_get(cfg, ["filters", "skip_seen_strategy_ids"], True))

    expand_stocks = bool(_cfg_get(cfg, ["output", "expand_stocks"], False))
    output_scanned_strategy_ids_csv = bool(_cfg_get(cfg, ["output", "output_scanned_strategy_ids_csv"], False))
    output_replay_compare_csv = bool(_cfg_get(cfg, ["output", "output_replay_compare_csv"], False))

    replay_from_scanned_csv_path = str(_cfg_get(cfg, ["replay", "from_scanned_csv_path"], "") or "").strip()
    replay_from_scanned_csv_path = replay_from_scanned_csv_path or None

    use_manual_strategy_ids = bool(_cfg_get(cfg, ["manual", "use_manual_strategy_ids"], False))
    manual_strategy_ids = _cfg_get(cfg, ["manual", "strategy_ids"], [])
    if not isinstance(manual_strategy_ids, list):
        manual_strategy_ids = []

    # 交易日（None 表示当天）
    if trade_date is None:
        trade_date = time.strftime("%Y-%m-%d")

    MIN_OUTPUT_COUNT = int(cfg_min_count)
    MAX_PAGES = int(cfg_max_pages)
    PAGE_NUM = int(cfg_page_num)

    MWR1_MAX = float(mwr1_max_cfg)
    MWR0_MIN = float(mwr0_min_cfg)

    OUTPUT_CSV_FILENAME = time.strftime("%Y%m%d_%H%M%S") + ".csv"
    EXPAND_STOCKS = expand_stocks
    OUTPUT_SCANNED_STRATEGY_IDS_CSV = output_scanned_strategy_ids_csv
    REPLAY_FROM_SCANNED_CSV_PATH = replay_from_scanned_csv_path
    OUTPUT_REPLAY_COMPARE_CSV = output_replay_compare_csv
    USE_MANUAL_STRATEGY_IDS = use_manual_strategy_ids
    MANUAL_STRATEGY_IDS: list[int] = [int(x) for x in manual_strategy_ids if str(x).strip().isdigit()]

    with requests.Session() as session:
        session.headers.update(_build_headers())

        # 访问网页首页一次，让 Cookie/会话更像“正常用户”
        try:
            session.get("https://backtest.10jqka.com.cn/", timeout=15)
        except Exception:
            pass
        _human_sleep(1.0, 2.8)

        final_rows: list[dict[str, Any]] = []
        scanned_strategy_records: list[dict[str, Any]] = []
        replay_compare_rows: list[dict[str, Any]] = []
        seen_strategy_ids: set[int] = set()
        if REPLAY_FROM_SCANNED_CSV_PATH:
            scanned_records = _read_scanned_strategy_csv(REPLAY_FROM_SCANNED_CSV_PATH)
            final_rows, replay_compare_rows = _replay_from_scanned_records(
                session,
                scanned_records=scanned_records,
                trade_date=trade_date,
                mwr0_min=MWR0_MIN,
                mwr1_max=MWR1_MAX,
                skip_seen_strategy_ids=skip_seen_strategy_ids,
                enable_list_winrate_prefilter=enable_list_winrate_prefilter,
            )
        else:
            strategy_ids = MANUAL_STRATEGY_IDS if USE_MANUAL_STRATEGY_IDS else None
            if strategy_ids:
                final_rows = _collect_rows_for_strategy_ids(
                    session,
                    strategy_ids=strategy_ids,
                    trade_date=trade_date,
                    mwr0_min=MWR0_MIN,
                    mwr1_max=MWR1_MAX,
                    scanned_strategy_records=scanned_strategy_records,
                    seen_strategy_ids=seen_strategy_ids,
                    skip_seen_strategy_ids=skip_seen_strategy_ids,
                )
            else:
                effective_sort_types = sort_types_cfg or sort_types or ["winRate","gain", "profit", "hot"]
                for st in effective_sort_types:
                    rows = _collect_rows_for_sort(
                        session,
                        sort_type=st,
                        keyword=str(keyword_cfg or ""),
                        trade_date=trade_date,
                        min_output_count=MIN_OUTPUT_COUNT,
                        max_pages=MAX_PAGES,
                        page_num=PAGE_NUM,
                        mwr0_min=MWR0_MIN,
                        mwr1_max=MWR1_MAX,
                        scanned_strategy_records=scanned_strategy_records,
                        seen_strategy_ids=seen_strategy_ids,
                        skip_seen_strategy_ids=skip_seen_strategy_ids,
                        enable_list_winrate_prefilter=enable_list_winrate_prefilter,
                    )
                    final_rows.extend(rows)

        # =============================
        # 输出：控制台 + CSV
        # =============================
        import os

        # 整理为可写入 CSV 的结构（final_rows 已是最终口径）
        csv_rows: list[dict[str, Any]] = []

        for r in final_rows:
            st = r.get("sort_type")
            q = r.get("query")
            pid = r.get("property_id")
            day_buy = r.get("dayBuyStockNum")
            upper = r.get("upperIncome")
            fall = r.get("fallIncome")
            lower = r.get("lowerIncome")
            may = r.get("maxAnnualYield")
            mwr = r.get("maxWinRate")

            stock_map = r.get("mainboardStockMap") if isinstance(r, dict) else None
            if not isinstance(stock_map, dict):
                stock_map = {}

            may0, may1 = _split_first_two(may)
            mwr0, _, mwr1, _ = _parse_max_win_rate(mwr)

            # 控制台输出
            print(f"策略：{q}")
            print(f"策略ID：{pid}")
            print(f"策略地址：https://backtest.10jqka.com.cn/backtest/app.html#/strategysquare/{pid}")
            print(f"排序方式: {st}")
            print(f"交易日: {trade_date}")
            print(f"单日买入数: {day_buy}")
            print(f"止盈: 收益率 ≥ {upper} % 时坚定持有;直到最高收益回落 ≤ {fall} %")
            print(f"止损: 收益率 ≤ -{lower} %")
            print(f"最大预期: 年化收益率{may0}, 回测持股周期 {may1} 天")
            print(f"最大胜率: 胜率 {mwr0}, 回测持股周期 {mwr1} 天")
            print(f"入选股票：{stock_map}")
            print("-" * 60)

            base_row: dict[str, Any] = {
                "sort_type": st,
                "trade_date": trade_date,
                "query": q,
                "property_id": pid,
                "url": f"https://backtest.10jqka.com.cn/backtest/app.html#/strategysquare/{pid}",
                "dayBuyStockNum": day_buy,
                "upperIncome": upper,
                "fallIncome": fall,
                "lowerIncome": lower,
                "maxAnnualYield0": may0,
                "maxAnnualYield1": may1,
                "maxWinRate0": mwr0,
                "maxWinRate1": mwr1,
            }

            if EXPAND_STOCKS:
                # 展开股票为多列（stock_1_code, stock_1_name ...）
                for i, (code, name) in enumerate(stock_map.items(), start=1):
                    base_row[f"stock_{i}_code"] = code
                    base_row[f"stock_{i}_name"] = name
            else:
                base_row["stocks"] = json.dumps(stock_map, ensure_ascii=False)

            csv_rows.append(base_row)

        # 写 CSV
        if csv_rows:
            out_dir = _get_csv_output_dir()
            out_path = str(out_dir / OUTPUT_CSV_FILENAME)
            _write_dict_csv(out_path, csv_rows)
            print(f"CSV 已输出: {out_path}")

            if OUTPUT_SCANNED_STRATEGY_IDS_CSV and scanned_strategy_records:
                scanned_filename = _derive_csv_filename(OUTPUT_CSV_FILENAME, "_scanned_strategy_ids")
                scanned_path = str(out_dir / scanned_filename)

                scanned_fieldnames = ["mode", "sort_type", "page", "strategy_id", "list_winRate", "prefilter_pass"]
                _write_dict_csv(scanned_path, scanned_strategy_records, fieldnames=scanned_fieldnames)
                print(f"扫描到的 strategy_id 列表已输出: {scanned_path}")

            if OUTPUT_REPLAY_COMPARE_CSV and replay_compare_rows:
                replay_filename = _derive_csv_filename(OUTPUT_CSV_FILENAME, "_replay_compare")
                replay_path = str(out_dir / replay_filename)

                replay_fieldnames = [
                    "strategy_id",
                    "sort_type",
                    "page",
                    "list_winRate",
                    "old_prefilter_pass",
                    "new_prefilter_pass",
                    "replay_mwr0",
                    "replay_mwr0_num",
                    "replay_mwr1",
                    "replay_mwr1_num",
                    "replay_pass",
                    "replay_reason",
                ]
                _write_dict_csv(replay_path, replay_compare_rows, fieldnames=replay_fieldnames)
                print(f"回放对比表已输出: {replay_path}")

            # 统计：每只股票出现次数（按 sort_type 去重：同一 sort_type 内只算一次）
            from collections import defaultdict

            stock_to_sort_types: dict[tuple[str, str], set[str]] = defaultdict(set)
            for row in csv_rows:
                st = str(row.get("sort_type") or "")
                stocks_json = row.get("stocks")
                if isinstance(stocks_json, str) and stocks_json.strip():
                    try:
                        sm = json.loads(stocks_json)
                    except Exception:
                        sm = None
                    if isinstance(sm, dict):
                        for code, name in sm.items():
                            if not code:
                                continue
                            stock_to_sort_types[(str(code), str(name))].add(st)
                    continue

                # 兼容 EXPAND_STOCKS=True 的情况
                for k, v in row.items():
                    if not (isinstance(k, str) and k.endswith("_code")):
                        continue
                    code = str(v) if v is not None else ""
                    if not code:
                        continue
                    name = str(row.get(k.replace("_code", "_name")) or "")
                    stock_to_sort_types[(code, name)].add(st)

            stock_count_rows: list[dict[str, Any]] = []
            for (code, name), sts in stock_to_sort_types.items():
                sts_sorted = sorted(s for s in sts if s)
                stock_count_rows.append(
                    {
                        "stock_code": code,
                        "stock_name": name,
                        "count": len(sts_sorted),
                        "sort_types": ",".join(sts_sorted),
                    }
                )

            stock_count_rows.sort(key=lambda r: (-int(r.get("count") or 0), str(r.get("stock_code") or "")))

            counts_filename = _derive_csv_filename(OUTPUT_CSV_FILENAME, "_stock_counts")
            counts_path = str(out_dir / counts_filename)

            if stock_count_rows:
                _write_dict_csv(counts_path, stock_count_rows, fieldnames=["stock_code", "stock_name", "count", "sort_types"])
                print(f"股票出现次数统计已输出: {counts_path}")
            else:
                print("股票出现次数统计为空：没有解析到 stocks")
        else:
            print("没有可输出的结果（主板过滤后为空），未生成 CSV")


if __name__ == "__main__":
    main()
