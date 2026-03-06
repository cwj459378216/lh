# -*- coding: utf-8 -*-
"""
loader.py — 数据加载模块

职责：
  1. 加载通达信日线 CSV
  2. 加载分钟线 CSV
  3. 加载 AkShare 实时快照（全市场批量，带限频保护）
  4. 加载股票池配置
"""

import csv
import os
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tail_trade import config
from tail_trade.logger import get_logger

log = get_logger("loader")


# ═══════════════════════════════════════════════════════════════════
# 股票池
# ═══════════════════════════════════════════════════════════════════

def load_stock_pool() -> dict[str, str]:
    """
    加载股票池，返回 {code: name} 字典。

    优先读取 stock_pool.csv，其次使用 config.STOCK_POOL_INLINE。
    CSV 格式：code,name  （首行表头）
    """
    pool: dict[str, str] = {}

    # 1) 从 CSV 文件读取
    csv_path = config.STOCK_POOL_FILE
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code") or "").strip()
                name = (row.get("name") or "").strip()
                if code:
                    pool[code] = name
        log.info(f"从 stock_pool.csv 加载 {len(pool)} 只股票")

    # 2) 合并 inline 配置
    for code, name in config.STOCK_POOL_INLINE.items():
        if code not in pool:
            pool[code] = name

    if not pool:
        log.warning("股票池为空！请在 stock_pool.csv 或 config.STOCK_POOL_INLINE 中配置")

    return pool


# ═══════════════════════════════════════════════════════════════════
# 日线数据
# ═══════════════════════════════════════════════════════════════════

def load_daily_csv(code: str, data_dir: str | None = None) -> pd.DataFrame:
    """
    读取单只股票的日线 CSV。

    Args:
        code:     股票代码，如 '000001.SZ'
        data_dir: CSV 所在目录，默认 config.DAILY_DATA_DIR

    Returns:
        DataFrame，列: trade_date(datetime), open, high, low, close, volume, amount
        按日期升序排列。
    """
    data_dir = data_dir or config.DAILY_DATA_DIR
    csv_path = os.path.join(data_dir, f"{code}.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"日线 CSV 不存在: {csv_path}")

    # 兼容 utf-8-sig / utf-8 / gbk
    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise RuntimeError(f"无法解析日线 CSV: {csv_path}")

    # 标准化列名
    df.columns = [c.strip().lower().lstrip("\ufeff") for c in df.columns]

    # 确保核心列存在
    required = {"trade_date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"日线 CSV 缺少列: {missing}  文件: {csv_path}")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("trade_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════
# 分钟线数据
# ═══════════════════════════════════════════════════════════════════

def load_minute_csv(code: str, data_dir: str | None = None) -> pd.DataFrame:
    """
    读取单只股票的 1 分钟线 CSV。

    CSV 格式：datetime,open,high,low,close,volume,amount
    datetime 列示例：2024-01-15 09:31:00

    Args:
        code:     股票代码，如 '000001.SZ'
        data_dir: 分钟线 CSV 目录，默认 config.MINUTE_DATA_DIR

    Returns:
        DataFrame，列: datetime(Timestamp), date(date), time(str HH:MM),
                       open, high, low, close, volume, amount
    """
    data_dir = data_dir or config.MINUTE_DATA_DIR
    csv_path = os.path.join(data_dir, f"{code}.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"分钟线 CSV 不存在: {csv_path}")

    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise RuntimeError(f"无法解析分钟线 CSV: {csv_path}")

    df.columns = [c.strip().lower().lstrip("\ufeff") for c in df.columns]

    # 兼容不同列名
    if "datetime" not in df.columns and "time" in df.columns and "date" in df.columns:
        df["datetime"] = df["date"].astype(str) + " " + df["time"].astype(str)
    elif "datetime" not in df.columns:
        # 尝试第一列作为 datetime
        df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")

    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_day_minute_bars(minute_df: pd.DataFrame, trade_date) -> pd.DataFrame:
    """
    从完整分钟线 DataFrame 中提取某一天的分钟 K 线。

    Args:
        minute_df:  完整的分钟线 DataFrame（由 load_minute_csv 返回）
        trade_date: 目标日期（date 对象或字符串 YYYY-MM-DD）

    Returns:
        该日的分钟 K 线 DataFrame
    """
    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()

    mask = minute_df["date"] == trade_date
    return minute_df.loc[mask].copy().reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# AkShare 实时快照（全市场批量获取）
# ═══════════════════════════════════════════════════════════════════

# 缓存：避免短时间内重复调用
_snapshot_cache: dict[str, Any] = {"data": None, "ts": 0.0}
# 连续失败计数（供外部读取）
snapshot_consecutive_failures: int = 0


def load_realtime_snapshot(force: bool = False) -> dict[str, dict[str, Any]]:
    """
    从 AkShare 获取 A 股全市场快照，返回 {CODE.SZ/SH: {...}} 字典。

    增强功能：
      - 内存缓存（TTL 由 config.SNAPSHOT_CACHE_TTL 控制）
      - 请求失败自动指数退避重试（最多 config.SNAPSHOT_MAX_RETRIES 次）
      - 连续失败计数，供监控模块读取告警

    返回值每只股票包含：
        name, price, open, high, low, prev_close, volume, amount, change_pct
    """
    global snapshot_consecutive_failures

    now = time.time()
    cache_ttl = config.SNAPSHOT_CACHE_TTL
    if not force and _snapshot_cache["data"] and (now - _snapshot_cache["ts"] < cache_ttl):
        return _snapshot_cache["data"]

    try:
        import akshare as ak
    except ImportError:
        raise RuntimeError("请安装 akshare: pip install akshare")

    # ── 指数退避重试 ──
    max_retries = config.SNAPSHOT_MAX_RETRIES
    base_delay = config.SNAPSHOT_RETRY_BASE_DELAY
    df = None
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"正在获取 AkShare 全市场快照 (第 {attempt}/{max_retries} 次)...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 0:
                break  # 成功
            last_err = RuntimeError("快照数据为空")
        except Exception as e:
            last_err = e
            log.warning(f"快照请求失败 (第 {attempt} 次): {e}")

        if attempt < max_retries:
            delay = base_delay * (2 ** (attempt - 1))  # 5s → 10s → 20s
            log.info(f"等待 {delay:.0f} 秒后重试...")
            time.sleep(delay)

    if df is None or len(df) == 0:
        snapshot_consecutive_failures += 1
        raise RuntimeError(
            f"AkShare 快照失败（已重试 {max_retries} 次，"
            f"连续失败 {snapshot_consecutive_failures} 轮）: {last_err}"
        )

    # 成功，重置失败计数
    snapshot_consecutive_failures = 0

    result: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        code_raw = str(row.get("代码", "")).strip()
        if len(code_raw) != 6 or not code_raw.isdigit():
            continue

        # 推断交易所
        suffix = _detect_exchange(code_raw)
        if not suffix:
            continue

        symbol = f"{code_raw}.{suffix}"
        try:
            result[symbol] = {
                "name": str(row.get("名称", "")),
                "price": _safe_float(row.get("最新价")),
                "open": _safe_float(row.get("今开")),
                "high": _safe_float(row.get("最高")),
                "low": _safe_float(row.get("最低")),
                "prev_close": _safe_float(row.get("昨收")),
                "volume": _safe_float(row.get("成交量")),       # 手
                "amount": _safe_float(row.get("成交额")),       # 元
                "change_pct": _safe_float(row.get("涨跌幅")),   # 百分比，如 3.21
            }
        except Exception:
            continue

    _snapshot_cache["data"] = result
    _snapshot_cache["ts"] = time.time()

    log.info(f"快照已更新，共 {len(result)} 只股票")
    return result


def _detect_exchange(code: str) -> str:
    """根据纯数字代码推断交易所后缀。"""
    if code.startswith(("000", "001", "002", "003", "200", "300", "301")):
        return "SZ"
    if code.startswith(("600", "601", "603", "605", "688", "900")):
        return "SH"
    return ""


def _safe_float(val) -> float | None:
    """安全转换为浮点数。"""
    if val is None:
        return None
    try:
        v = float(val)
        return v if v == v else None  # 排除 NaN
    except (ValueError, TypeError):
        return None
