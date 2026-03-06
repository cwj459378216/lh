# -*- coding: utf-8 -*-
"""
fetch_minute_data.py — 下载分钟线数据（供回测使用）

数据来源（可切换）：
  1. pytdx（推荐）：通达信行情协议，无限频限制，可获取较长历史
  2. AkShare（备选）：东方财富接口，仅保留最近 5 个交易日，有限频风险

使用方法：
    # 默认使用 pytdx 下载
    python -m tail_trade.fetch_minute_data --pool
    python -m tail_trade.fetch_minute_data --codes 000001.SZ,600519.SH

    # 强制使用 AkShare
    python -m tail_trade.fetch_minute_data --pool --source akshare --delay 3

    # 指定 pytdx 下载天数
    python -m tail_trade.fetch_minute_data --pool --days 30
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tail_trade import config
from tail_trade.data.loader import load_stock_pool, _detect_exchange
from tail_trade.logger import get_logger

log = get_logger("fetch_minute")


# ═══════════════════════════════════════════════════════════════════
# pytdx 数据源（推荐，无限频限制）
# ═══════════════════════════════════════════════════════════════════

_pytdx_api = None  # 复用连接


def _get_pytdx_api():
    """获取或创建 pytdx 连接（带自动选优服务器）。"""
    global _pytdx_api
    if _pytdx_api is not None:
        return _pytdx_api

    try:
        from pytdx.hq import TdxHq_API
    except ImportError:
        raise RuntimeError("请安装 pytdx: pip install pytdx")

    api = TdxHq_API()

    ip = config.PYTDX_BEST_IP
    port = config.PYTDX_BEST_PORT

    if not ip or port == 0:
        # 自动选优服务器
        log.info("pytdx: 正在自动选择最优服务器...")
        try:
            from pytdx.util.best_ip import select_best_ip
            best = select_best_ip()
            ip = best.get("ip", "119.147.212.81")
            port = int(best.get("port", 7709))
            log.info(f"pytdx: 最优服务器 {ip}:{port}")
        except Exception:
            ip, port = "119.147.212.81", 7709
            log.warning(f"pytdx: 自动选优失败，使用默认 {ip}:{port}")

    try:
        api.connect(ip, port)
        _pytdx_api = api
        log.info(f"pytdx: 已连接 {ip}:{port}")
    except Exception as e:
        raise RuntimeError(f"pytdx 连接失败: {e}")

    return _pytdx_api


def _code_to_pytdx_market(code: str) -> tuple[int, str]:
    """
    将 000001.SZ 格式转为 pytdx 的 (market, symbol) 格式。
    market: 0=深圳, 1=上海
    """
    parts = code.split(".")
    symbol = parts[0]
    suffix = parts[1].upper() if len(parts) > 1 else _detect_exchange(symbol)
    market = 0 if suffix == "SZ" else 1
    return market, symbol


def fetch_minute_data_pytdx(
    code: str,
    days: int = 10,
) -> pd.DataFrame | None:
    """
    通过 pytdx 下载单只股票的 1 分钟 K 线。

    pytdx 每次最多返回 800 条，通过分页获取更多数据。

    Args:
        code: 股票代码（如 000001.SZ）
        days: 需要获取的交易日数（约数，多取一些再截断）

    Returns:
        标准化 DataFrame 或 None
    """
    try:
        api = _get_pytdx_api()
    except Exception as e:
        log.error(f"pytdx 连接失败: {e}")
        return None

    market, symbol = _code_to_pytdx_market(code)

    # 每交易日约 240 根 1 分钟 K 线，每页 800 条
    total_bars_needed = days * 240
    page_size = 800
    all_data = []

    offset = 0
    while offset < total_bars_needed:
        try:
            bars = api.get_security_bars(
                category=8,     # 8 = 1 分钟 K 线
                market=market,
                code=symbol,
                start=offset,
                count=min(page_size, total_bars_needed - offset),
            )
        except Exception as e:
            log.warning(f"pytdx 分页请求失败 (offset={offset}): {e}")
            # 尝试重连一次
            try:
                _reconnect_pytdx()
                api = _get_pytdx_api()
                bars = api.get_security_bars(8, market, symbol, offset, page_size)
            except Exception:
                break

        if not bars:
            break
        all_data.extend(bars)
        offset += page_size

        # 如果返回数量少于请求数量，说明已到最早数据
        if len(bars) < page_size:
            break

    if not all_data:
        log.warning(f"[pytdx] {code} 无分钟数据")
        return None

    df = pd.DataFrame(all_data)

    # pytdx 返回列: datetime, open, close, high, low, vol, amount, ...
    # 标准化列名
    rename_map = {}
    if "vol" in df.columns:
        rename_map["vol"] = "volume"
    if "datetime" not in df.columns and "date" in df.columns:
        # pytdx 有时候用 year/month/day/hour/minute 拆分
        if "year" in df.columns:
            df["datetime"] = df.apply(
                lambda r: f"{r['year']:04d}-{r['month']:02d}-{r['day']:02d} "
                          f"{r['hour']:02d}:{r['minute']:02d}:00",
                axis=1,
            )

    if rename_map:
        df = df.rename(columns=rename_map)

    df["datetime"] = pd.to_datetime(df["datetime"])

    # 只保留标准列
    keep = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0
    df = df[keep]

    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"[pytdx] {code} 获取 {len(df)} 条分钟数据")
    return df


def _reconnect_pytdx():
    """断线重连 pytdx。"""
    global _pytdx_api
    if _pytdx_api:
        try:
            _pytdx_api.disconnect()
        except Exception:
            pass
    _pytdx_api = None
    _get_pytdx_api()


def close_pytdx():
    """关闭 pytdx 连接。"""
    global _pytdx_api
    if _pytdx_api:
        try:
            _pytdx_api.disconnect()
        except Exception:
            pass
        _pytdx_api = None


# ═══════════════════════════════════════════════════════════════════
# AkShare 数据源（备选，有限频风险）
# ═══════════════════════════════════════════════════════════════════

def fetch_minute_data_akshare(
    code: str,
    period: str = "1",
    adjust: str = "",
    max_retries: int = 2,
) -> pd.DataFrame | None:
    """
    通过 AkShare 下载单只股票的分钟 K 线（带重试）。

    Args:
        code:        股票代码（如 000001.SZ → 只取纯数字 000001）
        period:      K 线周期，'1'=1分钟, '5'=5分钟, '15', '30', '60'
        adjust:      复权类型，''=不复权, 'qfq'=前复权, 'hfq'=后复权
        max_retries: 失败重试次数

    Returns:
        DataFrame 或 None
    """
    try:
        import akshare as ak
    except ImportError:
        log.error("请安装 akshare: pip install akshare")
        return None

    # 提取纯数字代码
    symbol = code.split(".")[0] if "." in code else code

    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                adjust=adjust,
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            log.warning(f"[akshare] {code} 第 {attempt} 次请求失败: {e}")
            if attempt < max_retries:
                delay = 3 * attempt
                log.info(f"  等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                log.error(f"[akshare] {code} 下载失败，已重试 {max_retries} 次")
                return None

    if df is None or df.empty:
        log.warning(f"[akshare] {code} 无分钟数据")
        return None

    # 标准化列名
    col_map = {
        "时间": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=col_map)

    keep = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0
    df = df[keep]

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ═══════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════

def fetch_minute_data(
    code: str,
    source: str | None = None,
    days: int = 10,
    period: str = "1",
) -> pd.DataFrame | None:
    """
    统一分钟线数据下载入口，自动选择数据源。

    Args:
        code:   股票代码
        source: 数据源 'pytdx' 或 'akshare'，None 则使用 config
        days:   pytdx 模式下载天数
        period: akshare 模式 K 线周期

    Returns:
        标准化 DataFrame 或 None
    """
    src = (source or config.MINUTE_DATA_SOURCE).lower().strip()

    if src == "pytdx":
        df = fetch_minute_data_pytdx(code, days=days)
        if df is not None and not df.empty:
            return df
        log.warning(f"[{code}] pytdx 失败，回退到 akshare")
        return fetch_minute_data_akshare(code, period=period)

    elif src == "akshare":
        return fetch_minute_data_akshare(code, period=period)

    else:
        log.warning(f"未知数据源 '{src}'，使用 pytdx")
        return fetch_minute_data_pytdx(code, days=days)


def save_minute_csv(df: pd.DataFrame, code: str, output_dir: str) -> str:
    """保存分钟 K 线到 CSV（自动合并已有数据）。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{code}.csv")

    # 如果文件已存在，合并去重
    if os.path.isfile(path):
        existing = pd.read_csv(path, encoding="utf-8-sig")
        existing["datetime"] = pd.to_datetime(existing["datetime"])
        df = pd.concat([existing, df]).drop_duplicates(subset=["datetime"]).sort_values("datetime")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="下载分钟线数据（支持 pytdx / AkShare）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--codes", type=str, default="",
        help="股票代码列表（逗号分隔，如 000001.SZ,600519.SH）",
    )
    parser.add_argument(
        "--pool", action="store_true",
        help="使用 stock_pool.csv 中的全部股票",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        choices=["pytdx", "akshare"],
        help=f"数据源（默认 {config.MINUTE_DATA_SOURCE}）\n"
             f"  pytdx:   通达信协议，无限频，可获取长历史\n"
             f"  akshare: 东财接口，仅近 5 个交易日，有限频风险",
    )
    parser.add_argument(
        "--days", type=int, default=10,
        help="pytdx 模式下载的交易日数（默认 10）",
    )
    parser.add_argument(
        "--period", type=str, default="1",
        choices=["1", "5", "15", "30", "60"],
        help="akshare 模式 K 线周期（分钟），默认 1",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（默认 config.MINUTE_DATA_DIR）",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="每只股票之间的延时（秒）\n"
             "pytdx: 0.3 秒即可；akshare 建议 ≥ 2 秒",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or config.MINUTE_DATA_DIR
    source = args.source or config.MINUTE_DATA_SOURCE

    # 确定股票列表
    codes: list[str] = []
    if args.pool:
        pool = load_stock_pool()
        codes = list(pool.keys())
    elif args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        print("请指定 --codes 或 --pool")
        return

    log.info(f"准备下载 {len(codes)} 只股票的分钟 K 线")
    log.info(f"数据源: {source} | 输出目录: {output_dir}")
    if source == "pytdx":
        log.info(f"pytdx 模式: 下载最近 {args.days} 个交易日")

    success, fail = 0, 0
    for i, code in enumerate(codes, 1):
        log.info(f"[{i}/{len(codes)}] 下载 {code} ...")
        df = fetch_minute_data(code, source=source, days=args.days, period=args.period)
        if df is not None and not df.empty:
            path = save_minute_csv(df, code, output_dir)
            log.info(f"  ✅ 保存 {len(df)} 条 → {path}")
            success += 1
        else:
            fail += 1

        if i < len(codes):
            time.sleep(args.delay)

    # 清理 pytdx 连接
    close_pytdx()

    log.info(f"下载完成: 成功 {success}, 失败 {fail}")


if __name__ == "__main__":
    main()
