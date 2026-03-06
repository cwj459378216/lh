# -*- coding: utf-8 -*-
"""
backtest.py — 历史回测模块

功能：
  1. 遍历历史交易日，逐日判断尾盘买入信号
  2. 统计触发次数、胜率、平均收益
  3. 支持对每次触发调用 notifier 打印/发送消息
  4. 输出详细回测报告

使用方法：
    python -m tail_trade.backtest --code 000001.SZ --start 2024-01-01 --end 2024-12-31

依赖：日线 CSV + 分钟线 CSV
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tail_trade import config
from tail_trade.data.loader import (
    load_daily_csv,
    load_minute_csv,
    load_stock_pool,
    get_day_minute_bars,
)
from tail_trade.strategy import (
    TailStrategyParams,
    SignalResult,
    check_tail_signal_minute,
)
from tail_trade.notifier import notify_signal
from tail_trade.logger import get_logger

log = get_logger("backtest")


# ═══════════════════════════════════════════════════════════════════
# 单只股票回测
# ═══════════════════════════════════════════════════════════════════

def backtest_single(
    code: str,
    name: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    params: TailStrategyParams | None = None,
    notify: bool = False,
    daily_dir: str | None = None,
    minute_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    对单只股票执行历史回测。

    Args:
        code:       股票代码（如 000001.SZ）
        name:       股票名称
        start_date: 起始日期 YYYY-MM-DD
        end_date:   结束日期 YYYY-MM-DD
        params:     策略参数
        notify:     是否在触发时发送企业微信通知
        daily_dir:  日线数据目录
        minute_dir: 分钟线数据目录

    Returns:
        信号记录列表，每条包含 SignalResult.to_dict() + next_day_return
    """
    p = params or TailStrategyParams()
    records: list[dict[str, Any]] = []

    # ── 加载数据 ──
    try:
        daily_df = load_daily_csv(code, daily_dir)
    except FileNotFoundError:
        log.warning(f"[{code}] 日线数据不存在，跳过")
        return records

    try:
        minute_df = load_minute_csv(code, minute_dir)
    except FileNotFoundError:
        log.warning(f"[{code}] 分钟线数据不存在，跳过")
        return records

    # ── 日期过滤 ──
    if start_date:
        start_dt = pd.Timestamp(start_date)
        daily_df = daily_df[daily_df["trade_date"] >= start_dt]
    if end_date:
        end_dt = pd.Timestamp(end_date)
        daily_df = daily_df[daily_df["trade_date"] <= end_dt]

    if daily_df.empty:
        log.info(f"[{code}] 指定日期范围内无数据")
        return records

    daily_df = daily_df.reset_index(drop=True)
    log.info(
        f"[{code}] {name} 回测区间: "
        f"{daily_df['trade_date'].iloc[0].strftime('%Y-%m-%d')} ~ "
        f"{daily_df['trade_date'].iloc[-1].strftime('%Y-%m-%d')}  "
        f"共 {len(daily_df)} 个交易日"
    )

    # ── 逐日遍历 ──
    for i in range(1, len(daily_df)):
        prev_row = daily_df.iloc[i - 1]
        curr_row = daily_df.iloc[i]
        trade_date = curr_row["trade_date"]
        trade_date_str = trade_date.strftime("%Y-%m-%d")
        prev_close = prev_row["close"]

        # 获取当日分钟 K 线
        day_bars = get_day_minute_bars(minute_df, trade_date.date())
        if day_bars.empty:
            continue  # 无分钟数据则跳过

        # ── 策略判定 ──
        signal = check_tail_signal_minute(
            day_minute_bars=day_bars,
            prev_close=prev_close,
            code=code,
            name=name,
            trade_date=trade_date_str,
            params=p,
        )

        if not signal.triggered:
            continue

        # ── 计算次日收益 ──
        next_day_return = None
        if i + 1 < len(daily_df):
            next_close = daily_df.iloc[i + 1]["close"]
            buy_price = curr_row["close"]  # 以收盘价买入
            if buy_price > 0:
                next_day_return = (next_close - buy_price) / buy_price

        rec = signal.to_dict()
        rec["next_day_return"] = (
            f"{next_day_return * 100:+.2f}%" if next_day_return is not None else "N/A"
        )
        rec["next_day_return_raw"] = next_day_return
        records.append(rec)

        # ── 可选发送通知 ──
        if notify:
            notify_signal(
                code=code,
                name=name,
                price=signal.price,
                daily_gain=signal.daily_gain,
                tail_rise=signal.tail_rise,
                tail_volume_ratio=signal.tail_volume_ratio,
                extra={"回测日期": trade_date_str},
            )

    log.info(f"[{code}] 回测完成，触发信号 {len(records)} 次")
    return records


# ═══════════════════════════════════════════════════════════════════
# 多只股票批量回测
# ═══════════════════════════════════════════════════════════════════

def backtest_batch(
    stock_pool: dict[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    params: TailStrategyParams | None = None,
    notify: bool = False,
) -> pd.DataFrame:
    """
    对股票池中所有股票进行批量回测。

    Returns:
        所有触发信号的 DataFrame
    """
    if stock_pool is None:
        stock_pool = load_stock_pool()

    all_records: list[dict[str, Any]] = []

    for code, name in stock_pool.items():
        records = backtest_single(
            code=code,
            name=name,
            start_date=start_date,
            end_date=end_date,
            params=params,
            notify=notify,
        )
        all_records.extend(records)

    if not all_records:
        log.info("回测结束，无信号触发")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    return df


# ═══════════════════════════════════════════════════════════════════
# 回测统计报告
# ═══════════════════════════════════════════════════════════════════

def print_backtest_report(result_df: pd.DataFrame) -> None:
    """
    打印回测统计报告。

    指标：
      - 总触发次数
      - 胜率（次日收益 > 0 的比例）
      - 平均次日收益
      - 最大单次收益 / 亏损
      - 累计收益（简单累加）
    """
    if result_df.empty:
        print("\n═══ 回测报告 ═══")
        print("无信号触发")
        return

    total = len(result_df)

    # 过滤有次日收益的记录
    has_return = result_df["next_day_return_raw"].notna()
    valid = result_df.loc[has_return, "next_day_return_raw"]

    if valid.empty:
        print("\n═══ 回测报告 ═══")
        print(f"触发次数: {total}")
        print("无可计算的次日收益数据")
        return

    win_count = (valid > 0).sum()
    win_rate = win_count / len(valid) * 100
    avg_return = valid.mean() * 100
    max_gain = valid.max() * 100
    max_loss = valid.min() * 100
    cumulative = valid.sum() * 100

    print("\n" + "═" * 50)
    print("          📊 尾盘策略回测报告")
    print("═" * 50)
    print(f"  总触发次数 :  {total}")
    print(f"  有效统计   :  {len(valid)}（有次日收益数据）")
    print(f"  胜率       :  {win_rate:.1f}%（{win_count}/{len(valid)}）")
    print(f"  平均次日收益:  {avg_return:+.2f}%")
    print(f"  最大单次盈利:  {max_gain:+.2f}%")
    print(f"  最大单次亏损:  {max_loss:+.2f}%")
    print(f"  累计收益   :  {cumulative:+.2f}%（简单累加）")
    print("═" * 50)

    # 按股票汇总
    if "code" in result_df.columns:
        print("\n── 按股票汇总 ──")
        grouped = result_df.groupby("code")
        for code, grp in grouped:
            cnt = len(grp)
            g_valid = grp["next_day_return_raw"].dropna()
            if len(g_valid) > 0:
                wr = (g_valid > 0).sum() / len(g_valid) * 100
                ar = g_valid.mean() * 100
                print(f"  {code}: 触发 {cnt} 次 | 胜率 {wr:.0f}% | 平均收益 {ar:+.2f}%")
            else:
                print(f"  {code}: 触发 {cnt} 次 | 无次日数据")

    # 按月份汇总
    if "trade_date" in result_df.columns:
        print("\n── 按月份汇总 ──")
        result_df["_month"] = pd.to_datetime(result_df["trade_date"]).dt.to_period("M")
        for month, grp in result_df.groupby("_month"):
            cnt = len(grp)
            g_valid = grp["next_day_return_raw"].dropna()
            if len(g_valid) > 0:
                ar = g_valid.mean() * 100
                print(f"  {month}: {cnt} 次 | 均收益 {ar:+.2f}%")
        result_df.drop(columns=["_month"], inplace=True, errors="ignore")

    print()


# ═══════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="尾盘买入策略 — 历史回测",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--code", type=str, default="",
        help="单只股票代码，如 000001.SZ（留空则使用股票池）",
    )
    parser.add_argument(
        "--name", type=str, default="",
        help="股票名称（可选）",
    )
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--daily-dir", type=str, default=None,
        help="日线 CSV 目录（默认使用 config 配置）",
    )
    parser.add_argument(
        "--minute-dir", type=str, default=None,
        help="分钟线 CSV 目录（默认使用 config 配置）",
    )
    parser.add_argument(
        "--gain-max", type=float, default=None,
        help="当日涨幅上限（小数，如 0.05 = 5%%）",
    )
    parser.add_argument(
        "--tail-rise", type=float, default=None,
        help="尾盘涨幅下限（小数，如 0.005 = 0.5%%）",
    )
    parser.add_argument(
        "--vol-ratio", type=float, default=None,
        help="尾盘量比下限（如 1.5）",
    )
    parser.add_argument(
        "--notify", action="store_true",
        help="触发时发送企业微信通知",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="将信号记录输出到 CSV 文件",
    )

    args = parser.parse_args()

    # 构造自定义参数
    params = TailStrategyParams()
    if args.gain_max is not None:
        params.daily_gain_max = args.gain_max
    if args.tail_rise is not None:
        params.tail_price_rise_min = args.tail_rise
    if args.vol_ratio is not None:
        params.tail_volume_ratio_min = args.vol_ratio

    # 执行回测
    if args.code:
        records = backtest_single(
            code=args.code,
            name=args.name,
            start_date=args.start,
            end_date=args.end,
            params=params,
            notify=args.notify,
            daily_dir=args.daily_dir,
            minute_dir=args.minute_dir,
        )
        result_df = pd.DataFrame(records) if records else pd.DataFrame()
    else:
        result_df = backtest_batch(
            start_date=args.start,
            end_date=args.end,
            params=params,
            notify=args.notify,
        )

    # 输出报告
    print_backtest_report(result_df)

    # 可选导出 CSV
    if args.output and not result_df.empty:
        result_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        log.info(f"信号记录已导出: {args.output}")


if __name__ == "__main__":
    main()
