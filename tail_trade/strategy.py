# -*- coding: utf-8 -*-
"""
strategy.py — 尾盘买入策略逻辑

策略条件（全部满足时触发买入信号）：
  1. 当日涨幅 < DAILY_GAIN_MAX（默认 5%）
  2. 尾盘最后 N 分钟价格上涨 > TAIL_PRICE_RISE_MIN（默认 0.5%）
  3. 尾盘量比 > TAIL_VOLUME_RATIO_MIN（默认 1.5）

量比定义：
  tail_volume_ratio = (尾盘每分钟均量) / (全日每分钟均量)
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tail_trade import config
from tail_trade.logger import get_logger

log = get_logger("strategy")


# ═══════════════════════════════════════════════════════════════════
# 策略参数（便于动态调参）
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TailStrategyParams:
    """尾盘策略可调参数集合。"""
    daily_gain_max: float = config.DAILY_GAIN_MAX
    tail_start: str = config.TAIL_START_TIME      # "14:50"
    tail_end: str = config.TAIL_END_TIME            # "15:00"
    tail_minutes: int = config.TAIL_MINUTES
    tail_price_rise_min: float = config.TAIL_PRICE_RISE_MIN
    tail_volume_ratio_min: float = config.TAIL_VOLUME_RATIO_MIN
    total_trading_minutes: int = config.TOTAL_TRADING_MINUTES


# ═══════════════════════════════════════════════════════════════════
# 信号结果
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    """策略判断结果。"""
    triggered: bool = False          # 是否触发买入信号
    code: str = ""                   # 股票代码
    name: str = ""                   # 股票名称
    trade_date: str = ""             # 交易日
    price: float = 0.0               # 当前/收盘价
    prev_close: float = 0.0          # 昨日收盘
    daily_gain: float = 0.0          # 当日涨幅（小数）
    tail_start_price: float = 0.0    # 尾盘起始价
    tail_end_price: float = 0.0      # 尾盘结束价
    tail_rise: float = 0.0           # 尾盘涨幅（小数）
    tail_volume: float = 0.0         # 尾盘总成交量
    tail_volume_ratio: float = 0.0   # 尾盘量比
    reasons: list[str] = field(default_factory=list)  # 未触发原因

    def to_dict(self) -> dict[str, Any]:
        """转为字典（方便日志和表格输出）。"""
        return {
            "triggered": self.triggered,
            "code": self.code,
            "name": self.name,
            "trade_date": self.trade_date,
            "price": self.price,
            "prev_close": self.prev_close,
            "daily_gain": f"{self.daily_gain * 100:+.2f}%",
            "tail_rise": f"{self.tail_rise * 100:+.2f}%",
            "tail_volume_ratio": f"{self.tail_volume_ratio:.2f}",
            "reasons": "; ".join(self.reasons) if self.reasons else "",
        }


# ═══════════════════════════════════════════════════════════════════
# 核心策略函数（基于分钟线）
# ═══════════════════════════════════════════════════════════════════

def check_tail_signal_minute(
    day_minute_bars: pd.DataFrame,
    prev_close: float,
    code: str = "",
    name: str = "",
    trade_date: str = "",
    params: TailStrategyParams | None = None,
) -> SignalResult:
    """
    基于 **当日分钟 K 线** 判断是否触发尾盘买入信号。

    Args:
        day_minute_bars: 当日所有分钟 K 线（需含 time, close, volume 列）
                         time 格式 "HH:MM"
        prev_close:      前一交易日收盘价
        code:            股票代码
        name:            股票名称
        trade_date:      日期字符串
        params:          策略参数

    Returns:
        SignalResult
    """
    p = params or TailStrategyParams()
    result = SignalResult(code=code, name=name, trade_date=trade_date)

    if day_minute_bars.empty:
        result.reasons.append("当日无分钟数据")
        return result

    if prev_close <= 0:
        result.reasons.append("昨日收盘价无效")
        return result

    # ── 1. 当日收盘价 & 涨幅 ──
    day_close = day_minute_bars["close"].iloc[-1]
    result.price = day_close
    result.prev_close = prev_close
    daily_gain = (day_close - prev_close) / prev_close
    result.daily_gain = daily_gain

    if daily_gain >= p.daily_gain_max:
        result.reasons.append(f"当日涨幅 {daily_gain*100:.2f}% ≥ {p.daily_gain_max*100:.1f}%")

    # ── 2. 分离尾盘 K 线 ──
    tail_mask = (day_minute_bars["time"] >= p.tail_start) & (day_minute_bars["time"] < p.tail_end)
    tail_bars = day_minute_bars.loc[tail_mask]

    if tail_bars.empty:
        result.reasons.append(f"无 {p.tail_start}-{p.tail_end} 尾盘数据")
        return result

    # 尾盘起始价 = 尾盘第一根 K 线的开盘价
    tail_start_price = tail_bars["open"].iloc[0]
    # 尾盘结束价 = 尾盘最后一根 K 线的收盘价
    tail_end_price = tail_bars["close"].iloc[-1]
    result.tail_start_price = tail_start_price
    result.tail_end_price = tail_end_price

    # 尾盘涨幅
    if tail_start_price > 0:
        tail_rise = (tail_end_price - tail_start_price) / tail_start_price
    else:
        tail_rise = 0.0
    result.tail_rise = tail_rise

    if tail_rise < p.tail_price_rise_min:
        result.reasons.append(
            f"尾盘涨幅 {tail_rise*100:.2f}% < {p.tail_price_rise_min*100:.1f}%"
        )

    # ── 3. 尾盘量比 ──
    tail_volume = tail_bars["volume"].sum()
    tail_minutes = len(tail_bars)
    result.tail_volume = tail_volume

    # 全日非尾盘 K 线
    non_tail = day_minute_bars.loc[~tail_mask]
    if not non_tail.empty:
        total_volume = day_minute_bars["volume"].sum()
        total_minutes = len(day_minute_bars)
        daily_avg_per_min = total_volume / total_minutes if total_minutes > 0 else 1
        tail_avg_per_min = tail_volume / tail_minutes if tail_minutes > 0 else 0
        volume_ratio = tail_avg_per_min / daily_avg_per_min if daily_avg_per_min > 0 else 0
    else:
        volume_ratio = 0.0

    result.tail_volume_ratio = volume_ratio

    if volume_ratio < p.tail_volume_ratio_min:
        result.reasons.append(
            f"尾盘量比 {volume_ratio:.2f} < {p.tail_volume_ratio_min:.1f}"
        )

    # ── 综合判定 ──
    if not result.reasons:
        result.triggered = True
        log.info(
            f"✅ 信号触发 | {code} {name} | "
            f"涨幅={daily_gain*100:+.2f}% 尾盘涨={tail_rise*100:+.2f}% 量比={volume_ratio:.2f}"
        )
    return result


# ═══════════════════════════════════════════════════════════════════
# 简化策略函数（基于实时快照 + 基线快照）
# ═══════════════════════════════════════════════════════════════════

def check_tail_signal_snapshot(
    prev_close: float,
    current_price: float,
    baseline_price: float,
    baseline_volume: float,
    current_volume: float,
    minutes_from_open_to_baseline: float,
    tail_elapsed_minutes: float,
    code: str = "",
    name: str = "",
    params: TailStrategyParams | None = None,
) -> SignalResult:
    """
    基于 **实时快照对比** 判断尾盘信号（用于 monitor 模块）。

    快照策略：
      - baseline: 尾盘开始时的价格和成交量（14:50 的快照）
      - current:  当前最新价格和成交量

    Args:
        prev_close:                   昨日收盘价
        current_price:                当前最新价
        baseline_price:               尾盘起始价（14:50 的快照价格）
        baseline_volume:              尾盘起始时的全日累计成交量
        current_volume:               当前全日累计成交量
        minutes_from_open_to_baseline: 开盘到基线时刻的交易分钟数
        tail_elapsed_minutes:          尾盘已经过分钟数
        code:                          股票代码
        name:                          股票名称
        params:                        策略参数

    Returns:
        SignalResult
    """
    p = params or TailStrategyParams()
    result = SignalResult(code=code, name=name)

    if prev_close <= 0 or baseline_price <= 0:
        result.reasons.append("价格数据无效")
        return result

    # ── 1. 当日涨幅 ──
    daily_gain = (current_price - prev_close) / prev_close
    result.price = current_price
    result.prev_close = prev_close
    result.daily_gain = daily_gain

    if daily_gain >= p.daily_gain_max:
        result.reasons.append(f"当日涨幅 {daily_gain*100:.2f}% ≥ {p.daily_gain_max*100:.1f}%")

    # ── 2. 尾盘涨幅 ──
    tail_rise = (current_price - baseline_price) / baseline_price
    result.tail_start_price = baseline_price
    result.tail_end_price = current_price
    result.tail_rise = tail_rise

    if tail_rise < p.tail_price_rise_min:
        result.reasons.append(
            f"尾盘涨幅 {tail_rise*100:.2f}% < {p.tail_price_rise_min*100:.1f}%"
        )

    # ── 3. 尾盘量比 ──
    tail_volume = max(current_volume - baseline_volume, 0)
    result.tail_volume = tail_volume

    if tail_elapsed_minutes > 0 and minutes_from_open_to_baseline > 0:
        daily_avg = baseline_volume / minutes_from_open_to_baseline
        tail_avg = tail_volume / tail_elapsed_minutes
        volume_ratio = tail_avg / daily_avg if daily_avg > 0 else 0
    else:
        volume_ratio = 0

    result.tail_volume_ratio = volume_ratio

    if volume_ratio < p.tail_volume_ratio_min:
        result.reasons.append(
            f"尾盘量比 {volume_ratio:.2f} < {p.tail_volume_ratio_min:.1f}"
        )

    # ── 综合判定 ──
    if not result.reasons:
        result.triggered = True

    return result
