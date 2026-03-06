# -*- coding: utf-8 -*-
"""
monitor.py — 实时尾盘监控模块

功能：
  1. 在尾盘时段定时轮询 AkShare 快照
  2. 维护基线快照（14:50 时刻）
  3. 与最新快照对比，判断尾盘买入信号
  4. 触发信号后发送企业微信通知
  5. 支持多股票批量监控

使用方法：
    python -m tail_trade.monitor

注意：
  - AkShare 使用全市场批量接口（stock_zh_a_spot_em），单次获取所有股票
  - 带内存缓存 + 限频保护，避免被限制
  - 建议轮询间隔 ≥ 30 秒
"""

import argparse
import os
import sys
import time
from datetime import datetime, date, timedelta
from typing import Any

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tail_trade import config
from tail_trade.data.loader import (
    load_stock_pool,
    load_daily_csv,
    load_realtime_snapshot,
    snapshot_consecutive_failures,
)
from tail_trade.strategy import (
    TailStrategyParams,
    SignalResult,
    check_tail_signal_snapshot,
)
from tail_trade.notifier import notify_signal
from tail_trade.logger import get_logger

log = get_logger("monitor")


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _time_str_to_minutes(t: str) -> int:
    """将 'HH:MM' 转为从 00:00 开始的分钟数。"""
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _current_time_str() -> str:
    """返回当前时间的 HH:MM 字符串。"""
    return datetime.now().strftime("%H:%M")


def _is_in_time_range(current: str, start: str, end: str) -> bool:
    """判断 current 是否在 [start, end) 时间范围内。"""
    c, s, e = _time_str_to_minutes(current), _time_str_to_minutes(start), _time_str_to_minutes(end)
    return s <= c < e


def _calc_trading_minutes_since_open(time_str: str) -> float:
    """
    计算从开盘到指定时间的交易分钟数。

    A 股交易时段：09:30-11:30（120 min） + 13:00-15:00（120 min）
    """
    t = _time_str_to_minutes(time_str)
    morning_open = _time_str_to_minutes("09:30")
    morning_close = _time_str_to_minutes("11:30")
    afternoon_open = _time_str_to_minutes("13:00")
    afternoon_close = _time_str_to_minutes("15:00")

    if t <= morning_open:
        return 0
    elif t <= morning_close:
        return t - morning_open
    elif t <= afternoon_open:
        return 120  # 上午全部 120 分钟
    elif t <= afternoon_close:
        return 120 + (t - afternoon_open)
    else:
        return 240  # 全天


def _get_prev_close(code: str, snapshot_data: dict[str, Any] | None = None) -> float:
    """
    获取昨日收盘价。

    优先从快照的 prev_close 字段获取，否则从日线 CSV 最后一行获取。
    """
    # 从快照获取
    if snapshot_data and snapshot_data.get("prev_close"):
        return snapshot_data["prev_close"]

    # 从日线 CSV 获取
    try:
        daily_df = load_daily_csv(code)
        if not daily_df.empty:
            return daily_df["close"].iloc[-1]
    except Exception:
        pass

    return 0.0


# ═══════════════════════════════════════════════════════════════════
# 监控主类
# ═══════════════════════════════════════════════════════════════════

class TailMonitor:
    """
    尾盘实时监控器。

    工作流程：
      1. 在 MONITOR_START_TIME（如 14:45）开始轮询
      2. 在 TAIL_START_TIME（如 14:50）记录基线快照
      3. 此后每次轮询对比基线，判断是否触发
      4. 在 MONITOR_END_TIME（如 15:01）结束
      5. 每只股票每天最多通知一次
    """

    def __init__(
        self,
        stock_pool: dict[str, str] | None = None,
        params: TailStrategyParams | None = None,
        poll_interval: int | None = None,
    ):
        self.stock_pool = stock_pool or load_stock_pool()
        self.params = params or TailStrategyParams()
        self.poll_interval = poll_interval or config.MONITOR_POLL_INTERVAL

        # 基线快照：{code: {price, volume, time_str}}
        self.baseline: dict[str, dict[str, Any]] = {}

        # 今日已通知的股票（避免重复通知）
        self.notified_today: set[str] = set()

        # 今日触发的信号记录
        self.signals_today: list[SignalResult] = []

        # 连续轮询失败计数
        self._consecutive_poll_failures: int = 0
        self._max_consecutive_failures: int = config.MONITOR_MAX_CONSECUTIVE_FAILURES
        self._alert_sent: bool = False  # 是否已发送过告警

    def run(self) -> None:
        """
        启动监控主循环。
        """
        log.info("=" * 50)
        log.info("🚀 尾盘监控启动")
        log.info(f"   股票池: {len(self.stock_pool)} 只")
        log.info(f"   轮询间隔: {self.poll_interval} 秒")
        log.info(f"   监控时段: {config.MONITOR_START_TIME} - {config.MONITOR_END_TIME}")
        log.info(f"   尾盘区间: {self.params.tail_start} - {self.params.tail_end}")
        log.info("=" * 50)

        if not self.stock_pool:
            log.error("股票池为空，退出")
            return

        # 主循环：等待到监控开始时间
        while True:
            now_str = _current_time_str()

            # 还没到监控时间，等待
            if _time_str_to_minutes(now_str) < _time_str_to_minutes(config.MONITOR_START_TIME):
                wait = _time_str_to_minutes(config.MONITOR_START_TIME) - _time_str_to_minutes(now_str)
                log.info(f"等待监控开始... 距离 {config.MONITOR_START_TIME} 还有 {wait} 分钟")
                time.sleep(min(wait * 60, 60))
                continue

            # 超过监控结束时间，退出
            if _time_str_to_minutes(now_str) >= _time_str_to_minutes(config.MONITOR_END_TIME):
                log.info("监控时段已结束")
                self._print_summary()
                break

            # ── 执行一轮检查 ──
            self._poll_once()

            # ── 动态调整间隔：连续失败时自动延长，减少无效请求 ──
            if self._consecutive_poll_failures > 0:
                backoff = min(self.poll_interval * (2 ** self._consecutive_poll_failures), 300)
                log.warning(
                    f"连续失败 {self._consecutive_poll_failures} 次，"
                    f"扩大间隔至 {backoff:.0f} 秒"
                )
                time.sleep(backoff)
            else:
                log.info(f"等待 {self.poll_interval} 秒后下次轮询...")
                time.sleep(self.poll_interval)

    def _poll_once(self) -> None:
        """执行单次轮询（带连续失败告警）。"""
        now_str = _current_time_str()
        log.info(f"── 轮询 {now_str} ──")

        try:
            snapshot = load_realtime_snapshot(force=True)
        except Exception as e:
            self._consecutive_poll_failures += 1
            log.error(
                f"获取快照失败 (连续第 {self._consecutive_poll_failures} 次): {e}"
            )

            # 连续失败超限 → 发送告警
            if (
                self._consecutive_poll_failures >= self._max_consecutive_failures
                and not self._alert_sent
            ):
                from tail_trade.notifier import send_wechat_text
                send_wechat_text(
                    f"⚠️ 尾盘监控告警\n"
                    f"AkShare 快照连续失败 {self._consecutive_poll_failures} 次\n"
                    f"最近错误: {e}\n"
                    f"监控可能中断，请检查网络或 AkShare 状态"
                )
                self._alert_sent = True
                log.error("已发送连续失败告警通知")
            return

        # 成功获取快照，重置失败计数
        if self._consecutive_poll_failures > 0:
            log.info(f"快照恢复正常（之前连续失败 {self._consecutive_poll_failures} 次）")
        self._consecutive_poll_failures = 0
        self._alert_sent = False

        for code, name in self.stock_pool.items():
            if code in self.notified_today:
                continue  # 已通知过

            data = snapshot.get(code)
            if not data:
                continue

            price = data.get("price")
            volume = data.get("volume")
            if not price or not volume:
                continue

            # ── 记录基线 ──
            tail_started = _time_str_to_minutes(now_str) >= _time_str_to_minutes(self.params.tail_start)

            if code not in self.baseline and tail_started:
                self.baseline[code] = {
                    "price": price,
                    "volume": volume,
                    "time_str": now_str,
                    "name": name,
                }
                log.info(f"  📌 基线记录 | {code} {name} | 价={price} 量={volume}")
                continue  # 基线刚记录，下轮再判断

            # ── 判断信号 ──
            if code in self.baseline and tail_started:
                bl = self.baseline[code]
                minutes_to_baseline = _calc_trading_minutes_since_open(bl["time_str"])
                tail_elapsed = _time_str_to_minutes(now_str) - _time_str_to_minutes(bl["time_str"])

                if tail_elapsed <= 0:
                    continue

                prev_close = _get_prev_close(code, data)
                if prev_close <= 0:
                    continue

                signal = check_tail_signal_snapshot(
                    prev_close=prev_close,
                    current_price=price,
                    baseline_price=bl["price"],
                    baseline_volume=bl["volume"],
                    current_volume=volume,
                    minutes_from_open_to_baseline=minutes_to_baseline,
                    tail_elapsed_minutes=tail_elapsed,
                    code=code,
                    name=name,
                    params=self.params,
                )

                if signal.triggered:
                    log.info(
                        f"  ✅ 信号触发 | {code} {name} | "
                        f"价={price} 涨幅={signal.daily_gain*100:+.2f}% "
                        f"尾盘涨={signal.tail_rise*100:+.2f}% 量比={signal.tail_volume_ratio:.2f}"
                    )
                    self.signals_today.append(signal)
                    self.notified_today.add(code)

                    # 发送企业微信通知
                    notify_signal(
                        code=code,
                        name=name,
                        price=price,
                        daily_gain=signal.daily_gain,
                        tail_rise=signal.tail_rise,
                        tail_volume_ratio=signal.tail_volume_ratio,
                        extra={"监控时间": now_str},
                    )
                else:
                    log.debug(
                        f"  ❌ {code} {name} 未触发: "
                        + "; ".join(signal.reasons)
                    )

    def _print_summary(self) -> None:
        """打印今日监控汇总。"""
        print("\n" + "═" * 50)
        print("          📋 今日尾盘监控汇总")
        print("═" * 50)
        if self.signals_today:
            for s in self.signals_today:
                print(
                    f"  ✅ {s.code} {s.name} | 价={s.price:.2f} | "
                    f"涨幅={s.daily_gain*100:+.2f}% | "
                    f"尾盘涨={s.tail_rise*100:+.2f}% | 量比={s.tail_volume_ratio:.2f}"
                )
        else:
            print("  今日无信号触发")
        print("═" * 50 + "\n")


# ═══════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="尾盘买入策略 — 实时监控",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--codes", type=str, default="",
        help="要监控的股票代码（逗号分隔），留空则使用股票池",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help=f"轮询间隔秒数（默认 {config.MONITOR_POLL_INTERVAL}）",
    )
    parser.add_argument(
        "--gain-max", type=float, default=None,
        help="当日涨幅上限（小数）",
    )
    parser.add_argument(
        "--tail-rise", type=float, default=None,
        help="尾盘涨幅下限（小数）",
    )
    parser.add_argument(
        "--vol-ratio", type=float, default=None,
        help="尾盘量比下限",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="测试模式：立即执行一次轮询（不等待时间窗口）",
    )

    args = parser.parse_args()

    # 构造股票池
    stock_pool = None
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        stock_pool = {c: "" for c in codes}

    # 构造参数
    params = TailStrategyParams()
    if args.gain_max is not None:
        params.daily_gain_max = args.gain_max
    if args.tail_rise is not None:
        params.tail_price_rise_min = args.tail_rise
    if args.vol_ratio is not None:
        params.tail_volume_ratio_min = args.vol_ratio

    monitor = TailMonitor(
        stock_pool=stock_pool,
        params=params,
        poll_interval=args.interval,
    )

    if args.test:
        # 测试模式：直接执行一次轮询
        log.info("=== 测试模式：执行单次轮询 ===")
        monitor._poll_once()
        monitor._print_summary()
    else:
        monitor.run()


if __name__ == "__main__":
    main()
