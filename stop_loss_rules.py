"""止损/卖出规则模块（从 backtest_select_stocks_local.py 抽取）。

目的：
- 将回测中的“是否触发卖出”判断逻辑集中管理，便于迭代与复用。
- 不处理成交、资金、日志，仅输出“是否触发/原因”。

注意：
- 该模块不依赖 select_stocks_local.py，只依赖 pandas。
- 持仓字典 pos 建议沿用 backtest 中的结构：
  {
    'shares': int,
    'entry_price': float,
    'buy_date': pd.Timestamp | str,
    'peak_close': float | None,
  }
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class StopLossConfig:
    """止损/卖出规则参数。"""

    stop_loss_drawdown: float = 0.03
    enable_three_days_down_exit: bool = False

    # 持股N天内“累计最高涨幅”未达到阈值则卖出
    enable_early_underperform_exit: bool = True
    early_exit_hold_days: int = 7
    early_exit_min_return: float = 0.03


def get_close(symbol_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """取指定日期收盘价。"""
    if symbol_df is None or symbol_df.empty:
        return None
    row = symbol_df[symbol_df['trade_date'] == date]
    if row.empty:
        return None
    val = float(row['close'].iloc[0])
    return val if val > 0 else None


def is_last_n_days_all_down(symbol_df: pd.DataFrame, date: pd.Timestamp, n: int = 3) -> bool:
    """判断给定日期及其前面共 n-1 个交易日，收盘价是否连续下跌（严格单调下降）。"""
    if symbol_df is None or symbol_df.empty:
        return False
    df = symbol_df[symbol_df['trade_date'] <= date].sort_values('trade_date')
    if len(df) < n:
        return False
    tail = df.tail(n)
    closes = tail['close'].to_list()
    for i in range(1, len(closes)):
        if closes[i] >= closes[i - 1]:
            return False
    return True


def evaluate_exit_signal(
    cfg: StopLossConfig,
    symbol_df: pd.DataFrame,
    pos: dict,
    date: pd.Timestamp,
) -> tuple[bool, list[str]]:
    """评估某持仓在 date 当日是否触发卖出信号。

    返回:
    - (should_exit, reasons)
      reasons 可能包含: STOP_LOSS / THREE_DAYS_DOWN / EARLY_UNDERPERFORM

    说明:
    - 本函数会“更新/初始化 pos['peak_close']”（与原回测保持一致）。
    - 若 peak_close 为 None，则初始化后直接返回不触发（相当于从下一次更新开始比较）。
    """

    close = get_close(symbol_df, date)
    if close is None:
        return False, []

    reasons: list[str] = []

    # 规则1：持股N天内“累计最高涨幅”未达到阈值则卖出
    if cfg.enable_early_underperform_exit:
        try:
            buy_dt = pd.to_datetime(pos.get('buy_date'))
            hold_days = (pd.to_datetime(date) - buy_dt).days
            if hold_days is not None and hold_days == int(cfg.early_exit_hold_days):
                entry = float(pos.get('entry_price') or 0)
                if entry > 0:
                    peak_close_now = max(float(pos.get('peak_close') or close), close)
                    max_ret = (peak_close_now - entry) / entry
                    if max_ret < float(cfg.early_exit_min_return):
                        reasons.append('EARLY_UNDERPERFORM')
        except Exception:
            pass

    # peak_close 初始化逻辑：初始化当天不触发止损
    if 'peak_close' not in pos or pos.get('peak_close') is None:
        pos['peak_close'] = close
        return False, []

    pos['peak_close'] = max(float(pos['peak_close']), close)
    peak_close = float(pos['peak_close']) if pos.get('peak_close') else 0.0
    drawdown = (peak_close - close) / peak_close if peak_close > 0 else 0.0

    if drawdown >= float(cfg.stop_loss_drawdown):
        reasons.append('STOP_LOSS')

    if cfg.enable_three_days_down_exit:
        if is_last_n_days_all_down(symbol_df, date, n=3):
            reasons.append('THREE_DAYS_DOWN')

    return (len(reasons) > 0), reasons
