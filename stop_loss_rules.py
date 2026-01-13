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

    # 动态回撤止损：盈利每增加 step_profit_pct，允许最大回撤增加 step_drawdown_pct
    # 例如：基础 10%盈利->允许回撤3%；20%->4%；30%->5%
    enable_dynamic_drawdown: bool = True
    dynamic_profit_step_pct: float = 0.10
    dynamic_drawdown_base_pct: float = 0.03
    dynamic_drawdown_step_pct: float = 0.01

    # 跌破“信号日开盘价”止损（买入信号日的 open 作为底线）
    enable_signal_open_stop: bool = True

    # 持股N天内“累计最高涨幅”未达到阈值则卖出
    enable_early_underperform_exit: bool = True
    early_exit_hold_days: int = 5
    early_exit_min_return: float = 0.03

    # 新增：早期弱势卖出展期
    # - 在 early_exit_hold_days 交易日时，如果其中“收红天数”>= red_days_threshold，则将考核日延后 extend_days 个交易日
    enable_early_exit_extend_on_red_days: bool = False
    early_exit_extend_red_days_threshold: int = 3
    early_exit_extend_days: int = 2


def get_close(symbol_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """取指定日期收盘价。"""
    if symbol_df is None or symbol_df.empty:
        return None
    row = symbol_df[symbol_df['trade_date'] == date]
    if row.empty:
        return None
    val = float(row['close'].iloc[0])
    return val if val > 0 else None


def get_open(symbol_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """取指定日期开盘价。"""
    if symbol_df is None or symbol_df.empty or 'open' not in symbol_df.columns:
        return None
    row = symbol_df[symbol_df['trade_date'] == date]
    if row.empty:
        return None
    val = float(row['open'].iloc[0])
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


def _count_trading_days_inclusive(symbol_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """统计 [start, end] 区间内的交易日数量（按 symbol_df.trade_date）。"""
    if symbol_df is None or symbol_df.empty:
        return 0
    df = symbol_df[['trade_date']].dropna().copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna().sort_values('trade_date')
    m = (df['trade_date'] >= pd.to_datetime(start)) & (df['trade_date'] <= pd.to_datetime(end))
    return int(m.sum())


def _count_red_days_inclusive(symbol_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """统计 [start, end] 区间内“收红天数”(close > open) 的交易日数量。"""
    if symbol_df is None or symbol_df.empty or 'open' not in symbol_df.columns or 'close' not in symbol_df.columns:
        return 0
    df = symbol_df[['trade_date', 'open', 'close']].dropna().copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna().sort_values('trade_date')
    m = (df['trade_date'] >= pd.to_datetime(start)) & (df['trade_date'] <= pd.to_datetime(end))
    sub = df.loc[m]
    if sub.empty:
        return 0
    red = (pd.to_numeric(sub['close'], errors='coerce') > pd.to_numeric(sub['open'], errors='coerce'))
    return int(red.fillna(False).sum())


def _get_nth_trading_day(symbol_df: pd.DataFrame, start: pd.Timestamp, n: int) -> pd.Timestamp | None:
    """返回从 start(含) 起第 n 个交易日对应日期（n=1 表示 start 当天）。"""
    if symbol_df is None or symbol_df.empty:
        return None
    df = symbol_df[['trade_date']].dropna().copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna().sort_values('trade_date')
    sub = df[df['trade_date'] >= pd.to_datetime(start)]['trade_date'].reset_index(drop=True)
    if n <= 0 or len(sub) < n:
        return None
    return pd.to_datetime(sub.iloc[n - 1])


def _dynamic_drawdown_threshold(cfg: StopLossConfig, entry_price: float, peak_close: float) -> float:
    """根据累计最大盈利计算允许回撤阈值。

    规则：
    - 盈利<10%: 使用基础 stop_loss_drawdown（保持兼容旧配置）
    - 盈利每跨越 10%（10/20/30...），允许回撤 = 3%/4%/5%...
    """
    try:
        if entry_price <= 0 or peak_close <= 0:
            return float(cfg.stop_loss_drawdown)

        max_profit = (peak_close - entry_price) / entry_price
        step = float(cfg.dynamic_profit_step_pct)
        if step <= 0:
            return float(cfg.stop_loss_drawdown)

        # floor 到 0,1,2,... 对应 0~9.99%、10~19.99%、20~29.99%...
        k = int(max_profit // step)

        if k <= 0:
            return float(cfg.stop_loss_drawdown)

        thr = float(cfg.dynamic_drawdown_base_pct) + (k - 1) * float(cfg.dynamic_drawdown_step_pct)
        return float(max(thr, 0.0))
    except Exception:
        return float(cfg.stop_loss_drawdown)


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

    # 用于输出的原因中文映射（纯中文模式）
    def _reason_cn(code: str) -> str:
        m = {
            'BELOW_SIGNAL_OPEN': '跌破信号日开盘价止损',
            'EARLY_UNDERPERFORM': '早期弱势卖出',
            'THREE_DAYS_DOWN': '连续三天下跌卖出',
        }
        return m.get(code, code)

    # 规则2：跌破“信号日开盘价”止损
    # - signal_date: 选股信号触发日（买入发生在下一交易日开盘）
    # - signal_open: 提前缓存，避免 signal_date 那天没 open 时重复查
    if cfg.enable_signal_open_stop:
        try:
            sig_open = pos.get('signal_open')
            if sig_open is None:
                sig_dt = pos.get('signal_date')
                if sig_dt is not None:
                    sig_open = get_open(symbol_df, pd.to_datetime(sig_dt))
                    if sig_open is not None:
                        pos['signal_open'] = float(sig_open)
            if sig_open is not None and float(sig_open) > 0:
                if float(close) < float(sig_open):
                    reasons.append(_reason_cn('BELOW_SIGNAL_OPEN'))
        except Exception:
            pass

    # 规则3：购买后 N 个交易日内，累计最高涨幅未达到阈值则卖出（从自然日改为交易日）
    if cfg.enable_early_underperform_exit:
        try:
            buy_dt = pd.to_datetime(pos.get('buy_date'))
            hold_td = _count_trading_days_inclusive(symbol_df, buy_dt, pd.to_datetime(date))

            # 计算“最终考核交易日数”：默认 N
            eval_td = int(cfg.early_exit_hold_days)

            # 若开启展期：在第 N 个交易日时，如果区间内收红天数>=阈值，则延后 K 个交易日再考核
            if bool(cfg.enable_early_exit_extend_on_red_days):
                try:
                    if hold_td == int(cfg.early_exit_hold_days):
                        red_cnt = _count_red_days_inclusive(symbol_df, buy_dt, pd.to_datetime(date))
                        if red_cnt >= int(cfg.early_exit_extend_red_days_threshold):
                            eval_td = int(cfg.early_exit_hold_days) + int(cfg.early_exit_extend_days)
                            pos['early_exit_extended'] = True
                            pos['early_exit_extended_red_cnt'] = int(red_cnt)
                except Exception:
                    pass

            # 到达考核日才真正判断（只触发一次）
            if hold_td == eval_td:
                if not bool(pos.get('early_exit_checked')):
                    pos['early_exit_checked'] = True

                    entry = float(pos.get('entry_price') or 0)
                    if entry > 0:
                        peak_close_now = max(float(pos.get('peak_close') or close), close)
                        max_ret = (peak_close_now - entry) / entry
                        if max_ret < float(cfg.early_exit_min_return):
                            reasons.append(_reason_cn('EARLY_UNDERPERFORM'))
        except Exception:
            pass

    # peak_close 初始化逻辑：初始化当天不触发止损
    if 'peak_close' not in pos or pos.get('peak_close') is None:
        pos['peak_close'] = close
        return (len(reasons) > 0), reasons

    pos['peak_close'] = max(float(pos['peak_close']), close)
    peak_close = float(pos['peak_close']) if pos.get('peak_close') else 0.0

    # 规则1：回撤止损（支持动态回撤）
    entry_price = float(pos.get('entry_price') or 0.0)
    if cfg.enable_dynamic_drawdown:
        dd_thr = _dynamic_drawdown_threshold(cfg, entry_price, peak_close)
    else:
        dd_thr = float(cfg.stop_loss_drawdown)

    drawdown = (peak_close - close) / peak_close if peak_close > 0 else 0.0
    if drawdown >= dd_thr:
        reasons.append(f"回撤止损(阈值{dd_thr*100:.1f}%)")

    if cfg.enable_three_days_down_exit:
        if is_last_n_days_all_down(symbol_df, date, n=3):
            reasons.append(_reason_cn('THREE_DAYS_DOWN'))

    return (len(reasons) > 0), reasons
