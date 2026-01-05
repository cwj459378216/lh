import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 使用主脚本中的配置与方法，确保一致性
import select_stocks_local as sel

@dataclass
class BacktestConfig:
    data_dir: str = sel.CFG.data_dir
    # 基础输出目录，不带时间戳
    out_dir: str = os.path.join(os.path.dirname(__file__), 'output', 'backtest')
    months: int = sel.CFG.months
    range_lower: float = sel.CFG.range_lower
    range_upper: float = sel.CFG.range_upper
    near_low_tol: float = sel.CFG.near_low_tol
    limitup_months: int = sel.CFG.limitup_months
    limitup_threshold: float = sel.CFG.limitup_threshold
    min_limitup_count: int = sel.CFG.min_limitup_count
    vol_days: int = sel.CFG.vol_days
    vol_factor: float = sel.CFG.vol_factor
    only_10pct_a: bool = sel.CFG.only_10pct_a
    # 回测范围
    start_date: str = '20250901'
    end_date: str = '20260103'
    # 回测交易参数
    initial_capital: float = 37000.0          # 初始资金
    buy_mode: str = 'fixed'                     # 买入模式: 'fixed' 固定金额, 'ratio' 按百分比
    buy_ratio: float = 0.5                      # 按百分比买入时使用的资金比例（相对当前可用资金）
    buy_fixed_amount: float = 3000.0           # 固定金额买入时，每次使用的金额
    stop_loss_drawdown: float = 0.03            # 收盘价回撤止损：相对“最高收盘价”回撤 3%（从买入后第一天收盘价开始）
    enable_three_days_down_exit: bool = False    # 是否启用“连续三天下跌卖出”规则

CFG = BacktestConfig()


def _available_trading_days(data_dir: str) -> list[pd.Timestamp]:
    # 根据任意一个CSV汇总交易日（取并集）
    files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith('.csv')]
    files.sort()
    dates = set()
    pbar = tqdm(total=len(files), desc='扫描交易日', dynamic_ncols=True)
    for fn in files:
        fp = os.path.join(data_dir, fn)
        df = sel.load_csv(fp)
        if not df.empty and 'trade_date' in df.columns:
            for d in df['trade_date']:
                dates.add(pd.to_datetime(d))
        pbar.update(1)
    pbar.close()
    return sorted(dates)


def _load_symbol_map(data_dir: str) -> dict:
    """预加载所有标的的 trade_date、open、close，便于价格查询。"""
    files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith('.csv')]
    files.sort()
    out = {}
    pbar = tqdm(total=len(files), desc='加载价格数据', dynamic_ncols=True)
    for fn in files:
        stem = os.path.splitext(fn)[0]
        df = sel.load_csv(os.path.join(data_dir, fn))
        if not df.empty:
            # 增加对开盘价的预加载，后面用于第二天开盘买入
            cols = [c for c in ['trade_date', 'open', 'close'] if c in df.columns]
            out[stem] = df[cols].copy()
        pbar.update(1)
    pbar.close()
    return out


def _get_close(symbol_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    if symbol_df is None or symbol_df.empty:
        return None
    row = symbol_df[symbol_df['trade_date'] == date]
    if row.empty:
        return None
    val = float(row['close'].iloc[0])
    return val if val > 0 else None


def _get_next_open(symbol_df: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    """给定某天，返回该标的"下一交易日"的开盘价及该日期。

    如果不存在下一行或缺少 open 列，则返回 (None, None)。
    """
    if symbol_df is None or symbol_df.empty or 'open' not in symbol_df.columns:
        return None, None
    df = symbol_df.sort_values('trade_date')
    # 找到当前日期所在行的索引位置
    idx = df.index[df['trade_date'] == date]
    if len(idx) == 0:
        return None, None
    pos = df.index.get_loc(idx[0])
    if isinstance(pos, slice):
        # 理论上不应出现重复日期，如有则取第一条
        pos = pos.start
    next_pos = pos + 1
    if next_pos >= len(df):
        return None, None
    row = df.iloc[next_pos]
    val = float(row.get('open', 0) or 0)
    if val <= 0:
        return None, None
    return pd.to_datetime(row['trade_date']), val


def _is_last_n_days_all_down(symbol_df: pd.DataFrame, date: pd.Timestamp, n: int = 3) -> bool:
    """判断给定日期及其前面共 n-1 个交易日，收盘价是否连续下跌（严格单调下降）。

    要求 symbol_df 中已经包含 date 这一行；如果数据不足 n 天或缺少某天价格，则返回 False。
    """
    if symbol_df is None or symbol_df.empty:
        return False
    # 取出 <= date 的记录，并按日期排序
    df = symbol_df[symbol_df['trade_date'] <= date].sort_values('trade_date')
    if len(df) < n:
        return False
    tail = df.tail(n)
    closes = tail['close'].to_list()
    # 严格递减
    for i in range(1, len(closes)):
        if closes[i] >= closes[i-1]:
            return False
    return True


def main():
    cfg = CFG
    # 在基础 out_dir 下按当前时间（精确到分钟）创建本次回测的子目录
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    run_out_dir = os.path.join(cfg.out_dir, ts)
    os.makedirs(run_out_dir, exist_ok=True)

    start = pd.to_datetime(cfg.start_date)
    end = pd.to_datetime(cfg.end_date)

    all_days = _available_trading_days(cfg.data_dir)
    test_days = [d for d in all_days if start <= d <= end]
    if not test_days:
        print('指定区间内无交易日')
        return

    # 交易账户状态
    cash = cfg.initial_capital
    positions: dict[str, dict] = {}

    # 数据映射
    price_map = _load_symbol_map(cfg.data_dir)
    # 预加载完整CSV用于选股
    preloaded_all = {}
    files = [fn for fn in os.listdir(cfg.data_dir) if fn.lower().endswith('.csv')]
    for fn in files:
        stem = os.path.splitext(fn)[0]
        df_full = sel.load_csv(os.path.join(cfg.data_dir, fn))
        if not df_full.empty:
            preloaded_all[stem] = df_full

    equity_curve = []
    trade_log = []

    pbar = tqdm(total=len(test_days), desc='回测', dynamic_ncols=True)
    for d in test_days:
        # 用主逻辑选股（截止当天），静默模式
        df_sel = sel.scan_dir(
            data_dir=cfg.data_dir,
            months_lookback=cfg.months,
            range_lower=cfg.range_lower,
            range_upper=cfg.range_upper,
            near_low_tol=cfg.near_low_tol,
            limitup_lookback_months=cfg.limitup_months,
            limitup_threshold=cfg.limitup_threshold,
            volume_spike_days=cfg.vol_days,
            volume_spike_factor=cfg.vol_factor,
            only_10pct_a=cfg.only_10pct_a,
            end_date=d,
            preloaded=preloaded_all,
            quiet=True,
        )
        if not df_sel.empty:
            df_sel = df_sel[df_sel['limit_up_days_1y'] >= int(cfg.min_limitup_count)].copy()

        # 开始执行止损检查（按当日收盘价）
        exits = []
        for sym, pos in list(positions.items()):
            sym_df = price_map.get(sym)
            close = _get_close(sym_df, d)
            if close is None:
                continue

            # 收盘价最高点回撤止损：从“买入后第一天的收盘价”开始
            # - 买入当天(成交日)不初始化 peak_close
            # - 遇到首个有效收盘价（通常是成交日当天的 close）时，初始化 peak_close
            if 'peak_close' not in pos or pos['peak_close'] is None:
                pos['peak_close'] = close
                # peak_close 刚初始化当天，不触发止损判断（相当于从下一次更新开始才比较）
                continue

            pos['peak_close'] = max(float(pos['peak_close']), close)
            peak_close = float(pos['peak_close']) if pos['peak_close'] else 0.0
            drawdown = (peak_close - close) / peak_close if peak_close > 0 else 0.0

            # 连续三天下跌卖出条件（可配置开关）
            three_days_down = False
            if cfg.enable_three_days_down_exit:
                three_days_down = _is_last_n_days_all_down(sym_df, d, n=3)

            if drawdown >= cfg.stop_loss_drawdown or three_days_down:
                # 按当日收盘价清仓
                proceeds = pos['shares'] * close
                cash += proceeds
                pnl = (close - pos['entry_price']) * pos['shares']
                if cfg.enable_three_days_down_exit:
                    reason = 'STOP_LOSS' if drawdown >= cfg.stop_loss_drawdown and not three_days_down else (
                        'THREE_DAYS_DOWN' if three_days_down and drawdown < cfg.stop_loss_drawdown else 'STOP_LOSS_AND_THREE_DAYS_DOWN')
                else:
                    reason = 'STOP_LOSS'
                trade_log.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'symbol': sym,
                    'action': 'SELL',
                    'price': close,
                    'shares': pos['shares'],
                    'pnl': round(pnl, 2),
                    'reason': reason,
                })
                exits.append(sym)
        for sym in exits:
            positions.pop(sym, None)

        # 买入：对当日选股，若未持有，则按配置的买入模式买入（按下一交易日开盘价成交）
        if not df_sel.empty:
            for _, row in df_sel.iterrows():
                sym = row['symbol']
                if sym in positions:
                    continue
                sym_df = price_map.get(sym)
                # 使用下一交易日开盘价作为买入价
                buy_date, open_price = _get_next_open(sym_df, d)
                if buy_date is None or open_price is None or open_price <= 0:
                    continue

                # 根据买入模式计算本次使用资金
                if cfg.buy_mode == 'ratio':
                    buy_cash = cash * cfg.buy_ratio
                else:
                    buy_cash = cfg.buy_fixed_amount

                buy_cash = min(buy_cash, cash)
                if buy_cash <= 0:
                    continue

                shares = int(buy_cash // open_price)
                if shares <= 0:
                    continue

                cost = shares * open_price
                cash -= cost
                positions[sym] = {
                    'shares': shares,
                    'entry_price': open_price,
                    'buy_date': buy_date,   # 成交日（下一交易日）
                    'peak_close': None,     # 不用成本价占位，等待“买入后第一天的收盘价”初始化
                }
                trade_log.append({
                    'date': buy_date.strftime('%Y-%m-%d'),
                    'symbol': sym,
                    'action': 'BUY',
                    'price': open_price,
                    'shares': shares,
                    'pnl': 0.0
                })

        # 计算当日权益（持仓按收盘价估值）
        equity = cash
        for sym, pos in positions.items():
            sym_df = price_map.get(sym)
            close = _get_close(sym_df, d)
            if close is None:
                # 若无当日价格，则优先用已初始化的 peak_close 兜底，否则用成本价
                close = float(pos.get('peak_close') or pos['entry_price'])
            equity += pos['shares'] * close
        equity_curve.append({'date': d.strftime('%Y-%m-%d'), 'equity': round(equity, 2), 'cash': round(cash, 2), 'positions': len(positions)})

        # 输出每日选择文件
        out_file = os.path.join(run_out_dir, f"selection_{d.strftime('%Y%m%d')}.csv")
        if df_sel.empty:
            # 不输出空CSV
            pass
        else:
            df_sel.sort_values('range_pct').to_csv(out_file, index=False, encoding='utf-8-sig')

        # 简化进度输出，仅显示当前处理到的日期
        pbar.set_postfix_str(d.strftime('%Y-%m-%d'))
        pbar.update(1)
    pbar.close()

    # 汇总输出
    df_equity = pd.DataFrame(equity_curve).sort_values('date')
    df_trades = pd.DataFrame(trade_log)
    df_equity.to_csv(os.path.join(run_out_dir, 'equity_curve.csv'), index=False, encoding='utf-8-sig')
    df_trades.to_csv(os.path.join(run_out_dir, 'trade_log.csv'), index=False, encoding='utf-8-sig')

    # 资金占用与收益率统计
    # 资金占用率 = 1 - 平均(cash / equity)
    if not df_equity.empty:
        df_equity['cash_ratio'] = df_equity['cash'] / df_equity['equity'].replace(0, pd.NA)
        avg_cash_ratio = float(df_equity['cash_ratio'].dropna().mean()) if df_equity['cash_ratio'].notna().any() else 0.0
        avg_capital_usage = 1.0 - avg_cash_ratio
        start_equity = float(df_equity['equity'].iloc[0])
        end_equity = float(df_equity['equity'].iloc[-1])
        total_return = (end_equity - start_equity) / start_equity if start_equity > 0 else 0.0
        print(f"资金平均占用率: {avg_capital_usage*100:.2f}% | 总收益率: {total_return*100:.2f}%")

    # 胜率统计
    if not df_trades.empty:
        win_trades = df_trades[df_trades['pnl'] > 0]
        win_rate = len(win_trades) / len(df_trades) * 100
        print(f"胜率: {win_rate:.2f}%")

    print(f"回测完成：{len(test_days)} 天 | 期末权益: {df_equity['equity'].iloc[-1]:.2f} | 输出目录: {run_out_dir}")


if __name__ == '__main__':
    main()
