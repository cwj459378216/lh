import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 使用主脚本中的配置与方法，确保一致性
import select_stocks_local as sel

# 止损/卖出规则抽取为独立模块
from stop_loss_rules import StopLossConfig, evaluate_exit_signal

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
    end_date: str = '20260106'
    # 回测交易参数
    initial_capital: float = 100000.0          # 初始资金
    buy_mode: str = 'fixed'                     # 买入模式: 'fixed' 固定金额, 'ratio' 按百分比
    buy_ratio: float = 0.5                      # 按百分比买入时使用的资金比例（相对当前可用资金）
    buy_fixed_amount: float = 4000.0           # 固定金额买入时，每次使用的金额

    # 卖出成交价模式：
    # - 'close': 当天收盘价卖出
    # - 'next_open': 下一交易日开盘价卖出（原逻辑）
    sell_price_mode: str = 'close'

    stop_loss_drawdown: float = 0.03            # 收盘价回撤止损：相对“最高收盘价”回撤 3%（从买入后第一天收盘价开始）
    enable_three_days_down_exit: bool = False    # 是否启用“连续三天下跌卖出”规则

    # 新增：持股N天内，如果涨幅低于阈值则卖出（按当日收盘触发，下一交易日开盘成交）
    enable_early_underperform_exit: bool = True
    early_exit_hold_days: int = 7
    early_exit_min_return: float = 0.03  # 3%

    # 新增：回测结束时，未卖出持仓是否用“最后一日收盘价”进行统计性平仓
    # - True: 若无法按 sell_price_mode 获取成交价，则退化为用最后收盘价（或 peak_close/成本价兜底）
    # - False: 若无法成交则不做平仓记录，期末权益也不把这些仓位折算为现金
    close_open_positions_at_end: bool = True

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


def _get_sell_price(cfg: BacktestConfig, symbol_df: pd.DataFrame, signal_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    """根据配置决定卖出成交价与成交日。

    - cfg.sell_price_mode == 'close': 使用 signal_date 当天收盘价成交
    - cfg.sell_price_mode == 'next_open': 使用 signal_date 的下一交易日开盘价成交

    返回 (sell_date, sell_price)。若无法取到价格则返回 (None, None)。
    """
    mode = (cfg.sell_price_mode or 'next_open').lower().strip()
    if mode == 'close':
        sell_price = _get_close(symbol_df, signal_date)
        if sell_price is None or sell_price <= 0:
            return None, None
        return pd.to_datetime(signal_date), float(sell_price)

    # 默认 next_open
    return _get_next_open(symbol_df, signal_date)


def main():
    cfg = CFG
    # 在基础 out_dir 下按当前时间（精确到分钟）创建本次回测的子目录
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    run_out_dir = os.path.join(cfg.out_dir, ts)
    os.makedirs(run_out_dir, exist_ok=True)

    # 构建止损规则配置（从回测配置映射）
    sl_cfg = StopLossConfig(
        stop_loss_drawdown=cfg.stop_loss_drawdown,
        enable_three_days_down_exit=cfg.enable_three_days_down_exit,
        enable_early_underperform_exit=cfg.enable_early_underperform_exit,
        early_exit_hold_days=cfg.early_exit_hold_days,
        early_exit_min_return=cfg.early_exit_min_return,
    )

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

    # 统计：因资金/单笔金额不足导致无法买入的次数
    buy_skip_insufficient_cash = 0

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
            pos_1y_min_pct=sel.CFG.pos_1y_min_pct,
            pos_1y_max_pct=sel.CFG.pos_1y_max_pct,
            min_limitup_count=cfg.min_limitup_count,
        )
        # scan_dir 内已统一处理 min_limitup_count，无需额外过滤

        # 开始执行止损检查（按当日收盘价触发；成交价可配置：当日收盘 或 次日开盘）
        exits = []
        for sym, pos in list(positions.items()):
            sym_df = price_map.get(sym)

            should_exit, reasons = evaluate_exit_signal(sl_cfg, sym_df, pos, d)
            if not should_exit:
                continue

            # 卖出：根据配置决定成交价
            sell_date, sell_price = _get_sell_price(cfg, sym_df, d)
            if sell_date is None or sell_price is None or sell_price <= 0:
                # 无可用价格则无法成交，跳过
                continue

            proceeds = pos['shares'] * sell_price
            cash += proceeds
            pnl = (sell_price - pos['entry_price']) * pos['shares']

            reason = '_AND_'.join(reasons) if reasons else 'SELL'

            trade_log.append({
                'date': sell_date.strftime('%Y-%m-%d'),
                'symbol': sym,
                'action': 'SELL',
                'price': sell_price,
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

                # 固定金额模式：按100股一手取整买入（向下取整到100的整数倍）
                lot_size = 100

                # 资金不足（或单笔金额设置太小）：连 1 手都买不起
                if buy_cash <= 0 or buy_cash < open_price * lot_size:
                    buy_skip_insufficient_cash += 1
                    continue

                shares = int(buy_cash // open_price)
                shares = (shares // lot_size) * lot_size

                if shares <= 0:
                    buy_skip_insufficient_cash += 1
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
                    'pnl': 0.0,
                    # 记录“下单日（信号日）”及当日指标，供后续买入汇总直接使用
                    'signal_date': d.strftime('%Y-%m-%d'),
                    'pos_in_1y': (row.get('pos_in_1y') if isinstance(row, dict) else row.get('pos_in_1y')),
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

    # 回测结束：是否强制平仓（按配置决定）
    if positions:
        last_day = test_days[-1]

        if cfg.close_open_positions_at_end:
            # 强制平仓（卖出成交价同样按配置；若缺价则退化为最后一日收盘价用于统计）
            for sym, pos in list(positions.items()):
                sym_df = price_map.get(sym)
                sell_date, sell_price = _get_sell_price(cfg, sym_df, last_day)
                if sell_date is None or sell_price is None or sell_price <= 0:
                    # 若缺少可用卖出价，则退化为用最后一日收盘价进行统计性平仓（不改变历史交易过程，仅用于回测汇总）
                    sell_date = last_day
                    sell_price = _get_close(sym_df, last_day) or float(pos.get('peak_close') or pos['entry_price'])

                proceeds = pos['shares'] * sell_price
                cash += proceeds
                pnl = (sell_price - pos['entry_price']) * pos['shares']
                trade_log.append({
                    'date': sell_date.strftime('%Y-%m-%d'),
                    'symbol': sym,
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': pos['shares'],
                    'pnl': round(pnl, 2),
                    'reason': 'FORCE_LIQUIDATION',
                })
                positions.pop(sym, None)

            # 追加一条期末权益记录（以强制平仓后的现金作为最终权益）
            equity_curve.append({
                'date': sell_date.strftime('%Y-%m-%d'),
                'equity': round(cash, 2),
                'cash': round(cash, 2),
                'positions': 0,
            })
        else:
            # 不做期末强制平仓：保留持仓到结束
            # 为了让汇总更清晰，这里仅追加一条“期末仍有持仓”的权益记录（与循环最后一天的权益值一致）
            equity_curve.append({
                'date': last_day.strftime('%Y-%m-%d'),
                'equity': round(equity_curve[-1]['equity'], 2) if equity_curve else round(cash, 2),
                'cash': round(cash, 2),
                'positions': len(positions),
            })

    # 汇总输出
    df_equity = pd.DataFrame(equity_curve).sort_values('date')
    df_trades = pd.DataFrame(trade_log)

    # 输出列名改为中文标题
    df_equity_cn = df_equity.rename(columns={
        'date': '日期',
        'equity': '总权益',
        'cash': '现金',
        'positions': '持仓数量',
        'cash_ratio': '现金占比',
    })
    df_trades_cn = df_trades.rename(columns={
        'date': '日期',
        'symbol': '代码',
        'action': '操作',
        'price': '价格',
        'shares': '股数',
        'pnl': '盈亏',
        'reason': '原因',
    })

    df_equity_cn.to_csv(os.path.join(run_out_dir, '权益曲线.csv'), index=False, encoding='utf-8-sig')
    df_trades_cn.to_csv(os.path.join(run_out_dir, '交易记录.csv'), index=False, encoding='utf-8-sig')

    # 额外输出：买入明细 + 是否盈利（不改变交易逻辑，仅基于 trade_log 事后整理）
    if not df_trades.empty:
        df_buy = df_trades[df_trades['action'] == 'BUY'].copy()
        df_sell = df_trades[df_trades['action'] == 'SELL'].copy()

        if not df_buy.empty:
            df_buy = df_buy.rename(columns={'date': 'buy_date', 'price': 'buy_price', 'shares': 'buy_shares'})
            df_buy['buy_date'] = pd.to_datetime(df_buy['buy_date'], errors='coerce')
            df_buy['buy_price'] = pd.to_numeric(df_buy['buy_price'], errors='coerce')
            df_buy['buy_shares'] = pd.to_numeric(df_buy['buy_shares'], errors='coerce')
            df_buy['buy_amount'] = (df_buy['buy_price'] * df_buy['buy_shares']).round(2)

            if not df_sell.empty:
                df_sell = df_sell.rename(columns={'date': 'sell_date', 'price': 'sell_price', 'shares': 'sell_shares', 'pnl': 'sell_pnl'})
                df_sell['sell_date'] = pd.to_datetime(df_sell['sell_date'], errors='coerce')
                df_sell['sell_price'] = pd.to_numeric(df_sell['sell_price'], errors='coerce')
                df_sell['sell_shares'] = pd.to_numeric(df_sell['sell_shares'], errors='coerce')
                df_sell['sell_pnl'] = pd.to_numeric(df_sell['sell_pnl'], errors='coerce')

                # 同一标的可能多次买卖：按时间顺序为每次交易分配序号并配对
                df_buy = df_buy.sort_values(['symbol', 'buy_date']).copy()
                df_sell = df_sell.sort_values(['symbol', 'sell_date']).copy()
                df_buy['trade_no'] = df_buy.groupby('symbol').cumcount() + 1
                df_sell['trade_no'] = df_sell.groupby('symbol').cumcount() + 1

                df_buy_summary = df_buy.merge(
                    df_sell[['symbol', 'trade_no', 'sell_date', 'sell_price', 'sell_pnl', 'reason']],
                    on=['symbol', 'trade_no'],
                    how='left',
                )
            else:
                df_buy = df_buy.sort_values(['symbol', 'buy_date']).copy()
                df_buy['trade_no'] = df_buy.groupby('symbol').cumcount() + 1
                df_buy_summary = df_buy.copy()
                df_buy_summary['sell_date'] = pd.NaT
                df_buy_summary['sell_price'] = pd.NA
                df_buy_summary['sell_pnl'] = pd.NA
                df_buy_summary['reason'] = pd.NA

            # 是否挣钱：有 sell_pnl 才能判断，否则为 OPEN
            df_buy_summary['is_profit'] = df_buy_summary['sell_pnl'].apply(
                lambda x: ('OPEN' if pd.isna(x) else ('YES' if float(x) > 0 else ('NO' if float(x) < 0 else 'EVEN')))
            )

            # 持股天数：卖出日期 - 买入日期（未卖出则为空）
            _buy_dt = pd.to_datetime(df_buy_summary.get('buy_date'), errors='coerce')
            _sell_dt = pd.to_datetime(df_buy_summary.get('sell_date'), errors='coerce')
            df_buy_summary['hold_days'] = (_sell_dt - _buy_dt).dt.days

            # 前一天涨幅：以买入日期的“前一交易日”计算涨幅
            # - prev1: 买入日前一交易日
            # - prev2: prev1 的前一交易日
            # 涨幅 = (close(prev1) - close(prev2)) / close(prev2)
            def _prev_day_pct(row):
                try:
                    sym = row.get('symbol')
                    buy_date = row.get('buy_date')
                    if pd.isna(buy_date) or sym is None:
                        return pd.NA
                    sym_df = price_map.get(sym)
                    if sym_df is None or sym_df.empty:
                        return pd.NA

                    dfp = sym_df[['trade_date', 'close']].dropna().sort_values('trade_date').reset_index(drop=True)
                    buy_dt = pd.to_datetime(buy_date)
                    # 买入日所在位置
                    pos_list = dfp.index[dfp['trade_date'] == buy_dt].to_list()
                    if not pos_list:
                        return pd.NA
                    pos = int(pos_list[0])
                    if pos < 2:
                        return pd.NA
                    close_prev1 = float(dfp.loc[pos - 1, 'close'])
                    close_prev2 = float(dfp.loc[pos - 2, 'close'])
                    if close_prev2 <= 0:
                        return pd.NA
                    return round((close_prev1 - close_prev2) / close_prev2, 6)
                except Exception:
                    return pd.NA

            df_buy_summary['prev_day_pct'] = df_buy_summary.apply(_prev_day_pct, axis=1)

            # 前一天的价格在历史的位置 pos_in_1y：直接使用买入信号日的 pos_in_1y（已在 trade_log 里记录，无需重复计算）
            # 说明：买入是按“信号日 d 的下一交易日开盘”成交，因此“前一天”即信号日 d
            if 'signal_date' in df_buy_summary.columns:
                df_buy_summary['signal_date'] = pd.to_datetime(df_buy_summary['signal_date'], errors='coerce').dt.strftime('%Y-%m-%d')

            # 将 pos_in_1y 作为“前一天位置pos_in_1y”输出
            if 'pos_in_1y' in df_buy_summary.columns:
                df_buy_summary['prev_day_pos_in_1y'] = pd.to_numeric(df_buy_summary['pos_in_1y'], errors='coerce')
            else:
                df_buy_summary['prev_day_pos_in_1y'] = pd.NA

            # 格式化日期输出
            if 'buy_date' in df_buy_summary.columns:
                df_buy_summary['buy_date'] = df_buy_summary['buy_date'].dt.strftime('%Y-%m-%d')
            if 'sell_date' in df_buy_summary.columns:
                df_buy_summary['sell_date'] = pd.to_datetime(df_buy_summary['sell_date'], errors='coerce').dt.strftime('%Y-%m-%d')

            cols = [
                'symbol',
                'trade_no',
                'buy_date',
                'buy_price',
                'buy_shares',
                'buy_amount',
                'prev_day_pct',
                'prev_day_pos_in_1y',
                'sell_date',
                'sell_price',
                'sell_pnl',
                'hold_days',
                'is_profit',
                'reason',
            ]
            cols = [c for c in cols if c in df_buy_summary.columns]
            df_buy_summary = df_buy_summary[cols].sort_values(['symbol', 'trade_no'])

            # 买入汇总也改为中文标题
            df_buy_summary_cn = df_buy_summary.rename(columns={
                'symbol': '代码',
                'trade_no': '序号',
                'buy_date': '买入日期',
                'buy_price': '买入价',
                'buy_shares': '买入股数',
                'buy_amount': '买入金额',
                'prev_day_pct': '前一天涨幅',
                'prev_day_pos_in_1y': '前一天位置pos_in_1y',
                'sell_date': '卖出日期',
                'sell_price': '卖出价',
                'sell_pnl': '卖出盈亏',
                'hold_days': '持股天数',
                'is_profit': '是否盈利',
                'reason': '原因',
            })
            df_buy_summary_cn.to_csv(os.path.join(run_out_dir, '买入汇总.csv'), index=False, encoding='utf-8-sig')

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

    # 胜率统计（按“卖出/平仓”为一笔）
    if not df_trades.empty:
        df_sells = df_trades[df_trades['action'] == 'SELL'].copy()
        if not df_sells.empty:
            win_cnt = int((df_sells['pnl'] > 0).sum())
            loss_cnt = int((df_sells['pnl'] < 0).sum())
            even_cnt = int((df_sells['pnl'] == 0).sum())
            total_cnt = len(df_sells)
            # 口径：胜率 = 盈利笔数 / (盈利笔数 + 亏损笔数)，不把持平计入分母
            denom = win_cnt + loss_cnt
            win_rate = (win_cnt / denom * 100) if denom > 0 else 0.0
            print(f"平仓统计: 盈利 {win_cnt} | 亏损 {loss_cnt} | 持平 {even_cnt} | 平仓总笔数 {total_cnt}")
            print(f"胜率(按平仓, 不含持平): {win_rate:.2f}%")

    print(f"回测完成：{len(test_days)} 天 | 期末权益: {df_equity['equity'].iloc[-1]:.2f} | 输出目录: {run_out_dir}")
    print(f"无法买入（资金不足/单笔金额不足导致买不起1股）的股票次数: {buy_skip_insufficient_cash}")


if __name__ == '__main__':
    main()
