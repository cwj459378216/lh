import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
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
    start_date: str = '20200101'
    end_date: str = '20260106'
    # 回测交易参数
    initial_capital: float = 40000.0          # 初始资金
    buy_mode: str = 'ratio'                     # 买入模式: 'fixed' 固定金额, 'ratio' 按百分比
    buy_fixed_amount: float = 10000.0           # 固定金额买入时，每次使用的金额

    # ratio 买入策略附加约束：最多持有 N 只；每只目标仓位 = total_equity * per_position_ratio
    ratio_max_positions: int = 3
    ratio_per_position: float = 0.33

    # 卖出成交价模式：
    # - 'close': 当天收盘价卖出
    # - 'next_open': 下一交易日开盘价卖出（原逻辑）
    sell_price_mode: str = 'close'

    # 新增：最少持股天数（按“成交买入日 -> 卖出成交日”的自然日差计算）
    # - 默认 1：至少持有 1 天，避免买卖同一天
    # - 设为 0：允许同日卖出（不建议）
    min_hold_days: int = 1

    # --- 交易成本（A股常见）
    # 你的费率：万0.85（0.000085），取消最低5元，按比例但最低 0.1 元
    commission_rate: float = 0.000085
    commission_min: float = 0.1
    # 印花税：仅卖出收取（你的是 0.05%）
    stamp_tax_rate_sell: float = 0.0005

    stop_loss_drawdown: float = 0.03            # 收盘价回撤止损：相对“最高收盘价”回撤 3%（从买入后第一天收盘价开始）
    enable_three_days_down_exit: bool = False    # 是否启用“连续三天下跌卖出”规则

    # 新增：动态“早期弱势卖出”
    # 规则：每 step_days 个交易日，最低涨幅门槛增加 step_min_return
    enable_early_underperform_exit: bool = True
    early_exit_step_days: int = 5
    early_exit_step_min_return: float = 0.03  # 每 5 个交易日 +3%

    # 新增：回测结束时，未卖出持仓是否用“最后一日收盘价”进行统计性平仓
    # - True: 若无法按 sell_price_mode 获取成交价，则退化为用最后收盘价（或 peak_close/成本价兜底）
    # - False: 若无法成交则不做平仓记录，期末权益也不把这些仓位折算为现金
    close_open_positions_at_end: bool = True

    # ratio 模式：当日候选多于可买名额时的“优先级排序”配置
    # 说明：列表从前到后为优先级（先按第1个字段，再按第2个字段...）。
    # 可选字段（当前已支持）：
    # - 'score'（评分，数值越大越优先；若缺失则退化为 raw_score，仍缺失则记为 0）
    # - 'raw_score'（原始评分，数值越大越优先）
    # - 'last3_up'（是否三连涨信号，True 优先）
    # - 'limit_up_days_1y'（近一年涨停次数，数值越大越优先）
    # 默认：优先选评分高的
    # ratio_candidate_priority = ['score', 'last3_up', 'limit_up_days_1y']
    ratio_candidate_priority: list[str] = field(default_factory=lambda: ['score'])

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


def _same_trading_day(d1: pd.Timestamp | None, d2: pd.Timestamp | None) -> bool:
    if d1 is None or d2 is None:
        return False
    return pd.to_datetime(d1).normalize() == pd.to_datetime(d2).normalize()


def _get_sell_price_no_same_day(cfg: BacktestConfig, symbol_df: pd.DataFrame, signal_date: pd.Timestamp, buy_date: pd.Timestamp | None) -> tuple[pd.Timestamp | None, float | None]:
    """获取卖出价，但保证卖出日 != 买入成交日。

    规则：买卖不能同一天。
    处理：
    - 先按 cfg.sell_price_mode 取成交价
    - 如果成交日与 buy_date 同一天：返回 (None, None) 表示本次不允许成交
    """
    sell_date, sell_price = _get_sell_price(cfg, symbol_df, signal_date)
    if _same_trading_day(sell_date, buy_date):
        return None, None
    return sell_date, sell_price


def _calc_commission(cfg: BacktestConfig, amount: float) -> float:
    """佣金：按比例且有最低值。amount 为成交额（正数）。"""
    amt = max(float(amount or 0.0), 0.0)
    if amt <= 0:
        return 0.0
    fee = amt * float(cfg.commission_rate)
    return float(max(fee, float(cfg.commission_min)))


def _calc_stamp_tax_sell(cfg: BacktestConfig, amount: float) -> float:
    """印花税：仅卖出收取。amount 为成交额（正数）。"""
    amt = max(float(amount or 0.0), 0.0)
    if amt <= 0:
        return 0.0
    return float(amt * float(cfg.stamp_tax_rate_sell))


def _apply_candidate_priority(df_sel: pd.DataFrame, priority: list[str] | None) -> pd.DataFrame:
    """ratio 模式下，候选过多时的排序策略。

    priority: 字段名列表，按顺序作为排序 key。
    - score/raw_score: 数值大优先
    - last3_up: True 优先
    - limit_up_days_1y: 大优先
    """
    if df_sel is None or df_sel.empty:
        return df_sel

    pr = priority
    if pr is None:
        # 默认：先按评分从高到低
        pr = ['score']

    df = df_sel.copy()
    sort_cols: list[str] = []
    ascendings: list[bool] = []

    for key in pr:
        k = (key or '').strip()
        if not k:
            continue

        if k in ('score', 'raw_score'):
            # 兼容：优先用 score（0~100 固定标尺）；缺失则用 raw_score
            if k == 'score':
                if 'score' in df.columns:
                    src = 'score'
                elif 'raw_score' in df.columns:
                    src = 'raw_score'
                else:
                    src = None
            else:
                src = 'raw_score' if 'raw_score' in df.columns else None

            if src is None:
                df['_pri_score'] = 0.0
            else:
                df['_pri_score'] = pd.to_numeric(df[src], errors='coerce').fillna(0.0)
            sort_cols.append('_pri_score')
            ascendings.append(False)
            continue

        if k == 'last3_up':
            if 'last3_up' not in df.columns:
                df['last3_up'] = False
            # True(1) 优先
            df['_pri_last3_up'] = df['last3_up'].fillna(False).astype(int)
            sort_cols.append('_pri_last3_up')
            ascendings.append(False)
            continue

        if k in ('limit_up_days_1y', 'limitup_cnt', 'limit_up_days'):
            col = 'limit_up_days_1y' if 'limit_up_days_1y' in df.columns else None
            if col is None:
                # 兼容旧列名
                for _c in ['limitup_cnt', 'limit_up_days']:
                    if _c in df.columns:
                        col = _c
                        break
            if col is None:
                df['_pri_limitup'] = 0
            else:
                df['_pri_limitup'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            sort_cols.append('_pri_limitup')
            ascendings.append(False)
            continue

        # 未来扩展：未识别字段先忽略

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascendings)

    # 清理临时列
    for c in ['_pri_score', '_pri_last3_up', '_pri_limitup']:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


def main():
    print('开始回测...')
    try:
        cfg = CFG
        # 在基础 out_dir 下按当前时间（精确到分钟）创建本次回测的子目录
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        run_out_dir = os.path.join(cfg.out_dir, ts)
        os.makedirs(run_out_dir, exist_ok=True)
        print(f"输出目录: {run_out_dir}")

        # 构建止损规则配置（从回测配置映射）
        sl_cfg = StopLossConfig(
            stop_loss_drawdown=cfg.stop_loss_drawdown,
            enable_three_days_down_exit=cfg.enable_three_days_down_exit,
            enable_early_underperform_exit=cfg.enable_early_underperform_exit,
            early_exit_step_days=getattr(cfg, 'early_exit_step_days', 5),
            early_exit_step_min_return=getattr(cfg, 'early_exit_step_min_return', 0.03),
        )

        start = pd.to_datetime(cfg.start_date)
        end = pd.to_datetime(cfg.end_date)

        print('扫描交易日...')
        all_days = _available_trading_days(cfg.data_dir)
        test_days = [d for d in all_days if start <= d <= end]
        if not test_days:
            print('指定区间内无交易日')
            return

        # 交易账户状态
        cash = cfg.initial_capital
        positions: dict[str, dict] = {}

        # 数据映射
        print('加载价格数据...')
        price_map = _load_symbol_map(cfg.data_dir)

        # 预加载完整CSV用于选股
        print('预加载K线数据用于选股...')
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
            # ratio 模式：如果已满仓（达到持仓上限），则跳过当日选股扫描与买入逻辑
            # 仅继续执行止损/卖出与权益更新；当后续有卖出释放名额后再恢复扫描。
            _skip_scan_today = False
            if (cfg.buy_mode or '').lower().strip() == 'ratio':
                try:
                    if len(positions) >= int(cfg.ratio_max_positions):
                        _skip_scan_today = True
                except Exception:
                    _skip_scan_today = False

            # 用主逻辑选股（截止当天），静默模式
            if _skip_scan_today:
                df_sel = pd.DataFrame()
            else:
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

                # 最少持股天数限制：未达到则不允许卖出
                buy_exec_date = pos.get('buy_date')

                should_exit, reasons = evaluate_exit_signal(sl_cfg, sym_df, pos, d)
                if not should_exit:
                    continue

                # 针对“跌破信号日开盘价止损”：要求在触发日的下一交易日开盘卖出
                # 其他止损原因按 cfg.sell_price_mode 执行，但必须满足“买卖不同日”
                if any(('跌破信号日开盘价止损' in r) for r in (reasons or [])):
                    sell_date, sell_price = _get_next_open(sym_df, d)
                else:
                    sell_date, sell_price = _get_sell_price_no_same_day(cfg, sym_df, d, buy_exec_date)

                # 规则：买卖不能同一天（最终成交日校验）
                if _same_trading_day(sell_date, buy_exec_date):
                    continue

                if sell_date is None or sell_price is None or sell_price <= 0:
                    # 无可用价格或不允许成交，跳过
                    continue

                # 最少持股天数（二次校验：以成交日为准）
                if buy_exec_date is not None and int(getattr(cfg, 'min_hold_days', 1) or 0) > 0:
                    _held_days2 = (pd.to_datetime(sell_date).normalize() - pd.to_datetime(buy_exec_date).normalize()).days
                    if _held_days2 < int(cfg.min_hold_days):
                        continue

                proceeds = pos['shares'] * sell_price
                sell_commission = _calc_commission(cfg, proceeds)
                sell_stamp_tax = _calc_stamp_tax_sell(cfg, proceeds)
                sell_cost = sell_commission + sell_stamp_tax

                cash += (proceeds - sell_cost)
                pnl = (sell_price - pos['entry_price']) * pos['shares'] - float(pos.get('entry_cost', 0.0)) - sell_cost

                reason = '_AND_'.join(reasons) if reasons else 'SELL'

                trade_log.append({
                    'date': sell_date.strftime('%Y-%m-%d'),
                    'symbol': sym,
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': pos['shares'],
                    'pnl': round(pnl, 2),
                    'reason': reason,
                    'fees': round(sell_cost, 2),
                })
                exits.append(sym)
            for sym in exits:
                positions.pop(sym, None)

            # 买入：对当日选股，若未持有，则按配置的买入模式买入（按下一交易日开盘价成交）
            if not df_sel.empty:
                # ratio 模式：若当日候选太多，按配置的优先级排序，并只取剩余名额
                if (cfg.buy_mode or '').lower().strip() == 'ratio':
                    remaining_slots = int(cfg.ratio_max_positions) - len(positions)
                    if remaining_slots <= 0:
                        df_sel = df_sel.iloc[0:0]
                    else:
                        # 默认优先级：last3_up（是否三连涨信号）
                        pri = getattr(cfg, 'ratio_candidate_priority', None)
                        if pri is None:
                            pri = ['last3_up']
                        df_sel = _apply_candidate_priority(df_sel, pri)

                        # 候选超过名额：只取前 N 个
                        if len(df_sel) > remaining_slots:
                            df_sel = df_sel.head(remaining_slots).copy()

                for _, row in df_sel.iterrows():
                    sym = row['symbol']
                    if sym in positions:
                        continue

                    # ratio 策略：最多持有 N 只
                    if (cfg.buy_mode or '').lower().strip() == 'ratio':
                        if len(positions) >= int(cfg.ratio_max_positions):
                            continue

                    sym_df = price_map.get(sym)
                    # 使用下一交易日开盘价作为买入价
                    buy_date, open_price = _get_next_open(sym_df, d)
                    if buy_date is None or open_price is None or open_price <= 0:
                        continue

                    # 根据买入模式计算本次使用资金
                    if (cfg.buy_mode or '').lower().strip() == 'ratio':
                        # 每个票占 20% 总权益（按信号日收盘估值的 total_equity；再受限于可用现金）
                        total_equity = cash
                        for _sym, _pos in positions.items():
                            _sym_df = price_map.get(_sym)
                            _close = _get_close(_sym_df, d)
                            if _close is None:
                                _close = float(_pos.get('peak_close') or _pos['entry_price'])
                            total_equity += _pos['shares'] * _close

                        buy_cash = float(total_equity) * float(cfg.ratio_per_position)
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
                    buy_commission = _calc_commission(cfg, cost)
                    total_cost = cost + buy_commission

                    # 由于佣金包含最低值，total_cost 可能略高于 buy_cash/cash，再次校验
                    if total_cost > cash:
                        buy_skip_insufficient_cash += 1
                        continue

                    cash -= total_cost

                    # 保存信号日的关键指标（用于买入汇总，不改变交易逻辑）
                    _row_get = (row.get if hasattr(row, 'get') else (lambda k, default=None: row[k] if k in row else default))
                    signal_pos_in_1y = _row_get('pos_in_1y', pd.NA)
                    signal_last3_up = _row_get('last3_up', pd.NA)
                    # 新增：评分信息（来自选股结果；选股侧已产出 raw_score/score_reason，score(0-100) 需在回测侧汇总后归一化）
                    signal_raw_score = _row_get('raw_score', pd.NA)
                    signal_score_reason = _row_get('score_reason', pd.NA)

                    # 兼容可能的列名：market_cap / float_market_cap / 流通市值
                    signal_float_mktcap = (
                        _row_get('float_market_cap', pd.NA)
                        if _row_get('float_market_cap', None) is not None else _row_get('market_cap', pd.NA)
                    )

                    positions[sym] = {
                        'shares': shares,
                        'entry_price': open_price,
                        'buy_date': buy_date,   # 成交日（下一交易日）
                        'peak_close': None,
                        'entry_cost': float(buy_commission),
                        'signal_date': d,
                        'signal_open': None,
                        # signal-day metrics for later summary
                        'signal_pos_in_1y': signal_pos_in_1y,
                        'signal_last3_up': signal_last3_up,
                        'signal_float_mktcap': signal_float_mktcap,
                        # 新增：评分
                        'signal_raw_score': signal_raw_score,
                        'signal_score_reason': signal_score_reason,
                    }

                    trade_log.append({
                        'date': buy_date.strftime('%Y-%m-%d'),
                        'symbol': sym,
                        'action': 'BUY',
                        'price': open_price,
                        'shares': shares,
                        'pnl': 0.0,
                        'fees': round(buy_commission, 2),
                        # 记录“下单日（信号日）”及当日指标，供后续买入汇总直接使用
                        'signal_date': d.strftime('%Y-%m-%d'),
                        'pos_1y_min_pct': (_row_get('pos_1y_min_pct', pd.NA)),
                        'pos_1y_max_pct': (_row_get('pos_1y_max_pct', pd.NA)),
                        'pos_in_1y': signal_pos_in_1y,
                        'last3_up': signal_last3_up,
                        'float_market_cap': signal_float_mktcap,
                        # 新增：评分
                        'raw_score': signal_raw_score,
                        'score_reason': signal_score_reason,
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

            # 不再输出每日选股文件（selection_YYYYMMDD.csv）

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

                    # 强制平仓的成交日：优先用“最后一日的下一交易日开盘”来避免同日买卖
                    # （否则：如果最后一日刚好是买入成交日，会出现 sell_date == buy_date，导致持仓天数为 0）
                    sell_date, sell_price = _get_next_open(sym_df, last_day)

                    # 如果没有下一交易日数据，再退化为最后一日按配置取价（仅用于回测汇总）
                    if sell_date is None or sell_price is None or sell_price <= 0:
                        sell_date, sell_price = _get_sell_price(cfg, sym_df, last_day)

                    if sell_date is None or sell_price is None or sell_price <= 0:
                        # 若缺少可用卖出价，则退化为用最后一日收盘价进行统计性平仓（不改变历史交易过程，仅用于回测汇总）
                        sell_date = last_day
                        sell_price = _get_close(sym_df, last_day) or float(pos.get('peak_close') or pos['entry_price'])

                    # 兜底：避免出现 sell_date 与买入成交日同一天（持仓=0天）
                    buy_exec_date = pos.get('buy_date')
                    if _same_trading_day(sell_date, buy_exec_date):
                        # 再尝试用最后一日的“下一交易日开盘”（有可能上面已经是 next_open，但这里再兜一次）
                        _sd2, _sp2 = _get_next_open(sym_df, last_day)
                        if _sd2 is not None and _sp2 is not None and _sp2 > 0 and (not _same_trading_day(_sd2, buy_exec_date)):
                            sell_date, sell_price = _sd2, _sp2

                    proceeds = pos['shares'] * sell_price
                    sell_commission = _calc_commission(cfg, proceeds)
                    sell_stamp_tax = _calc_stamp_tax_sell(cfg, proceeds)
                    sell_cost = sell_commission + sell_stamp_tax

                    cash += (proceeds - sell_cost)
                    pnl = (sell_price - pos['entry_price']) * pos['shares'] - float(pos.get('entry_cost', 0.0)) - sell_cost
                    trade_log.append({
                        'date': sell_date.strftime('%Y-%m-%d'),
                        'symbol': sym,
                        'action': 'SELL',
                        'price': sell_price,
                        'shares': pos['shares'],
                        'pnl': round(pnl, 2),
                        'reason': 'FORCE_LIQUIDATION',
                        'fees': round(sell_cost, 2),
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
            'fees': '税费',
        })

        df_equity_cn.to_csv(os.path.join(run_out_dir, '权益曲线.csv'), index=False, encoding='utf-8-sig')
        df_trades_cn.to_csv(os.path.join(run_out_dir, '交易记录.csv'), index=False, encoding='utf-8-sig')

        # 额外输出：买入明细 + 是否盈利（不改变交易逻辑，仅基于 trade_log 事后整理）
        if not df_trades.empty:
            df_buy = df_trades[df_trades['action'] == 'BUY'].copy()
            df_sell = df_trades[df_trades['action'] == 'SELL'].copy()

            if not df_buy.empty:
                df_buy = df_buy.rename(columns={'date': 'buy_date', 'price': 'buy_price', 'shares': 'buy_shares', 'fees': 'buy_fees'})
                df_buy['buy_date'] = pd.to_datetime(df_buy['buy_date'], errors='coerce')
                df_buy['buy_price'] = pd.to_numeric(df_buy['buy_price'], errors='coerce')
                df_buy['buy_shares'] = pd.to_numeric(df_buy['buy_shares'], errors='coerce')
                df_buy['buy_amount'] = (df_buy['buy_price'] * df_buy['buy_shares']).round(2)

                # === FIFO 配对（按成交股数，先进先出，严格按时间序列） ===
                # 说明：之前按“先把全部 BUY 入队，再按日期处理 SELL”会导致：
                # - SELL 可能消耗到“未来”的 BUY，从而出现 hold_days 为 -1 或 0。
                # 正确做法：按 symbol 内的成交时间（buy_date/sell_date）合并排序，依次推进。
                fifo_pairs: list[dict] = []

                # 预处理 SELL
                df_sell_fifo = pd.DataFrame()
                if not df_sell.empty:
                    df_sell_fifo = df_sell.rename(columns={'date': 'sell_date', 'price': 'sell_price', 'shares': 'sell_shares', 'pnl': 'sell_pnl', 'fees': 'sell_fees'}).copy()
                    df_sell_fifo['sell_date'] = pd.to_datetime(df_sell_fifo['sell_date'], errors='coerce')
                    df_sell_fifo['sell_price'] = pd.to_numeric(df_sell_fifo['sell_price'], errors='coerce')
                    df_sell_fifo['sell_shares'] = pd.to_numeric(df_sell_fifo['sell_shares'], errors='coerce')
                    df_sell_fifo['sell_pnl'] = pd.to_numeric(df_sell_fifo['sell_pnl'], errors='coerce')
                    df_sell_fifo['sell_fees'] = pd.to_numeric(df_sell_fifo['sell_fees'], errors='coerce')

                # 针对每个 symbol 做 FIFO（按时间推进）
                for sym in sorted(set(df_buy['symbol'].unique()).union(set(df_sell_fifo['symbol'].unique() if not df_sell_fifo.empty else []))):
                    buy_queue: list[dict] = []

                    g_buy = df_buy[df_buy['symbol'] == sym].copy()
                    g_sell = df_sell_fifo[df_sell_fifo['symbol'] == sym].copy() if not df_sell_fifo.empty else pd.DataFrame()

                    events = []
                    if not g_buy.empty:
                        for _, b in g_buy.iterrows():
                            events.append({'dt': b.get('buy_date'), 'kind': 'BUY', 'row': b})
                    if not g_sell.empty:
                        for _, s in g_sell.iterrows():
                            events.append({'dt': s.get('sell_date'), 'kind': 'SELL', 'row': s})

                    # 同一天/同一时刻：先 BUY 再 SELL（符合日内先买再卖的直觉；且我们交易层已禁止同日卖出）
                    events.sort(key=lambda x: (pd.to_datetime(x['dt'], errors='coerce'), 0 if x['kind'] == 'BUY' else 1))

                    for ev in events:
                        if ev['kind'] == 'BUY':
                            b = ev['row']
                            b_shares = int(b.get('buy_shares') or 0)
                            if b_shares <= 0:
                                continue
                            buy_queue.append({'buy_row': b.to_dict(), 'remain': b_shares})
                            continue

                        # SELL：消耗 buy_queue
                        s = ev['row']
                        sell_total_shares = int(s.get('sell_shares') or 0)
                        if sell_total_shares <= 0:
                            continue

                        sell_date = s.get('sell_date')
                        sell_remain = sell_total_shares
                        sell_pnl_total = float(s.get('sell_pnl') or 0.0)
                        sell_fees_total = float(s.get('sell_fees') or 0.0)

                        while sell_remain > 0 and buy_queue:
                            node = buy_queue[0]
                            # 防御：如果队首 BUY 发生在 SELL 之后，说明数据/排序异常；此时不配对，避免出现负持仓天数。
                            _bdate = pd.to_datetime(node['buy_row'].get('buy_date'), errors='coerce')
                            _sdate = pd.to_datetime(sell_date, errors='coerce')
                            if pd.notna(_bdate) and pd.notna(_sdate) and _bdate > _sdate:
                                break

                            take = min(int(node['remain']), sell_remain)
                            if take <= 0:
                                buy_queue.pop(0)
                                continue

                            ratio = take / float(sell_total_shares)
                            pair = dict(node['buy_row'])
                            pair.update({
                                'sell_date': sell_date,
                                'sell_price': s.get('sell_price'),
                                'sell_shares': take,
                                'sell_pnl': sell_pnl_total * ratio,
                                'sell_fees': sell_fees_total * ratio,
                                'reason': s.get('reason'),
                            })
                            fifo_pairs.append(pair)

                            node['remain'] -= take
                            sell_remain -= take
                            if node['remain'] <= 0:
                                buy_queue.pop(0)

                # 未平仓的 BUY（队列剩余）
                    for node in buy_queue:
                        b = dict(node['buy_row'])
                        b['buy_shares'] = int(node['remain'])
                        b['buy_amount'] = round(float(b.get('buy_price') or 0.0) * float(b.get('buy_shares') or 0.0), 2)
                        b['sell_date'] = pd.NaT
                        b['sell_price'] = pd.NA
                        b['sell_shares'] = pd.NA
                        b['sell_pnl'] = pd.NA
                        b['reason'] = pd.NA
                        b['sell_fees'] = pd.NA
                        fifo_pairs.append(b)

                df_buy_summary = pd.DataFrame(fifo_pairs)

                # 新增：持仓天数 / 是否盈利 / 异常标记（不改变交易逻辑，仅用于统计与分析）
                if not df_buy_summary.empty:
                    # 持仓天数
                    _bd = pd.to_datetime(df_buy_summary.get('buy_date'), errors='coerce')
                    _sd = pd.to_datetime(df_buy_summary.get('sell_date'), errors='coerce')
                    df_buy_summary['持仓天数'] = (_sd.dt.normalize() - _bd.dt.normalize()).dt.days

                    # 是否盈利：基于卖出盈亏 sell_pnl（未平仓 -> 缺失）
                    _pnl = pd.to_numeric(df_buy_summary.get('sell_pnl'), errors='coerce')
                    df_buy_summary['是否盈利'] = pd.Series(pd.NA, index=df_buy_summary.index, dtype='object')
                    df_buy_summary.loc[_pnl > 0, '是否盈利'] = '盈利'
                    df_buy_summary.loc[_pnl < 0, '是否盈利'] = '亏损'
                    df_buy_summary.loc[_pnl == 0, '是否盈利'] = '持平'

                    # 持仓天数异常：sell_date 存在但持仓天数 < 0
                    df_buy_summary['持仓天数异常'] = False
                    _has_sell = _sd.notna()
                    df_buy_summary.loc[_has_sell & (pd.to_numeric(df_buy_summary['持仓天数'], errors='coerce') < 0), '持仓天数异常'] = True

                # --- 新增：买入明细评分（归一化到0~100）
                # 说明：fifo_pairs 里会带上 BUY 时记录的 raw_score/score_reason
                if not df_buy_summary.empty:
                    if 'raw_score' in df_buy_summary.columns:
                        df_buy_summary['原始评分'] = pd.to_numeric(df_buy_summary['raw_score'], errors='coerce')
                        df_buy_summary['评分'] = sel._min_max_normalize(df_buy_summary['原始评分'], vmin=0.0, vmax=100.0, flat_default=50.0).round(2)
                    else:
                        df_buy_summary['原始评分'] = pd.NA
                        df_buy_summary['评分'] = pd.NA

                    if 'score_reason' in df_buy_summary.columns:
                        df_buy_summary['评分原因'] = df_buy_summary['score_reason']
                    else:
                        df_buy_summary['评分原因'] = pd.NA

                # 输出列（全部中文）
                df_buy_summary_out = df_buy_summary.rename(columns={
                    'symbol': '代码',
                    'trade_no': '序号',
                    'buy_date': '买入日期',
                    'buy_price': '买入价格',
                    'buy_shares': '买入股数',
                    'buy_amount': '买入金额',
                    'buy_fees': '买入税费',
                    'sell_date': '卖出日期',
                    'sell_price': '卖出价格',
                    'sell_shares': '卖出股数',
                    'sell_pnl': '卖出盈亏',
                    'reason': '卖出原因',
                    'sell_fees': '卖出税费',
                    'signal_date': '信号日期',
                })

                # 新增：前一日一年内位置(%)（信号日计算的 pos_in_1y 就是“前一日一年内位置(%)”的口径）
                if 'pos_in_1y' in df_buy_summary_out.columns and '前一日一年内位置(%)' not in df_buy_summary_out.columns:
                    df_buy_summary_out['前一日一年内位置(%)'] = pd.to_numeric(df_buy_summary_out['pos_in_1y'], errors='coerce').round(2)

                # 新增：是否三连涨信号（由 last3_up 转换：True->是，False->否）
                if 'last3_up' in df_buy_summary_out.columns and '是否三连涨信号' not in df_buy_summary_out.columns:
                    _v = df_buy_summary_out['last3_up']
                    _b = _v.fillna(False).astype(bool)
                    df_buy_summary_out['是否三连涨信号'] = _b.map(lambda x: '是' if bool(x) else '否')

                cols_cn = [
                    '代码', '序号', '信号日期',
                    '买入日期', '买入价格', '买入股数', '买入金额', '买入税费',
                    '卖出日期', '卖出价格', '卖出股数', '卖出盈亏', '卖出原因', '卖出税费',
                    '持仓天数', '是否盈利', '持仓天数异常',
                    '前一日一年内位置(%)', '是否三连涨信号', '流通市值',
                    # 新增：评分
                    '原始评分', '评分', '评分原因',
                ]
                cols_cn = [c for c in cols_cn if c in df_buy_summary_out.columns]
                df_buy_summary_out = df_buy_summary_out[cols_cn]

                # 输出为 CSV 文件（支持中文标题）
                buy_summary_path = os.path.join(run_out_dir, '买入明细.csv')
                df_buy_summary_out.to_csv(buy_summary_path, index=False, encoding='utf-8-sig')
                print(f"买入明细已输出：{buy_summary_path}")

        # 汇总统计
        summary_rows = []

        # 资金使用率（正确口径）：用“权益曲线”计算资金占用（总权益-现金=持仓市值）
        # 回撤（正确口径）：按权益曲线计算最大回撤
        if not df_equity.empty:
            _eq = df_equity.sort_values('date').copy()
            _eq['equity'] = pd.to_numeric(_eq.get('equity'), errors='coerce')
            _eq['cash'] = pd.to_numeric(_eq.get('cash'), errors='coerce')

            # 平均资金占用率 = 平均(持仓市值 / 总权益)
            _eq['hold_value'] = (_eq['equity'] - _eq['cash']).clip(lower=0)
            _eq['capital_usage_ratio'] = (_eq['hold_value'] / _eq['equity']).where(_eq['equity'] > 0)
            avg_capital_usage = float(_eq['capital_usage_ratio'].mean()) if _eq['capital_usage_ratio'].notna().any() else 0.0

            # 总收益率
            last_equity_val = float(_eq['equity'].iloc[-1])
            total_return = (last_equity_val - float(cfg.initial_capital)) / float(cfg.initial_capital)

            # 最大回撤（权益曲线）
            _eq['peak_equity'] = _eq['equity'].cummax()
            _eq['drawdown'] = (_eq['equity'] - _eq['peak_equity']) / _eq['peak_equity']
            max_drawdown = float(_eq['drawdown'].min()) if _eq['drawdown'].notna().any() else 0.0

            print(f"资金平均占用率: {avg_capital_usage*100:.2f}% | 总收益率: {total_return*100:.2f}% | 最大回撤: {abs(max_drawdown)*100:.2f}%")

            summary_rows.append({'指标': '资金平均占用率', '数值': round(avg_capital_usage * 100, 4), '单位': '%'})
            summary_rows.append({'指标': '总收益率', '数值': round(total_return * 100, 4), '单位': '%'})
            summary_rows.append({'指标': '最大回撤', '数值': round(abs(max_drawdown) * 100, 4), '单位': '%'})

        # 胜率统计（按“卖出/平仓”为一笔）（同时写入CSV）
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

                # 控制台输出（中文）
                print(f"平仓统计: 盈利 {win_cnt} | 亏损 {loss_cnt} | 持平 {even_cnt} | 平仓总笔数 {total_cnt}")
                print(f"胜率(按平仓, 不含持平): {win_rate:.2f}%")

                # 汇总表（中文字段）
                summary_rows.append({'指标': '平仓盈利笔数', '数值': win_cnt, '单位': '笔'})
                summary_rows.append({'指标': '平仓亏损笔数', '数值': loss_cnt, '单位': '笔'})
                summary_rows.append({'指标': '平仓持平笔数', '数值': even_cnt, '单位': '笔'})
                summary_rows.append({'指标': '平仓总笔数', '数值': total_cnt, '单位': '笔'})
                summary_rows.append({'指标': '胜率(按平仓, 不含持平)', '数值': round(win_rate, 4), '单位': '%'})

        # 回测完成信息（中文）（同时写入CSV）
        last_equity_val = float(df_equity['equity'].iloc[-1]) if not df_equity.empty else float(cash)
        print(f"回测完成：{len(test_days)} 天 | 期末权益: {last_equity_val:.2f} | 输出目录: {run_out_dir}")
        print(f"无法买入（资金不足/单笔金额不足导致买不起1股）的股票次数: {buy_skip_insufficient_cash}")

        summary_rows.append({'指标': '回测天数', '数值': len(test_days), '单位': '天'})
        summary_rows.append({'指标': '期末权益', '数值': round(last_equity_val, 2), '单位': '元'})
        summary_rows.append({'指标': '输出目录', '数值': run_out_dir, '单位': ''})
        summary_rows.append({'指标': '无法买入次数(资金不足/单笔金额不足)', '数值': buy_skip_insufficient_cash, '单位': '次'})

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(os.path.join(run_out_dir, '回测统计.csv'), index=False, encoding='utf-8-sig')

        return run_out_dir
    except Exception as e:
        print('回测运行异常：', repr(e))
        raise

if __name__ == '__main__':
    # 直接运行脚本时的入口
    main()
