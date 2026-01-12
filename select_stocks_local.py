import os
# import argparse  # 不再使用命令行参数
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from tqdm import tqdm
import requests
import json
from dataclasses import dataclass

# 统一参数配置（集中管理默认值）
@dataclass
class Config:
    # 路径与输出（硬编码）：数据源目录与结果文件路径
    data_dir: str = os.path.join(os.path.dirname(__file__), '通达信', 'data', 'pytdx', 'daily_raw')
    # 默认文件名改为带时间戳，避免多次单独运行覆盖
    default_out: str = os.path.join(
        os.path.dirname(__file__),
        'output',
        f"selection_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    date_subdir: bool = True  # 是否按当天日期(YYYYMMDD)创建子文件夹保存输出
    # 市值过滤（单位：元）：总市值区间与是否跳过过滤
    mktcap_min: float = 8e9   # 市值下限，默认80亿
    mktcap_max: float = 15e9  # 市值上限，默认150亿
    skip_mktcap: bool = True # 跳过市值/估值过滤开关
    # 选股核心参数（全部硬编码）：窗口、振幅、接近低点、涨停统计与阈值、放量参数、A股10%前缀过滤
    months: int = 3                   # 近几个月作为振幅窗口
    range_lower: float = 0.10         # 振幅下限（比例）
    range_upper: float = 0.20         # 振幅上限（比例）
    near_low_tol: float = 0.10        # 接近近低容差（比例）
    limitup_months: int = 12          # 涨停统计窗口（月）
    limitup_threshold: float = 0.098  # 涨停阈值（收盘涨幅近似≥9.8%）
    min_limitup_count: int = 3        # 正价涨停次数最少要求（近一年）
    vol_days: int = 5                 # 放量考察天数（当前逻辑已不用最近n天，而是当日 vs 前一日）
    vol_factor: float = 2           # 放量倍数阈值（当日成交量≥前一日X倍）
    only_10pct_a: bool = True         # 是否仅保留10%涨停上限A股常见前缀
    # 估值过滤（PE-TTM）：估值上下限，None 表示不限制
    pe_min: float | None = None       # PE(TTM)下限，例如10.0
    pe_max: float | None = None       # PE(TTM)上限，例如40.0
    # 形态附加条件：最近N天全部收红，或最近N天内存在单日涨幅≥阈值
    use_extra_cond: bool = True       # 是否启用附加形态条件过滤
    last_n_days_red_n: int = 3        # 最近收红天数要求的窗口（默认3天），也作为“连续上涨天数”
    up_pct_days_n: int = 3            # 最近涨幅检查天数（当前逻辑由“任意一日”改为“当日”）
    up_pct_threshold: float = 0.03     # 单日涨幅阈值（比例，默认3%）
    # 当日最高价与收盘价允许的最大偏离比例（默认 10%）
    high_close_tol: float = 0.10
    # 启动质量开关：是否启用启动质量过滤
    enable_start_quality: bool = True
    # 1年区间位置过滤（单位：%）。None 表示不启用。
    # pos_1y_min_pct: float | None = 30.0
    # pos_1y_max_pct: float | None = 65.0
    pos_1y_min_pct: float | None = None
    pos_1y_max_pct: float | None = None
    # 单股测试：仅扫描该股票文件名（如 "603598.SH"），None 表示不限制
    test_single_symbol: str | None = None

CFG = Config()

# 数据目录格式：

# CSV 文件名形如 000001.SZ.csv
# 列为 trade_date,open,high,low,close,volume,amount
# 默认筛选条件与 Tushare 版一致：

# 近3个月振幅在 10%~20%
# 收盘接近近低 3% 内
# 最近一年涨停日数 > 5（按收盘涨幅≥9.8%近似）
# 最近5日存在放量（当日成交量≥前一日2倍）
# 仅 A 股 10% 涨停上限常见前缀（沪:600/601/603/605，深:000/001/002/003）
# 使用示例（你的数据目录）： powershell: python select_stocks_local.py --data-dir "E:\work\SynologyDrive\量化交易\通达信\通达信\data\pytdx\daily_raw" --out ".\output\selection_local.csv"

# 如需调整阈值，可加参数：

# --months 3
# --range-lower 0.10 --range-upper 0.20
# --near-low-tol 0.03
# --limitup-months 12 --limitup-threshold 0.098
# --vol-days 5 --vol-factor 2.0
# 关闭仅10% A股过滤：--no-only-10pct-a

# 仅保留 A 股中 10% 涨停上限常见前缀（不含 B 股）
ALLOW_PREFIXES_10P_SH = ("600", "601", "603", "605")  # 沪A
ALLOW_PREFIXES_10P_SZ = ("000", "001", "002", "003")  # 深A


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    # 规范列
    exp = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    missing = [c for c in exp if c not in df.columns]
    if missing:
        return pd.DataFrame()

    # 解析日期并排序（兼容 trade_date 为 int/float 导致被误解析成 1970 时间戳的问题）
    # 预期格式：YYYYMMDD（如 20260112）
    td_raw = df['trade_date']
    td_str = td_raw.astype(str).str.strip()
    td_str = td_str.str.replace(r'\.0$', '', regex=True)

    td = pd.to_datetime(td_str, format='%Y%m%d', errors='coerce')
    # 若不是 YYYYMMDD（例如带 '-' 或 'YYYY/MM/DD'），则回退通用解析
    mask = td.isna() & td_str.ne('')
    if bool(mask.any()):
        td.loc[mask] = pd.to_datetime(td_str.loc[mask], errors='coerce')

    df['trade_date'] = td
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)

    # 数值列
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def is_10pct_a_share(code: str, market: str) -> bool:
    if market == 'SH':
        return code.startswith(ALLOW_PREFIXES_10P_SH)
    else:
        return code.startswith(ALLOW_PREFIXES_10P_SZ)


def calc_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    if df.empty:
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    high = float(df['high'].max())
    low = float(df['low'].min())
    last_close = float(df['close'].iloc[-1])
    if low <= 0:
        return (high, low, last_close, float('nan'))
    rng = (high - low) / low
    return (high, low, last_close, rng)


def near_low_enough(last_close: float, low: float, tol: float) -> bool:
    import math
    if any(math.isnan(x) for x in [last_close, low]) or low <= 0:
        return False
    return abs(last_close - low) / low <= tol


def count_limit_up_days(df: pd.DataFrame, threshold: float = 0.098) -> int:
    if df is None or df.empty:
        return 0
    pre_close = df['close'].shift(1)
    pct = (df['close'] - pre_close) / pre_close
    cond = (pre_close > 0) & (pct >= (threshold - 1e-4))
    return int(cond.sum())


def has_volume_spike_last_n_days(df: pd.DataFrame, n: int = 5, factor: float = 2.0) -> bool:
    if df is None or df.empty or 'volume' not in df.columns:
        return False
    sub = df.tail(n + 1).copy()
    if len(sub) < 2:
        return False
    ratios = sub['volume'] / sub['volume'].shift(1)
    return bool((ratios.iloc[1:] >= factor).any())


def _last_n_days_red(df: pd.DataFrame, n: int = CFG.last_n_days_red_n) -> bool:
    """最近 n 天全部收红（收盘价 > 开盘价）。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n)
    if len(sub) < n:
        return False
    return bool((sub['close'] > sub['open']).all())


def _has_single_day_up_pct(df: pd.DataFrame, n: int = CFG.up_pct_days_n, threshold: float = CFG.up_pct_threshold) -> bool:
    """[旧逻辑] 最近 n 天内存在单日涨幅 ≥ threshold（按收盘相对前一日收盘）。保留以兼容历史，如不需要可删除。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n + 1).copy()
    if len(sub) < 2:
        return False
    pre_close = sub['close'].shift(1)
    pct = (sub['close'] - pre_close) / pre_close
    pct = pct.iloc[1:]  # 排除第一天无前收
    return bool((pct >= threshold - 1e-4).any())


def _is_today_volume_spike(df: pd.DataFrame, factor: float = CFG.vol_factor) -> bool:
    """当日成交量是否相对前一日放量：volume_today >= factor * volume_yesterday。"""
    if df is None or len(df) < 2 or 'volume' not in df.columns:
        return False
    today_vol = df['volume'].iloc[-1]
    prev_vol = df['volume'].iloc[-2]
    if prev_vol <= 0:
        return False
    return bool(today_vol >= factor * prev_vol)


def _is_today_up_pct(df: pd.DataFrame, threshold: float = CFG.up_pct_threshold) -> bool:
    """当日收盘相对前一日收盘涨幅是否 ≥ threshold。"""
    if df is None or len(df) < 2 or 'close' not in df.columns:
        return False
    prev_close = df['close'].iloc[-2]
    today_close = df['close'].iloc[-1]
    if prev_close <= 0:
        return False
    pct = (today_close - prev_close) / prev_close
    return bool(pct >= threshold - 1e-4)


def _is_last_n_days_all_up(df: pd.DataFrame, n: int = CFG.last_n_days_red_n) -> bool:
    """最近 n 天连续上涨：每一天的收盘价都高于前一日收盘价。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n + 1)
    if len(sub) < n + 1:
        return False
    pre_close = sub['close'].shift(1)
    pct = (sub['close'] - pre_close) / pre_close
    pct = pct.iloc[1:]  # 最近 n 天
    return bool((pct > 0).all())


def scan_dir(data_dir: str,
             months_lookback: int,
             range_lower: float,
             range_upper: float,
             near_low_tol: float,
             limitup_lookback_months: int,
             limitup_threshold: float,
             volume_spike_days: int,
             volume_spike_factor: float,
             only_10pct_a: bool,
             end_date: pd.Timestamp | None = None,
             preloaded: dict[str, pd.DataFrame] | None = None,
             quiet: bool = False,
             pos_1y_min_pct: float | None = None,
             pos_1y_max_pct: float | None = None,
             min_limitup_count: int | None = None) -> pd.DataFrame:
    # 允许传入截断日期（用于回测），默认使用今天
    end_dt = (pd.to_datetime(end_date) if end_date is not None else pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
    end = end_dt.date()
    start_3m = end - relativedelta(months=months_lookback)
    start_1y = end - relativedelta(months=limitup_lookback_months)
    s3 = pd.to_datetime(start_3m.strftime('%Y-%m-%d'))
    s1 = pd.to_datetime(start_1y.strftime('%Y-%m-%d'))
    e = pd.to_datetime(end.strftime('%Y-%m-%d'))

    # 准备迭代的源：文件或预加载数据
    if preloaded is None:
        files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith('.csv')]
        files.sort()
        iterable = [(os.path.splitext(fn)[0], None, os.path.join(data_dir, fn)) for fn in files]
        pbar = None if quiet else tqdm(total=len(files), desc='筛选(本地CSV)', dynamic_ncols=True)
    else:
        keys = sorted(preloaded.keys())
        iterable = [(stem, preloaded.get(stem), None) for stem in keys]
        pbar = None if quiet else tqdm(total=len(keys), desc='筛选(预加载)', dynamic_ncols=True)

    results = []
    found = 0

    for stem, df_pre, fp in iterable:
        # 如果设置了单股测试，只保留指定 symbol（例如 "603598.SH"）
        if CFG.test_single_symbol is not None and stem != CFG.test_single_symbol:
            if pbar: pbar.update(1)
            continue
        # 解析 symbol/code/market
        parts = stem.split('.')
        if len(parts) != 2:
            if pbar: pbar.update(1)
            continue
        code, market = parts[0], parts[1].upper()
        if only_10pct_a and not is_10pct_a_share(code, market):
            if pbar: pbar.update(1)
            continue

        # 加载数据
        if df_pre is not None:
            df = df_pre
        else:
            df = load_csv(fp) if fp is not None else pd.DataFrame()
        if df.empty or len(df) < 10:
            if pbar: pbar.update(1)
            continue

        # 窗口切片（按传入的 e 截断）
        df_cut = df[df['trade_date'] <= e]
        if df_cut.empty or len(df_cut) < 10:
            if pbar: pbar.update(1)
            continue
        bars_all = df_cut[(df_cut['trade_date'] >= s1) & (df_cut['trade_date'] <= e)]
        bars_3m = df_cut[(df_cut['trade_date'] >= s3) & (df_cut['trade_date'] <= e)]
        if bars_3m.empty or len(bars_3m) < 10:
            if pbar: pbar.update(1)
            continue

        high, low, last_close, rng = calc_metrics(bars_3m)
        if pd.isna(rng):
            if pbar: pbar.update(1)
            continue

        # 计算当日 K 线实体位置约束：最高价与收盘价偏离不超过 CFG.high_close_tol
        today_high = float(bars_3m['high'].iloc[-1])
        today_low = float(bars_3m['low'].iloc[-1])
        today_close = float(bars_3m['close'].iloc[-1])
        high_close_ok = True
        if today_close > 0:
            high_close_ok = (today_high - today_close) / today_close <= CFG.high_close_tol + 1e-4

        # 计算今年区间内位置（按近一年窗口 bars_all）
        pos_pct = None
        if not bars_all.empty:
            high_1y = float(bars_all['high'].max())
            low_1y = float(bars_all['low'].min())
            if high_1y > low_1y and low_1y > 0:
                pos_pct = (last_close - low_1y) / (high_1y - low_1y)
                # 夹在 0~1 范围
                pos_pct = max(0.0, min(1.0, pos_pct))

        # 1年区间位置过滤（pos_in_1y，单位%）：默认要求 30~65
        pos_ok = True
        if pos_1y_min_pct is not None or pos_1y_max_pct is not None:
            if not isinstance(pos_pct, float):
                pos_ok = False
            else:
                pos_pct100 = pos_pct * 100.0
                if pos_1y_min_pct is not None:
                    pos_ok = pos_ok and (pos_pct100 >= pos_1y_min_pct - 1e-9)
                if pos_1y_max_pct is not None:
                    pos_ok = pos_ok and (pos_pct100 <= pos_1y_max_pct + 1e-9)

        limitup_cnt = count_limit_up_days(bars_all, limitup_threshold)

        # 统一涨停次数门槛：优先使用传入参数，否则退化为全局配置 CFG.min_limitup_count
        _min_lu = CFG.min_limitup_count if min_limitup_count is None else int(min_limitup_count)

        vol_spike = _is_today_volume_spike(bars_all, volume_spike_factor)
        today_up = _is_today_up_pct(bars_all, CFG.up_pct_threshold)
        cont_up_n = _is_last_n_days_all_up(bars_all, CFG.last_n_days_red_n)
        extra_cond_val = (vol_spike and today_up) or cont_up_n
        extra_cond = (not CFG.use_extra_cond) or extra_cond_val

        # 启动质量：当日上涨 + 放量 + 收盘价不低于当日中轴
        start_quality = True
        if CFG.enable_start_quality:
            mid_price = (today_high + today_low) / 2.0 if today_high > 0 and today_low > 0 else float('inf')
            start_quality = bool(today_up and vol_spike and today_close >= mid_price - 1e-9)

        # 使用配置的 min_limitup_count 作为统一门槛
        passed = (
            range_lower - 1e-9 <= rng <= range_upper + 1e-9
            and near_low_enough(last_close, low, near_low_tol)
            and limitup_cnt >= _min_lu
            and vol_spike
            and extra_cond
            and high_close_ok
            and start_quality
            and pos_ok
        )

        # 单股测试时，如果没通过，打印各条件方便排查
        if CFG.test_single_symbol is not None and stem == CFG.test_single_symbol and not passed and not quiet:
            try:
                print(
                    f"[DEBUG] {stem} 未入选原因: "
                    f"振幅={rng:.4f} 是否在区间[{range_lower},{range_upper}]内={range_lower - 1e-9 <= rng <= range_upper + 1e-9}; "
                    f"接近近低={near_low_enough(last_close, low, near_low_tol)}; "
                    f"近一年涨停天数={limitup_cnt} 是否≥配置门槛{_min_lu}={limitup_cnt >= _min_lu}; "
                    f"当日放量={vol_spike}; 当日涨幅达标={today_up}; 近{CFG.last_n_days_red_n}天连续上涨={cont_up_n}; "
                    f"当日高收价偏离≤{CFG.high_close_tol*100:.1f}%={high_close_ok}; "
                    f"启动质量(涨+量+收盘≥中轴)={start_quality}; "
                    f"附加条件整体={extra_cond}"
                )
            except Exception:
                pass

        if passed:
            results.append({
                'symbol': stem,
                'code': code,
                'market': market,
                'high': round(high, 3),
                'low': round(low, 3),
                'last_close': round(last_close, 3),
                'range_pct': round(rng * 100, 2),
                'limit_up_days_1y': int(limitup_cnt),
                'vol_spike_5d': bool(vol_spike),
                'pos_in_1y': (round(pos_pct * 100, 2) if isinstance(pos_pct, float) else None),
                'last3_up': _is_last_n_days_all_up(bars_all, CFG.last_n_days_red_n),
                'today_up_ge_3pct': _is_today_up_pct(bars_all, CFG.up_pct_threshold),
                # 可选：输出启动质量标志
                'start_quality': bool(start_quality),
            })
            found += 1
            if not quiet:
                try:
                    tqdm.write(
                        f"入选: {stem} | 高:{round(high,3)} 低:{round(low,3)} 现:{round(last_close,3)} 振幅:{round(rng*100,2)}% "
                        f"涨停:{limitup_cnt} 放量:{'是' if vol_spike else '否'} 位置(1年):{(round(pos_pct*100,2) if isinstance(pos_pct,float) else 'N/A')}% "
                        f"近{CFG.last_n_days_red_n}天连续上涨:{'是' if _is_last_n_days_all_up(bars_all,CFG.last_n_days_red_n) else '否'} "
                        f"当日涨幅≥{int(CFG.up_pct_threshold*100)}%:{'是' if _is_today_up_pct(bars_all,CFG.up_pct_threshold) else '否'} "
                        f"附加条件:{'未启用' if not CFG.use_extra_cond else '启用'}"
                    )
                except Exception:
                    pass
        if pbar:
            pbar.set_postfix(found=found, code=stem)
            pbar.update(1)

    if pbar:
        pbar.close()
    return pd.DataFrame(results)


def _fetch_market_cap_eastmoney(code: str, market: str, timeout: float = 5.0) -> float | None:
    """通过东方财富接口获取总市值(元)。返回 float 或 None。
    市场编码：SH->1，SZ->0
    字段 f20: 总市值(元)
    """
    try:
        secid = ('1' if market.upper() == 'SH' else '0') + f'.{code}'
        url = f'https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f20'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        data = j.get('data') or {}
        val = data.get('f20')
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _fetch_pe_ttm_eastmoney(code: str, market: str, timeout: float = 5.0) -> float | None:
    """通过东方财富接口获取 PE(TTM)。字段 f9。返回 float 或 None。"""
    try:
        secid = ('1' if market.upper() == 'SH' else '0') + f'.{code}'
        url = f'https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f9'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        data = j.get('data') or {}
        val = data.get('f9')
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _filter_by_market_cap_and_valuation(rows: list[dict], mc_min: float, mc_max: float, pe_min: float | None, pe_max: float | None) -> list[dict]:
    """按总市值与估值过滤。"""
    out = []
    pbar = tqdm(total=len(rows), desc='市值/估值过滤', dynamic_ncols=True)
    for r in rows:
        code = r.get('code', '')
        market = r.get('market', '')
        mc = _fetch_market_cap_eastmoney(code, market)
        pe = _fetch_pe_ttm_eastmoney(code, market)
        r['market_cap'] = mc
        r['pe_ttm'] = pe
        ok_mc = (mc is not None and mc_min - 1e-6 <= mc <= mc_max + 1e-6)
        ok_pe = True
        if pe_min is not None:
            ok_pe = ok_pe and (pe is not None and pe >= pe_min - 1e-9)
        if pe_max is not None:
            ok_pe = ok_pe and (pe is not None and pe <= pe_max + 1e-9)
        if ok_mc and ok_pe:
            out.append(r)
        pbar.set_postfix(code=f"{code}.{market}", mc=mc, pe=pe)
        pbar.update(1)
    pbar.close()
    return out


def main():
    # 使用硬编码的配置
    data_dir = CFG.data_dir
    out_path = CFG.default_out

    # 可选命令行参数：指定截止日期与静默模式
    end_date_arg = None
    quiet_arg = False
    try:
        import argparse
        parser = argparse.ArgumentParser(description='本地CSV选股')
        parser.add_argument('--end-date', type=str, help='指定筛选截止日期(YYYYMMDD)，默认今天')
        parser.add_argument('--quiet', action='store_true', help='静默模式，减少日志输出')
        args, _ = parser.parse_known_args()
        if args.end_date:
            try:
                end_date_arg = pd.to_datetime(args.end_date)
            except Exception:
                print('end-date 参数格式错误，应为 YYYYMMDD，例如 20250901')
        quiet_arg = bool(args.quiet)
    except Exception:
        pass

    date_subdir = CFG.date_subdir
    mktcap_min = CFG.mktcap_min
    mktcap_max = CFG.mktcap_max
    skip_mktcap = CFG.skip_mktcap
    months = CFG.months
    range_lower = CFG.range_lower
    range_upper = CFG.range_upper
    near_low_tol = CFG.near_low_tol
    limitup_months = CFG.limitup_months
    limitup_threshold = CFG.limitup_threshold
    min_limitup_count = CFG.min_limitup_count
    vol_days = CFG.vol_days
    vol_factor = CFG.vol_factor
    only_10pct_a = CFG.only_10pct_a
    pe_min = CFG.pe_min
    pe_max = CFG.pe_max

    # 处理按日期子目录
    if date_subdir:
        date_str = datetime.today().strftime('%Y%m%d') if end_date_arg is None else pd.to_datetime(end_date_arg).strftime('%Y%m%d')
        out_dir = os.path.dirname(out_path) if out_path.lower().endswith('.csv') else out_path
        out_dir = os.path.join(out_dir, date_str)
        base_name = os.path.basename(out_path) if out_path.lower().endswith('.csv') else os.path.basename(CFG.default_out)
        out_path = os.path.join(out_dir, base_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 扫描并筛选
    df = scan_dir(
        data_dir=data_dir,
        months_lookback=months,
        range_lower=range_lower,
        range_upper=range_upper,
        near_low_tol=near_low_tol,
        limitup_lookback_months=limitup_months,
        limitup_threshold=limitup_threshold,
        volume_spike_days=vol_days,
        volume_spike_factor=vol_factor,
        only_10pct_a=only_10pct_a,
        end_date=end_date_arg,
        quiet=quiet_arg,
        pos_1y_min_pct=CFG.pos_1y_min_pct,
        pos_1y_max_pct=CFG.pos_1y_max_pct,
        min_limitup_count=min_limitup_count,
    )

    if df.empty:
        print('无符合条件的标的')
        return

    # 市值/估值过滤（使用硬编码参数）
    if not skip_mktcap:
        rows = df.to_dict(orient='records')
        rows = _filter_by_market_cap_and_valuation(rows, mktcap_min, mktcap_max, pe_min, pe_max)
        df = pd.DataFrame(rows)
        if df.empty:
            print('市值/估值过滤后无标的（检查区间是否合理）')
            return

    df = df.sort_values('range_pct').reset_index(drop=True)

    # 输出前将列名改为中文
    col_map = {
        'symbol': '标的',
        'code': '代码',
        'market': '市场',
        'high': '最高价',
        'low': '最低价',
        'last_close': '最新收盘价',
        'range_pct': '振幅(%)',
        'limit_up_days_1y': '近一年涨停天数',
        'vol_spike_5d': '当日放量',
        'pos_in_1y': '一年区间位置(%)',
        'last3_up': '近3天连续上涨',
        'today_up_ge_3pct': '当日涨幅≥3%'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if not quiet_arg:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        print(df)

    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已保存到: {out_path} | 入选数量: {len(df)} | 截止日期: {(pd.to_datetime(end_date_arg).strftime('%Y-%m-%d') if end_date_arg is not None else datetime.today().strftime('%Y-%m-%d'))}")


if __name__ == '__main__':
    main()
