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
    default_out: str = os.path.join(os.path.dirname(__file__), 'output', 'selection_local.csv')
    date_subdir: bool = True  # 是否按当天日期(YYYYMMDD)创建子文件夹保存输出
    # 市值过滤（单位：元）：总市值区间与是否跳过过滤
    mktcap_min: float = 8e9   # 市值下限，默认80亿
    mktcap_max: float = 15e9  # 市值上限，默认150亿
    skip_mktcap: bool = True # 跳过市值/估值过滤开关
    # 选股核心参数（全部硬编码）：窗口、振幅、接近低点、涨停统计与阈值、放量参数、A股10%前缀过滤
    months: int = 3                   # 近几个月作为振幅窗口
    range_lower: float = 0.10         # 振幅下限（比例）
    range_upper: float = 0.20         # 振幅上限（比例）
    near_low_tol: float = 0.03        # 接近近低容差（比例）
    limitup_months: int = 12          # 涨停统计窗口（月）
    limitup_threshold: float = 0.098  # 涨停阈值（收盘涨幅近似≥9.8%）
    min_limitup_count: int = 2        # 正价涨停次数最少要求（近一年）
    vol_days: int = 5                 # 放量考察天数
    vol_factor: float = 2.0           # 放量倍数阈值（当日成交量≥前一日X倍）
    only_10pct_a: bool = True         # 是否仅保留10%涨停上限A股常见前缀
    # 估值过滤（PE-TTM）：估值上下限，None 表示不限制
    pe_min: float | None = None       # PE(TTM)下限，例如10.0
    pe_max: float | None = None       # PE(TTM)上限，例如40.0
    # 形态附加条件：最近N天全部收红，或最近N天内存在单日涨幅≥阈值
    last_n_days_red_n: int = 3        # 最近收红天数要求的窗口（默认3天）
    up_pct_days_n: int = 3            # 检查单日涨幅的最近天数窗口（默认3天）
    up_pct_threshold: float = 0.03     # 单日涨幅阈值（比例，默认3%）

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
    # 解析日期并排序
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
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
    """最近 n 天内存在单日涨幅 ≥ threshold（按收盘相对前一日收盘）。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n + 1).copy()
    if len(sub) < 2:
        return False
    pre_close = sub['close'].shift(1)
    pct = (sub['close'] - pre_close) / pre_close
    pct = pct.iloc[1:]  # 排除第一天无前收
    return bool((pct >= threshold - 1e-4).any())


def scan_dir(data_dir: str,
             months_lookback: int,
             range_lower: float,
             range_upper: float,
             near_low_tol: float,
             limitup_lookback_months: int,
             limitup_threshold: float,
             volume_spike_days: int,
             volume_spike_factor: float,
             only_10pct_a: bool) -> pd.DataFrame:
    end = datetime.today().date()
    start_3m = end - relativedelta(months=months_lookback)
    start_1y = end - relativedelta(months=limitup_lookback_months)
    s3 = pd.to_datetime(start_3m.strftime('%Y-%m-%d'))
    s1 = pd.to_datetime(start_1y.strftime('%Y-%m-%d'))
    e = pd.to_datetime(end.strftime('%Y-%m-%d'))

    files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith('.csv')]
    files.sort()

    results = []
    pbar = tqdm(total=len(files), desc='筛选(本地CSV)', dynamic_ncols=True)
    found = 0

    for fn in files:
        fp = os.path.join(data_dir, fn)
        # 解析 symbol/code/market
        stem = os.path.splitext(fn)[0]  # e.g., 000001.SZ
        parts = stem.split('.')
        if len(parts) != 2:
            pbar.update(1)
            continue
        code, market = parts[0], parts[1]
        market = market.upper()
        if only_10pct_a and not is_10pct_a_share(code, market):
            pbar.update(1)
            continue

        df = load_csv(fp)
        if df.empty or len(df) < 10:
            pbar.update(1)
            continue

        # 窗口切片
        bars_all = df[(df['trade_date'] >= s1) & (df['trade_date'] <= e)]
        bars_3m = df[(df['trade_date'] >= s3) & (df['trade_date'] <= e)]
        if bars_3m.empty or len(bars_3m) < 10:
            pbar.update(1)
            continue

        high, low, last_close, rng = calc_metrics(bars_3m)
        if pd.isna(rng):
            pbar.update(1)
            continue

        # 计算今年区间内位置（按近一年窗口 bars_all）
        pos_pct = None
        if not bars_all.empty:
            high_1y = float(bars_all['high'].max())
            low_1y = float(bars_all['low'].min())
            if high_1y > low_1y and low_1y > 0:
                pos_pct = (last_close - low_1y) / (high_1y - low_1y)
                # 夹在 0~1 范围
                pos_pct = max(0.0, min(1.0, pos_pct))

        limitup_cnt = count_limit_up_days(bars_all, limitup_threshold)
        vol_spike = has_volume_spike_last_n_days(bars_all, volume_spike_days, volume_spike_factor)
        # 新增条件：最近N天收红，或者最近N天内存在单日涨幅≥阈值（统一参数）
        extra_cond = _last_n_days_red(bars_all, CFG.last_n_days_red_n) or _has_single_day_up_pct(bars_all, CFG.up_pct_days_n, CFG.up_pct_threshold)

        if (
            range_lower - 1e-9 <= rng <= range_upper + 1e-9
            and near_low_enough(last_close, low, near_low_tol)
            and limitup_cnt > 5
            and vol_spike
            and extra_cond
        ):
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
                'last3_red': _last_n_days_red(bars_all, CFG.last_n_days_red_n),
                'last3_has_up3pct': _has_single_day_up_pct(bars_all, CFG.up_pct_days_n, CFG.up_pct_threshold),
            })
            found += 1
            try:
                tqdm.write(
                    f"入选: {stem} | 高:{round(high,3)} 低:{round(low,3)} 现:{round(last_close,3)} 振幅:{round(rng*100,2)}% "
                    f"涨停:{limitup_cnt} 放量:{'是' if vol_spike else '否'} 位置(1年):{(round(pos_pct*100,2) if isinstance(pos_pct,float) else 'N/A')}% "
                    f"近3天收红:{'是' if _last_n_days_red(bars_all,3) else '否'} 单日≥3%:{'是' if _has_single_day_up_pct(bars_all,3,0.03) else '否'}"
                )
            except Exception:
                pass
        pbar.set_postfix(found=found, code=stem)
        pbar.update(1)

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
        date_str = datetime.today().strftime('%Y%m%d')
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

    # 使用可配置的正价涨停次数要求（硬编码）
    df = df[df['limit_up_days_1y'] >= int(min_limitup_count)].copy()
    if df.empty:
        print('正价涨停次数过滤后无标的')
        return

    df = df.sort_values('range_pct').reset_index(drop=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(df)

    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已保存到: {out_path} | 入选数量: {len(df)}")


if __name__ == '__main__':
    main()
