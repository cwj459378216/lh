import os
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from tqdm import tqdm

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


def load_cap_csv(path: str) -> pd.DataFrame:
    """读取流通市值映射 CSV，支持列(code/symbol, cap)，cap单位可为亿元或元。
    规范输出列：code(6位)、cap_billion(以亿元计)。"""
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame()
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # 提取 code
    if 'code' in df.columns:
        out = df[['code']].copy()
        out['code'] = out['code'].str[-6:].str.zfill(6)
    elif 'symbol' in df.columns:
        out = pd.DataFrame({'code': df['symbol'].str[:6]})
    else:
        return pd.DataFrame()
    # 提取 cap
    if 'cap' in df.columns:
        cap_raw = pd.to_numeric(df['cap'], errors='coerce')
    elif 'float_cap' in df.columns:
        cap_raw = pd.to_numeric(df['float_cap'], errors='coerce')
    else:
        return pd.DataFrame()
    # 自动识别单位：若平均值>1000，则可能为“元”，转换为“亿元”。
    avg = pd.to_numeric(cap_raw, errors='coerce').dropna().mean()
    if pd.isna(avg):
        return pd.DataFrame()
    if avg and avg > 1000:  # 以元为单位，转亿元
        cap_billion = cap_raw / 1e8
    else:  # 已是亿元
        cap_billion = cap_raw
    out['cap_billion'] = pd.to_numeric(cap_billion, errors='coerce')
    return out.dropna(subset=['cap_billion']).reset_index(drop=True)


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
             cap_map: pd.DataFrame | None,
             cap_min_billion: float,
             cap_max_billion: float) -> pd.DataFrame:
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

        # 市值过滤（可选）
        if cap_map is not None and not cap_map.empty:
            row = cap_map[cap_map['code'] == code]
            if row.empty:
                pbar.update(1)
                continue
            cap_bil = float(row['cap_billion'].iloc[0])
            if not (cap_min_billion <= cap_bil <= cap_max_billion):
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

        limitup_cnt = count_limit_up_days(bars_all, limitup_threshold)
        vol_spike = has_volume_spike_last_n_days(bars_all, volume_spike_days, volume_spike_factor)

        if (
            range_lower - 1e-9 <= rng <= range_upper + 1e-9
            and near_low_enough(last_close, low, near_low_tol)
            and limitup_cnt > 5
            and vol_spike
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
            })
            found += 1
            try:
                tqdm.write(f"入选: {stem} | 高:{round(high,3)} 低:{round(low,3)} 现:{round(last_close,3)} 振幅:{round(rng*100,2)}% 涨停:{limitup_cnt} 放量:{'是' if vol_spike else '否'}")
            except Exception:
                pass
        pbar.set_postfix(found=found, code=stem)
        pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='本地 CSV（pytdx 导出）选股：参考 select_stocks.py 逻辑')
    parser.add_argument('--data-dir', type=str, required=True, help='CSV 数据目录（包含 *.csv，如 000001.SZ.csv）')
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(__file__), 'output', 'selection_local.csv'))
    # 在输出目录下按日期创建子文件夹
    parser.add_argument('--date-subdir', action='store_true', default=True, help='在输出目录下按当天日期(YYYYMMDD)创建子文件夹并保存文件，默认开启')
    # 新增市值过滤参数（单位：亿元）
    parser.add_argument('--cap-csv', type=str, default=None, help='流通市值映射 CSV（含 code/symbol 与 cap 或 float_cap 列，cap单位为亿元或元自动识别）')
    parser.add_argument('--cap-min', type=float, default=80.0, help='市值下限（亿元），默认80')
    parser.add_argument('--cap-max', type=float, default=120.0, help='市值上限（亿元），默认120')
    parser.add_argument('--months', type=int, default=3, help='近几个月作为振幅窗口，默认3')
    parser.add_argument('--range-lower', type=float, default=0.10, help='振幅下限（比例），默认0.10')
    parser.add_argument('--range-upper', type=float, default=0.20, help='振幅上限（比例），默认0.20')
    parser.add_argument('--near-low-tol', type=float, default=0.03, help='接近近低容差（比例），默认0.03')
    parser.add_argument('--limitup-months', type=int, default=12, help='涨停统计窗口（月），默认12')
    parser.add_argument('--limitup-threshold', type=float, default=0.098, help='涨停阈值，默认0.098')
    parser.add_argument('--vol-days', type=int, default=5, help='放量考察天数，默认5')
    parser.add_argument('--vol-factor', type=float, default=2.0, help='放量倍数阈值，默认2.0')
    parser.add_argument('--only-10pct-a', action='store_true', default=True, help='仅 A 股 10% 涨停上限常见前缀（默认开启）')
    parser.add_argument('--no-only-10pct-a', dest='only_10pct_a', action='store_false', help='关闭仅10% A股过滤')
    args = parser.parse_args()

    # 处理按日期子目录
    out_path = args.out
    if args.date_subdir:
        date_str = datetime.today().strftime('%Y%m%d')
        out_dir = os.path.dirname(out_path) if out_path.lower().endswith('.csv') else out_path
        out_dir = os.path.join(out_dir, date_str)
        base_name = os.path.basename(out_path) if out_path.lower().endswith('.csv') else 'selection_local.csv'
        out_path = os.path.join(out_dir, base_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 读取市值映射
    cap_map = None
    if args.cap_csv:
        cap_map = load_cap_csv(args.cap_csv)
        if cap_map is None or cap_map.empty:
            tqdm.write('警告：未能读取有效的市值映射，跳过市值过滤。')
            cap_map = None

    df = scan_dir(
        data_dir=args.data_dir,
        months_lookback=args.months,
        range_lower=args.range_lower,
        range_upper=args.range_upper,
        near_low_tol=args.near_low_tol,
        limitup_lookback_months=args.limitup_months,
        limitup_threshold=args.limitup_threshold,
        volume_spike_days=args.vol_days,
        volume_spike_factor=args.vol_factor,
        only_10pct_a=args.only_10pct_a,
        cap_map=cap_map,
        cap_min_billion=args.cap_min,
        cap_max_billion=args.cap_max,
    )

    if df.empty:
        print('无符合条件的标的')
        return

    df = df.sort_values('range_pct').reset_index(drop=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(df)

    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已保存到: {out_path} | 入选数量: {len(df)}")


if __name__ == '__main__':
    main()
