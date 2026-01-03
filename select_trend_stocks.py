import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from tqdm import tqdm
import requests
from dataclasses import dataclass

# 统一参数配置（硬编码）
@dataclass
class Config:
    data_dir: str = os.path.join(os.path.dirname(__file__), '通达信', 'data', 'pytdx', 'daily_raw')
    default_out: str = os.path.join(os.path.dirname(__file__), 'output', 'trend_selection.csv')
    date_subdir: bool = True
    # 趋势判断参数（更严格）
    lookback_months: int = 9        # 近9个月作为趋势窗口
    ma_window: int = 30             # 移动平均窗口提高为30日
    ma_slope_min: float = 0.001     # MA斜率下限提高，确保明显上升
    close_above_ma_days: int = 12   # 最近至少12天收盘高于MA
    higher_highs_lows_days: int = 30 # 近30天高低点抬升
    recent_up_days: int = 3         # 最近3天至少有
    recent_up_days_required: int = 3 # 3天全部收红
    # 市值过滤（单位：元）
    enable_mktcap_filter: bool = False
    mktcap_min: float = 8e9
    mktcap_max: float = 30e9
    # A股10%涨停上限前缀过滤
    only_10pct_a: bool = True
    # 近N日无涨停要求
    no_limitup_days: int = 5        # 最近N天不得出现涨停
    limitup_threshold: float = 0.098 # 涨停阈值（收盘涨幅近似≥9.8%）
    # 最近一月涨幅上限
    month_gain_days: int = 22       # 最近一月按22个交易日近似
    max_month_gain_pct: float = 0.30 # 最近一月涨幅不得超过30%

CFG = Config()

ALLOW_PREFIXES_10P_SH = ("600", "601", "603", "605")
ALLOW_PREFIXES_10P_SZ = ("000", "001", "002", "003")


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    exp = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    if any(c not in df.columns for c in exp):
        return pd.DataFrame()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def is_10pct_a_share(code: str, market: str) -> bool:
    if market == 'SH':
        return code.startswith(ALLOW_PREFIXES_10P_SH)
    return code.startswith(ALLOW_PREFIXES_10P_SZ)


def ma_trend(df: pd.DataFrame, window: int) -> tuple[float, int]:
    if df is None or df.empty:
        return (float('nan'), 0)
    s = df['close'].rolling(window=window, min_periods=window).mean()
    if s.isna().all():
        return (float('nan'), 0)
    # 简单斜率：末值-首值 / 天数
    last = s.iloc[-1]
    first = s.dropna().iloc[0]
    slope = (last - first) / max(1, len(s.dropna()))
    # 最近N天收盘高于MA的天数
    recent = df.tail(CFG.close_above_ma_days)
    recent_ma = s.tail(len(recent))
    above_days = int(((recent['close'] > recent_ma).fillna(False)).sum())
    return (float(slope), above_days)


def higher_highs_lows(df: pd.DataFrame, days: int) -> bool:
    sub = df.tail(days)
    if len(sub) < 5:
        return False
    # 将窗口分三段，检查逐段抬升
    k = len(sub) // 3
    a, b, c = sub.iloc[:k], sub.iloc[k:2*k], sub.iloc[2*k:]
    if len(a) == 0 or len(b) == 0 or len(c) == 0:
        return False
    hh = (a['high'].max() < b['high'].max() < c['high'].max())
    hl = (a['low'].min() < b['low'].min() < c['low'].min())
    return bool(hh and hl)


def recent_up_days(df: pd.DataFrame, n: int, need: int) -> bool:
    sub = df.tail(n)
    if len(sub) < n:
        return False
    up = (sub['close'] > sub['open']).sum()
    return int(up) >= int(need)


def _fetch_market_cap_eastmoney(code: str, market: str, timeout: float = 5.0) -> float | None:
    try:
        secid = ('1' if market.upper() == 'SH' else '0') + f'.{code}'
        url = f'https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f20'
        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json(); data = j.get('data') or {}
        val = data.get('f20')
        return float(val) if val is not None else None
    except Exception:
        return None


def _no_limitup_last_n_days(df: pd.DataFrame, n: int, threshold: float) -> bool:
    """检查最近 n 天无涨停（按收盘相对前收涨幅 ≥ threshold 视为涨停）。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n + 1).copy()
    if len(sub) < 2:
        return False
    pre_close = sub['close'].shift(1)
    pct = (sub['close'] - pre_close) / pre_close
    pct = pct.iloc[1:]  # 去掉第一天
    return bool((pct < threshold - 1e-4).all())


def _monthly_gain_within(df: pd.DataFrame, days: int, max_pct: float) -> tuple[bool, float]:
    """检查最近 days 天涨幅是否不超过 max_pct，返回(是否符合, 实际涨幅)。"""
    if df is None or df.empty:
        return (False, float('nan'))
    sub = df.tail(days)
    if len(sub) < 2:
        return (False, float('nan'))
    first = sub['close'].iloc[0]
    last = sub['close'].iloc[-1]
    if pd.isna(first) or pd.isna(last) or first <= 0:
        return (False, float('nan'))
    pct = (last / first) - 1.0
    return (pct <= max_pct + 1e-6, float(pct))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    data_dir = CFG.data_dir
    out_path = CFG.default_out
    if CFG.date_subdir:
        date_str = datetime.today().strftime('%Y%m%d')
        out_dir = os.path.join(os.path.dirname(out_path), date_str)
        out_path = os.path.join(out_dir, os.path.basename(out_path))
    ensure_dir(os.path.dirname(out_path))

    end = datetime.today().date()
    start = end - relativedelta(months=CFG.lookback_months)
    sdt = pd.to_datetime(start.strftime('%Y-%m-%d'))
    edt = pd.to_datetime(end.strftime('%Y-%m-%d'))

    files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith('.csv')]
    files.sort()

    results = []
    pbar = tqdm(total=len(files), desc='筛选(上涨趋势)', dynamic_ncols=True)

    for fn in files:
        stem = os.path.splitext(fn)[0]
        parts = stem.split('.')
        if len(parts) != 2:
            pbar.update(1); continue
        code, market = parts[0], parts[1].upper()
        if CFG.only_10pct_a and not is_10pct_a_share(code, market):
            pbar.update(1); continue

        df = load_csv(os.path.join(data_dir, fn))
        if df.empty or len(df) < 60:
            pbar.update(1); continue
        window_df = df[(df['trade_date'] >= sdt) & (df['trade_date'] <= edt)]
        if window_df.empty or len(window_df) < CFG.ma_window:
            pbar.update(1); continue

        slope, above_days = ma_trend(window_df, CFG.ma_window)
        hh_hl = higher_highs_lows(window_df, CFG.higher_highs_lows_days)
        recent_ok = recent_up_days(window_df, CFG.recent_up_days, CFG.recent_up_days_required)

        # 额外严格条件：最新收盘高于近60日最高价的98%
        recent_high = window_df['close'].rolling(60).max().iloc[-1]
        close_breakout = (recent_high is not None) and (window_df['close'].iloc[-1] >= 0.98 * recent_high)
        # 新增条件：最近N天没有涨停
        no_limitup = _no_limitup_last_n_days(window_df, CFG.no_limitup_days, CFG.limitup_threshold)
        # 新增条件：最近一月涨幅不得超过30%
        month_ok, month_pct = _monthly_gain_within(window_df, CFG.month_gain_days, CFG.max_month_gain_pct)

        # 基本趋势条件（更严格）
        trend_ok = (
            slope > CFG.ma_slope_min - 1e-9
            and above_days >= CFG.close_above_ma_days
            and hh_hl
            and recent_ok
            and close_breakout
            and no_limitup
            and month_ok
        )

        # 市值过滤（可选）
        mc = None
        if CFG.enable_mktcap_filter:
            mc = _fetch_market_cap_eastmoney(code, market)
            if (mc is None) or not (CFG.mktcap_min - 1e-6 <= mc <= CFG.mktcap_max + 1e-6):
                pbar.update(1); continue

        if trend_ok:
            results.append({
                'symbol': stem,
                'code': code,
                'market': market,
                'ma_window': CFG.ma_window,
                'ma_slope': round(slope, 6),
                'close_above_ma_days': int(above_days),
                'higher_highs_lows': bool(hh_hl),
                'recent_up_days_ok': bool(recent_ok),
                'close_breakout': bool(close_breakout),
                'no_limitup_last_5d': bool(no_limitup),
                'month_gain_pct': round(month_pct, 4),
                'market_cap': mc,
            })
        pbar.set_postfix(code=stem)
        pbar.update(1)

    pbar.close()
    df_out = pd.DataFrame(results).sort_values(['ma_slope','close_above_ma_days','close_breakout'], ascending=[False, False, False]).reset_index(drop=True)
    if df_out.empty:
        print('无符合上涨趋势的标的')
        return
    print(df_out)
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已保存到: {out_path} | 标的数量: {len(df_out)}")


if __name__ == '__main__':
    main()
