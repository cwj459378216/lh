import os
import sys
import time
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from tqdm import tqdm

# 也支持读取本地通达信日线文件（vipdoc/sh|sz/lday/*.day）
# python download_all_data_pytdx.py --tdx-lday "D:\tdx\vipdoc\sh\lday" --out ".\通达信\data\pytdx\daily_raw"
# python download_all_data_pytdx.py --tdx-lday "D:\tdx\vipdoc" --cap-csv ".\output\caps.csv" --out ".\通达信\data\pytdx\daily_raw"
try:
    from pytdx.reader import TdxDailyBarReader
except Exception:
    TdxDailyBarReader = None

# 仅限定为 A 股中典型 10% 涨停板标的（不含 B 股）
ALLOW_PREFIXES_10P_SH = ("600", "601", "603", "605")  # 上交所 A 股
ALLOW_PREFIXES_10P_SZ = ("000", "001", "002", "003")  # 深交所 A 股

# 列映射（pytdx -> 英文）
COL_KEEP = ["datetime", "open", "close", "high", "low", "vol", "amount"]


class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.min_interval = 60.0 / max(1, calls_per_minute)
        self._last = 0.0

    def wait(self):
        now = time.time()
        dt = now - self._last
        need = self.min_interval - dt
        if need > 0:
            time.sleep(need)
        self._last = time.time()


# -------- 本地 vipdoc 工具函数 --------

def _resolve_lday_dirs(root_or_lday: str) -> list[str]:
    """输入 vipdoc 路径或其下 sh/sz/lday 路径，返回存在的 lday 目录列表。"""
    p = os.path.normpath(root_or_lday)
    dirs: list[str] = []
    base = p
    name = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    if name == 'lday' and parent in ('sh', 'sz'):
        base = os.path.dirname(os.path.dirname(p))  # 回到 vipdoc
    elif name in ('sh', 'sz'):
        base = os.path.dirname(p)
    # 组装 sh/sz lday
    sh_lday = os.path.join(base, 'sh', 'lday')
    sz_lday = os.path.join(base, 'sz', 'lday')
    for d in (sh_lday, sz_lday):
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def _local_security_list(lday_dirs: list[str]) -> pd.DataFrame:
    """从本地 lday 文件名提取代码和市场，仅保留 A 股 10% 涨停板前缀标的。"""
    rows = []
    for d in lday_dirs:
        market_dir = os.path.basename(os.path.dirname(d)).lower()
        market = 1 if market_dir == 'sh' else 0  # 0=SZ,1=SH
        for fn in os.listdir(d):
            if not fn.lower().endswith('.day'):
                continue
            stem = os.path.splitext(fn)[0]
            code = stem[-6:]
            if not code.isdigit():
                continue
            # 白名单：仅保留典型10% 涨停板前缀（A股）
            if market == 1:
                if not code.startswith(ALLOW_PREFIXES_10P_SH):
                    continue
            else:
                if not code.startswith(ALLOW_PREFIXES_10P_SZ):
                    continue
            symbol = f"{code}.{ 'SZ' if market==0 else 'SH'}"
            rows.append({'market': market, 'code': code, 'name': '', 'symbol': symbol, 'filepath': os.path.join(d, fn)})
    df = pd.DataFrame(rows)
    return df.sort_values(['market', 'code']).reset_index(drop=True)


def _fetch_daily_from_file(file_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    if TdxDailyBarReader is None:
        tqdm.write('缺少 TdxDailyBarReader，无法读取本地 day 文件。')
        return pd.DataFrame()
    try:
        reader = TdxDailyBarReader()
        df = reader.get_df(file_path)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # 标准化列
    if 'date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    elif 'datetime' in df.columns:
        df['trade_date'] = pd.to_datetime(df['datetime'], errors='coerce').dt.date
    else:
        # 尝试索引
        try:
            df['trade_date'] = pd.to_datetime(df.index, errors='coerce').date
        except Exception:
            return pd.DataFrame()
    # 数值列
    rename_map = {}
    if 'vol' in df.columns:
        rename_map['vol'] = 'volume'
    if 'volume' in df.columns:
        # 若已存在则保持
        pass
    if 'amount' in df.columns:
        rename_map['amount'] = 'amount'
    for c in ['open', 'high', 'low', 'close']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'vol' in df.columns:
        df['volume'] = pd.to_numeric(df['vol'], errors='coerce')
    elif 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    # 选列与排序
    keep = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    for k in keep:
        if k not in df.columns:
            # 某些列可能缺失，填充为 NaN
            df[k] = pd.NA
    df = df[keep]
    df = df.dropna(subset=['trade_date']).sort_values('trade_date')
    # 过滤区间
    sdt = datetime.strptime(start_date, '%Y%m%d').date()
    edt = datetime.strptime(end_date, '%Y%m%d').date()
    df = df[(df['trade_date'] >= sdt) & (df['trade_date'] <= edt)].reset_index(drop=True)
    return df


def _load_names_csv(path: str) -> pd.DataFrame:
    """读取本地名称映射 CSV，支持列(code,name)或(symbol,name)。"""
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame()
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if 'code' in df.columns and 'name' in df.columns:
        out = df[['code', 'name']].copy()
        out['code'] = out['code'].str[-6:].str.zfill(6)
        return out
    if 'symbol' in df.columns and 'name' in df.columns:
        out = df[['symbol', 'name']].copy()
        out['code'] = out['symbol'].str[:6]
        return out[['code', 'name']]
    return pd.DataFrame()


def _apply_name_filter(df: pd.DataFrame, names_df: pd.DataFrame | None) -> pd.DataFrame:
    """按名称关键字过滤 ST/*ST/退（需要本地名称映射）。"""
    if names_df is None or names_df.empty:
        tqdm.write('未提供名称映射，跳过 ST 关键字过滤，仅按代码前缀过滤。')
        return df
    m = df.merge(names_df, on='code', how='left')
    m['name'] = m['name'].fillna('')
    mask = ~m['name'].str.upper().apply(lambda nm: any(k in nm for k in EXCLUDE_NAME_KEYWORDS))
    m = m[mask]
    m.rename(columns={'name': 'name'}, inplace=True)
    return m.drop_duplicates(subset=['symbol']).reset_index(drop=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _clear_output_dir(path: str):
    """删除输出目录下已有的 CSV 文件，保留子目录。"""
    try:
        for fn in os.listdir(path):
            fp = os.path.join(path, fn)
            if os.path.isfile(fp) and fn.lower().endswith('.csv'):
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass


def _load_cap_csv(path: str) -> pd.DataFrame:
    """读取流通市值映射 CSV，支持列(code/symbol, cap/float_cap)。
    统一输出：code(6位)、cap_billion(亿元)。"""
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame()
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # 解析代码
    if 'code' in df.columns:
        codes = df['code'].str[-6:].str.zfill(6)
    elif 'symbol' in df.columns:
        codes = df['symbol'].str[:6]
    else:
        return pd.DataFrame()
    # 解析市值列
    cap_col = None
    for c in ['cap', 'float_cap', 'market_cap', 'float_market_cap']:
        if c in df.columns:
            cap_col = c
            break
    if cap_col is None:
        return pd.DataFrame()
    cap_raw = pd.to_numeric(df[cap_col], errors='coerce')
    avg = pd.to_numeric(cap_raw, errors='coerce').dropna().mean()
    if pd.isna(avg):
        return pd.DataFrame()
    # 大于1000则认为单位为元，转换为亿元
    cap_billion = cap_raw / 1e8 if avg > 1000 else cap_raw
    out = pd.DataFrame({'code': codes, 'cap_billion': pd.to_numeric(cap_billion, errors='coerce')})
    return out.dropna(subset=['cap_billion']).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='仅使用本地通达信 vipdoc 日线文件导出最近一年数据（A股10%涨停上限，原始价量）')
    parser.add_argument('--tdx-lday', type=str, required=True, help='通达信 vipdoc 目录，或其下的 sh/lday、sz/lday 路径，例如 D:\\tdx\\vipdoc 或 D:\\tdx\\vipdoc\\sh\\lday')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'data', 'pytdx', 'daily_raw'), help='输出目录')
    parser.add_argument('--overwrite', action='store_true', help='已存在是否覆盖')
    parser.add_argument('--limit', type=int, default=None, help='仅导出前 N 个标的')
    # 新增：市值映射 CSV（将市值加入到每个导出的行中，列名为 cap_billion）
    parser.add_argument('--cap-csv', type=str, default=None, help='流通市值映射 CSV（含 code/symbol 与 cap/float_cap，单位元或亿元自动识别）')
    # 移除名称关键字(ST)过滤相关参数
    args = parser.parse_args()

    ensure_dir(args.out)
    # 下载前清理历史 CSV
    _clear_output_dir(args.out)

    # 读取市值映射
    cap_map = None
    if args.cap_csv:
        cap_map = _load_cap_csv(args.cap_csv)
        if cap_map is None or cap_map.empty:
            tqdm.write('警告：未能读取有效的市值映射，导出时将不包含市值列。')
            cap_map = None

    end = datetime.today().date()
    start = end - relativedelta(months=12)
    start_str = start.strftime('%Y%m%d')
    end_str = end.strftime('%Y%m%d')

    # 本地模式
    lday_dirs = _resolve_lday_dirs(args.tdx_lday)
    if not lday_dirs:
        print(f"未找到 lday 目录，请检查路径：{args.tdx_lday}")
        sys.exit(1)

    uni = _local_security_list(lday_dirs)

    if uni.empty:
        print('符合条件的标的为空。')
        sys.exit(1)

    if args.limit:
        uni = uni.head(args.limit)

    print(f"[vipdoc] 标的数: {len(uni)} | 区间: {start_str} ~ {end_str} | 源: {', '.join(lday_dirs)} | 输出: {args.out}")

    pbar = tqdm(total=len(uni), desc='导出中(vipdoc)', dynamic_ncols=True)
    saved = 0
    failed: list[str] = []
    for _, r in uni.iterrows():
        market, code, symbol, fpath = r['market'], r['code'], r['symbol'], r['filepath']
        out_fp = os.path.join(args.out, f"{symbol}.csv")
        if os.path.exists(out_fp) and not args.overwrite:
            pbar.update(1)
            continue
        df = _fetch_daily_from_file(fpath, start_str, end_str)
        if df.empty:
            failed.append(symbol)
            pbar.update(1)
            continue
        # 合并市值（将同一标的的市值写入每一行的 cap_billion 列）
        if cap_map is not None:
            row = cap_map[cap_map['code'] == code]
            if not row.empty:
                cap_bil = float(row['cap_billion'].iloc[0])
                df['cap_billion'] = cap_bil
        df.to_csv(out_fp, index=False, encoding='utf-8-sig')
        saved += 1
        try:
            tqdm.write(f"保存: {symbol} -> {out_fp} ({len(df)} 行)")
        except Exception:
            pass
        pbar.set_postfix(saved=saved, code=symbol)
        pbar.update(1)
    pbar.close()

    if failed:
        fail_fp = os.path.join(args.out, f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(fail_fp, 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed))
        print(f"完成。成功保存: {saved} / {len(uni)}，失败: {len(failed)}，失败清单: {fail_fp}")
    else:
        print(f"完成。成功保存: {saved} / {len(uni)}")


if __name__ == '__main__':
    main()
