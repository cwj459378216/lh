import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

import select_stocks_local as sel

"""
根据 select_stocks_local 的筛选结果，发送到微信并记录。

新增：
- 在选股通知时维护一个可人工维护的持仓表 CSV：portfolio_maintain.csv
  列包括（中文表头）：
    股票名称(股票代码)(symbol_cn)、底仓价格(收盘价)(底仓价格)、当前价格(收盘价)(当前价格)、是否平仓(是否平仓)、收入(收入)、购买日期(购买日期)
  - 新入选股票如果表中不存在，则追加一行；
    * 底仓价格 使用选股日的收盘价；
    * 当前价格 初始与底仓价格相同；
    * 购买日期 为选股日的后一个自然日（示意，可手工调整为实际交易日）；
    * 是否平仓、收入 初始为空，方便手工填写或由其他脚本维护；
  - 已存在的股票不覆盖原有手工填写内容。
- 增加开关参数：ENABLE_WECHAT_NOTIFY，用于控制是否真正发送到微信。
python notify_selection_wechat.py --no-wechat --end-date 20251225
"""


# ===== 配置区 =====

# 是否使用 select_stocks_local.CFG 的路径
DATA_DIR = sel.CFG.data_dir
# 输出 / 日志目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'notify_logs')
# 持仓维护表路径
PORTFOLIO_MAINTAIN_PATH = os.path.join(os.path.dirname(__file__), 'output', 'portfolio_maintain.csv')
# 微信通知标题前缀
MSG_TITLE_PREFIX = '选股通知'

# Server酱 sendkey（请替换为你的 sendkey）
SEND_KEY = 'SCT128639T02UHHDlc8xxG2DSsndvMfSiU'  # 你提供的新 sendkey

# 是否启用微信发送（如仅想生成文件、不推送，可设为 False）
ENABLE_WECHAT_NOTIFY = True

# 每次购买金额（元）：为避免每次都恰好是整数 5000，这里支持在一个区间内随机（默认围绕 5000 小幅波动）
BUY_AMOUNT_BASE = 5000.0  # 基准金额
BUY_AMOUNT_JITTER = 30.0  # 随机扰动范围：最终金额在 [BASE-JITTER, BASE+JITTER]
BUY_AMOUNT_ROUND_TO = 1.0  # 取整粒度：1.0=精确到元；0.1=精确到角；0.01=精确到分


def _generate_buy_amount() -> float:
    """生成一个不总是等于 5000 的购买金额（带随机扰动）。

    说明：
    - 默认在 5000±30 的范围内随机；
    - 为避免恰好等于 5000，会在碰到等于基准金额时做一次极小偏移；
    - 通过 BUY_AMOUNT_ROUND_TO 控制精度。
    """
    import random

    low = BUY_AMOUNT_BASE - BUY_AMOUNT_JITTER
    high = BUY_AMOUNT_BASE + BUY_AMOUNT_JITTER
    if high <= low:
        amount = BUY_AMOUNT_BASE
    else:
        amount = random.uniform(low, high)

    # 按粒度取整
    step = float(BUY_AMOUNT_ROUND_TO) if BUY_AMOUNT_ROUND_TO else 1.0
    amount = round(amount / step) * step

    # 避免刚好等于 5000
    if abs(amount - BUY_AMOUNT_BASE) < 1e-9:
        amount = amount + step

    # 防止出现非正数
    return max(step, float(amount))


def send_wechat_message(text: str, as_of: datetime) -> None:
    """发送微信消息：通过 Server酱 HTTP 接口；失败则只打印。"""
    print('\n===== WECHAT MESSAGE BEGIN =====')
    print(text)
    print('===== WECHAT MESSAGE END =====\n')

    if not ENABLE_WECHAT_NOTIFY:
        print('提示: 已关闭微信推送(ENABLE_WECHAT_NOTIFY=False)，仅本地打印。')
        return

    if not SEND_KEY:
        print('提示: 未配置 SEND_KEY，暂不发送微信，仅本地打印。')
        return

    title = f"{MSG_TITLE_PREFIX} - {as_of.strftime('%Y-%m-%d')}"
    try:
        base_url = f"https://sctapi.ftqq.com/{SEND_KEY}.send"
        params = {
            'title': title,
            'desp': text,
        }
        resp = requests.get(base_url, params=params, timeout=5)
        print('推送返回状态:', resp.status_code)
        print('推送返回内容:', resp.text[:200])
    except Exception as e:
        print(f"发送微信消息失败: {e}")


def format_selection_message(df: pd.DataFrame, as_of: datetime) -> str:
    """将选股结果格式化为微信文本。"""
    if df.empty:
        return f"{MSG_TITLE_PREFIX}\n日期: {as_of.strftime('%Y-%m-%d')}\n无符合条件的标的。"

    lines = [
        f"{MSG_TITLE_PREFIX}",
        f"日期: {as_of.strftime('%Y-%m-%d')}",
        f"数量: {len(df)}",
        "",
    ]
    # 尽量使用中文列名；如果不存在则退回英文列名
    col_symbol = '标的' if '标的' in df.columns else ('symbol' if 'symbol' in df.columns else None)
    col_last_close = '最新收盘价' if '最新收盘价' in df.columns else ('last_close' if 'last_close' in df.columns else None)
    col_range = '振幅(%)' if '振幅(%)' in df.columns else ('range_pct' if 'range_pct' in df.columns else None)
    col_limitup = '近一年涨停天数' if '近一年涨停天数' in df.columns else ('limit_up_days_1y' if 'limit_up_days_1y' in df.columns else None)

    for _, row in df.iterrows():
        parts = []
        if col_symbol:
            parts.append(str(row[col_symbol]))
        if col_last_close is not None and not pd.isna(row[col_last_close]):
            parts.append(f"价:{row[col_last_close]:.2f}")
        if col_range is not None and not pd.isna(row[col_range]):
            parts.append(f"振幅:{row[col_range]:.2f}%")
        if col_limitup is not None and not pd.isna(row[col_limitup]):
            parts.append(f"涨停天数:{int(row[col_limitup])}")
        lines.append(' | '.join(parts))

    return '\n'.join(lines)


def log_selection(df: pd.DataFrame, as_of: datetime) -> str:
    """将当日选股结果记录到本地 CSV，返回保存路径。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = as_of.strftime('%Y%m%d')
    ts_str = as_of.strftime('%Y%m%d_%H%M%S')
    # 以日期为子目录，避免文件过多
    date_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(date_dir, exist_ok=True)
    path = os.path.join(date_dir, f'selection_{ts_str}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def update_portfolio_maintain(df: pd.DataFrame, as_of: datetime) -> str:
    """根据今日入选标的，更新/创建持仓维护表 CSV（使用中文列名）。

    列：
      - 股票代码
      - 底仓价格（选股日收盘价）
      - 当前价格（收盘价）
      - 是否平仓
      - 收入
      - 购买日期（后一个交易日/示意）
      - 购买金额（默认围绕 BUY_AMOUNT_BASE 随机打散）
    """
    os.makedirs(os.path.dirname(PORTFOLIO_MAINTAIN_PATH), exist_ok=True)

    cols = ['股票代码', '底仓价格', '当前价格', '是否平仓', '收入', '购买日期', '购买金额']
    if os.path.exists(PORTFOLIO_MAINTAIN_PATH):
        try:
            pf = pd.read_csv(PORTFOLIO_MAINTAIN_PATH, dtype={'股票代码': str})
        except Exception:
            pf = pd.DataFrame(columns=cols)
    else:
        pf = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in pf.columns:
            pf[c] = None

    if df.empty:
        pf.to_csv(PORTFOLIO_MAINTAIN_PATH, index=False, encoding='utf-8-sig')
        return PORTFOLIO_MAINTAIN_PATH

    # 选股结果中的代码和收盘价
    if 'code' in df.columns:
        codes = df['code'].astype(str).tolist()
    elif 'symbol' in df.columns:
        # 形如 600001.SH，取完整写入
        codes = df['symbol'].astype(str).tolist()
    elif '标的' in df.columns:
        codes = df['标的'].astype(str).tolist()
    else:
        codes = []

    # 收盘价列
    if 'last_close' in df.columns:
        closes = df['last_close'].tolist()
    elif '最新收盘价' in df.columns:
        closes = df['最新收盘价'].tolist()
    else:
        closes = [None] * len(codes)

    # 购买日期：选股日的后一个自然日（你也可以后续手工调整为实际交易日）
    buy_date_str = (as_of + timedelta(days=1)).strftime('%Y-%m-%d')

    existing_codes = set(str(s) for s in pf['股票代码'].dropna().astype(str)) if not pf.empty else set()

    new_rows = []
    for code, close in zip(codes, closes):
        if code in existing_codes:
            continue
        base_price = float(close) if close is not None and not pd.isna(close) else None
        new_rows.append({
            '股票代码': code,
            '底仓价格': base_price,
            '当前价格': base_price,
            '是否平仓': None,
            '收入': None,
            '购买日期': buy_date_str,
            '购买金额': _generate_buy_amount(),
        })

    if new_rows:
        pf = pd.concat([pf, pd.DataFrame(new_rows)], ignore_index=True)

    pf.to_csv(PORTFOLIO_MAINTAIN_PATH, index=False, encoding='utf-8-sig')
    return PORTFOLIO_MAINTAIN_PATH


def run_once(end_date: str | None = None) -> None:
    """执行一次完整流程：选股 -> 生成消息 -> 发送微信 -> 记录本地 -> 更新持仓维护表。"""
    if end_date:
        as_of = pd.to_datetime(end_date)
    else:
        as_of = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

    df = sel.scan_dir(
        data_dir=DATA_DIR,
        months_lookback=sel.CFG.months,
        range_lower=sel.CFG.range_lower,
        range_upper=sel.CFG.range_upper,
        near_low_tol=sel.CFG.near_low_tol,
        limitup_lookback_months=sel.CFG.limitup_months,
        limitup_threshold=sel.CFG.limitup_threshold,
        volume_spike_days=sel.CFG.vol_days,
        volume_spike_factor=sel.CFG.vol_factor,
        only_10pct_a=sel.CFG.only_10pct_a,
        end_date=as_of,
        quiet=True,
    )

    if df.empty:
        print(f"[{as_of.strftime('%Y-%m-%d')}] 无符合条件的标的。")
    else:
        df = df[df['limit_up_days_1y'] >= int(sel.CFG.min_limitup_count)].copy()
        if df.empty:
            print(f"[{as_of.strftime('%Y-%m-%d')}] 涨停次数过滤后无标的。")

    saved_path = log_selection(df, as_of)
    print(f"选股结果已记录到: {saved_path}")

    pf_path = update_portfolio_maintain(df, as_of)
    print(f"持仓维护表路径: {pf_path}")

    msg = format_selection_message(df, as_of)
    send_wechat_message(msg, as_of)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='根据本地选股结果发送微信并记录')
    parser.add_argument('--end-date', type=str, help='指定筛选截止日期(YYYYMMDD)，默认今天')
    parser.add_argument('--no-wechat', action='store_true', help='仅生成文件，不发送微信通知')
    args, _ = parser.parse_known_args()

    if args.no_wechat:
        ENABLE_WECHAT_NOTIFY = False

    run_once(end_date=args.end_date)
