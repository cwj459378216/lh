import os
from datetime import datetime

import pandas as pd
import requests

"""
根据回测或实盘生成的价格序列（来自 trade_log.csv），结合持仓维护表 portfolio_maintain.csv，
计算每只股票的峰值回撤，并在超过阈值时自动标记平仓与收入；同时根据 trade_log 中的 SELL 记录发送卖出通知到微信。

核心点：
- 平仓判断以 portfolio_maintain.csv 为基准，只处理其中未平仓(closed 为空或为假)的股票；
- 对于每只未平仓股票，从 trade_log.csv 中取出该 symbol 在截至指定日期 as_of 之前的全部价格(price)记录，
  计算 peak_price / current_price / drawdown，并在 drawdown >= AUTO_CLOSE_DRAWDOWN 时自动平仓；
- 卖出通知仍然基于 trade_log 中 action=='SELL' 的真实卖出记录，和自动平仓逻辑相互独立。
python notify_sell_wechat.py --date 20251230 --no-wechat
"""

# ===== 配置区 =====

BASE_DIR = os.path.dirname(__file__)
# 回测输出基目录（里面按时间戳有多个子目录）
BACKTEST_BASE_DIR = os.path.join(BASE_DIR, 'output', 'backtest')
# 卖出通知日志目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'notify_logs_sell')
# 持仓维护表路径
PORTFOLIO_MAINTAIN_PATH = os.path.join(BASE_DIR, 'output', 'portfolio_maintain.csv')
# 卖出通知标题前缀
MSG_TITLE_PREFIX = '卖出通知'

# 与选股通知共用同一个 Server酱 sendkey
SEND_KEY = 'SCT128639T02UHHDlc8xxG2DSsndvMfSiU'

# 是否启用微信发送
ENABLE_WECHAT_NOTIFY = True

# 超过该回撤比例，自动视为平仓
AUTO_CLOSE_DRAWDOWN = 0.03


def _find_latest_backtest_dir() -> str | None:
    """找到 output/backtest 下最新的时间戳子目录。"""
    if not os.path.isdir(BACKTEST_BASE_DIR):
        return None
    subs = [d for d in os.listdir(BACKTEST_BASE_DIR) if os.path.isdir(os.path.join(BACKTEST_BASE_DIR, d))]
    if not subs:
        return None
    subs.sort()
    return os.path.join(BACKTEST_BASE_DIR, subs[-1])


def _load_trades(trade_log_path: str) -> pd.DataFrame:
    if not os.path.isfile(trade_log_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(trade_log_path)
        # 规范日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()


def _load_portfolio() -> pd.DataFrame:
    """加载持仓维护表；若不存在则返回空表。"""
    if not os.path.isfile(PORTFOLIO_MAINTAIN_PATH):
        return pd.DataFrame(columns=['symbol', 'base_price', 'current_price', 'closed', 'income', 'peak_price', 'drawdown', 'shares', 'buy_date'])
    try:
        df = pd.read_csv(PORTFOLIO_MAINTAIN_PATH, dtype={'symbol': str})
    except Exception:
        df = pd.DataFrame(columns=['symbol', 'base_price', 'current_price', 'closed', 'income', 'peak_price', 'drawdown', 'shares', 'buy_date'])
    # 确保必要列存在
    for col in ['symbol', 'base_price', 'current_price', 'closed', 'income', 'peak_price', 'drawdown', 'shares', 'buy_date']:
        if col not in df.columns:
            df[col] = None
    return df


def _update_portfolio_with_drawdown(pf: pd.DataFrame, df_trades: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """以 portfolio_maintain 中的持仓为基准，更新每只股票的 peak_price / current_price / drawdown；
    当回撤超过阈值时自动计算收入并标记 closed。
    """
    if pf.empty or df_trades.empty:
        return pf

    # 仅考虑截至 as_of 的记录
    trades = df_trades[df_trades['date'] <= as_of].copy()
    if trades.empty or 'symbol' not in trades.columns or 'price' not in trades.columns:
        return pf

    # 确保 income/peak_price/drawdown/closed/shares 列存在
    for col in ['income', 'peak_price', 'drawdown', 'closed', 'shares']:
        if col not in pf.columns:
            pf[col] = None

    # 简单归一化 closed 字段，便于判断是否已平仓
    def _is_closed(val) -> bool:
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s in {'1', 'true', '是', '已平仓', 'closed', 'y'}

    for idx, row in pf.iterrows():
        sym = str(row.get('symbol', '')).strip()
        if not sym:
            continue
        if _is_closed(row.get('closed')):
            continue  # 已平仓的不再更新

        # 只用该股票的价格序列
        sym_trades = trades[trades['symbol'] == sym]
        if sym_trades.empty:
            continue
        prices = pd.to_numeric(sym_trades['price'], errors='coerce').dropna()
        if prices.empty:
            continue
        peak = float(prices.max())
        current = float(prices.iloc[-1])
        if peak <= 0:
            continue
        drawdown = (peak - current) / peak

        pf.at[idx, 'peak_price'] = round(peak, 4)
        pf.at[idx, 'current_price'] = round(current, 4)
        pf.at[idx, 'drawdown'] = round(drawdown, 4)

        # 回撤达到/超过阈值则视为触发平仓点
        if drawdown >= AUTO_CLOSE_DRAWDOWN:
            base_price = row.get('base_price')
            shares = row.get('shares')
            try:
                base_price_val = float(base_price) if pd.notna(base_price) else None
                shares_val = int(shares) if pd.notna(shares) else None
            except Exception:
                base_price_val, shares_val = None, None

            if base_price_val is not None and shares_val is not None and shares_val > 0:
                income = (current - base_price_val) * shares_val
                pf.at[idx, 'income'] = round(income, 2)
                pf.at[idx, 'closed'] = '是'

    # 写回文件
    os.makedirs(os.path.dirname(PORTFOLIO_MAINTAIN_PATH), exist_ok=True)
    pf.to_csv(PORTFOLIO_MAINTAIN_PATH, index=False, encoding='utf-8-sig')
    return pf


def send_wechat_message(text: str, as_of: datetime) -> None:
    print('\n===== WECHAT SELL MESSAGE BEGIN =====')
    print(text)
    print('===== WECHAT SELL MESSAGE END =====\n')

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
        print(f"发送微信卖出消息失败: {e}")


def format_sell_message(df: pd.DataFrame, as_of: datetime) -> str:
    """将卖出记录格式化为微信文本。"""
    if df.empty:
        return f"{MSG_TITLE_PREFIX}\n日期: {as_of.strftime('%Y-%m-%d')}\n当日无卖出记录。"

    lines = [
        f"{MSG_TITLE_PREFIX}",
        f"日期: {as_of.strftime('%Y-%m-%d')}",
        f"卖出笔数: {len(df)}",
        "",
    ]
    for _, row in df.iterrows():
        sym = row.get('symbol', '')
        price = row.get('price', '')
        shares = row.get('shares', '')
        pnl = row.get('pnl', '')
        reason = row.get('reason', '')
        parts = [str(sym)]
        if pd.notna(price):
            parts.append(f"价:{float(price):.2f}")
        if pd.notna(shares):
            parts.append(f"股数:{int(shares)}")
        if pd.notna(pnl):
            parts.append(f"盈亏:{float(pnl):.2f}")
        if pd.notna(reason) and str(reason):
            parts.append(f"原因:{reason}")
        lines.append(' | '.join(parts))

    return '\n'.join(lines)


def log_sell(df: pd.DataFrame, as_of: datetime) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = as_of.strftime('%Y%m%d')
    ts_str = as_of.strftime('%Y%m%d_%H%M%S')
    date_dir = os.path.join(OUTPUT_DIR, date_str)
    os.makedirs(date_dir, exist_ok=True)
    path = os.path.join(date_dir, f'sell_{ts_str}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def run_once(date_str: str | None = None, trade_log_path: str | None = None) -> None:
    """执行一次流程：读取 trade_log -> 更新持仓表回撤 -> 过滤卖出 -> 发送微信 -> 记录本地。"""
    if date_str:
        as_of = pd.to_datetime(date_str)
    else:
        as_of = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

    # 找 trade_log.csv
    if trade_log_path is None:
        latest_dir = _find_latest_backtest_dir()
        if latest_dir is None:
            print('未找到回测输出目录。')
            return
        trade_log_path = os.path.join(latest_dir, 'trade_log.csv')

    df_trades = _load_trades(trade_log_path)
    if df_trades.empty:
        print(f"未找到有效的 trade_log: {trade_log_path}")
        return

    # 先加载并更新持仓维护表中的 peak_price / drawdown
    pf = _load_portfolio()
    pf = _update_portfolio_with_drawdown(pf, df_trades, as_of)
    print(f"已根据 trade_log 更新持仓表回撤信息: {PORTFOLIO_MAINTAIN_PATH}")

    # 过滤指定日期的 SELL 记录
    if 'date' not in df_trades.columns or 'action' not in df_trades.columns:
        print('trade_log 缺少必要字段 date/action。')
        return

    df_trades['date'] = pd.to_datetime(df_trades['date'], errors='coerce')
    mask = (df_trades['date'] == as_of) & (df_trades['action'] == 'SELL')
    df_sell = df_trades[mask].copy()

    if df_sell.empty:
        print(f"[{as_of.strftime('%Y-%m-%d')}] 当日无 SELL 记录。")
    else:
        print(f"[{as_of.strftime('%Y-%m-%d')}] 卖出笔数: {len(df_sell)}")

    saved_path = log_sell(df_sell, as_of)
    print(f"卖出记录已保存到: {saved_path}")

    msg = format_sell_message(df_sell, as_of)
    send_wechat_message(msg, as_of)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='根据 trade_log.csv 发送卖出通知到微信并记录，并更新持仓表回撤信息')
    parser.add_argument('--date', type=str, help='指定日期(YYYYMMDD)，默认今天')
    parser.add_argument('--log', type=str, help='指定 trade_log.csv 路径，默认取最新回测目录')
    parser.add_argument('--no-wechat', action='store_true', help='仅更新文件，不发送微信通知')
    args, _ = parser.parse_known_args()

    if args.no_wechat:
        ENABLE_WECHAT_NOTIFY = False

    run_once(date_str=args.date, trade_log_path=args.log)
