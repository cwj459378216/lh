"""update_stop_loss_table.py

功能：
1) 止损表模式（旧）：
     根据输入 CSV 的【购买时间、代码】读取本地日线数据，结合 `stop_loss_rules.py` 计算并回写：
     - 收盘最高价、止损价格、持仓时间、是否平仓、平仓日期、平仓原因

2) 买入明细模式（新增/自动识别）：
     读取 backtest 输出的 `买入明细.csv`，若某行未填写【卖出日期/卖出价格/卖出原因】则：
     - 按 `stop_loss_rules.evaluate_exit_signal` 逐日评估到 --end-date
     - 触发后回填：卖出日期/卖出价格/卖出股数/卖出盈亏/卖出原因/卖出税费

说明：
- 价格数据默认读取 `select_stocks_local.CFG.data_dir`（pytdx daily_raw 数据目录）。
- 股票代码会自动补全市场后缀（.SH/.SZ），规则：
    - 6开头 -> SH
    - 0/2/3开头 -> SZ
    
    python ./update_stop_loss_table.py --input ./output/选股维护表单.csv --end-date 20260120 --update-selection-form --only-fill-empty
"""
# python ./update_stop_loss_table.py --input ./output/买入明细.csv --end-date 20260120
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import pandas as pd

import select_stocks_local as sel
from stop_loss_rules import StopLossConfig, evaluate_exit_signal, get_close, get_open


def _guess_market_suffix(code: str) -> str:
    """根据纯数字代码猜测 .SH/.SZ 后缀。"""
    code = str(code).strip()
    if code.startswith("6"):
        return "SH"
    return "SZ"


def _to_symbol(code: str) -> str:
    code = str(code).strip()
    if "." in code:
        # 已经是 600000.SH 之类
        parts = code.split(".")
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1].upper()}"
        return code
    return f"{code}.{_guess_market_suffix(code)}"


def _load_symbol_df(data_dir: str, symbol: str) -> pd.DataFrame:
    fp = os.path.join(data_dir, f"{symbol}.csv")
    return sel.load_csv(fp)


def _calc_commission(amount: float, rate: float, minimum: float) -> float:
    amt = max(float(amount or 0.0), 0.0)
    if amt <= 0:
        return 0.0
    fee = amt * float(rate)
    return float(max(fee, float(minimum)))


def _calc_stamp_tax_sell(amount: float, rate: float) -> float:
    amt = max(float(amount or 0.0), 0.0)
    if amt <= 0:
        return 0.0
    return float(amt * float(rate))


def _get_next_open(symbol_df: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    """取 date 下一交易日开盘价（若无则返回 (None, None)）。"""
    if symbol_df is None or symbol_df.empty:
        return None, None

    df = symbol_df[['trade_date'] + (["open"] if 'open' in symbol_df.columns else [])].dropna().copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna().sort_values('trade_date')

    d0 = pd.to_datetime(date).normalize()
    sub = df[df['trade_date'] > d0]
    if sub.empty:
        return None, None

    d1 = pd.to_datetime(sub['trade_date'].iloc[0])
    px = get_open(symbol_df, d1)
    if px is None:
        return None, None
    return d1, float(px)


def _same_day(d1: pd.Timestamp | None, d2: pd.Timestamp | None) -> bool:
    if d1 is None or d2 is None:
        return False
    return pd.to_datetime(d1).normalize() == pd.to_datetime(d2).normalize()


def _calc_row(data_dir: str, sl_cfg: StopLossConfig, buy_date: pd.Timestamp, code: str, end_date: pd.Timestamp) -> dict:
    symbol = _to_symbol(code)
    df = _load_symbol_df(data_dir, symbol)

    out: dict = {
        "购买时间": buy_date.strftime("%Y%m%d"),
        "代码": str(code).strip(),
        "持仓成本": "",
        "收盘最高价": "",
        "止损价格": "",
        "持仓时间": "",
        "是否平仓": "",
        "平仓日期": "",
        "平仓原因": "",
    }

    if df is None or df.empty:
        return out

    # 选取买入日至 end_date 的数据（含端点）
    df2 = df[(df["trade_date"] >= buy_date) & (df["trade_date"] <= end_date)].copy()
    if df2.empty:
        return out

    peak_close = float(df2["close"].max())
    stop_price = peak_close * (1 - float(sl_cfg.stop_loss_drawdown)) if peak_close > 0 else 0.0

    # 持仓时间：自然日
    hold_days = int((end_date - buy_date).days)

    # 逐日评估是否触发平仓
    # entry_price：按你的规则，持仓成本=购买当天的开盘价；若购买日缺失则取区间首日开盘价（再缺失则回退到首日收盘价）
    buy_day_df = df2[df2["trade_date"] == buy_date]
    if not buy_day_df.empty and "open" in buy_day_df.columns:
        entry_price = float(buy_day_df["open"].iloc[0])
    elif "open" in df2.columns:
        entry_price = float(df2["open"].iloc[0])
    else:
        entry_price = float(df2["close"].iloc[0])

    out["持仓成本"] = round(entry_price, 4)

    pos = {
        "entry_price": entry_price,
        "buy_date": buy_date,
        "peak_close": None,
        # 对齐止损规则2：把“信号日”按买入日回填（用于表格用途：跌破买入日开盘价）
        # 如果你有单独的信号日字段，可改为真正的信号日
        "signal_date": buy_date,
        "signal_open": float(entry_price) if entry_price else None,
    }

    closed = False
    close_date: pd.Timestamp | None = None
    close_reason: str = ""

    for d in df2["trade_date"].sort_values().tolist():
        should_exit, reasons = evaluate_exit_signal(sl_cfg, df2, pos, pd.to_datetime(d))
        if should_exit:
            closed = True
            close_date = pd.to_datetime(d)
            close_reason = "_AND_".join(reasons) if reasons else "SELL"
            break

    out["收盘最高价"] = round(peak_close, 4)
    out["止损价格"] = round(stop_price, 4)
    out["持仓时间"] = hold_days
    out["是否平仓"] = "是" if closed else "否"
    if closed and close_date is not None:
        out["平仓日期"] = close_date.strftime("%Y%m%d")
        out["平仓原因"] = close_reason

    return out


def _is_buy_detail_df(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    cols = set([str(c).strip() for c in df.columns])
    # backtest 输出的“买入明细.csv”特征列
    return ('买入日期' in cols) and ('卖出日期' in cols) and ('买入价格' in cols) and ('卖出原因' in cols) and ('代码' in cols)


def _is_selection_form_df(df: pd.DataFrame) -> bool:
    """识别 output/选股维护表单.csv 结构。"""
    if df is None or df.empty:
        return False
    cols = set([str(c).strip() for c in df.columns])
    return ('信号日' in cols) and ('股票代码' in cols) and ('是否平仓' in cols) and ('平仓日期' in cols) and ('平仓原因' in cols)


def _is_blank(v) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none"}


def _update_selection_form_row(
    data_dir: str,
    sl_cfg: StopLossConfig,
    r: pd.Series,
    end_date: pd.Timestamp,
    only_fill_empty: bool = False,
) -> dict:
    """维护“选股维护表单”：更新 是否平仓/平仓日期/平仓原因。"""
    out = r.to_dict()

    if only_fill_empty:
        v1 = r.get('是否平仓', '')
        v2 = r.get('平仓日期', '')
        v3 = r.get('平仓原因', '')
        # 任一已有值则不覆盖
        if (not _is_blank(v1)) or (not _is_blank(v2)) or (not _is_blank(v3)):
            return out

    code = str(r.get('股票代码', '')).strip()
    if not code:
        return out

    signal_date = pd.to_datetime(r.get('信号日', ''), errors='coerce')
    if pd.isna(signal_date):
        return out

    symbol = _to_symbol(code)
    df = _load_symbol_df(data_dir, symbol)
    if df is None or df.empty:
        # 数据缺失：不强行写“否”，避免误导
        return out

    df_all = df[df['trade_date'] <= end_date].copy()
    if df_all.empty:
        return out

    df_iter = df_all[(df_all['trade_date'] >= signal_date) & (df_all['trade_date'] <= end_date)].copy()
    if df_iter.empty:
        return out

    # entry_price：按信号日开盘价；若缺失则回退到信号日收盘价
    entry_price = get_open(df_all, signal_date)
    if entry_price is None:
        entry_price = get_close(df_all, signal_date)
    if entry_price is None or float(entry_price) <= 0:
        return out

    pos = {
        'entry_price': float(entry_price),
        'buy_date': signal_date,
        'peak_close': None,
        'signal_date': signal_date,
        'signal_open': float(entry_price),
    }

    closed = False
    close_date: pd.Timestamp | None = None
    close_reason: str = ''

    for d in df_iter['trade_date'].sort_values().tolist():
        d = pd.to_datetime(d)
        should_exit, reasons = evaluate_exit_signal(sl_cfg, df_all, pos, d)
        if should_exit:
            closed = True
            close_date = d
            close_reason = '_AND_'.join([str(x) for x in (reasons or []) if str(x).strip()]) or 'SELL'
            break

    out['是否平仓'] = '是' if closed else '否'
    if closed and close_date is not None:
        out['平仓日期'] = pd.to_datetime(close_date).strftime('%Y%m%d')
        out['平仓原因'] = close_reason
    else:
        out['平仓日期'] = ''
        out['平仓原因'] = ''

    return out


def _update_buy_detail_row(
    data_dir: str,
    sl_cfg: StopLossConfig,
    r: pd.Series,
    end_date: pd.Timestamp,
) -> dict:
    # 读入字段（尽量温和）
    code = str(r.get('代码', '')).strip()
    if not code:
        return r.to_dict()

    buy_date = pd.to_datetime(r.get('买入日期', ''), errors='coerce')
    if pd.isna(buy_date):
        return r.to_dict()

    # 已有平仓信息则不覆盖
    sell_date_raw = str(r.get('卖出日期', '')).strip()
    sell_reason_raw = str(r.get('卖出原因', '')).strip()
    sell_price_raw = str(r.get('卖出价格', '')).strip()
    if sell_date_raw or sell_reason_raw or sell_price_raw:
        return r.to_dict()

    buy_price = pd.to_numeric(r.get('买入价格', 0), errors='coerce')
    buy_shares = pd.to_numeric(r.get('买入股数', 0), errors='coerce')
    buy_fees = pd.to_numeric(r.get('买入税费', 0), errors='coerce')

    if pd.isna(buy_price) or float(buy_price) <= 0:
        return r.to_dict()
    if pd.isna(buy_shares) or int(buy_shares) <= 0:
        return r.to_dict()

    symbol = _to_symbol(code)
    df = _load_symbol_df(data_dir, symbol)
    if df is None or df.empty:
        return r.to_dict()

    # 关键：规则“跌破信号日开盘价止损”需要能读取 signal_date 当天的 open。
    # 因此评估止损与取价都使用 df_all（截止到 end_date 的全量数据）。
    df_all = df[df['trade_date'] <= end_date].copy()
    if df_all.empty:
        return r.to_dict()

    # 遍历评估区间：从买入日到 end_date（含端点）
    df_iter = df_all[(df_all['trade_date'] >= buy_date) & (df_all['trade_date'] <= end_date)].copy()
    if df_iter.empty:
        return r.to_dict()

    # 成交规则：完全遵循 backtest CFG.sell_price_mode（close/next_open）
    # 持股天数规则：遵循 backtest CFG.min_hold_days（不满足则跳过本次触发，继续往后找）
    try:
        import backtest_select_stocks_local as bt

        sell_price_mode = str(getattr(bt.CFG, 'sell_price_mode', 'close') or 'close').lower().strip()
        min_hold_days = int(getattr(bt.CFG, 'min_hold_days', 1) or 0)
        commission_rate = float(getattr(bt.CFG, 'commission_rate', 0.000085))
        commission_min = float(getattr(bt.CFG, 'commission_min', 0.1))
        stamp_tax_rate_sell = float(getattr(bt.CFG, 'stamp_tax_rate_sell', 0.0005))
    except Exception:
        sell_price_mode = 'close'
        min_hold_days = 1
        commission_rate = 0.000085
        commission_min = 0.1
        stamp_tax_rate_sell = 0.0005

    # pos 结构对齐 stop_loss_rules
    signal_date = pd.to_datetime(r.get('信号日期', buy_date), errors='coerce')
    if pd.isna(signal_date):
        signal_date = buy_date

    pos = {
        'shares': int(buy_shares),
        'entry_price': float(buy_price),
        'buy_date': buy_date,
        'peak_close': None,
        'signal_date': signal_date,
        'signal_open': None,
    }

    closed = False
    reasons: list[str] = []
    exec_date: pd.Timestamp | None = None
    exec_price: float | None = None

    for d in df_iter['trade_date'].sort_values().tolist():
        d = pd.to_datetime(d)

        should_exit, rs = evaluate_exit_signal(sl_cfg, df_all, pos, d)
        if not should_exit:
            continue

        if sell_price_mode == 'next_open':
            _sd, _sp = _get_next_open(df_all, d)
        else:
            _sd, _sp = d, get_close(df_all, d)

        if _sd is None or _sp is None or float(_sp) <= 0:
            continue

        # 买卖不能同日
        if _same_day(_sd, buy_date):
            continue

        # 最少持股天数（以成交日为准）
        if min_hold_days > 0:
            try:
                held_days = (pd.to_datetime(_sd).normalize() - pd.to_datetime(buy_date).normalize()).days
                if held_days < int(min_hold_days):
                    continue
            except Exception:
                pass

        closed = True
        exec_date, exec_price = pd.to_datetime(_sd), float(_sp)
        reasons = rs or []
        break

    if not closed or exec_date is None or exec_price is None:
        return r.to_dict()

    proceeds = float(exec_price) * int(buy_shares)
    sell_commission = _calc_commission(proceeds, commission_rate, commission_min)
    sell_stamp_tax = _calc_stamp_tax_sell(proceeds, stamp_tax_rate_sell)
    sell_fees = sell_commission + sell_stamp_tax

    # 盈亏口径：与 backtest 一致（扣买入佣金 + 卖出成本）
    buy_fees_v = 0.0 if pd.isna(buy_fees) else float(buy_fees)
    pnl = (float(exec_price) - float(buy_price)) * int(buy_shares) - buy_fees_v - float(sell_fees)

    out = r.to_dict()
    out['卖出日期'] = pd.to_datetime(exec_date).strftime('%Y-%m-%d')
    out['卖出价格'] = round(float(exec_price), 4)
    out['卖出股数'] = int(buy_shares)
    out['卖出税费'] = round(float(sell_fees), 2)
    out['卖出盈亏'] = round(float(pnl), 2)
    out['卖出原因'] = '_AND_'.join([str(x) for x in reasons if str(x).strip()]) or 'SELL'
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join("wh", "qyb.csv"))
    parser.add_argument("--output", default="")
    parser.add_argument("--end-date", default=datetime_today_yyyymmdd())

    # 维护“选股维护表单.csv”模式
    parser.add_argument(
        "--update-selection-form",
        action="store_true",
        help="维护 output/选股维护表单.csv：更新 是否平仓/平仓日期/平仓原因",
    )
    parser.add_argument(
        "--only-fill-empty",
        action="store_true",
        help="只填空白：当平仓三列已有任一值时不覆盖",
    )

    # 止损策略参数（默认沿用 stop_loss_rules.py 默认值）
    parser.add_argument("--stop-loss-drawdown", type=float, default=StopLossConfig().stop_loss_drawdown)
    parser.add_argument("--enable-three-days-down-exit", action="store_true", default=StopLossConfig().enable_three_days_down_exit)
    parser.add_argument("--disable-early-underperform-exit", action="store_true")
    # 早期弱势卖出：每 step_days 个交易日，最低涨幅门槛增加 step_min_return
    # 兼容旧参数名：--early-exit-hold-days/--early-exit-min-return
    parser.add_argument(
        "--early-exit-step-days",
        "--early-exit-hold-days",
        dest="early_exit_step_days",
        type=int,
        default=StopLossConfig().early_exit_step_days,
    )
    parser.add_argument(
        "--early-exit-step-min-return",
        "--early-exit-min-return",
        dest="early_exit_step_min_return",
        type=float,
        default=StopLossConfig().early_exit_step_min_return,
    )

    args = parser.parse_args()

    end_date = pd.to_datetime(args.end_date)

    input_path = args.input
    output_path = args.output.strip() if str(args.output).strip() else input_path

    sl_cfg = StopLossConfig(
        stop_loss_drawdown=float(args.stop_loss_drawdown),
        enable_three_days_down_exit=bool(args.enable_three_days_down_exit),
        enable_early_underperform_exit=not bool(args.disable_early_underperform_exit),
        early_exit_step_days=int(args.early_exit_step_days),
        early_exit_step_min_return=float(args.early_exit_step_min_return),
    )

    data_dir = sel.CFG.data_dir

    df_in = pd.read_csv(input_path, dtype=str)
    df_in = df_in.fillna("")

    # 强制：选股维护表单模式
    if bool(args.update_selection_form) or _is_selection_form_df(df_in):
        required_cols = ['信号日', '股票代码', '原始评分', '是否平仓', '平仓日期', '平仓原因']
        for c in required_cols:
            if c not in df_in.columns:
                df_in[c] = ""

        rows = []
        for _, r in df_in.iterrows():
            rows.append(_update_selection_form_row(data_dir, sl_cfg, r, end_date, only_fill_empty=bool(args.only_fill_empty)))

        df_out = pd.DataFrame(rows, columns=[c for c in df_in.columns])
        df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已更新选股维护表单平仓信息: {output_path}")
        print(f"止损参数: {asdict(sl_cfg)}")
        return

    # 自动识别：买入明细模式
    if _is_buy_detail_df(df_in):
        # 兼容：缺列则补齐
        required_cols = [
            '代码', '信号日期',
            '买入日期', '买入价格', '买入股数', '买入金额', '买入税费',
            '卖出日期', '卖出价格', '卖出股数', '卖出盈亏', '卖出原因', '卖出税费',
        ]
        for c in required_cols:
            if c not in df_in.columns:
                df_in[c] = ""

        rows = []
        for _, r in df_in.iterrows():
            rows.append(_update_buy_detail_row(data_dir, sl_cfg, r, end_date))

        df_out = pd.DataFrame(rows, columns=[c for c in df_in.columns])
        df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已更新买入明细平仓信息: {output_path}")
        print(f"止损参数: {asdict(sl_cfg)}")
        return

    # 兼容用户的旧表头（如果缺少列则补）
    required_cols = ["购买时间", "代码", "持仓成本", "收盘最高价", "止损价格", "持仓时间", "是否平仓", "平仓日期", "平仓原因"]
    for c in required_cols:
        if c not in df_in.columns:
            df_in[c] = ""

    rows = []
    for _, r in df_in.iterrows():
        buy_time = str(r.get("购买时间", "")).strip()
        code = str(r.get("代码", "")).strip()
        if not buy_time or not code:
            # 空行原样返回
            rows.append({c: r.get(c, "") for c in required_cols})
            continue

        buy_date = pd.to_datetime(buy_time)
        row_out = _calc_row(data_dir, sl_cfg, buy_date, code, end_date)
        rows.append(row_out)

    df_out = pd.DataFrame(rows, columns=required_cols)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"已更新: {output_path}")
    print(f"止损参数: {asdict(sl_cfg)}")


def datetime_today_yyyymmdd() -> str:
    return pd.Timestamp.today().strftime("%Y%m%d")


if __name__ == "__main__":
    main()
