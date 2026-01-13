"""update_stop_loss_table.py

根据 `wh/止损表.csv` 中的【购买时间、代码】读取本地日线数据，
结合 `stop_loss_rules.py` 中的止损策略，计算并回写以下字段：

- 收盘最高价: 从买入日(含)到当前评估日(含)的最大收盘价
- 止损价格: 收盘最高价 * (1 - 回撤止损比例)
- 持仓时间: 从买入日到当前评估日的自然日天数（end_date - buy_date）
- 是否平仓: 是否在区间内触发止损/规则并在触发日记为平仓

用法（PowerShell 示例）：
  python update_stop_loss_table.py --end-date 20260106

说明：
- 价格数据默认读取 `select_stocks_local.CFG.data_dir`（pytdx daily_raw 数据目录）。
- 股票代码会自动补全市场后缀（.SH/.SZ），规则：
  - 6开头 -> SH
  - 0/2/3开头 -> SZ
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import pandas as pd

import select_stocks_local as sel
from stop_loss_rules import StopLossConfig, evaluate_exit_signal


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join("wh", "qyb.csv"))
    parser.add_argument("--output", default=os.path.join("wh", "qyb.csv"))
    parser.add_argument("--end-date", default=datetime_today_yyyymmdd())

    # 止损策略参数（默认沿用 stop_loss_rules.py 默认值）
    parser.add_argument("--stop-loss-drawdown", type=float, default=StopLossConfig().stop_loss_drawdown)
    parser.add_argument("--enable-three-days-down-exit", action="store_true", default=StopLossConfig().enable_three_days_down_exit)
    parser.add_argument("--disable-early-underperform-exit", action="store_true")
    # 默认从 7 改为 5 个交易日
    parser.add_argument("--early-exit-hold-days", type=int, default=StopLossConfig().early_exit_hold_days)
    parser.add_argument("--early-exit-min-return", type=float, default=StopLossConfig().early_exit_min_return)

    args = parser.parse_args()

    end_date = pd.to_datetime(args.end_date)

    sl_cfg = StopLossConfig(
        stop_loss_drawdown=float(args.stop_loss_drawdown),
        enable_three_days_down_exit=bool(args.enable_three_days_down_exit),
        enable_early_underperform_exit=not bool(args.disable_early_underperform_exit),
        early_exit_hold_days=int(args.early_exit_hold_days),
        early_exit_min_return=float(args.early_exit_min_return),
    )

    data_dir = sel.CFG.data_dir

    df_in = pd.read_csv(args.input, dtype=str)
    df_in = df_in.fillna("")

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
    df_out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"已更新: {args.output}")
    print(f"止损参数: {asdict(sl_cfg)}")


def datetime_today_yyyymmdd() -> str:
    return pd.Timestamp.today().strftime("%Y%m%d")


if __name__ == "__main__":
    main()
