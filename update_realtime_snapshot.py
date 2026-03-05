#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量更新通达信 CSV（daily_raw）中的当天实时数据。

数据来源：akshare 快照（优先使用 EastMoney 全市场快照）。
更新策略：
- 对每个 `*.SH.csv` / `*.SZ.csv` 文件，追加或覆盖当天一行：
  trade_date=今天，open=今开，high=最高，low=最低，close=最新价，volume=成交量，amount=成交额。
"""

import os
import sys
import csv
import datetime as dt
from typing import Dict, Any

try:
    from utils.trading_calendar import is_cn_trading_day
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.trading_calendar import is_cn_trading_day


STANDARD_HEADERS = ["trade_date", "open", "high", "low", "close", "volume", "amount"]


def _parse_symbols_arg(symbols_raw: str) -> set[str]:
    """Parse symbols from comma/space-separated string."""
    if not symbols_raw:
        return set()
    tokens = [t.strip() for t in symbols_raw.replace(",", " ").split() if t.strip()]
    out: set[str] = set()
    for t in tokens:
        sym = _normalize_symbol(t)
        if sym:
            out.add(sym)
    return out


def _normalize_symbol(code: str) -> str | None:
    """Normalize code to CODE.SZ/SH format."""
    s = str(code or "").strip().upper()
    if not s:
        return None

    if "." in s:
        parts = s.split(".")
        if len(parts) == 2:
            p0, p1 = parts[0].strip(), parts[1].strip()
            if p0 in ("SZ", "SH") and len(p1) == 6 and p1.isdigit():
                return f"{p1}.{p0}"
            if p1 in ("SZ", "SH") and len(p0) == 6 and p0.isdigit():
                return f"{p0}.{p1}"
        return None

    if s.startswith(("SZ", "SH")) and len(s) >= 8:
        exch = s[:2]
        code_no = s[2:8]
        if code_no.isdigit():
            return f"{code_no}.{exch}"

    if len(s) == 6 and s.isdigit():
        suffix = detect_exchange(s)
        if suffix:
            return f"{s}.{suffix}"
    return None


def _read_csv_rows_smart(path: str) -> list[dict]:
    """Read CSV rows with encoding fallback (utf-8-sig -> gbk)."""
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError(f"无法读取CSV: {path}")


def _load_symbols_from_selection_form(path: str, only_unsold: bool) -> set[str]:
    """Load symbols from selection maintain form."""
    rows = _read_csv_rows_smart(path)
    out: set[str] = set()
    for r in rows:
        code_raw = str(r.get("股票代码") or r.get("代码") or "").strip()
        if not code_raw:
            continue

        if only_unsold:
            closed_flag = str(r.get("是否平仓", "")).strip()
            close_date = str(r.get("平仓日期", "")).strip()
            close_reason = str(r.get("平仓原因", "")).strip()
            if closed_flag == "是" or close_date or close_reason:
                continue

        sym = _normalize_symbol(code_raw)
        if sym:
            out.add(sym)
    return out


def _normalize_header(name: str) -> str:
    return (name or "").strip().lstrip("\ufeff").lower()


def detect_exchange(code_no_prefix: str) -> str:
    """根据纯数字代码推断交易所后缀（SZ/SH）。"""
    if code_no_prefix.startswith(("000", "001", "002", "003", "200", "300", "301")):
        return "SZ"
    if code_no_prefix.startswith(("600", "601", "603", "605", "688", "900")):
        return "SH"
    # 兜底：常见 A 股外的情况不处理
    return ""


def load_spot_snapshot() -> Dict[str, Dict[str, Any]]:
    """从 akshare 获取 A 股快照，返回以 `CODE.SZ/SH` 为键的字典。"""
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError(f"未安装 akshare 或初始化失败: {e}")

    df = None
    # 优先 Eastmoney 全市场接口
    try:
        df = ak.stock_zh_a_spot_em()
        source = "em"
    except Exception:
        # 备用：旧版快照
        try:
            df = ak.stock_zh_a_spot()
            source = "spot"
        except Exception as e:
            raise RuntimeError(f"拉取快照失败: {e}")

    if df is None or len(df) == 0:
        raise RuntimeError("快照数据为空")

    # 标准化列名映射
    # Eastmoney 风格（中文列）
    if "代码" in df.columns:
        code_col = "代码"
        price_col = "最新价"
        open_col = "今开"
        high_col = "最高"
        low_col = "最低"
        vol_col = "成交量"
        amt_col = "成交额"
    else:
        # 旧英文列风格
        code_col = "symbol" if "symbol" in df.columns else "代码"
        price_col = "trade" if "trade" in df.columns else ("最新价" if "最新价" in df.columns else None)
        open_col = "open" if "open" in df.columns else ("今开" if "今开" in df.columns else None)
        high_col = "high" if "high" in df.columns else ("最高" if "最高" in df.columns else None)
        low_col = "low" if "low" in df.columns else ("最低" if "最低" in df.columns else None)
        vol_col = "volume" if "volume" in df.columns else ("成交量" if "成交量" in df.columns else None)
        amt_col = "amount" if "amount" in df.columns else ("成交额" if "成交额" in df.columns else None)

    missing = [k for k, v in {
        "code": code_col,
        "price": price_col,
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "volume": vol_col,
        "amount": amt_col,
    }.items() if v is None]
    if missing:
        raise RuntimeError(f"快照列缺失: {missing}")

    result: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        code_raw = str(row[code_col]).strip().lower()

        if code_raw.startswith("sz") or code_raw.startswith("sh"):
            # 如 sz000001/sh600000
            exch = code_raw[:2].upper()
            code_no = code_raw[2:]
            suffix = "SZ" if exch == "SZ" else "SH"
        else:
            code_no = code_raw
            suffix = detect_exchange(code_no.upper())

        if not suffix or not code_no or len(code_no) != 6:
            continue

        symbol = f"{code_no.upper()}.{suffix}"
        try:
            result[symbol] = {
                "close": float(row[price_col]) if row[price_col] is not None else None,
                "open": float(row[open_col]) if row[open_col] is not None else None,
                "high": float(row[high_col]) if row[high_col] is not None else None,
                "low": float(row[low_col]) if row[low_col] is not None else None,
                "volume": float(row[vol_col]) if row[vol_col] is not None else None,
                "amount": float(row[amt_col]) if row[amt_col] is not None else None,
                "_source": source,
            }
        except Exception:
            # 某些行可能存在非数值内容，跳过该条
            continue

    return result


def update_csv_file(csv_path: str, today: str, snapshot_row: Dict[str, Any]) -> bool:
    """
    用当天快照更新单个 CSV 文件。
    返回是否发生了写入（追加或覆盖）。
    """
    if not snapshot_row:
        return False

    updated = False
    rows = []
    headers = STANDARD_HEADERS

    # 读取全量（兼容 UTF-8 BOM：避免出现 \ufefftrade_date 导致匹配失败）
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_headers = reader.fieldnames or []
        normalized_to_raw = {_normalize_header(h): h for h in raw_headers}

        def get_value(row: Dict[str, Any], *candidates: str) -> str:
            for c in candidates:
                raw = normalized_to_raw.get(_normalize_header(c))
                if raw is not None and raw in row:
                    return row.get(raw)
                if c in row:
                    return row.get(c)
            return ""

        for r in reader:
            rows.append({
                "trade_date": get_value(r, "trade_date", "date"),
                "open": get_value(r, "open", "Open"),
                "high": get_value(r, "high", "High"),
                "low": get_value(r, "low", "Low"),
                "close": get_value(r, "close", "Close"),
                "volume": get_value(r, "volume", "Volume"),
                "amount": get_value(r, "amount", "Amount"),
            })

    # 先删除任何已存在的当天数据行
    filtered_rows = []
    for r in rows:
        rdate = (r.get("trade_date") or r.get("date") or "").strip()
        if rdate != today:
            filtered_rows.append(r)
    rows = filtered_rows

    new_record = {
        "trade_date": today,
        "open": snapshot_row.get("open"),
        "high": snapshot_row.get("high"),
        "low": snapshot_row.get("low"),
        "close": snapshot_row.get("close"),
        "volume": snapshot_row.get("volume"),
        "amount": snapshot_row.get("amount"),
    }

    # 将数值转为字符串，保持 CSV 一致性
    for k, v in list(new_record.items()):
        if isinstance(v, (int, float)):
            new_record[k] = f"{float(v)}"
        elif v is None:
            new_record[k] = ""

    # 追加新的当天记录（统一在末尾）
    rows.append(new_record)
    updated = True

    # 写回文件
    if updated:
        # 使用标准头写回（去掉 BOM），避免后续再次出现 \ufefftrade_date
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STANDARD_HEADERS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in STANDARD_HEADERS})

    return updated


def main():
    import argparse
    parser = argparse.ArgumentParser(description="更新 daily_raw CSV 的当天实时快照")
    parser.add_argument(
        "--data-dir",
        default=os.path.join("通达信", "data", "pytdx", "daily_raw"),
        help="CSV 目录，默认为 通达信/data/pytdx/daily_raw",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().strftime("%Y-%m-%d"),
        help="指定日期（YYYY-MM-DD），默认今天",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将更新的文件与数据，不实际写入",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="仅更新指定股票列表（逗号/空格分隔，支持 000001 或 000001.SZ）",
    )
    parser.add_argument(
        "--selection-form",
        default="",
        help="从选股维护表单读取股票代码并仅更新这些股票",
    )
    parser.add_argument(
        "--only-unsold",
        action="store_true",
        help="配合 --selection-form: 仅更新未平仓股票",
    )
    args = parser.parse_args()

    try:
        data_dir = os.path.abspath(args.data_dir)
        today = args.date
        today_dt = dt.datetime.strptime(today, "%Y-%m-%d").date()

        if not is_cn_trading_day(today_dt):
            print(f"非交易日，跳过更新: {today}")
            return

        if not os.path.isdir(data_dir):
            raise RuntimeError(f"目录不存在: {data_dir}")

        if args.only_unsold and (not str(args.selection_form).strip()):
            raise RuntimeError("--only-unsold 需要配合 --selection-form 使用")

        target_symbols = set()
        symbols_arg = str(args.symbols).strip()
        selection_form_arg = str(args.selection_form).strip()

        target_symbols |= _parse_symbols_arg(symbols_arg)

        if selection_form_arg:
            form_path = os.path.abspath(selection_form_arg)
            if not os.path.exists(form_path):
                raise RuntimeError(f"选股维护表单不存在: {form_path}")
            target_symbols |= _load_symbols_from_selection_form(form_path, bool(args.only_unsold))

        if (symbols_arg or selection_form_arg) and (not target_symbols):
            print("未找到有效股票，跳过更新")
            return

        if target_symbols:
            print(f"仅更新 {len(target_symbols)} 只股票的实时数据", flush=True)

        print("拉取快照中……", flush=True)
        snapshot = load_spot_snapshot()
        print(f"快照股票数: {len(snapshot)}")

        # 遍历 CSV 文件
        updated_count = 0
        skipped_count = 0
        missing_count = 0
        failed_count = 0
        missing_file_count = 0

        if target_symbols:
            symbols = sorted(target_symbols)
            for symbol in symbols:
                csv_path = os.path.join(data_dir, f"{symbol}.csv")
                if not os.path.exists(csv_path):
                    missing_file_count += 1
                    continue

                row = snapshot.get(symbol)
                if row is None:
                    missing_count += 1
                    continue

                if args.dry_run:
                    print(f"[DRY] {symbol}: {row}")
                    skipped_count += 1
                    continue

                try:
                    if update_csv_file(csv_path, today, row):
                        updated_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"更新失败: {symbol} ({csv_path}) -> {e}", file=sys.stderr)
        else:
            files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            for fname in files:
                # 文件名形如 000001.SZ.csv
                base = os.path.splitext(fname)[0]  # 000001.SZ
                symbol = base
                row = snapshot.get(symbol)
                csv_path = os.path.join(data_dir, fname)

                if row is None:
                    missing_count += 1
                    continue

                if args.dry_run:
                    print(f"[DRY] {symbol}: {row}")
                    skipped_count += 1
                    continue

                try:
                    if update_csv_file(csv_path, today, row):
                        updated_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"更新失败: {symbol} ({csv_path}) -> {e}", file=sys.stderr)

        msg = (
            f"更新完成：写入 {updated_count} 个文件；跳过 {skipped_count}；未匹配 {missing_count}；失败 {failed_count}。"
        )
        if missing_file_count > 0:
            msg = msg.rstrip("。") + f"；缺失文件 {missing_file_count}。"
        print(msg)
        if failed_count > 0:
            sys.exit(1)
    except Exception as e:
        print(f"更新失败：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
