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


STANDARD_HEADERS = ["trade_date", "open", "high", "low", "close", "volume", "amount"]


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
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    today = args.date

    if not os.path.isdir(data_dir):
        print(f"目录不存在: {data_dir}")
        sys.exit(1)

    print("拉取快照中……")
    snapshot = load_spot_snapshot()
    print(f"快照股票数: {len(snapshot)}")

    # 遍历 CSV 文件
    updated_count = 0
    skipped_count = 0
    missing_count = 0
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

        if update_csv_file(csv_path, today, row):
            updated_count += 1

    print(f"更新完成：写入 {updated_count} 个文件；跳过 {skipped_count}；未匹配 {missing_count}。")


if __name__ == "__main__":
    main()
