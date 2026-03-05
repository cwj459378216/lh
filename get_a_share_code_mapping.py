"""获取A股股票代码与名称映射。

依赖：akshare、pandas
用法：
  python get_a_share_code_mapping.py --out csv/stock_code_mapping.csv
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable

import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _detect_exchange(code: str) -> str | None:
    code = (code or "").strip()
    if not code.isdigit() or len(code) != 6:
        return None
    if code.startswith("6"):
        return "SH"
    if code.startswith(("0", "2", "3")):
        return "SZ"
    return None


def _normalize_rows(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["exchange"] = df["code"].apply(_detect_exchange)
    df = df[df["exchange"].notna()].copy()
    df["symbol"] = df["code"] + "." + df["exchange"]
    df["name"] = df["name"].astype(str).str.strip()
    return df.sort_values(["exchange", "code"]).reset_index(drop=True)


def fetch_with_akshare() -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError(f"未安装 akshare 或初始化失败: {e}")

    rows: list[dict] = []

    # 首选：全市场行情快照（包含代码/名称）
    try:
        spot = ak.stock_zh_a_spot_em()
        if spot is not None and len(spot) > 0:
            code_col = "代码" if "代码" in spot.columns else "symbol"
            name_col = "名称" if "名称" in spot.columns else "name"
            for _, r in spot.iterrows():
                code = str(r.get(code_col, "")).strip()
                name = str(r.get(name_col, "")).strip()
                if code:
                    rows.append({"code": code, "name": name})
            df = _normalize_rows(rows)
            if not df.empty:
                return df
    except Exception:
        pass

    # 备用：基础代码名称表
    try:
        base = ak.stock_info_a_code_name()
        if base is not None and len(base) > 0:
            code_col = "code" if "code" in base.columns else "代码"
            name_col = "name" if "name" in base.columns else "名称"
            for _, r in base.iterrows():
                code = str(r.get(code_col, "")).strip()
                name = str(r.get(name_col, "")).strip()
                if code:
                    rows.append({"code": code, "name": name})
            df = _normalize_rows(rows)
            if not df.empty:
                return df
    except Exception:
        pass

    raise RuntimeError("拉取A股代码映射失败：akshare数据源不可用或返回为空")


def main() -> None:
    parser = argparse.ArgumentParser(description="获取A股股票代码与名称映射")
    parser.add_argument(
        "--out",
        default=os.path.join("csv", "stock_code_mapping.csv"),
        help="输出CSV路径（默认 csv/stock_code_mapping.csv）",
    )
    args = parser.parse_args()

    df = fetch_with_akshare()

    out_path = args.out
    out_dir = os.path.dirname(out_path) or "."
    _ensure_dir(out_dir)
    df.to_csv(out_path, index=False)
    print(f"已写入: {out_path} | {len(df)} 条")


if __name__ == "__main__":
    main()
