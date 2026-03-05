"""run_daily_workflow.py

流程化控制脚本：
1) 运行 update_realtime_snapshot.py 更新当日实时数据，并打印更新统计
2) 运行 select_stocks_local.py 筛选当日符合条件的股票，并打印选股结果（前 N 行）
3) 运行 update_stop_loss_table.py 更新 output/选股维护表单.csv 的平仓信息，并打印“截至 end-date 已平仓”明细

用法示例：
  # 不传 --signal-date/--end-date 时，默认使用“当天”日期
    python run_daily_workflow.py

  # 指定日期
    python run_daily_workflow.py --signal-date 20260119 --end-date 20260120

    # 指定只跑某个流程
    python run_daily_workflow.py --flow stoploss
    python run_daily_workflow.py --flow select

    # 跳过 update_realtime_snapshot（直接用现有数据继续跑后续步骤）
    python run_daily_workflow.py --flow stoploss --skip-snapshot
    python run_daily_workflow.py --flow select --skip-snapshot

    # 测试通知（不执行任何脚本，仅发一条消息验证 webhook/定时器）
    python run_daily_workflow.py --flow test

说明：
- 本脚本通过子进程调用现有脚本，尽量不侵入原逻辑。
- “当日平仓”定义：在选股维护表单中，平仓日期 <= end-date 且 是否平仓 == 是。

新增：支持拆分为两个“定时子流程”
- flow=stoploss: update_realtime_snapshot(仅未平仓股票) -> update_stop_loss_table
- flow=select:   update_realtime_snapshot -> select_stocks_local
- flow=test:     仅发送测试通知

每个流程结束后（成功/失败）都会发送企业微信通知（若配置了 webhook）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from datetime import datetime

import json
import pandas as pd
import urllib.request


ROOT = os.path.dirname(os.path.abspath(__file__))

# 持仓上限（可配置）
MAX_HOLDINGS = 4

# 企业微信机器人 webhook（写死默认值；如需变更，直接改这里即可）
DEFAULT_WECOM_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=de57fc57-503b-4b2d-b62f-f5a0cb92bd59"


def _run(cmd: list[str], cwd: str | None = None) -> None:
    # 直接透传 stdout/stderr，便于观察原脚本输出
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_capture(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    """运行命令并捕获输出（用于做兼容性回退判断）。"""
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )
    out_lines: list[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        out_lines.append(line)
        print(line, end="")
    p.wait()
    out = "".join(out_lines)
    return int(p.returncode), out


def _run_quiet(cmd: list[str], cwd: str | None = None) -> None:
    """静默运行：只在失败时抛错，不打印子脚本的详细过程输出。"""
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        out = p.stdout or ''
        # 失败时把输出带出来，便于排查
        raise subprocess.CalledProcessError(int(p.returncode), cmd, output=out)


def _read_csv_smart(path: str) -> pd.DataFrame:
    # 兼容 utf-8-sig / utf-8 / gbk（部分 Windows CSV）
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError(f"无法读取CSV: {path}")


def _print_df(df: pd.DataFrame, title: str, max_rows: int = 50) -> None:
    print(f"\n=== {title} (rows={len(df)}) ===")
    if df is None or df.empty:
        print("(empty)")
        return
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.max_colwidth",
        60,
    ):
        print(df.head(max_rows).to_string(index=False))


def _extract_snapshot_summary(output: str) -> str:
    """从 update_realtime_snapshot 输出中提取“更新完成：...”统计行。"""
    if not output:
        return ""
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("更新完成"):
            return s
    return ""


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_selected_for_signal_date(selection_form: str, signal_date: str) -> pd.DataFrame:
    """当没有 selection_local_*.csv 时，从维护表单中提取当日('信号日')的股票列表。"""
    if not os.path.exists(selection_form):
        return pd.DataFrame()
    df = _read_csv_smart(selection_form)
    if df is None or df.empty:
        return pd.DataFrame()

    if '信号日' not in df.columns:
        return pd.DataFrame()

    df2 = df.copy()
    df2['信号日'] = df2['信号日'].astype(str).str.strip().str.replace('-', '', regex=False)
    df2 = df2[df2['信号日'] == str(signal_date).strip()]

    # 只保留关键信息列（存在就留）
    keep = [c for c in ['信号日', '股票代码', '股票名', '原始评分'] if c in df2.columns]
    if keep:
        df2 = df2[keep]
    return df2.reset_index(drop=True)


def _load_code_name_map() -> dict[str, str]:
    """从 csv/stock_code_mapping.csv 读取 symbol->name 映射。"""
    map_path = os.path.join(ROOT, "csv", "stock_code_mapping.csv")
    if not os.path.exists(map_path):
        return {}
    try:
        df_map = pd.read_csv(map_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        return {}
    if df_map is None or df_map.empty:
        return {}
    if "symbol" not in df_map.columns or "name" not in df_map.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in df_map.iterrows():
        sym = str(r.get("symbol", "") or "").strip().upper()
        name = str(r.get("name", "") or "").strip()
        if sym and name and sym not in out:
            out[sym] = name
    return out


def _attach_stock_name(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "股票代码" not in df.columns:
        return df
    name_map = _load_code_name_map()
    if not name_map:
        return df
    df2 = df.copy()
    if "股票名" not in df2.columns:
        df2["股票名"] = ""
    name_series = df2["股票名"].astype(str).fillna("")
    fill_mask = name_series.str.strip() == ""
    if fill_mask.any():
        df2.loc[fill_mask, "股票名"] = df2.loc[fill_mask, "股票代码"].map(
            lambda x: name_map.get(_normalize_code(x), "")
        )
    return df2


def _row_key(code: str, signal_date: str) -> str:
    return f"{_normalize_code(code)}|{str(signal_date or '').strip().replace('-', '')}"


def _normalize_code(val: str) -> str:
    return str(val or "").strip().upper()


def _load_unsold_holdings(selection_form: str) -> set[str]:
    if not os.path.exists(selection_form):
        return set()
    df = _read_csv_smart(selection_form)
    if df is None or df.empty:
        return set()
    if "股票代码" not in df.columns:
        return set()

    out: set[str] = set()
    for _, r in df.iterrows():
        code = _normalize_code(r.get("股票代码", ""))
        if not code:
            continue
        closed_flag = str(r.get("是否平仓", "")).strip()
        close_date = str(r.get("平仓日期", "")).strip()
        close_reason = str(r.get("平仓原因", "")).strip()
        if closed_flag == "是" or close_date or close_reason:
            continue
        out.add(code)
    return out


def _pick_top_by_score(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty or n <= 0:
        return df.head(0) if df is not None else pd.DataFrame()
    if "原始评分" in df.columns:
        df2 = df.copy()
        df2["__score"] = pd.to_numeric(df2["原始评分"], errors="coerce")
        df2 = df2.sort_values("__score", ascending=False, na_position="last")
        df2 = df2.drop(columns=["__score"], errors="ignore")
        return df2.head(n)
    return df.head(n)


def _is_unsold_row(r: pd.Series) -> bool:
    closed_flag = str(r.get("是否平仓", "")).strip()
    close_date = str(r.get("平仓日期", "")).strip()
    close_reason = str(r.get("平仓原因", "")).strip()
    return not (closed_flag == "是" or close_date or close_reason)


def _cap_holdings_in_form(selection_form: str, signal_date: str, max_holdings: int = MAX_HOLDINGS) -> int:
    if not os.path.exists(selection_form):
        return 0
    df = _read_csv_smart(selection_form)
    if df is None or df.empty:
        return 0
    if "股票代码" not in df.columns:
        return 0

    df2 = df.copy()
    df2["__unsold"] = df2.apply(_is_unsold_row, axis=1)
    df2["__signal"] = df2.get("信号日", "").astype(str).str.strip().str.replace("-", "", regex=False)

    unsold = df2[df2["__unsold"]].copy()
    if unsold.empty:
        return 0

    today_unsold = unsold[unsold["__signal"] == str(signal_date).strip()].copy()
    old_unsold = unsold[unsold["__signal"] != str(signal_date).strip()].copy()

    if len(old_unsold) >= max_holdings:
        keep_pool = old_unsold.copy()
        if len(keep_pool) > max_holdings:
            if "原始评分" in keep_pool.columns:
                keep_pool["__score"] = pd.to_numeric(keep_pool["原始评分"], errors="coerce")
                keep_pool = keep_pool.sort_values("__score", ascending=False, na_position="last")
            keep_pool = keep_pool.head(max_holdings)
    else:
        need = max_holdings - len(old_unsold)
        if "原始评分" in today_unsold.columns:
            today_unsold["__score"] = pd.to_numeric(today_unsold["原始评分"], errors="coerce")
            today_unsold = today_unsold.sort_values("__score", ascending=False, na_position="last")
        keep_pool = pd.concat([old_unsold, today_unsold.head(need)], axis=0)

    keep_unsold_idx = set(keep_pool.index.tolist())
    df_keep = df2[(~df2["__unsold"]) | (df2.index.isin(keep_unsold_idx))].copy()
    df_keep = df_keep.drop(columns=["__unsold", "__score", "__signal"], errors="ignore")
    df_keep.to_csv(selection_form, index=False, encoding="utf-8-sig")
    return len(keep_unsold_idx)


def _send_wecom_text(
    webhook_url: str,
    content: str,
    timeout: int = 10,
    msgtype: str = "text",
) -> tuple[int | None, str]:
    """发送企业微信机器人消息（默认 text）。返回 (errcode, errmsg)。"""
    msgtype = str(msgtype or "text").strip().lower()
    if msgtype != "markdown":
        msgtype = "text"
    if msgtype == "markdown":
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
    else:
        payload = {
            "msgtype": "text",
            "text": {"content": content},
        }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    try:
        j = json.loads(body)
        return j.get("errcode"), j.get("errmsg")
    except Exception:
        return None, body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", required=False, default=date.today().strftime("%Y%m%d"), help="选股信号日 YYYYMMDD")
    parser.add_argument("--end-date", required=False, default=date.today().strftime("%Y%m%d"), help="止损/平仓评估截止日 YYYYMMDD")

    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="跳过 update_realtime_snapshot.py（直接用已有数据继续执行后续步骤）",
    )

    parser.add_argument(
        "--flow",
        choices=["stoploss", "select", "full", "test"],
        default="full",
        help="执行流程：stoploss(快照->止损表) / select(快照->选股) / full(原完整流程) / test(仅发通知)",
    )

    parser.add_argument("--selection-form", default=os.path.join(ROOT, "output", "选股维护表单.csv"), help="选股维护表单路径")
    # 选股时默认始终增量更新维护表单（写死启用，不再暴露为命令行参数）

    parser.add_argument("--print-selected", type=int, default=50, help="打印选股结果前 N 行")
    parser.add_argument("--print-closed", type=int, default=200, help="打印当日平仓前 N 行")

    parser.add_argument(
        "--wecom-webhook",
        default=DEFAULT_WECOM_WEBHOOK,
        help="企业微信机器人 webhook 完整URL（默认已写死；如需变更，修改 DEFAULT_WECOM_WEBHOOK 即可）。",
    )

    args = parser.parse_args()

    signal_date = str(args.signal_date).strip()
    end_date = str(args.end_date).strip()
    selection_form = os.path.abspath(args.selection_form)

    # webhook 写死默认值；仍允许通过参数覆盖（但通常无需传）
    wecom_webhook = str(getattr(args, 'wecom_webhook', '') or '').strip() or DEFAULT_WECOM_WEBHOOK

    flow = str(getattr(args, 'flow', 'full') or 'full').strip().lower()
    skip_snapshot = bool(getattr(args, 'skip_snapshot', False))

    msg_lines: list[str] = []
    status = "OK"
    err_text = ""
    suppress_notify = False

    # 为了满足“每个流程执行完成后都要通知”：无论成功/失败都 try 发送
    try:
        if flow == "test":
            print("[TEST] flow=test：仅发送测试通知，不执行 update/select/stoploss")
            msg_lines.append("[TEST] notify-only")
            return 0

        # 1) 更新实时快照（可按需跳过）
        if skip_snapshot:
            print("1. 跳过 update_realtime_snapshot（--skip-snapshot）")
            msg_lines.append("数据更新成功(跳过)")
        else:
            snap_cmd = [sys.executable, os.path.join(ROOT, "update_realtime_snapshot.py")]
            if flow == "stoploss":
                snap_cmd += ["--selection-form", selection_form, "--only-unsold"]

            rc1, out1 = _run_capture(snap_cmd, cwd=ROOT)
            if rc1 != 0:
                msg_lines.append("数据更新失败")
                raise subprocess.CalledProcessError(rc1, snap_cmd)

            snap_summary = _extract_snapshot_summary(out1)
            if snap_summary:
                line1 = "数据更新成功"
                print(line1)
                msg_lines.append(line1)

        # 2) 选股（flow=select/full）
        if flow in ("select", "full"):
            holdings_before = _load_unsold_holdings(selection_form)
            holdings_before_count = len(holdings_before)

            select_cmd = [
                sys.executable,
                os.path.join(ROOT, "select_stocks_local.py"),
                "--end-date",
                signal_date,
                "--update-form",
            ]
            _run_quiet(select_cmd, cwd=ROOT)

            capped = _cap_holdings_in_form(selection_form, signal_date=signal_date, max_holdings=MAX_HOLDINGS)
            if capped:
                print(f"(选股维护表单持仓上限={MAX_HOLDINGS}，当前持仓={capped})")

            # 尝试打印最新 selection_local_*.csv（如果存在）
            out_dir = os.path.join(ROOT, "output")
            latest_selection = None
            if os.path.isdir(out_dir):
                cands = [
                    os.path.join(out_dir, f)
                    for f in os.listdir(out_dir)
                    if f.startswith("selection_local_") and f.lower().endswith(".csv")
                ]
                if cands:
                    latest_selection = max(cands, key=lambda p: os.path.getmtime(p))

            df_selected: pd.DataFrame
            if latest_selection and os.path.exists(latest_selection):
                df_selected = _read_csv_smart(latest_selection)
            else:
                df_selected = _load_selected_for_signal_date(selection_form, signal_date)

            df_selected = _attach_stock_name(df_selected)

            holdings = _load_unsold_holdings(selection_form)
            holdings_count = len(holdings)

            if flow == "select" and holdings_before_count > MAX_HOLDINGS:
                print(f"2.=== 持仓数={holdings_before_count}，超过 {MAX_HOLDINGS}")
                msg_lines.append(f"持仓数={holdings_before_count}，超过上限={MAX_HOLDINGS}")
            else:
                print("2.=== 选中股票")

                df_send_base = df_selected.copy() if df_selected is not None else pd.DataFrame()
                if not df_send_base.empty and "股票代码" in df_send_base.columns:
                    df_send_base["__code"] = df_send_base["股票代码"].apply(_normalize_code)
                    if holdings_before:
                        df_send_base = df_send_base[~df_send_base["__code"].isin(holdings_before)]
                    df_send_base = df_send_base.drop(columns=["__code"], errors="ignore")

                df_send_base = _attach_stock_name(df_send_base)

                capacity = max(0, MAX_HOLDINGS - holdings_before_count)
                if capacity < len(df_send_base):
                    df_send_base = _pick_top_by_score(df_send_base, capacity)

                total_selected = int(len(df_selected)) if df_selected is not None else 0
                picked_count = int(len(df_send_base))
                msg_lines.append(f"一共选股{total_selected}个，入选{picked_count}个")

                # 企业微信文本限制较紧：只发简表（最多 30 行）
                df_send = df_send_base.copy().head(30) if df_send_base is not None else pd.DataFrame()

                for _, r in df_send.iterrows():
                    code = str(r.get('股票代码', '')).strip()
                    name = str(r.get('股票名', '')).strip() if '股票名' in df_send.columns else ''
                    if not name:
                        name = code
                    if code:
                        msg_lines.append(f"🔴{name}：{code}")

                _print_df(df_selected, "选中股票", max_rows=int(args.print_selected))

        # 3) 更新平仓信息（flow=stoploss/full）
        if flow in ("stoploss", "full"):
            df_form_before = None
            if os.path.exists(selection_form):
                try:
                    df_form_before = _read_csv_smart(selection_form)
                except Exception:
                    df_form_before = None
            stop_loss_cmd = [
                sys.executable,
                os.path.join(ROOT, "update_stop_loss_table.py"),
                "--input",
                selection_form,
                "--end-date",
                end_date,
            ]
            _run_quiet(stop_loss_cmd, cwd=ROOT)

            if not os.path.exists(selection_form):
                print(f"\n(维护表单不存在，无法打印当日平仓: {selection_form})")
            else:
                df_form = _read_csv_smart(selection_form)

                # 仅保留“是否平仓”从否/空到是的行
                newly_closed_keys: set[str] = set()
                if df_form_before is not None and not df_form_before.empty:
                    try:
                        before = df_form_before.copy()
                        after = df_form.copy()

                        if "信号日" in before.columns and "信号日" in after.columns:
                            before["信号日"] = before["信号日"].astype(str).str.strip().str.replace("-", "", regex=False)
                            after["信号日"] = after["信号日"].astype(str).str.strip().str.replace("-", "", regex=False)

                        before_map: dict[str, str] = {}
                        if "股票代码" in before.columns and "信号日" in before.columns and "是否平仓" in before.columns:
                            for _, r in before.iterrows():
                                k = _row_key(r.get("股票代码", ""), r.get("信号日", ""))
                                before_map[k] = str(r.get("是否平仓", "")).strip()

                        if "股票代码" in after.columns and "信号日" in after.columns and "是否平仓" in after.columns:
                            for _, r in after.iterrows():
                                k = _row_key(r.get("股票代码", ""), r.get("信号日", ""))
                                after_flag = str(r.get("是否平仓", "")).strip()
                                before_flag = str(before_map.get(k, "")).strip()
                                if after_flag == "是" and before_flag != "是":
                                    newly_closed_keys.add(k)
                    except Exception:
                        newly_closed_keys = set()

                # 规范列名
                cols = {str(c).strip(): c for c in df_form.columns}
                must = ["是否平仓", "平仓日期"]
                for c in must:
                    if c not in cols:
                        print(f"\n(维护表单缺少列 {c}，无法过滤当日平仓)")
                        _print_df(df_form, f"维护表单: {os.path.relpath(selection_form, ROOT)}", max_rows=50)
                        break
                else:
                    # 当日平仓：平仓日期 <= end_date 且 是否平仓 == 是
                    df_closed = df_form.copy()
                    df_closed["是否平仓"] = df_closed["是否平仓"].astype(str).str.strip()
                    df_closed["平仓日期"] = df_closed["平仓日期"].astype(str).str.strip().str.replace("-", "", regex=False)

                    df_closed = df_closed[
                        (df_closed["是否平仓"] == "是")
                        & (df_closed["平仓日期"].astype(str) <= end_date)
                    ].copy()

                    if newly_closed_keys:
                        df_closed["__key"] = df_closed.apply(
                            lambda r: _row_key(r.get("股票代码", ""), r.get("信号日", "")),
                            axis=1,
                        )
                        df_closed = df_closed[df_closed["__key"].isin(newly_closed_keys)].copy()
                        df_closed = df_closed.drop(columns=["__key"], errors="ignore")

                    sort_cols = [c for c in ["平仓日期", "股票代码", "信号日", "平仓原因"] if c in df_closed.columns]
                    if sort_cols:
                        df_closed = df_closed.sort_values(sort_cols)

                    print("3. 当日平仓股票")

                    if df_closed is not None and (not df_closed.empty):
                        df_send2 = _attach_stock_name(df_closed.copy())
                        keep_cols2 = [c for c in ['股票代码', '股票名', '平仓原因'] if c in df_send2.columns]
                        if keep_cols2:
                            df_send2 = df_send2[keep_cols2]
                        df_send2 = df_send2.head(50)

                        for _, r in df_send2.iterrows():
                            code = str(r.get('股票代码', '')).strip()
                            name = str(r.get('股票名', '')).strip() if '股票名' in df_send2.columns else ''
                            if not name:
                                name = code
                            reason = str(r.get('平仓原因', '')).strip() or "SELL"
                            if name:
                                msg_lines.append(f"🟢{name}: {reason}")

                    _print_df(df_closed, f"当日平仓股票(end-date={end_date})", max_rows=int(args.print_closed))


    except Exception as e:
        status = "FAIL"
        err_text = str(e)
        raise
    finally:
        # 发送企业微信（无论成功/失败）
        if wecom_webhook and (not suppress_notify):
            if status != "OK" and msg_lines == ["数据更新失败"]:
                content = "数据更新失败"
            elif flow == "stoploss":
                clean_lines = [x for x in msg_lines if str(x).strip()]
                content = "\n".join(clean_lines)
                if not content:
                    content = "数据更新成功"
            elif flow in ("select", "stoploss"):
                clean_lines = [x for x in msg_lines if str(x).strip()]
                if flow == "select":
                    content = "\n".join(clean_lines)
                    if not content:
                        content = "当日无选中股票"
                else:
                    body_lines = [
                        x for x in clean_lines
                        if not str(x).strip().startswith("1.")
                        and not str(x).strip().startswith("3.")
                    ]
                    content = "\n".join(body_lines)
                    if not content:
                        content = "当日无平仓"
            else:
                header = f"[{status}] flow={flow} time={_now_str()} signal={signal_date} end={end_date}"
                content = "\n".join([header] + [x for x in msg_lines if str(x).strip()])
                if status != "OK" and err_text:
                    content = content + "\n" + f"error={err_text}"
            try:
                errcode, errmsg = _send_wecom_text(wecom_webhook, content)
                if errcode != 0:
                    print(f"\n(企业微信发送失败: errcode={errcode}, errmsg={errmsg})")
            except Exception as e2:
                print(f"\n(企业微信发送异常: {e2})")
        elif suppress_notify:
            print("\n(持仓数超过 3，跳过企业微信通知)")
        else:
            print("\n(未配置企业微信 webhook，跳过发送通知)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
