"""run_schedule_workflow.py

使用 schedule 库在本机常驻定时执行日常流程：
- 每天定时触发，但仅在【中国法定工作日】才真正执行：
    - python run_daily_workflow.py --flow stoploss
    - python run_daily_workflow.py --flow select
    - python run_daily_workflow.py --flow pc

说明：
- 这是一个常驻进程脚本，需要一直运行（可放在后台/用 launchd/pm2/supervisor 等托管）。
- 子流程内部已包含企业微信通知（成功/失败都会发送）。

运行：
  python run_schedule_workflow.py

可选：
  # 跳过 update_realtime_snapshot
  python run_schedule_workflow.py --skip-snapshot

    # 仅调度某一个流程（另一个不跑）
    python run_schedule_workflow.py --flow stoploss
    python run_schedule_workflow.py --flow select
    python run_schedule_workflow.py --flow pc

    # 立即触发一次（然后仍保持常驻）
    python run_schedule_workflow.py --flow stoploss --run-now

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from datetime import date as dt_date

import schedule

try:
    from chinese_calendar import is_workday  # type: ignore
except Exception:  # pragma: no cover
    is_workday = None


ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today() -> dt_date:
    return datetime.now().date()


def _is_cn_workday(d: dt_date) -> bool:
    """是否中国法定工作日。

    依赖 chinese_calendar；若不可用则退化为周一~周五。
    """
    if is_workday is not None:
        return bool(is_workday(d))
    # fallback：周一~周五
    return d.weekday() < 5


def _run_flow(flow: str, skip_snapshot: bool) -> None:
    today = _today()
    if not _is_cn_workday(today):
        print(f"\n[{_now()}] SKIP (not CN workday): {today} flow={flow}", flush=True)
        return

    cmd = [PY, os.path.join(ROOT, "run_daily_workflow.py"), "--flow", flow]
    if skip_snapshot:
        cmd.append("--skip-snapshot")

    print(f"\n[{_now()}] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="调度执行时跳过 update_realtime_snapshot.py（等同于给 run_daily_workflow.py 传 --skip-snapshot）",
    )
    parser.add_argument(
        "--flow",
        choices=["stoploss", "select", "pc", "all"],
        default="all",
        help="调度哪些流程：stoploss / select / pc / all(默认全部调度)",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="启动后立即触发一次所选 flow（随后仍保持常驻调度）",
    )
    args = parser.parse_args()

    flow = str(getattr(args, 'flow', 'all') or 'all').strip().lower()
    skip_snapshot = bool(getattr(args, 'skip_snapshot', False))

    # 05:00 pc（在线策略选股）
    if flow in ("pc", "all"):
        schedule.every().day.at("22:10").do(_run_flow, flow="pc", skip_snapshot=skip_snapshot)

    # 14:50 stoploss
    if flow in ("stoploss", "all"):
        schedule.every().day.at("14:50").do(_run_flow, flow="stoploss", skip_snapshot=skip_snapshot)
    # 15:20 select
    if flow in ("select", "all"):
        schedule.every().day.at("15:20").do(_run_flow, flow="select", skip_snapshot=skip_snapshot)

    if bool(getattr(args, 'run_now', False)):
        if flow in ("pc", "all"):
            _run_flow(flow="pc", skip_snapshot=skip_snapshot)
        if flow in ("stoploss", "all"):
            _run_flow(flow="stoploss", skip_snapshot=skip_snapshot)
        if flow in ("select", "all"):
            _run_flow(flow="select", skip_snapshot=skip_snapshot)

    print(f"[{_now()}] scheduler started (flow={flow}, skip_snapshot={skip_snapshot})", flush=True)
    print("jobs:")
    for j in schedule.get_jobs():
        print("-", j)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
