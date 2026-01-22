"""run_schedule_workflow.py

使用 schedule 库在本机常驻定时执行日常流程：
- 每天 02:50 执行：python run_daily_workflow.py --flow stoploss
- 每天 03:50 执行：python run_daily_workflow.py --flow select

说明：
- 这是一个常驻进程脚本，需要一直运行（可放在后台/用 launchd/pm2/supervisor 等托管）。
- 子流程内部已包含企业微信通知（成功/失败都会发送）。

运行：
  python run_schedule_workflow.py

可选：
  # 跳过 update_realtime_snapshot
  python run_schedule_workflow.py --skip-snapshot

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import schedule


ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_flow(flow: str, skip_snapshot: bool) -> None:
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
    args = parser.parse_args()

    # 14:50 stoploss
    schedule.every().day.at("14:50").do(_run_flow, flow="stoploss", skip_snapshot=bool(args.skip_snapshot))
    # 15:20 select
    schedule.every().day.at("15:20").do(_run_flow, flow="select", skip_snapshot=bool(args.skip_snapshot))

    print(f"[{_now()}] scheduler started (skip_snapshot={bool(args.skip_snapshot)})", flush=True)
    print("jobs:")
    for j in schedule.get_jobs():
        print("-", j)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
