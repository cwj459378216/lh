# -*- coding: utf-8 -*-
"""
config.py — 全局配置

可直接修改此文件，也可通过环境变量覆盖。
"""

import os

# ═══════════════════════════════════════════════════════════════════
# 企业微信 Webhook（在企业微信群 → 群机器人中获取）
# ═══════════════════════════════════════════════════════════════════
WECHAT_WEBHOOK = os.getenv(
    "WECHAT_WEBHOOK",
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY_HERE",
)

# ═══════════════════════════════════════════════════════════════════
# 数据目录（相对于本文件位置）
# ═══════════════════════════════════════════════════════════════════
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 日线 CSV 目录：直接复用已有的通达信 daily_raw
DAILY_DATA_DIR = os.getenv(
    "DAILY_DATA_DIR",
    os.path.join(_BASE_DIR, "..", "通达信", "data", "pytdx", "daily_raw"),
)

# 分钟线 CSV 目录（需自行下载或生成）
MINUTE_DATA_DIR = os.getenv(
    "MINUTE_DATA_DIR",
    os.path.join(_BASE_DIR, "data", "minute"),
)

# ═══════════════════════════════════════════════════════════════════
# 策略参数（尾盘买入条件）
# ═══════════════════════════════════════════════════════════════════
DAILY_GAIN_MAX = float(os.getenv("DAILY_GAIN_MAX", "0.05"))
"""当日涨幅上限，超过此值不触发（默认 5%）"""

TAIL_START_TIME = os.getenv("TAIL_START_TIME", "14:50")
"""尾盘起始时间 HH:MM"""

TAIL_END_TIME = os.getenv("TAIL_END_TIME", "15:00")
"""尾盘结束时间 HH:MM"""

TAIL_MINUTES = int(os.getenv("TAIL_MINUTES", "10"))
"""尾盘区间分钟数（默认 10）"""

TAIL_PRICE_RISE_MIN = float(os.getenv("TAIL_PRICE_RISE_MIN", "0.005"))
"""尾盘价格涨幅下限（默认 0.5%）"""

TAIL_VOLUME_RATIO_MIN = float(os.getenv("TAIL_VOLUME_RATIO_MIN", "1.5"))
"""尾盘量比下限：(尾盘每分钟均量) / (全日每分钟均量)"""

# ═══════════════════════════════════════════════════════════════════
# 实时监控参数
# ═══════════════════════════════════════════════════════════════════
MONITOR_POLL_INTERVAL = int(os.getenv("MONITOR_POLL_INTERVAL", "60"))
"""轮询间隔（秒），建议 ≥ 30 避免触发 AkShare 限频"""

MONITOR_START_TIME = os.getenv("MONITOR_START_TIME", "14:45")
"""监控开始时间（提前几分钟准备基线快照）"""

MONITOR_END_TIME = os.getenv("MONITOR_END_TIME", "15:01")
"""监控结束时间"""

# ═══════════════════════════════════════════════════════════════════
# AkShare 限频保护
# ═══════════════════════════════════════════════════════════════════
SNAPSHOT_MAX_RETRIES = int(os.getenv("SNAPSHOT_MAX_RETRIES", "3"))
"""快照请求最大重试次数"""

SNAPSHOT_RETRY_BASE_DELAY = float(os.getenv("SNAPSHOT_RETRY_BASE_DELAY", "5"))
"""首次重试等待秒数（之后指数退避：5s → 10s → 20s）"""

SNAPSHOT_CACHE_TTL = int(os.getenv("SNAPSHOT_CACHE_TTL", "15"))
"""快照内存缓存 TTL（秒）"""

MONITOR_MAX_CONSECUTIVE_FAILURES = int(os.getenv("MONITOR_MAX_CONSECUTIVE_FAILURES", "5"))
"""监控连续失败上限，超过后发送告警并暂停"""

# ═══════════════════════════════════════════════════════════════════
# pytdx 配置（分钟线数据源，无限频限制）
# ═══════════════════════════════════════════════════════════════════
PYTDX_BEST_IP = os.getenv("PYTDX_BEST_IP", "")   # 留空则自动选择
"""通达信行情服务器 IP（留空自动选优）"""

PYTDX_BEST_PORT = int(os.getenv("PYTDX_BEST_PORT", "0"))
"""通达信行情服务器端口（0 则自动选优）"""

MINUTE_DATA_SOURCE = os.getenv("MINUTE_DATA_SOURCE", "pytdx")
"""分钟线数据源：'pytdx'（推荐，无限频）或 'akshare'（仅近5日）"""

# ═══════════════════════════════════════════════════════════════════
# 股票池
# ═══════════════════════════════════════════════════════════════════
STOCK_POOL_FILE = os.getenv(
    "STOCK_POOL_FILE",
    os.path.join(_BASE_DIR, "stock_pool.csv"),
)
"""股票池 CSV，格式: code,name"""

# 也可以直接在这里硬编码一组股票（code → name）
STOCK_POOL_INLINE: dict[str, str] = {
    # "000001.SZ": "平安银行",
    # "600519.SH": "贵州茅台",
}

# ═══════════════════════════════════════════════════════════════════
# A 股交易时段（用于计算全日分钟数）
# ═══════════════════════════════════════════════════════════════════
MORNING_SESSION = ("09:30", "11:30")   # 上午 120 min
AFTERNOON_SESSION = ("13:00", "15:00") # 下午 120 min
TOTAL_TRADING_MINUTES = 240            # 全天共 240 根 1 分钟 K 线
