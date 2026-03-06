# -*- coding: utf-8 -*-
"""
notifier.py — 企业微信消息推送

功能：
  1. 封装企业微信群机器人 Webhook 发送
  2. 格式化尾盘买入信号消息
  3. 支持 Markdown 富文本格式
"""

import json
from typing import Any

import requests

from tail_trade.config import WECHAT_WEBHOOK
from tail_trade.logger import get_logger

log = get_logger("notifier")


# ═══════════════════════════════════════════════════════════════════
# 发送消息
# ═══════════════════════════════════════════════════════════════════

def send_wechat_text(content: str, webhook: str | None = None) -> bool:
    """
    发送纯文本消息到企业微信群。

    Args:
        content:  消息正文
        webhook:  Webhook URL，为 None 则使用 config 默认值

    Returns:
        是否发送成功
    """
    url = webhook or WECHAT_WEBHOOK
    if "YOUR_KEY_HERE" in url:
        log.warning("[通知] Webhook 未配置，仅打印到控制台")
        log.info(f"[企业微信] {content}")
        return False

    payload = {
        "msgtype": "text",
        "text": {"content": content},
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if data.get("errcode") == 0:
            log.info(f"[通知] 发送成功: {content[:60]}...")
            return True
        else:
            log.error(f"[通知] 发送失败: {data}")
            return False
    except Exception as e:
        log.error(f"[通知] 请求异常: {e}")
        return False


def send_wechat_markdown(content: str, webhook: str | None = None) -> bool:
    """
    发送 Markdown 格式消息到企业微信群。

    Args:
        content:  Markdown 正文
        webhook:  Webhook URL

    Returns:
        是否发送成功
    """
    url = webhook or WECHAT_WEBHOOK
    if "YOUR_KEY_HERE" in url:
        log.warning("[通知] Webhook 未配置，仅打印到控制台")
        log.info(f"[企业微信-MD]\n{content}")
        return False

    payload = {
        "msgtype": "markdown",
        "markdown": {"content": content},
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if data.get("errcode") == 0:
            log.info("[通知] Markdown 发送成功")
            return True
        else:
            log.error(f"[通知] Markdown 发送失败: {data}")
            return False
    except Exception as e:
        log.error(f"[通知] 请求异常: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# 格式化信号消息
# ═══════════════════════════════════════════════════════════════════

def format_signal_message(
    code: str,
    name: str,
    price: float,
    daily_gain: float,
    tail_rise: float,
    tail_volume_ratio: float,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    将信号数据格式化为企业微信 Markdown 消息。

    Args:
        code:               股票代码（如 000001.SZ）
        name:               股票名称
        price:              当前价格
        daily_gain:         当日涨幅（小数，如 0.032 = 3.2%）
        tail_rise:          尾盘涨幅（小数）
        tail_volume_ratio:  尾盘量比
        extra:              额外信息字典

    Returns:
        格式化后的 Markdown 字符串
    """
    lines = [
        f"## 🔔 尾盘买入信号",
        f"> **{code}** {name}",
        f"",
        f"- 当前价格：**{price:.2f}**",
        f"- 当日涨幅：**{daily_gain * 100:+.2f}%**",
        f"- 尾盘涨幅：**{tail_rise * 100:+.2f}%**",
        f"- 尾盘量比：**{tail_volume_ratio:.2f}**",
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"- {k}：{v}")
    return "\n".join(lines)


def notify_signal(
    code: str,
    name: str,
    price: float,
    daily_gain: float,
    tail_rise: float,
    tail_volume_ratio: float,
    extra: dict[str, Any] | None = None,
    webhook: str | None = None,
) -> bool:
    """
    发送尾盘买入信号通知（一站式调用）。

    Returns:
        是否发送成功
    """
    msg = format_signal_message(
        code, name, price, daily_gain, tail_rise, tail_volume_ratio, extra
    )
    return send_wechat_markdown(msg, webhook)
