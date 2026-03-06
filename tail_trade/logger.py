# -*- coding: utf-8 -*-
"""
logger.py — 统一日志模块

用法：
    from tail_trade.logger import get_logger
    log = get_logger("backtest")
    log.info("策略已启动")
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(
    name: str = "tail_trade",
    *,
    level: int = logging.INFO,
    log_dir: str | None = None,
    to_file: bool = True,
    to_console: bool = True,
) -> logging.Logger:
    """
    创建/获取一个带文件和控制台输出的 Logger。

    Args:
        name:       logger 名称
        level:      日志级别
        log_dir:    日志文件目录，默认 tail_trade/logs/
        to_file:    是否写入文件
        to_console: 是否输出到控制台

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加 handler

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 控制台 ──
    if to_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # ── 文件 ──
    if to_file:
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{today}.log"),
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
