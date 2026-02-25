from __future__ import annotations

from datetime import date, datetime, timedelta

try:
    from chinese_calendar import is_holiday  # type: ignore
except Exception:  # pragma: no cover
    is_holiday = None


def is_cn_trading_day(d: date) -> bool:
    """是否中国A股交易日（非节假日且非周末）。

    注意：调休上班的周末仍视为休市日。
    """
    if is_holiday is not None:
        return (not bool(is_holiday(d))) and (d.weekday() < 5)
    return d.weekday() < 5


def get_prev_trading_date(base: date | None = None) -> date:
    """获取上一个交易日（T-1）。"""
    if base is None:
        base = datetime.today().date()
    d = base - timedelta(days=1)
    while not is_cn_trading_day(d):
        d -= timedelta(days=1)
    return d


def get_prev_trading_date_str(base: date | None = None, fmt: str = "%Y-%m-%d") -> str:
    """获取上一个交易日字符串。"""
    return get_prev_trading_date(base).strftime(fmt)
