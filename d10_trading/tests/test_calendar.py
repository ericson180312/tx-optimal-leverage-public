"""TAIFEX 行事曆單元測試。"""
from __future__ import annotations

import datetime as dt

import pytest

from d10_trading.data import calendar
from d10_trading.data.calendar import CalendarCoverageError


def test_weekend_and_holiday():
    assert not calendar.is_trading_day(dt.date(2026, 7, 18))   # 週六
    assert not calendar.is_trading_day(dt.date(2026, 1, 1))    # 元旦
    assert calendar.is_trading_day(dt.date(2026, 7, 16))       # 平日


def test_prev_next_shift():
    fri = dt.date(2026, 7, 17)
    assert calendar.next_trading_day(fri) == dt.date(2026, 7, 20)   # 跳過週末
    assert calendar.prev_trading_day(dt.date(2026, 7, 20)) == fri
    assert calendar.shift_trading_days(fri, -3) == dt.date(2026, 7, 14)
    assert calendar.shift_trading_days(fri, 0) == fri


def test_coverage_error_for_unknown_year():
    with pytest.raises(CalendarCoverageError):
        calendar.is_trading_day(dt.date(2031, 1, 6))


def test_third_wednesday_matches_backtest_formula():
    """與回測 continuous.third_wednesday 同式（合約字串 YYYYMM）。"""
    from backtest_threshold import continuous as bt
    for ym in ["202601", "202606", "202612", "202502"]:
        ours = calendar.third_wednesday(int(ym[:4]), int(ym[4:]))
        ref = bt.third_wednesday(ym).date()
        assert ours == ref


def test_validate_against_actual():
    since = dt.date(2026, 6, 1)
    # 完整的 6 月實際交易日（6/19 端午休市）→ 無異常
    days, cur = [], since
    while cur <= dt.date(2026, 6, 30):
        if calendar.is_trading_day(cur):
            days.append(cur)
        cur += dt.timedelta(days=1)
    assert calendar.validate_against_actual(days, since) == []
    # 拿掉一個平日行情 → 偵測「漏列假日或臨時停市」
    missing = [d for d in days if d != dt.date(2026, 6, 10)]
    problems = calendar.validate_against_actual(missing, since)
    assert any("2026-06-10" in p for p in problems)
