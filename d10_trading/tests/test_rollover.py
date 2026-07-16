"""轉倉日判定單元測試（規格 §8-1：月差正負兩情境、假日順延、安全補課）。"""
from __future__ import annotations

import datetime as dt

from d10_trading.data import calendar
from d10_trading.strategy.rollover import decide_roll

# 2026-08 合約：8/1 週六 → 第三個週三 = 2026-08-19（非假日）
SETTLE = dt.date(2026, 8, 19)
EARLY = calendar.shift_trading_days(SETTLE, -5)     # 2026-08-12


def test_early_roll_when_spread_positive():
    d = decide_roll(EARLY, SETTLE, spread=25.0, holding_near=True)
    assert d.should_roll and d.reason == "early_roll"
    assert d.early_roll_date == EARLY


def test_no_early_roll_when_spread_negative():
    d = decide_roll(EARLY, SETTLE, spread=-10.0, holding_near=True)
    assert not d.should_roll


def test_expiry_roll_when_spread_negative():
    d = decide_roll(SETTLE, SETTLE, spread=-10.0, holding_near=True)
    assert d.should_roll and d.reason == "expiry_roll"


def test_late_roll_guard_when_spread_flips_positive():
    """月差在 T−5 之後才轉正 → 當日補轉，不帶部位進結算。"""
    day = calendar.next_trading_day(EARLY)
    d = decide_roll(day, SETTLE, spread=5.0, holding_near=True)
    assert d.should_roll and d.reason == "late_roll"
    # 結算日當天月差為正也必須轉（late_roll 涵蓋）
    d2 = decide_roll(SETTLE, SETTLE, spread=5.0, holding_near=True)
    assert d2.should_roll


def test_not_holding_near_never_rolls():
    assert not decide_roll(EARLY, SETTLE, 25.0, holding_near=False).should_roll
    assert not decide_roll(SETTLE, SETTLE, -5.0, holding_near=False).should_roll


def test_ordinary_day_no_roll():
    day = calendar.shift_trading_days(SETTLE, -10)
    assert not decide_roll(day, SETTLE, 25.0, holding_near=True).should_roll


def test_settlement_holiday_deferral():
    """2026-02 第三個週三 2/18 落在春節休市 → 順延至 2/23（週一）。"""
    assert calendar.third_wednesday(2026, 2) == dt.date(2026, 2, 18)
    assert calendar.settlement_date(2026, 2) == dt.date(2026, 2, 23)


def test_early_roll_counts_trading_days_not_calendar_days():
    """T−5 用交易日回推：2026-02 結算 2/23 → 往前 5 個交易日跨過春節連假。"""
    settle = calendar.settlement_date(2026, 2)
    early = calendar.shift_trading_days(settle, -5)
    # 春節休市 2/12–2/20（實際行情互驗）→ 回推：2/11、2/10、2/9、2/6、2/5
    assert early == dt.date(2026, 2, 5)
