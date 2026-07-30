"""到期哨兵測試：分級門檻、假日檔自動推導、推播節奏、批次整合。"""
from __future__ import annotations

import datetime as dt

import pytest

from d10_trading import config
from d10_trading.ops import watchdog
from d10_trading.ops.watchdog import ExpiryItem

MON = dt.date(2026, 8, 3)      # 週一
TUE = dt.date(2026, 8, 4)      # 週二


@pytest.fixture
def only_one_date(monkeypatch):
    """單一到期項（CA 2028-07-15），並停用假日檔項目以隔離測試。"""
    monkeypatch.setattr(config, "EXPIRY_DATES", {"測試憑證": dt.date(2028, 7, 15)})
    monkeypatch.setattr(watchdog, "holiday_deadline", lambda: None)


def test_levels_by_days_left(only_one_date):
    d = dt.date(2028, 7, 15)
    cases = [
        (d - dt.timedelta(days=61), 0, None),        # 尚未進入預警窗 → 不列入
        (d - dt.timedelta(days=60), 1, "WARNING"),   # 恰好 60 天
        (d - dt.timedelta(days=15), 1, "WARNING"),
        (d - dt.timedelta(days=14), 1, "ALERT"),     # 恰好 14 天
        (d, 1, "ALERT"),                             # 當天
        (d + dt.timedelta(days=1), 1, "CRITICAL"),   # 已過期
    ]
    for today, n, level in cases:
        items = watchdog.check_expiries(today)
        assert len(items) == n, f"{today} 應有 {n} 項"
        if level:
            assert items[0].level == level, f"{today} 應為 {level}"


def test_message_wording(only_one_date):
    d = dt.date(2028, 7, 15)
    future = watchdog.check_expiries(d - dt.timedelta(days=30))[0]
    assert "將於" in future.message() and "剩 30 天" in future.message()
    past = watchdog.check_expiries(d + dt.timedelta(days=3))[0]
    assert "已於" in past.message() and "逾期 3 天" in past.message()


def test_holiday_deadline_derived_from_covered_years():
    """假日檔期限自 covered_years 自動推導（目前檔案涵蓋到 2026）。"""
    assert watchdog.holiday_deadline() == dt.date(2026, 12, 31)


def test_holiday_item_warns_before_year_end(monkeypatch):
    monkeypatch.setattr(config, "EXPIRY_DATES", {})
    items = watchdog.check_expiries(dt.date(2026, 11, 15))    # 距 12/31 剩 46 天
    assert len(items) == 1
    it = items[0]
    assert it.name == watchdog.HOLIDAY_ITEM and it.level == "WARNING"
    assert "2027" in it.note and "fail-closed" in it.note


def test_items_sorted_most_urgent_first(monkeypatch):
    monkeypatch.setattr(watchdog, "holiday_deadline", lambda: None)
    monkeypatch.setattr(config, "EXPIRY_DATES", {
        "遠的": dt.date(2026, 9, 20),      # 剩 51 天
        "近的": dt.date(2026, 8, 10),      # 剩 10 天
    })
    items = watchdog.check_expiries(dt.date(2026, 7, 31))
    assert [i.name for i in items] == ["近的", "遠的"]
    assert items[0].level == "ALERT" and items[1].level == "WARNING"


def test_push_cadence():
    """WARNING 僅週一推播；ALERT/CRITICAL 每日推播。"""
    warn = ExpiryItem("x", dt.date(2026, 9, 1), 30, "WARNING")
    assert watchdog.should_push(warn, MON) is True
    assert watchdog.should_push(warn, TUE) is False
    for lv in ("ALERT", "CRITICAL"):
        it = ExpiryItem("x", dt.date(2026, 8, 5), 2, lv)
        assert watchdog.should_push(it, MON) and watchdog.should_push(it, TUE)


def test_real_config_dates_are_silent_today():
    """實際登錄的三個日期距今皆 > 60 天 → 今日不應有任何提醒（無雜訊）。"""
    monkeypatch_free = watchdog.check_expiries(dt.date(2026, 7, 31))
    names = [i.name for i in monkeypatch_free]
    assert "永豐 CA 憑證" not in names and "Vultr 信用卡" not in names


def test_watchdog_failure_never_breaks_batch(monkeypatch):
    """哨兵故障不得影響交易（規格優先序：正確性>安全性>可觀測性）。"""
    from d10_trading import main as main_mod

    def boom(today):
        raise RuntimeError("watchdog broken")

    monkeypatch.setattr(main_mod.watchdog, "check_expiries", boom)
    warnings: list[str] = []
    main_mod._check_expiries(dt.date(2026, 7, 31), warnings)   # 不拋出即通過
    assert warnings == []
