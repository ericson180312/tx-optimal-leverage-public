"""假日檔自動更新測試：解析規則、健全性、additive-only、臨時停市補登。

🔴 核心不變式：**只增不刪**。TWSE 年度表不含臨時停市（2026-07-10 實測），
覆蓋寫入會刪掉該筆 → prev_trading_day 誤判 → 序列插入假的一天 → 訊號相位偏移。
"""
from __future__ import annotations

import datetime as dt

import pytest

from d10_trading import config
from d10_trading.data import calendar, holiday_sync
from d10_trading.data.holiday_sync import SanityError


@pytest.fixture
def temp_holiday_file(tmp_path, monkeypatch):
    """獨立的假日檔副本，避免測試污染 repo 內的正式檔。"""
    src = config.HOLIDAY_FILE.read_text(encoding="utf-8")
    dst = tmp_path / "taifex_holidays.csv"
    dst.write_text(src, encoding="utf-8")
    monkeypatch.setattr(config, "HOLIDAY_FILE", dst)
    calendar.reload()
    yield dst
    calendar.reload()


# ── TWSE 解析規則 ───────────────────────────────────────────────────────────

def _fake_payload(rows):
    return {"stat": "ok", "data": rows}


def test_parse_excludes_trading_days_and_weekends(monkeypatch):
    """名稱含「交易日」者是有交易的特殊日；週末本就非交易日——皆須排除。"""
    rows = [
        ["2027-01-01", "中華民國開國紀念日", "放假1日"],
        ["2027-01-04", "國曆新年開始交易日", "開始交易"],      # 有交易 → 排除
        ["2027-02-06", "農曆春節", "適逢星期六"],              # 週六 → 排除
        ["2027-02-08", "農曆春節", "放假"],
        ["2026-12-31", "他年度資料", ""],                      # 非目標年 → 排除
    ]

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return _fake_payload(rows)

    monkeypatch.setattr(holiday_sync.requests, "get", lambda *a, **k: R())
    got = holiday_sync.fetch_twse_year(2027)
    assert set(got) == {dt.date(2027, 1, 1), dt.date(2027, 2, 8)}


def test_fetch_returns_empty_when_year_not_published(monkeypatch):
    """API 查無該年度時會回退其他年度——必須濾成空，觸發 unavailable。"""
    class R:
        def raise_for_status(self): pass
        def json(self): return _fake_payload([["2026-01-01", "元旦", ""]])

    monkeypatch.setattr(holiday_sync.requests, "get", lambda *a, **k: R())
    assert holiday_sync.fetch_twse_year(2027) == {}


# ── 健全性檢查 ──────────────────────────────────────────────────────────────

def _weekdays_from(year: int, start_month: int, n: int, step: int = 1):
    """自指定月份起挑 n 個平日（跳過週末），供測試資料使用。"""
    out, d = [], dt.date(year, start_month, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=step)
    return out


def _plausible_year(year=2027):
    """含春節連假（2/8–2/12 皆平日）的合理年度資料（18 天，全為平日）。"""
    days = {d: "農曆春節" for d in _weekdays_from(year, 2, 5)[:5]}
    # 其餘 13 天散落各月，一律取平日
    for m in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        for d in _weekdays_from(year, m, 2, step=13):
            days.setdefault(d, "假日")
        if len(days) >= 18:
            break
    return dict(sorted(days.items())[:18])


def test_sanity_accepts_plausible_data():
    holiday_sync.sanity_check(2027, _plausible_year())


def test_sanity_rejects_out_of_range_count():
    with pytest.raises(SanityError, match="超出合理區間"):
        holiday_sync.sanity_check(2027, {dt.date(2027, 1, 1): "元旦"})
    too_many = {dt.date(2027, 1, 1) + dt.timedelta(days=i * 3): "x"
                for i in range(30)}
    with pytest.raises(SanityError, match="超出合理區間"):
        holiday_sync.sanity_check(2027, too_many)


def test_sanity_rejects_wrong_year_or_weekend():
    bad = _plausible_year()
    bad[dt.date(2028, 1, 3)] = "他年"
    with pytest.raises(SanityError, match="非目標年度或週末"):
        holiday_sync.sanity_check(2027, bad)


def test_sanity_rejects_missing_lunar_new_year_block():
    """無連續 3 日休市（春節）→ 幾乎可斷定抓錯資料。"""
    scattered = {d: "假日" for m in range(1, 13)
                 for d in _weekdays_from(2027, m, 1)}     # 每月 1 天、全平日
    with pytest.raises(SanityError, match="連續 3 日"):
        holiday_sync.sanity_check(2027, scattered)


# ── 🔴 additive-only（核心不變式）──────────────────────────────────────────

def test_merge_never_deletes_adhoc_closure(temp_holiday_file):
    """TWSE 2026 表不含 07-10 颱風停市；合併後該筆必須仍在。"""
    assert dt.date(2026, 7, 10) in calendar._load_holiday_file()[0]
    twse_2026 = {dt.date(2026, 1, 1): "元旦", dt.date(2026, 5, 1): "勞動節"}
    holiday_sync.merge(twse_2026, cover_year=2026)
    holidays, _ = calendar._load_holiday_file()
    assert dt.date(2026, 7, 10) in holidays, "🔴 臨時停市被刪除——會導致訊號相位偏移"


def test_merge_keeps_existing_name(temp_holiday_file):
    """既有條目保留原名稱——人工註記優先於官方名稱。"""
    before = calendar._load_holiday_file()[0]
    assert dt.date(2026, 7, 10) in before
    holiday_sync.merge({dt.date(2026, 7, 10): "官方名稱"})
    text = temp_holiday_file.read_text(encoding="utf-8")
    assert "官方名稱" not in text and "臨時停市" in text


def test_merge_adds_and_extends_coverage(temp_holiday_file):
    data = _plausible_year(2027)
    added = holiday_sync.merge(data, cover_year=2027)
    assert len(added) == 18
    assert 2027 in calendar.covered_years()
    assert not calendar.is_trading_day(sorted(data)[0])       # 已加入者為休市
    assert calendar.is_trading_day(dt.date(2027, 1, 4))       # 未列入的平日照常


def test_merge_is_idempotent(temp_holiday_file):
    holiday_sync.merge(_plausible_year(2027), cover_year=2027)
    again = holiday_sync.merge(_plausible_year(2027), cover_year=2027)
    assert again == []                                        # 第二次無新增


# ── auto_sync_if_needed 流程 ───────────────────────────────────────────────

def test_sync_skipped_outside_warning_window(temp_holiday_file, monkeypatch):
    called = []
    monkeypatch.setattr(holiday_sync, "fetch_twse_year",
                        lambda y: called.append(y) or {})
    res = holiday_sync.auto_sync_if_needed(dt.date(2026, 7, 30))   # 距年底 154 天
    assert res.status == "not_needed" and called == []             # 完全不觸網


def test_sync_reports_unavailable_before_publication(temp_holiday_file, monkeypatch):
    monkeypatch.setattr(holiday_sync, "fetch_twse_year", lambda y: {})
    res = holiday_sync.auto_sync_if_needed(dt.date(2026, 11, 15))
    assert res.status == "unavailable" and res.year == 2027


def test_sync_adds_when_published(temp_holiday_file, monkeypatch):
    monkeypatch.setattr(holiday_sync, "fetch_twse_year",
                        lambda y: _plausible_year(y))
    res = holiday_sync.auto_sync_if_needed(dt.date(2026, 11, 15))
    assert res.status == "added" and res.year == 2027 and len(res.added) == 18
    assert 2027 in calendar.covered_years()
    assert "請人工複核" in res.message()


def test_sync_refuses_write_on_sanity_failure(temp_holiday_file, monkeypatch):
    """健全性不過 → 不寫入、維持 fail-closed。"""
    monkeypatch.setattr(holiday_sync, "fetch_twse_year",
                        lambda y: {dt.date(y, 1, 1): "只有一天"})
    res = holiday_sync.auto_sync_if_needed(dt.date(2026, 11, 15))
    assert res.status == "failed" and "超出合理區間" in res.detail
    assert 2027 not in calendar.covered_years()


def test_sync_survives_network_error(temp_holiday_file, monkeypatch):
    def boom(y):
        raise ConnectionError("network down")

    monkeypatch.setattr(holiday_sync, "fetch_twse_year", boom)
    res = holiday_sync.auto_sync_if_needed(dt.date(2026, 11, 15))
    assert res.status == "failed" and "ConnectionError" in res.detail
    assert 2027 not in calendar.covered_years()


# ── 臨時停市補登 ────────────────────────────────────────────────────────────

def _forget_date(path, iso: str) -> None:
    """自假日檔移除某日，重現「颱風剛過、尚未補登」的狀態。"""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(l for l in text.splitlines() if not l.startswith(iso)) + "\n",
        encoding="utf-8")
    calendar.reload()


def test_record_adhoc_closure(temp_holiday_file):
    """模擬 2026-07-10 尚未登錄時，由實際行情自動補上。"""
    _forget_date(temp_holiday_file, "2026-07-10")
    assert calendar.is_trading_day(dt.date(2026, 7, 10))      # 誤判為交易日

    # 實際行情：7/10 無交易，前後皆有
    actual = [dt.date(2026, 7, d) for d in (6, 7, 8, 9, 13, 14, 15)]
    added = holiday_sync.record_adhoc_closures(actual, since=dt.date(2026, 7, 1))
    assert added == [dt.date(2026, 7, 10)]
    assert not calendar.is_trading_day(dt.date(2026, 7, 10))   # 已修正


def test_adhoc_ignores_trailing_data_lag(temp_holiday_file):
    """資料延遲的最新日不得被誤判為停市（其後無行情佐證）。"""
    actual = [dt.date(2026, 7, d) for d in (6, 7, 8, 9)]       # 資料只到 7/9
    added = holiday_sync.record_adhoc_closures(actual, since=dt.date(2026, 7, 6))
    assert added == []                                          # 7/10 之後無資料 → 不判定


def test_adhoc_ignores_dates_before_data_starts(temp_holiday_file):
    """since 早於首個實際行情日時，資料起點之前的平日不得被誤判為停市。"""
    _forget_date(temp_holiday_file, "2026-07-10")
    actual = [dt.date(2026, 7, d) for d in (6, 7, 8, 9, 13)]
    added = holiday_sync.record_adhoc_closures(actual, since=dt.date(2026, 7, 1))
    assert added == [dt.date(2026, 7, 10)]                      # 只有真正的缺口
    assert dt.date(2026, 7, 1) not in added                     # 資料起點前不列入
