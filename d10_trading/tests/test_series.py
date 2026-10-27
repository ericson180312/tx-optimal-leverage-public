"""序列層單元測試：spot 建構、新鮮度、Shioaji 遞補、雙源核對、暫定列。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from d10_trading import config
from d10_trading.data import calendar, series
from d10_trading.data.series import (DataStaleError, append_provisional,
                              build_signal_series, prepare_series)
from d10_trading.tests.conftest import synth_cdf

TODAY = dt.date(2026, 7, 16)
PREV = calendar.prev_trading_day(TODAY)          # 2026-07-15
START = dt.date(2025, 1, 2)


def _cdf(price_fn=lambda d: 46000.0, end=PREV, basis=0.0):
    return synth_cdf(price_fn, START, end, basis=basis)


def test_spot_equals_price_path_when_no_gap(monkeypatch):
    """basis=0（各月同價）→ 還原連續價 = 價格路徑本身。"""
    path = {}

    def price_fn(d):
        path[d] = 40000.0 + (d.toordinal() % 37) * 10
        return path[d]

    ss = build_signal_series(_cdf(price_fn))
    dates = ss.dates
    expect = np.array([path[d] for d in dates])
    assert np.allclose(ss.spot, expect, rtol=1e-12)


def test_freshness_ok_and_provisional(monkeypatch):
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0,
                        today_price_fn=lambda m: 46100.0)
    assert ss.last_official_date == PREV
    assert ss.has_provisional and ss.dates[-1] == TODAY
    assert ss.spot[-1] == pytest.approx(ss.spot[-2] * 46100.0 / 46000.0)
    assert not [w for w in ss.warnings if "雙源" in w]


def test_stale_two_days_raises(monkeypatch):
    cdf = _cdf(end=calendar.prev_trading_day(PREV) - dt.timedelta(days=1))
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    with pytest.raises(DataStaleError):
        prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0)


def test_missing_yesterday_falls_back_to_shioaji(monkeypatch):
    cdf = _cdf(end=calendar.prev_trading_day(PREV))     # FinMind 缺 7/15
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46200.0)
    assert ss.last_official_date == PREV                # 已以 Shioaji 參考價遞補
    assert any("暫代" in w for w in ss.warnings)


def test_missing_yesterday_and_no_shioaji_stops(monkeypatch):
    cdf = _cdf(end=calendar.prev_trading_day(PREV))
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    with pytest.raises(DataStaleError):
        prepare_series(TODAY, prev_settlement_fn=lambda m: None)


def test_dual_source_divergence_warns(monkeypatch):
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY,
                        prev_settlement_fn=lambda m: 46000.0 * 1.002)
    assert any("雙源核對" in w for w in ss.warnings)


def test_dual_source_reference_no_false_alarm_2026_07_17(monkeypatch):
    """🔴 回歸：改用 reference（=結算價）後，7/17「收盤競價 vs 結算」不再誤報。

    實際事件：FinMind 結算 42604、Shioaji 13:45 K 棒收盤（收盤集合競價）42725，
    差 0.284% 曾誤觸雙源告警。改用 contract.reference（=前日結算 42604）比對後，
    差 0 → 無告警。此測試以 42604 為 FinMind 官方價、reference 回 42604。
    """
    cdf = _cdf(price_fn=lambda d: 42604.0)
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 42604.0)
    assert ss.last_official_price == 42604.0
    assert not [w for w in ss.warnings if "雙源核對" in w]
    # 對照：若仍拿收盤競價 42725 比對，會超過 0.1% 容忍
    assert abs(42725.0 / 42604.0 - 1.0) > config.DUAL_SOURCE_TOL


def test_dual_source_skipped_when_ref_not_trustworthy(monkeypatch):
    """盤後（reference 已滾動成當日）→ ref_trustworthy=False → 不做核對、不假警報。"""
    cdf = _cdf(price_fn=lambda d: 42604.0)
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    # 即使 reference 回一個差很多的值（模擬盤後滾動後的當日結算），也不告警
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 42725.0,
                        ref_trustworthy=False)
    assert not [w for w in ss.warnings if "雙源核對" in w]


def test_fallback_uses_reference_value(monkeypatch):
    """FinMind 缺前日 → 以 reference（結算價）補列，且訊息標明『參考價』。"""
    cdf = _cdf(price_fn=lambda d: 42604.0, end=calendar.prev_trading_day(PREV))
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 42611.0)
    assert ss.last_official_date == PREV
    assert ss.last_official_price == 42611.0          # 補進去的是 reference 值
    assert any("參考價" in w and "暫代" in w for w in ss.warnings)


def test_preview_failure_does_not_stop(monkeypatch):
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)

    def boom(month):
        raise RuntimeError("expired contract")

    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0,
                        today_price_fn=boom)
    assert not ss.has_provisional
    assert any("即時價查詢失敗" in w for w in ss.warnings)


def test_provisional_only_once():
    ss = build_signal_series(_cdf())
    append_provisional(ss, TODAY, 46100.0)
    with pytest.raises(AssertionError):
        append_provisional(ss, TODAY, 46100.0)


def test_exec_index_is_strictly_before_today():
    """T+1：正常 13:20 跑（序列到昨日）→ 執行索引 = 最後官方列。"""
    ss = build_signal_series(_cdf())
    idx = ss.exec_index(TODAY)
    assert ss.dates[idx] == PREV and idx == ss.n_official - 1
    append_provisional(ss, TODAY, 46100.0)
    assert ss.exec_index(TODAY) == idx            # 暫定列不影響執行索引
    assert ss.today_index(TODAY) == len(ss.dates) - 1


def test_exec_index_skips_today_official_row():
    """收盤後補跑：FinMind 已含今日官方列 → 執行訊號仍取上一交易日。"""
    ss = build_signal_series(_cdf(end=TODAY))     # 官方序列含今日
    assert ss.last_official_date == TODAY
    idx = ss.exec_index(TODAY)
    assert ss.dates[idx] == PREV                  # 嚴格早於今日
    assert ss.today_index(TODAY) == ss.n_official - 1   # 今日列作預覽


def test_settlement_of_returns_tx_settlement():
    """exec_vs_close 稽核用：TX 月合約在官方日的結算價；查無回 None。"""
    ss = build_signal_series(_cdf(price_fn=lambda d: 42600.0))
    month, d = ss.held_contract, ss.last_official_date
    assert ss.settlement_of(month, d) == pytest.approx(42600.0)
    assert ss.settlement_of(month, dt.date(2000, 1, 1)) is None   # 日期不在序列
    assert ss.settlement_of("209912", d) is None                  # 無此合約


def test_today_index_none_when_absent():
    ss = build_signal_series(_cdf())
    assert ss.today_index(TODAY) is None


# ── 序列持倉月已結算下市（每月結算後的例行過渡）─────────────────────────────

def test_expired_held_contract_single_clear_warning(monkeypatch):
    """🔴 2026-08-20 實況：TX 202608 於 8/19 結算，FinMind 資料也只到 8/19，
    連續序列的事後到期判定（last_seen < data_end）尚未把它判為到期 →
    仍以為持有 202608 → 查價必然失敗。

    應以「單一」明確訊息說明，而非兩則模糊警示；且預覽與雙源核對皆略過。
    """
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    calls = {"price": 0, "settle": 0}

    def price_fn(m):
        calls["price"] += 1
        raise RuntimeError("contract not found")

    def settle_fn(m):
        calls["settle"] += 1
        return 46000.0 * 1.05          # 若被拿去核對會產生假性不一致

    ss = prepare_series(TODAY, prev_settlement_fn=settle_fn,
                        today_price_fn=price_fn,
                        contract_exists_fn=lambda m: False)

    expired = [w for w in ss.warnings if "已結算下市" in w]
    assert len(expired) == 1 and ss.held_contract in expired[0]
    assert "自動恢復" in expired[0]
    # 舊的兩則模糊訊息不應再出現
    assert not [w for w in ss.warnings if "即時價" in w]
    # 雙源核對必須略過（否則會用錯月份比對出假警報）
    # 註：比對「雙源核對：」帶冒號的實際告警格式——新訊息本身也含「雙源核對」四字
    assert not [w for w in ss.warnings if "雙源核對：" in w]
    assert calls["price"] == 0, "不應再對已下市合約查價"
    assert not ss.has_provisional


def test_tradable_held_contract_keeps_normal_behaviour(monkeypatch):
    """合約仍可交易時行為不變（預覽正常追加、雙源核對照做）。"""
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0,
                        today_price_fn=lambda m: 46100.0,
                        contract_exists_fn=lambda m: True)
    assert ss.has_provisional
    assert not [w for w in ss.warnings if "已結算下市" in w]


def test_contract_exists_check_failure_falls_back_to_normal(monkeypatch):
    """判定函式自身出錯 → 保守視為可交易，走原流程（不得因此中斷）。"""
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)

    def boom(m):
        raise RuntimeError("broker down")

    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0,
                        today_price_fn=lambda m: 46100.0,
                        contract_exists_fn=boom)
    assert ss.has_provisional
    assert not [w for w in ss.warnings if "已結算下市" in w]


# ── 臨時停市查證（2026-10-23 颱風後新增）─────────────────────────────────────
# 停市後第一個交易日，「上一交易日」若是未補登的停市日，FinMind 本來就不會有那天；
# 不查證的話，單日遞補會以停市前的結算價補出一列假的平盤、偏移 5 日檢查相位。

def _lagging_cdf(monkeypatch, days_missing=1):
    end = PREV
    for _ in range(days_missing):
        end = calendar.prev_trading_day(end)
    cdf = _cdf(end=end)
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    return end


def test_verified_closure_recorded_instead_of_fake_row(monkeypatch, temp_holiday_file):
    last = _lagging_cdf(monkeypatch)                    # FinMind 無 7/15（視為颱風停市）
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0,
                        day_traded_fn=lambda d: d != PREV)
    assert ss.closures_recorded == [PREV]
    assert not calendar.is_trading_day(PREV)            # 已補登假日檔
    assert PREV not in ss.dates                         # 沒有假的平盤列
    assert ss.last_official_date == last == calendar.prev_trading_day(TODAY)
    assert not any("暫代" in w for w in ss.warnings)
    assert any("補登臨時停市" in w for w in ss.warnings)


def test_consecutive_closures_walk_back(monkeypatch, temp_holiday_file):
    """連續兩日停市：逐日往前查證並補登，不因「落後兩日」而誤判資料過期。"""
    last = _lagging_cdf(monkeypatch, days_missing=2)
    gap = {PREV, calendar.prev_trading_day(PREV)}
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0,
                        day_traded_fn=lambda d: d not in gap)
    assert set(ss.closures_recorded) == gap
    assert ss.last_official_date == last


def test_traded_day_keeps_normal_fallback(monkeypatch, temp_holiday_file):
    """查證為有成交（FinMind 單純晚到）→ 照原流程以參考價遞補，不動假日檔。"""
    _lagging_cdf(monkeypatch)
    before = temp_holiday_file.read_text(encoding="utf-8")
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46200.0,
                        day_traded_fn=lambda d: True)
    assert ss.closures_recorded == []
    assert ss.last_official_date == PREV and any("暫代" in w for w in ss.warnings)
    assert temp_holiday_file.read_text(encoding="utf-8") == before


def test_unknown_verification_keeps_normal_fallback(monkeypatch, temp_holiday_file):
    """查證失敗（None）不得當成停市——寧可照舊遞補，也不把交易日誤登為假日。"""
    _lagging_cdf(monkeypatch)
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46200.0,
                        day_traded_fn=lambda d: None)
    assert ss.closures_recorded == []
    assert calendar.is_trading_day(PREV)
    assert ss.last_official_date == PREV


def test_closure_not_recorded_when_sync_disabled(monkeypatch, temp_holiday_file):
    monkeypatch.setattr(config, "HOLIDAY_SYNC_ENABLED", False)
    _lagging_cdf(monkeypatch)
    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46200.0,
                        day_traded_fn=lambda d: False)
    assert ss.closures_recorded == []
    assert ss.last_official_date == PREV                # 退回原流程


def test_no_verification_when_series_is_current(monkeypatch):
    """序列已到上一交易日 → 不需查證（也不該多打 API）。"""
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: _cdf())

    def must_not_call(d):
        raise AssertionError("不應查證")

    ss = prepare_series(TODAY, prev_settlement_fn=lambda m: 46000.0,
                        day_traded_fn=must_not_call)
    assert ss.closures_recorded == []
