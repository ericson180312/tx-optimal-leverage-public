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
