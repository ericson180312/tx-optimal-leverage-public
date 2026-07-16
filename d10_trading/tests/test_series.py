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
    ss = prepare_series(TODAY, yesterday_close_fn=lambda m, d: 46000.0,
                        today_price_fn=lambda m: 46100.0)
    assert ss.last_official_date == PREV
    assert ss.has_provisional and ss.dates[-1] == TODAY
    assert ss.spot[-1] == pytest.approx(ss.spot[-2] * 46100.0 / 46000.0)
    assert not [w for w in ss.warnings if "雙源" in w]


def test_stale_two_days_raises(monkeypatch):
    cdf = _cdf(end=calendar.prev_trading_day(PREV) - dt.timedelta(days=1))
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    with pytest.raises(DataStaleError):
        prepare_series(TODAY, yesterday_close_fn=lambda m, d: 46000.0)


def test_missing_yesterday_falls_back_to_shioaji(monkeypatch):
    cdf = _cdf(end=calendar.prev_trading_day(PREV))     # FinMind 缺 7/15
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY, yesterday_close_fn=lambda m, d: 46200.0)
    assert ss.last_official_date == PREV                # 已以 Shioaji 收盤遞補
    assert any("暫代" in w for w in ss.warnings)


def test_missing_yesterday_and_no_shioaji_stops(monkeypatch):
    cdf = _cdf(end=calendar.prev_trading_day(PREV))
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    with pytest.raises(DataStaleError):
        prepare_series(TODAY, yesterday_close_fn=lambda m, d: None)


def test_dual_source_divergence_warns(monkeypatch):
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)
    ss = prepare_series(TODAY,
                        yesterday_close_fn=lambda m, d: 46000.0 * 1.002)
    assert any("雙源核對" in w for w in ss.warnings)


def test_preview_failure_does_not_stop(monkeypatch):
    cdf = _cdf()
    monkeypatch.setattr(series, "load_tx_contracts", lambda **kw: cdf)

    def boom(month):
        raise RuntimeError("expired contract")

    ss = prepare_series(TODAY, yesterday_close_fn=lambda m, d: 46000.0,
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


def test_today_index_none_when_absent():
    ss = build_signal_series(_cdf())
    assert ss.today_index(TODAY) is None
