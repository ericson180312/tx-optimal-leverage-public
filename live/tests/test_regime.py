"""regime 狀態機單元測試（規格 §8-1：緩衝帶邊界、檢查頻率、暖機預設）。"""
from __future__ import annotations

import datetime as dt

import numpy as np

from live import config
from live.strategy import regime


def lev(spot, **kw):
    return regime.trend_leverage(np.asarray(spot, float), **kw)


def test_warmup_defaults_to_bull():
    out = lev([100.0] * 50)                    # 不足 200 日：恆為 L_HIGH
    assert (out == config.L_HIGH).all()


def test_hysteresis_band_keeps_state():
    """遲滯：跌破下緣→BEAR；回到帶內→維持 BEAR；突破上緣→BULL。

    注意 MA 含當日：ma_win=4 下逐日 MA 為
      t=8: (100+100+100+95)/4=98.75  下緣 97.76 → 95 跌破 → BEAR
      t=9: (100+100+95+98)/4=98.25   帶 [97.27, 99.23] → 98 帶內 → 維持 BEAR
      t=10:(100+95+98+104)/4=99.25   上緣 100.24 → 104 突破 → BULL
    """
    path = [100.0] * 8 + [95.0, 98.0, 104.0]
    out = lev(path, ma_win=4, check_every=1)
    assert out[7] == config.L_HIGH
    assert out[8] == config.L_LOW
    assert out[9] == config.L_LOW
    assert out[10] == config.L_HIGH


def test_band_inside_vs_outside():
    """帶內不切換、帶外切換（MA 含當日的口徑）。"""
    ma_win, chk = 5, 1
    flat = [100.0] * 10
    # y=99：MA=(400+99)/5=99.8，下緣 98.802 → 99 仍在帶內 → 維持 BULL
    inside = lev(flat + [99.0], ma_win=ma_win, check_every=chk)
    assert inside[-1] == config.L_HIGH
    # y=97：MA=99.4，下緣 98.406 → 97 跌破 → BEAR
    outside = lev(flat + [97.0], ma_win=ma_win, check_every=chk)
    assert outside[-1] == config.L_LOW


def test_check_every_5_gates_switches():
    """跌破發生在非檢查日 → 延至下一個 t%5==0 檢查日才切換。"""
    n = 210
    spot = np.full(n, 100.0)
    spot[203:] = 90.0                           # 自 t=203 起跌破（203 非 5 倍數）
    out = lev(spot)
    assert out[203] == config.L_HIGH
    assert out[204] == config.L_HIGH
    assert out[205] == config.L_LOW             # 205 = 下一個檢查日
    # 檢查日集合 = 索引 % 5 == 0（且 t ≥ 199）
    switches = np.nonzero(np.diff(out))[0] + 1
    assert all(t % config.SIGNAL_CHECK_EVERY == 0 for t in switches)


def test_regime_switches_dates():
    dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(6)]
    levs = np.array([2.0, 2.0, 1.0, 1.0, 2.0, 2.0])
    sw = regime.regime_switches(dates, levs)
    assert sw == [(dates[2], regime.BULL, regime.BEAR),
                  (dates[4], regime.BEAR, regime.BULL)]


def test_identical_to_round3_reference_formula():
    """與 round3.signals.trend_leverage 在隨機路徑上逐位元一致（黃金測試前哨）。"""
    import round3.signals as r3sg
    rng = np.random.default_rng(7)
    spot = 8000.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.015, size=1500))
    ours = regime.trend_leverage(spot)
    ref = r3sg.trend_leverage(spot, config.L_LOW)
    assert np.array_equal(ours, ref)
