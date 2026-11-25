"""口數計算單元測試（規格 §8-1：含捨入、防呆上限、保險絲）。"""
from __future__ import annotations

import pytest

from d10_trading import config
from d10_trading.strategy import sizing


def test_spec_example_tmf():
    """規格 §4 範例：70 萬、指數 46,000、TMF 乘數 10。"""
    r2 = sizing.target_contracts(2.0, 700_000, 46_000, multiplier=10.0)
    assert r2.target_contracts == 3            # 3.04 → 3
    r1 = sizing.target_contracts(1.0, 700_000, 46_000, multiplier=10.0)
    assert r1.target_contracts == 2            # 1.52 → 2（half-up）
    assert r1.rounding_residual == pytest.approx(1.5217 - 2, abs=1e-3)


def test_round_half_up_not_bankers():
    # 1.5 → 2、2.5 → 3（銀行家捨入會把 2.5 捨成 2）
    assert sizing.target_contracts(1.5, 100_000, 10_000,
                                   multiplier=10.0).target_contracts == 2
    assert sizing.target_contracts(2.5, 100_000, 10_000,
                                   multiplier=10.0).target_contracts == 3


def test_rounding_residual_sign_recorded():
    r = sizing.target_contracts(2.0, 700_000, 46_000, multiplier=10.0)
    assert r.raw_contracts == pytest.approx(3.0435, abs=1e-3)
    assert r.rounding_residual == pytest.approx(r.raw_contracts - 3)
    assert r.achieved_leverage == pytest.approx(3 * 46_000 * 10 / 700_000)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        sizing.target_contracts(2.0, 0, 46_000)
    with pytest.raises(ValueError):
        sizing.target_contracts(2.0, 700_000, 0)


def test_caps(monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTRACTS", 4)
    sizing.check_caps(4, 4)                    # 恰在上限 → 通過
    with pytest.raises(sizing.ContractsCapExceeded):
        sizing.check_caps(5, 1)
    with pytest.raises(sizing.ContractsCapExceeded):
        sizing.check_caps(4, -5)


def test_fuse_boundary():
    assert sizing.fuse_triggered(config.FUSE_L)
    assert sizing.fuse_triggered(config.FUSE_L + 0.1)
    assert not sizing.fuse_triggered(config.FUSE_L - 0.01)


def test_current_leverage():
    assert sizing.current_leverage(3, 46_000, 700_000, multiplier=10.0) == \
        pytest.approx(1.9714, abs=1e-3)
    assert sizing.current_leverage(0, 46_000, 700_000, multiplier=10.0) == 0.0


# ── MAX_CONTRACTS 容量餘裕 ───────────────────────────────────────────────────

def test_capacity_headroom_math():
    """上限在 raw ≥ MAX+0.5 時綁定 → E_bind = (MAX+0.5)·P·m/L_HIGH。"""
    P, m, MAX, L = 47_303.0, 10.0, 4, 2.0
    e_bind, hr = sizing.capacity_headroom(719_716.0, P, multiplier=m,
                                          max_contracts=MAX, l_max=L)
    assert e_bind == pytest.approx((MAX + 0.5) * P * m / L)      # 1,064,317.5
    assert hr == pytest.approx(1 - 719_716.0 / e_bind, abs=1e-9)
    assert 0.30 < hr < 0.35                                       # 現況約 32%
    # 恰在 E_bind 上 → 餘裕 0；超過 → 負值
    assert sizing.capacity_headroom(e_bind, P, multiplier=m,
                                    max_contracts=MAX, l_max=L)[1] == pytest.approx(0)
    assert sizing.capacity_headroom(e_bind * 1.1, P, multiplier=m,
                                    max_contracts=MAX, l_max=L)[1] < 0


def test_capacity_binding_point_matches_target_contracts():
    """E_bind 處的目標口數應恰好超過上限——與 target_contracts 交叉驗證。"""
    P, m, MAX, L = 47_303.0, 10.0, 4, 2.0
    e_bind, _ = sizing.capacity_headroom(0.0, P, multiplier=m,
                                         max_contracts=MAX, l_max=L)
    assert sizing.target_contracts(L, e_bind * 0.999, P,
                                   multiplier=m).target_contracts == MAX
    assert sizing.target_contracts(L, e_bind * 1.001, P,
                                   multiplier=m).target_contracts == MAX + 1


def test_capacity_uses_l_high_not_current_leverage():
    """以 L_HIGH 計算（保守）：BEAR 期間 L 較低不應讓餘裕看似寬鬆。"""
    P = 47_303.0
    hi, _ = sizing.capacity_headroom(700_000.0, P, l_max=config.L_HIGH)
    lo, _ = sizing.capacity_headroom(700_000.0, P, l_max=config.L_LOW)
    assert hi < lo, "L 越低 → E_bind 越大 → 餘裕越寬；預設須用 L_HIGH"


# ── 口數遲滯帶（2026-11-24 新增）───────────────────────────────────────────────
# P=40,000、乘數 10、L=2 → raw = E / 200,000，權益直接對應 raw。
P, M = 40_000.0, 10.0


def _n(equity, held, band=0.1, L=2.0):
    return sizing.target_contracts(L, equity, P, multiplier=M, held=held, band=band)


def test_band_keeps_held_inside_and_moves_outside():
    r = _n(510_000, held=2)                       # raw 2.55：四捨五入 3，帶內 → 維持 2
    assert (r.target_contracts, r.band_hold) == (2, True)
    assert r.rounding_residual == pytest.approx(0.55)
    assert r.achieved_leverage == pytest.approx(2 * P * M / 510_000)
    assert _n(522_000, held=2).target_contracts == 3        # raw 2.61：跨出 → 3
    assert (_n(490_000, held=3).target_contracts, _n(490_000, held=3).band_hold) == (3, True)
    assert _n(478_000, held=3).target_contracts == 2        # raw 2.39：跨出 → 2
    assert _n(1_000_000, held=2).target_contracts == 5      # 遠超帶外 → 照四捨五入，不只差 1 口


def test_band_off_is_identical_to_plain_rounding():
    """band=0、held 為 None 或 0 時，結果與原本的純四捨五入逐筆相同。"""
    for i in range(30, 470):
        eq = i / 100 * 200_000                      # raw 0.30–4.69
        plain = sizing.target_contracts(2.0, eq, P, multiplier=M)
        assert plain.band_hold is False
        for held, band in [(h, 0.0) for h in range(5)] + [(None, 0.1), (0, 0.1)]:
            r = _n(eq, held=held, band=band)
            assert r.target_contracts == plain.target_contracts
            assert r.band_hold is False


def test_band_rejects_negative():
    with pytest.raises(ValueError):
        _n(500_000, held=2, band=-0.1)


def _falling_bear(band, days=40):
    """BEAR（L=1）持 1 口、raw 貼近 1.5，價格每日 −0.3% 單向下跌；回傳每日口數。"""
    price, n = 40_000.0, 1
    equity = 1.497 * M * price
    path = []
    for _ in range(days):
        nxt = price * 0.997
        equity += n * M * (nxt - price)
        price = nxt
        n = sizing.target_contracts(1.0, equity, price, multiplier=M,
                                    held=n, band=band).target_contracts
        path.append(n)
    return path


def test_bear_falling_market_pingpong_without_band():
    """持 1 口實際槓桿 <1、2 口 >1：下跌讓兩側都往 1.5 推，沒有遲滯帶就隔日加減。"""
    path = _falling_bear(band=0.0)
    changes = sum(a != b for a, b in zip(path, path[1:]))
    assert changes >= 10, path


def test_bear_falling_market_band_stops_pingpong():
    path = _falling_bear(band=0.1)
    changes = sum(a != b for a, b in zip(path, path[1:]))
    assert changes <= 1, path
