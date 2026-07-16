"""口數計算單元測試（規格 §8-1：含捨入、防呆上限、保險絲）。"""
from __future__ import annotations

import pytest

from live import config
from live.strategy import sizing


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
