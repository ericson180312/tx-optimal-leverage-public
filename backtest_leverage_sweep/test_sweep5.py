"""第五輪 pytest：強制平倉手算、臨界跌幅、跌停生存、存活者聚合、訊號凍結。

執行：conda run -n ML_env python -m pytest backtest_leverage_sweep/test_sweep5.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_multimarket import engine as eng, signals as sg

from backtest_leverage_sweep import margin, sweep5
from backtest_leverage_sweep.params import (
    L_HIGH_GRID, MAINT_MARGIN_RATE, TREND_BUFFER, TREND_MA, variant_key,
)


# ── §3-1 臨界跌幅 ───────────────────────────────────────────────────────────

def test_x_liq_matches_spec_expectations():
    """規格 §3-1 明示：L=2.0 → ~46%、L=2.5 → ~36%（近似式）。"""
    assert margin.x_liq_approx(2.0) == pytest.approx(0.465, abs=5e-4)
    assert margin.x_liq_approx(2.5) == pytest.approx(0.365, abs=5e-4)


def test_x_liq_exact_solves_engine_condition():
    """精確式必須真的滿足引擎條件 E == m·N。"""
    for L in L_HIGH_GRID:
        x = margin.x_liq_exact(L)
        E = 1.0 - L * x                     # 期初權益 1
        N = L * (1.0 - x)
        assert E == pytest.approx(MAINT_MARGIN_RATE * N, abs=1e-12)


def test_liquidation_precedes_ruin():
    """強平門檻恆早於權益歸零（x_liq < x_ruin=1/L）。"""
    for L in L_HIGH_GRID:
        assert margin.x_liq_exact(L) < margin.x_ruin(L)


def test_x_liq_monotone_decreasing_in_leverage():
    """槓桿越高 → 可承受跌幅越小（緩衝收縮）。"""
    xs = [margin.x_liq_exact(L) for L in L_HIGH_GRID]
    assert all(a > b for a, b in zip(xs, xs[1:]))


# ── §3-2 連續跌停生存窗口 ───────────────────────────────────────────────────

def test_limit_down_days_hand_computed():
    """手算：1−0.9^k。L=2.0 的 x_liq=48.2% 落在 k=6(46.9%) 與 k=7(52.2%) 之間。"""
    assert 1 - 0.9 ** 6 == pytest.approx(0.468559, abs=1e-6)
    assert 1 - 0.9 ** 7 == pytest.approx(0.521703, abs=1e-6)
    d = margin.limit_down_days(2.0)
    assert d["survive_days_exact"] == 6           # 第 7 個跌停才觸發
    assert margin.limit_down_days(2.5)["survive_days_exact"] == 4


def test_limit_down_days_non_increasing():
    """跌停生存日數不隨槓桿升高而增加。"""
    ds = [margin.limit_down_days(L)["survive_days_exact"] for L in L_HIGH_GRID]
    assert all(a >= b for a, b in zip(ds, ds[1:]))


# ── 引擎：強制平倉行為 ──────────────────────────────────────────────────────

def _run(ret, L, maint, e0=100.0):
    lev = np.full(len(ret), L)
    return eng.simulate({"kind": "signal", "L_exec": lev}, np.asarray(ret),
                        financing="index", costs="bp", rf_dep=0.0, rf_bor=0.0,
                        cost_rate=0.0, maint_margin_rate=maint, e0=e0)


def test_margin_disabled_is_behaviour_neutral():
    """maint_margin_rate=0（預設）→ 與未啟用完全一致，且 liquidated 全 False。"""
    ret = np.array([-0.10, -0.10, -0.10, 0.05])
    a = _run(ret, 2.0, 0.0)
    b = eng.simulate({"kind": "signal", "L_exec": np.full(len(ret), 2.0)}, ret,
                     financing="index", costs="bp", rf_dep=0.0, rf_bor=0.0,
                     cost_rate=0.0, e0=100.0)
    assert np.allclose(a.eq, b.eq)
    assert not a.liquidated.any()


def test_forced_liquidation_triggers_and_is_absorbing():
    """單日 −25%、L=2.5：權益 100→37.5，名目 187.5×0.75=140.6，m=0.30 → 強平。

    強平後曝險恆 0：其後市場反彈不得回血（吸收狀態）。
    """
    res = _run([-0.25, +0.50, +0.50], 2.5, 0.30)
    assert res.liquidated[0]
    eq = res.eq[:, 0]
    assert eq[1] == pytest.approx(eq[0])      # 平倉後不再參與市場
    assert eq[2] == pytest.approx(eq[0])
    assert res.lev[1, 0] == 0.0


def test_liquidation_preserves_residual_equity():
    """強平保留剩餘權益（非歸零），且剩餘 > 0。"""
    res = _run([-0.25], 2.5, 0.30)
    assert res.liquidated[0]
    assert res.eq[0, 0] > 0
    assert not res.bankrupt[0]                # 強平 ≠ 破產，分開計數


def test_no_liquidation_when_buffer_sufficient():
    """小幅下跌不應觸發（避免過度觸發的偽陽性）。"""
    res = _run([-0.05, -0.05], 2.0, MAINT_MARGIN_RATE)
    assert not res.liquidated.any()


def test_engine_liquidation_matches_analytic_threshold():
    """引擎觸發點與 margin.x_liq_exact 一致（跨越門檻前後各一）。"""
    for L in (2.0, 2.5):
        x = margin.x_liq_exact(L, MAINT_MARGIN_RATE)
        assert not _run([-(x - 0.005)], L, MAINT_MARGIN_RATE).liquidated[0]
        assert _run([-(x + 0.005)], L, MAINT_MARGIN_RATE).liquidated[0]


# ── §8 存活者口徑聚合 ───────────────────────────────────────────────────────

def test_aggregate_excludes_dead_paths():
    """假優勝者防線：歸零前暴賺的路徑不得進入報酬統計分子。"""
    m = pd.DataFrame({
        "cagr": [0.10, 0.12, 9.99], "cumret": [1.0, 1.2, 50.0],
        "sharpe": [0.5, 0.6, 3.0], "sortino": [0.7, 0.8, 4.0],
        "calmar": [0.3, 0.4, 2.0], "mdd": [-0.4, -0.45, -1.0],
        "uw_months": [10, 12, 30], "avg_lev": [1.6, 1.6, 1.7],
        "lev_at_trough": [1.0, 1.0, 2.5], "max_lev": [2.0, 2.0, 3.0],
        "cost_ratio": [0.1, 0.1, 0.2], "n_adj": [10, 11, 12],
        "bankrupt": [False, False, False], "liquidated": [False, False, True],
    })
    out = sweep5.aggregate_survivor({"D_L200": m}, "TX_FUT", 10).iloc[0]
    assert out["n_liquidated"] == 1
    assert out["survival_rate"] == pytest.approx(2 / 3)
    assert out["med_cagr"] == pytest.approx(0.11)       # 排除 9.99 的假優勝者
    assert out["med_cagr"] < 0.5


def test_aggregate_all_dead_returns_nan():
    m = pd.DataFrame({c: [1.0] for c in (
        "cagr", "cumret", "sharpe", "sortino", "calmar", "mdd", "uw_months",
        "avg_lev", "lev_at_trough", "max_lev", "cost_ratio", "n_adj")})
    m["bankrupt"] = [True]; m["liquidated"] = [False]
    out = sweep5.aggregate_survivor({"D_L250": m}, "NKX", 20).iloc[0]
    assert out["survival_rate"] == 0.0
    assert np.isnan(out["med_cagr"])


# ── 訊號：除 L_high 外一切凍結（規格 §1）────────────────────────────────────

def test_signal_switch_timing_identical_across_l_high():
    """L_high 只改變幅度、不改變切換時點（MA/buffer 凍結的直接後果）。"""
    rng = np.random.default_rng(0)
    spot = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 2000))
    V = sweep5.build_variant_signals(spot)
    base = V[variant_key(2.0)]
    for lh in L_HIGH_GRID[1:]:
        v = V[variant_key(lh)]
        # 降載日（=L_low=1.0）位置必須完全相同
        assert np.array_equal(base <= 1.0 + 1e-12, v <= 1.0 + 1e-12)


def test_l_high_2_0_reproduces_d10():
    """L_high=2.0 即第三／四輪的 D_10（凍結參數的一致性檢核）。"""
    rng = np.random.default_rng(1)
    spot = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, 1500))
    ours = sweep5.build_variant_signals(spot, l_grid=[2.0])[variant_key(2.0)]
    ref = sg.to_exec(sg.trend_leverage(spot, 1.0, ma_win=TREND_MA,
                                       buf=TREND_BUFFER, l_high=2.0))
    assert np.array_equal(ours, ref)


def test_no_lookahead_signal_is_shifted():
    """T+1 執行：執行槓桿必為訊號序列平移一日。"""
    rng = np.random.default_rng(2)
    spot = 100 * np.cumprod(1 + rng.normal(0.0, 0.01, 600))
    raw = sg.trend_leverage(spot, 1.0, ma_win=TREND_MA, buf=TREND_BUFFER, l_high=2.3)
    ex = sweep5.build_variant_signals(spot, l_grid=[2.3])[variant_key(2.3)]
    assert np.array_equal(ex[1:], raw[:-1])
