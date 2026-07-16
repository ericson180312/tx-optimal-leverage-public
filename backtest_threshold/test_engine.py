"""引擎 sanity check：與手算結果逐一對照（pytest 或直接執行皆可）。"""
from __future__ import annotations

import math

import numpy as np

from backtest_threshold.engine import simulate

INF = math.inf


def _sim(ret, X, Y, target, *, slip=0.0, fee=0.0, tax=0.0, rf=0.0,
         e0=100.0, is_roll=None, p_near=None, p_next=None, price=None,
         entry_price=10000.0, rebal_every=0, rebal_phase=0):
    T = len(ret)
    ret = np.asarray(ret, dtype=float)
    price = np.full(T, 10000.0) if price is None else np.asarray(price, float)
    is_roll = np.zeros(T, bool) if is_roll is None else np.asarray(is_roll, bool)
    p_near = np.full(T, np.nan) if p_near is None else np.asarray(p_near, float)
    p_next = np.full(T, np.nan) if p_next is None else np.asarray(p_next, float)
    return simulate(ret, price, is_roll, p_near, p_next,
                    np.array([X], float), np.array([Y], float), np.array([target], float),
                    entry_price, rebal_every=np.array([rebal_every], np.int64),
                    rebal_phase=np.array([rebal_phase], np.int64),
                    slippage_ticks=slip, fee=fee, tax=tax,
                    rf_daily=rf, e0=e0)


def test_buy_and_hold_no_cost():
    """BH 2x 無成本無利息：E_T = e0 * (1 + 2*(Π(1+r)-1))。"""
    r = [0.01, -0.02, 0.03, 0.015, -0.01]
    res = _sim(r, INF, -INF, 2.0)
    expect = 100.0 * (1.0 + 2.0 * (np.prod(1 + np.array(r)) - 1.0))
    assert abs(res.eq[-1, 0] - expect) < 1e-9
    assert res.n_up[0] == 0 and res.n_down[0] == 0 and not res.bankrupt[0]


def test_daily_rebalance_no_cost():
    """每日再平衡 2x 無成本：E_T = e0 * Π(1+2r)。"""
    r = [0.01, -0.02, 0.03, 0.015, -0.01]
    res = _sim(r, 2.0, 2.0, 2.0)
    expect = 100.0 * np.prod(1 + 2 * np.array(r))
    assert abs(res.eq[-1, 0] - expect) < 1e-9
    assert res.n_up[0] + res.n_down[0] == len(r)


def test_threshold_hand_computed():
    """門檻觸發手算對照：X=2.2 於第 2 天觸發減碼。"""
    res = _sim([-0.06, -0.03, 0.10], 2.2, 1.8, 2.0)
    # d1: E=100-200*0.06=88, N=188, L=2.1364 未觸發
    assert abs(res.eq[0, 0] - 88.0) < 1e-9
    assert res.trig[0, 0] == 0
    # d2: E=88-188*0.03=82.36, N=182.36, L=2.2142 ≥ 2.2 → N=164.72
    assert abs(res.eq[1, 0] - 82.36) < 1e-9
    assert res.trig[1, 0] == -1
    assert abs(res.dN[1, 0] - (182.36 - 164.72)) < 1e-9
    # d3: E=82.36+164.72*0.10=98.832, N=181.192, L=1.8333 未觸發（Y=1.8）
    assert abs(res.eq[2, 0] - 98.832) < 1e-9
    assert res.trig[2, 0] == 0
    assert res.n_down[0] == 1 and res.n_up[0] == 0
    assert abs(res.max_lev[0] - 182.36 / 82.36) < 1e-12


def test_rebalance_cost():
    """再平衡成本 = |ΔN|/(200×price)×(fee+slip×200) + |ΔN|×tax，即時扣權益。"""
    e0, tax, fee, slip = 1_000_000.0, 0.00002, 20.0, 1.0
    res = _sim([-0.10], 2.1, -INF, 2.0, slip=slip, fee=fee, tax=tax,
               e0=e0, price=[9000.0], entry_price=10000.0)
    entry_cost = (2 * e0) / (200 * 10000.0) * (fee + slip * 200) + 2 * e0 * tax
    E1 = (e0 - entry_cost) + 2 * e0 * (-0.10)          # 結算後
    N1 = 2 * e0 * 0.90
    assert N1 / E1 >= 2.1                               # 觸發減碼
    dN = abs(2 * E1 - N1)
    cost = dN / (200 * 9000.0) * (fee + slip * 200) + dN * tax
    assert abs(res.eq[0, 0] - (E1 - cost)) < 1e-6
    assert abs(res.lev[0, 0] - 2 * E1 / (E1 - cost)) < 1e-12


def test_roll_cost_keeps_exposure():
    """轉倉：兩腳成本、不改變曝險。"""
    e0, tax, fee, slip = 1_000_000.0, 0.00002, 20.0, 1.0
    res = _sim([0.0, 0.0], INF, -INF, 2.0, slip=slip, fee=fee, tax=tax, e0=e0,
               is_roll=[False, True], p_near=[np.nan, 10000.0], p_next=[np.nan, 10100.0],
               price=[10000.0, 10100.0], entry_price=10000.0)
    N = 2 * e0
    roll_cost = (N / (200 * 10000) + N / (200 * 10100)) * (fee + slip * 200) + 2 * N * tax
    entry_cost = N / (200 * 10000) * (fee + slip * 200) + N * tax
    assert abs(res.eq[1, 0] - (e0 - entry_cost - roll_cost)) < 1e-6
    assert abs(res.cost[1, 0] - roll_cost) < 1e-9      # 曝險不變、只計成本


def test_bankruptcy_no_rebound():
    """破產：權益歸零後不得反彈，後續報酬恆為 0。"""
    res = _sim([-0.60, 0.50, 0.50], INF, -INF, 2.0)
    assert res.bankrupt[0]
    assert (res.eq[:, 0] == 0.0).all()                  # d1 即歸零
    assert (res.lev[1:, 0] == 0.0).all()


def test_periodic_rebalance_timing():
    """每 5 日再平衡：只在第 5、10 個交易日觸發，其餘日不動。"""
    r = [0.01] * 12
    res = _sim(r, INF, -INF, 2.0, rebal_every=5)
    fired = np.flatnonzero(res.trig[:, 0] != 0)
    assert list(fired) == [4, 9], f"觸發日應為第 5、10 個交易日（index 4, 9），實得 {fired}"
    # 上漲時槓桿自然下滑（曝險成長慢於權益），調回 2.0 需買進 → 加碼
    assert res.n_up[0] == 2 and res.n_down[0] == 0
    assert res.trig[4, 0] == 1 and res.trig[9, 0] == 1


def test_periodic_rebalance_hand_computed():
    """每 2 日再平衡的手算對照（無成本、無利息）。"""
    res = _sim([0.10, 0.10], INF, -INF, 2.0, rebal_every=2)
    # d1: E=100+200*0.10=120, N=220 → 未到週期，不動
    assert abs(res.eq[0, 0] - 120.0) < 1e-9 and res.trig[0, 0] == 0
    # d2: E=120+220*0.10=142, N=242, L=1.7042 → 第2個交易日 → 調回 N=284（加碼）
    assert abs(res.eq[1, 0] - 142.0) < 1e-9
    assert res.trig[1, 0] == 1
    assert abs(res.dN[1, 0] - (284.0 - 242.0)) < 1e-9
    assert abs(res.lev[1, 0] - 2.0) < 1e-12


def test_periodic_phase_shift():
    """相位平移：phase=2 時，觸發日往後移 2 天。"""
    r = [0.01] * 12
    base = _sim(r, INF, -INF, 2.0, rebal_every=5, rebal_phase=0)
    shifted = _sim(r, INF, -INF, 2.0, rebal_every=5, rebal_phase=2)
    assert list(np.flatnonzero(base.trig[:, 0] != 0)) == [4, 9]
    assert list(np.flatnonzero(shifted.trig[:, 0] != 0)) == [6, 11]


def test_periodic_every_1_equals_daily_threshold():
    """rebal_every=1 應與門檻式 X=Y=target（每日再平衡）逐日等價。"""
    r = [0.01, -0.02, 0.03, 0.015, -0.01, 0.02]
    a = _sim(r, INF, -INF, 2.0, rebal_every=1, slip=1.0, fee=20.0, tax=0.00002, e0=1e6)
    b = _sim(r, 2.0, 2.0, 2.0, slip=1.0, fee=20.0, tax=0.00002, e0=1e6)
    assert np.allclose(a.eq, b.eq, rtol=0, atol=1e-9)
    assert a.n_up[0] + a.n_down[0] == b.n_up[0] + b.n_down[0] == len(r)


def test_periodic_and_threshold_single_adjustment():
    """門檻與週期同日觸發時只調整一次（不重複計成本）。"""
    # 大跌使 L 超過 X，同時當天正好是週期日
    res = _sim([-0.10], 2.1, -INF, 2.0, rebal_every=1, slip=1.0, fee=20.0,
               tax=0.00002, e0=1_000_000.0, price=[9000.0])
    only_thr = _sim([-0.10], 2.1, -INF, 2.0, slip=1.0, fee=20.0,
                    tax=0.00002, e0=1_000_000.0, price=[9000.0])
    assert abs(res.eq[0, 0] - only_thr.eq[0, 0]) < 1e-9   # 成本相同 → 只調一次
    assert res.n_up[0] + res.n_down[0] == 1


def test_idle_cash_interest():
    """r=0 時權益以日頻複利成長：E_T = e0 (1+rf)^T。"""
    rf = 0.0001
    res = _sim([0.0] * 10, INF, -INF, 1.0, rf=rf)
    assert abs(res.eq[-1, 0] - 100.0 * (1 + rf) ** 10) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
    print("engine sanity check: all passed")
