"""round6 的 sanity check：相位變體與第三輪逐位元一致、延遲對齊、規則 A–D 全分支。

執行：conda run -n ML_env python -m backtest_signal_timing.test_sweep6
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import signals as sg
from backtest_signal_timing import bootstrap6, sweep6
from backtest_signal_timing.params import K_GRID, PLATEAU_TOL, variant_key

OK = {"n": 0}


def check(label, cond):
    assert cond, f"FAIL: {label}"
    OK["n"] += 1
    print(f"  ok  {label}")


def _spot(seed=7, T=1500):
    rng = np.random.default_rng(seed)
    return 8000.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.015, size=T))


def test_phase0_identical_to_round3():
    """phase=0、K=5 與 signals.trend_leverage 逐位元相同（1D 與 2D）。"""
    s = _spot()
    ref = sg.trend_leverage(s, 1.0)
    check("1D 逐位元相同", np.array_equal(
        sweep6.trend_leverage_phased(s, check_every=5, phase=0), ref))
    m = np.column_stack([_spot(i) for i in range(4)])
    check("2D 逐位元相同", np.array_equal(
        sweep6.trend_leverage_phased(m, check_every=5, phase=0), sg.trend_leverage(m, 1.0)))
    for k in (1, 20):
        check(f"K={k} 逐位元相同", np.array_equal(
            sweep6.trend_leverage_phased(s, check_every=k, phase=0),
            sg.trend_leverage(s, 1.0, check_every=k)))


def test_switches_only_on_phase_days():
    """切換只發生在 (t − phase) % K == 0 的檢查日。"""
    s = _spot(3, 3000)
    for k, ph in [(5, 2), (10, 7), (20, 13), (3, 1)]:
        lev = sweep6.trend_leverage_phased(s, check_every=k, phase=ph)
        sw = np.flatnonzero(np.diff(lev) != 0) + 1
        check(f"K={k} φ={ph} 有切換且都在檢查日",
              len(sw) > 0 and all((t - ph) % k == 0 for t in sw))


def test_lag_alignment():
    sig = np.array([2, 2, 1, 1, 2, 2], float)
    check("lag=0 同日", np.array_equal(sweep6.apply_lag(sig, 0), sig))
    check("lag=1 等於 to_exec", np.array_equal(sweep6.apply_lag(sig, 1), sg.to_exec(sig)))
    check("lag=2 再延一日", np.array_equal(sweep6.apply_lag(sig, 2),
                                         np.array([2, 2, 2, 2, 1, 1], float)))


def test_variant_grid():
    grid = sweep6.variant_grid()
    check("123 個變體", len(grid) == 123)
    check("鍵名可解析", variant_key(5, 1, 0) == "D_K05_L1_P00")
    df = pd.DataFrame({"strategy": [variant_key(20, 2, 13), "D_10", "C_EWMA"],
                       "med_sharpe": [0.1, 0.2, 0.3]})
    t = sweep6.tag_variants(df)
    check("tag 只留變體並拆欄", len(t) == 1 and (t.K.iloc[0], t.lag.iloc[0], t.phase.iloc[0])
          == (20, 2, 13))


def _neutral(s_by_kl, markets=("TX_FUT", "SPX", "NKX", "HSI"), periods=(3, 5, 10, 20)):
    """合成 phase_neutral 表：s_by_kl[(K,lag)] = 純量或 {市場: 值}。"""
    rows = []
    for (k, lag), v in s_by_kl.items():
        for m in markets:
            for p in periods:
                rows.append({"K": k, "lag": lag, "market": m, "period_years": p,
                             "med_sharpe": v[m] if isinstance(v, dict) else v,
                             "med_cagr": 0.1, "med_mdd": -0.3, "worst_mdd": -0.4,
                             "med_avg_lev": 1.7, "med_time_derisked": 0.2,
                             "med_cost_ratio": 0.01, "med_n_adj": 10.0, "n_bankrupt": 0.0})
    return pd.DataFrame(rows)


def _decide(delta, *, noise=0.0, delta_p=0.0, boot=4):
    """delta[(K,lag)] 為相對基準 0.50 的差；未指定者差 0。"""
    base = {(k, 1): 0.5 for k in K_GRID}
    base.update({(5, 0): 0.5, (1, 0): 0.5, (5, 2): 0.5})
    for v, d in delta.items():
        base[v] = {m: 0.5 + x for m, x in d.items()} if isinstance(d, dict) else 0.5 + d
    summary = sweep6.summarize(sweep6.delta_table(_neutral(base)))
    return sweep6.decide(summary, noise, delta_p, lambda k, lag: boot)


def test_rule_a_branches():
    r = _decide({(1, 1): 0.03, (20, 1): -0.03})
    check("A：R_A<τ → 高原", r["verdict_a"] == "高原" and abs(r["r_a"] - 0.06) < 1e-9)
    r = _decide({(1, 1): 0.05, (2, 1): -0.02, (3, 1): -0.03, (10, 1): -0.04, (20, 1): -0.10})
    check("A：R_A≥τ、K=5 前 3 → 傾斜高原", r["verdict_a"] == "傾斜高原" and r["rank5"] == 2)
    r = _decide({(1, 1): 0.12, (2, 1): 0.10, (3, 1): 0.09, (10, 1): -0.01, (20, 1): -0.02})
    check("A：K=5 後 3 且條件全過 → 劣位", r["verdict_a"] == "5日為劣位")
    r = _decide({(1, 1): 0.12, (2, 1): 0.10, (3, 1): 0.09, (10, 1): -0.01}, boot=2)
    check("A：bootstrap 不同意 → 降級", r["verdict_a"] == "傾斜但無可行替代")
    r = _decide({(1, 1): 0.12, (2, 1): 0.01, (3, 1): 0.01, (10, 1): -0.01})
    check("A：只有 K=1 過門檻仍可判劣位（相鄰 K=2 的 D≥0）",
          r["verdict_a"] == "5日為劣位"
          and r["candidates"].set_index("K").loc[1, "ok"]
          and not r["candidates"].set_index("K").loc[2, "ok"])


def test_rule_b_branches():
    r = _decide({(5, 0): 0.05, (1, 0): 0.02})
    check("B：U<τ → 排除", r["verdict_b"] == "排除" and r["v_star"] == (5, 0))
    r = _decide({(1, 0): 0.10})
    check("B：U≥τ 且一致 → 上界有空間", r["verdict_b"] == "上界有空間"
          and r["v_star"] == (1, 0))
    r = _decide({(5, 0): 0.10}, boot=2)
    check("B：bootstrap 不同意 → 不一致", r["verdict_b"] == "不一致")
    # 只有兩個市場為正，且中位數 0.05+... 仍 ≥ τ：方向不足 → 不一致
    r = _decide({(5, 0): {"TX_FUT": 0.3, "SPX": 0.3, "NKX": -0.1, "HSI": -0.1}}, noise=0.0)
    check("B：方向不足 3 市場 → 不一致", r["verdict_b"] == "不一致" and r["u"] >= 0.08)
    r = _decide({(5, 2): -0.10})
    check("B：lag=2 退步 ≥τ → 延遲敏感", r["delay_sensitive"])


def test_rule_c_d_and_tau():
    r = _decide({(3, 1): -0.10, (10, 1): -0.09})
    check("C：K=5 高於兩鄰 >τ → 尖峰", r["spike"] and abs(r["spike_gap"] - 0.09) < 1e-9)
    r = _decide({(3, 1): -0.10, (10, 1): -0.05})
    check("C：只高於一鄰 → 未觸發", not r["spike"])
    r = _decide({}, delta_p=0.09)
    check("D：|ΔP|≥τ → 揭露", r["phase_flag"])
    r = _decide({}, delta_p=-0.05)
    check("D：|ΔP|<τ → 不揭露", not r["phase_flag"])
    r = _decide({(1, 1): 0.05, (20, 1): -0.05}, noise=0.15)
    check("τ＝max(0.08, N)：相位雜訊較大時門檻跟著放大",
          r["tau"] == 0.15 and r["verdict_a"] == "高原")
    r = _decide({})
    check("τ 下限 0.08", r["tau"] == PLATEAU_TOL)


def test_phase_noise_and_live_phase():
    rows = []
    for m in ("TX_FUT", "SPX"):
        for p in (3, 5):
            for ph, s in enumerate([0.50, 0.52, 0.54, 0.56, 0.58]):
                rows.append({"K": 5, "lag": 1, "phase": ph, "market": m,
                             "period_years": p, "med_sharpe": s})
    v = pd.DataFrame(rows)
    n, _ = sweep6.phase_noise(v)
    check("N＝每格相位全距的中位數", abs(n - 0.08) < 1e-9)
    check("ΔP＝實盤相位 − 相位中位數", abs(sweep6.live_phase_delta(v) - (-0.04)) < 1e-9)


def test_paired_differences():
    """配對差逐格相減：與「兩個中位數相減」不同（MDD 落在排序斷層時會差很多）。"""
    n = _neutral({(5, 1): 0.5, (1, 1): 0.5})
    base = n["K"] == 5
    n.loc[base, "med_mdd"] = [-.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9] * 2
    n.loc[~base, "med_mdd"] = n.loc[base, "med_mdd"].to_numpy() + 0.01
    s = sweep6.summarize(sweep6.delta_table(n)).set_index(["K", "lag"])
    check("ΔMDD 為逐格配對", abs(s.loc[(1, 1), "dmdd"] - 0.01) < 1e-12)
    check("基準 ΔMDD＝0", s.loc[(5, 1), "dmdd"] == 0)


def test_next_day_effect():
    """手算：切換日隔天的報酬差＝ΔL × r_{t+1}，與引擎無關，但要與定義一致。"""
    class M:
        key = "X"
    m = M()
    T = 700
    m.ret = np.full(T, 0.001)
    m.ret[400:440] = -0.03                        # 製造一次跌破
    m.spot = 100.0 * np.cumprod(1.0 + m.ret)
    m.T = T
    r = sweep6.next_day_effect({"X": m}).set_index("K").loc[5]
    sig = sweep6.trend_leverage_phased(m.spot, check_every=5, phase=0)
    t = int(np.flatnonzero(np.diff(sig))[0]) + 1
    check("有切換", r["n_switch"] >= 1)
    check("降槓桿隔天報酬", abs(r["next_ret_after_down"] - m.ret[t + 1]) < 1e-12)


def test_same_sign_counter():
    b = pd.DataFrame({"K": [1] * 4 + [20] * 4, "lag": [1] * 8,
                      "med_diff": [.1, .1, .1, -.1, -.1, -.1, .1, -.1]})
    f = bootstrap6.same_sign_counter(b)
    check("同向市場數", f(1, 1) == 3 and f(20, 1) == 1 and f(2, 1) == 0)


if __name__ == "__main__":
    for fn in [test_phase0_identical_to_round3, test_switches_only_on_phase_days,
               test_lag_alignment, test_variant_grid, test_rule_a_branches,
               test_rule_b_branches, test_rule_c_d_and_tau,
               test_phase_noise_and_live_phase, test_paired_differences,
               test_next_day_effect, test_same_sign_counter]:
        print(fn.__name__)
        fn()
    print(f"全部通過（{OK['n']} 項）")
