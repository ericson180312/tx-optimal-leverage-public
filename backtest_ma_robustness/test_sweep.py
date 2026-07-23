"""round4 掃描的 sanity check：regime 手算、MA200 複現第三輪 D_10、規則 A/B 機制。

執行：conda run -n ML_env python -m backtest_ma_robustness.test_sweep
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import signals as sg
from backtest_ma_robustness import sweep
from backtest_ma_robustness.params import BASELINE_MA, BOOT_BUFFER, variant_key

OK = {"n": 0}


def check(label, cond):
    assert cond, f"FAIL: {label}"
    OK["n"] += 1
    print(f"  ok  {label}")


def test_regime_basic():
    """手算 regime：滿載→降載→滿載，其中一段短命=假訊號。"""
    L = 2.0
    # 狀態序列（1=滿載2.0, 0=降載1.0）：滿滿滿 降降 滿(僅3日→<20假訊號) 降降降
    lev = np.array([2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1], float)
    r = sweep.regime_stats(lev, l_high=L, false_window=20)
    check("切換次數=3", r["n_switches"] == 3)
    check("regime 段數=4", r["n_regimes"] == 4)
    # 內部段：降降(2日)、滿滿滿(3日) 皆 ≤20 → 2 次假訊號
    check("假訊號=2", r["n_false_signals"] == 2)
    check("降載佔比=5/11", abs(r["share_derisked"] - 5 / 11) < 1e-9)


def test_regime_no_switch():
    """全程滿載 → 0 切換、0 假訊號。"""
    r = sweep.regime_stats(np.full(50, 2.0))
    check("無切換", r["n_switches"] == 0 and r["n_false_signals"] == 0)
    check("單一 regime 長度=50", r["avg_regime_days"] == 50.0)


def test_false_signal_window():
    """長命 regime（>20日）不算假訊號。"""
    lev = np.concatenate([np.full(5, 2.0), np.full(30, 1.0), np.full(5, 2.0)])
    r = sweep.regime_stats(lev, false_window=20)
    check("2 次切換", r["n_switches"] == 2)
    check("長 regime 非假訊號", r["n_false_signals"] == 0)


def test_variant_signal_matches_trend():
    """變體訊號 = trend_leverage(shift)。MA200/buf1% 應等於第三輪 D_10 的訊號。"""
    rng = np.random.default_rng(0)
    spot = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 800))
    V = sweep.build_variant_signals(spot, 1.0, mas=[200], buffers=[0.01])
    d10 = sg.to_exec(sg.trend_leverage(spot, 1.0, ma_win=200, buf=0.01, l_high=2.0))
    check("MA200/buf1% = D_10 執行訊號", np.allclose(V[variant_key(200, 0.01)], d10))


def test_ma_changes_signal():
    """不同 MA → 不同訊號（掃描維度真的有變化）。"""
    rng = np.random.default_rng(1)
    spot = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, 1500))
    s120 = sg.trend_leverage(spot, 1.0, ma_win=120, buf=0.01)
    s300 = sg.trend_leverage(spot, 1.0, ma_win=300, buf=0.01)
    check("MA120 ≠ MA300", not np.allclose(s120, s300))


def test_maximin_reproduces_d10():
    """整合檢核：MA200/buf1% 的 maximin 最差格 = 第三輪 D_10（同一組 10 策略排名）。"""
    import backtest_threshold.params as rp
    from backtest_threshold.continuous import build_continuous
    from backtest_threshold.data import load_tx_contracts
    from backtest_multimarket import evaluate as ev, strategies as st
    from backtest_multimarket.data_mkts import load_markets
    from backtest_multimarket.params import STRATEGIES, RANKING_MARKETS

    tx_cs = build_continuous(load_tx_contracts(), rp.ROLL_OFFSET_DAYS)
    mkts = load_markets(tx_cs)
    calibs = {k: st.calibrate_market(m.ret, m.spot) for k, m in mkts.items()}

    # 第三輪原生：10 策略的 maximin（D_10 的 worst_pct_rank）
    aggs = []
    for k in RANKING_MARKETS:
        L = st.exec_signals(calibs[k], "wf")
        a, _ = ev.run_market_all_periods(mkts[k], L, calibs[k]["calib_days"], STRATEGIES)
        aggs.append(a)
    agg3 = pd.concat(aggs, ignore_index=True)
    mm3 = ev.maximin_ranking(agg3, "med_sharpe")
    d10_worst = float(mm3[mm3["strategy"] == "D_10"]["worst_pct_rank"].iloc[0])

    # round4：MA200/buf1% 變體 vs 9 參照 cohort
    agg4 = sweep.run_sweep(mkts, calibs, buffers=[BOOT_BUFFER], markets=RANKING_MARKETS)
    cp = sweep.variant_cell_percentiles(agg4, [variant_key(200, BOOT_BUFFER)])
    mm4 = sweep.variant_maximin(cp)
    v_worst = float(mm4["worst_pct_rank"].iloc[0])
    check(f"MA200 複現 D_10 maximin（{v_worst:.3f} == {d10_worst:.3f}）",
          abs(v_worst - d10_worst) < 1e-9)


def test_decide_plateau_synthetic():
    """規則 A：人工造一個「完全等高」的 maximin → 應判高原、規則 B 不觸發。"""
    from backtest_ma_robustness.params import MA_GRID, BUFFER_GRID
    rows_mm, rows_cp = [], []
    for ma in MA_GRID:
        for buf in BUFFER_GRID:
            k = variant_key(ma, buf)
            rows_mm.append({"strategy": k, "worst_pct_rank": 0.40,
                            "mean_pct_rank": 0.70, "worst_cell": "HSI×5y",
                            "med_avg_lev": 1.6, "med_cagr": 0.15})
            for mk in ["TX_FUT", "SPX", "NKX", "HSI"]:
                for yr in [3, 5, 10, 20]:
                    rows_cp.append({"strategy": k, "market": mk, "period_years": yr,
                                    "med_sharpe": 0.5, "pct_rank": 0.4,
                                    "med_avg_lev": 1.6, "med_cagr": 0.15})
    dec = sweep.decide(pd.DataFrame(rows_mm), pd.DataFrame(rows_cp))
    check("等高 → 百分位極差=0", dec["spread_pct"] == 0.0)
    check("等高 → 判高原", dec["verdict_a"] == "高原")
    check("等高 → 規則B不觸發", dec["spike"] is False)


if __name__ == "__main__":
    for fn in [test_regime_basic, test_regime_no_switch, test_false_signal_window,
               test_variant_signal_matches_trend, test_ma_changes_signal,
               test_decide_plateau_synthetic, test_maximin_reproduces_d10]:
        print(fn.__name__)
        fn()
    print(f"\n全部 {OK['n']} 項檢核通過")
