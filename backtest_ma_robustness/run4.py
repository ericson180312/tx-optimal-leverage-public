"""第四輪主流程：資料 → MA×buffer 掃描 → maximin/規則A·B → regime → 壓力H1/H2
→ bootstrap → 圖表 → 報告。

執行：conda run -n ML_env python -m backtest_ma_robustness.run4
輸出：backtest_ma_robustness/output/ 下的 r4_*.csv、PNG 與 report.md。
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

import backtest_threshold.params as root_params
from backtest_threshold.continuous import build_continuous
from backtest_threshold.data import load_tx_contracts

from backtest_multimarket import strategies as st
from backtest_multimarket.data_mkts import load_markets

from backtest_ma_robustness import bootstrap4, charts4, stress as stress_mod, sweep
from backtest_ma_robustness.params import (
    BOOT_BUFFER, BUFFER_GRID, CACHE_DIR, L_LOW_SPOT, MA_GRID, OUTPUT_DIR,
    PERIOD_YEARS, RANKING_MARKETS, variant_key, variant_label,
)


def topography_grids(cellpct: pd.DataFrame, maximin: pd.DataFrame) -> dict:
    """(buffer, MA) 熱力圖矩陣：4 市場（各市場最差期間百分位）+ 綜合（真 maximin）。"""
    grids = {}
    mm = maximin.set_index("strategy")["worst_pct_rank"]
    for mk in RANKING_MARKETS:
        g = np.full((len(BUFFER_GRID), len(MA_GRID)), np.nan)
        sub = cellpct[cellpct["market"] == mk]
        worst = sub.groupby("strategy")["pct_rank"].min()
        for xi, ma in enumerate(MA_GRID):
            for yi, buf in enumerate(BUFFER_GRID):
                k = variant_key(ma, buf)
                if k in worst.index:
                    g[yi, xi] = worst[k]
        grids[mk] = g
    gc = np.full((len(BUFFER_GRID), len(MA_GRID)), np.nan)
    for xi, ma in enumerate(MA_GRID):
        for yi, buf in enumerate(BUFFER_GRID):
            k = variant_key(ma, buf)
            if k in mm.index:
                gc[yi, xi] = mm[k]
    grids["綜合 maximin"] = gc
    return grids


def leverage_normalized_table(maximin: pd.DataFrame, *, buf: float = BOOT_BUFFER) -> pd.DataFrame:
    """窗口 × 平均槓桿 × 每單位槓桿超額 CAGR（規格 §3-5、§5-2；Q4）。"""
    mm = maximin.set_index("strategy")
    rows = []
    for ma in MA_GRID:
        k = variant_key(ma, buf)
        if k not in mm.index:
            continue
        lev = float(mm.loc[k, "med_avg_lev"])
        cagr = float(mm.loc[k, "med_cagr"])
        rows.append({"ma": ma, "buffer": buf, "med_avg_lev": lev, "med_cagr": cagr,
                     "cagr_per_lev": cagr / lev if lev else np.nan,
                     "worst_pct_rank": float(mm.loc[k, "worst_pct_rank"]),
                     "mean_pct_rank": float(mm.loc[k, "mean_pct_rank"])})
    return pd.DataFrame(rows)


def buffer_interaction_table(maximin: pd.DataFrame) -> pd.DataFrame:
    """每個 MA 的最適 buffer（規格 §5-4）：是否隨窗口長度單調變化？（預期：長窗口配窄buffer）"""
    mm = maximin.set_index("strategy")
    rows = []
    for ma in MA_GRID:
        best_buf, best_val = None, -np.inf
        vals = {}
        for buf in BUFFER_GRID:
            k = variant_key(ma, buf)
            if k in mm.index:
                v = float(mm.loc[k, "mean_pct_rank"])
                vals[buf] = v
                if v > best_val:
                    best_val, best_buf = v, buf
        rows.append({"ma": ma, "best_buffer": best_buf,
                     **{f"mean_pct_b{int(b*1000):03d}": vals.get(b, np.nan)
                        for b in BUFFER_GRID}})
    return pd.DataFrame(rows)


def tx_200_vs_240(sweep_agg: pd.DataFrame, *, buf: float = BOOT_BUFFER) -> pd.DataFrame:
    """200 vs 240 在 TX_FUT 的直接對決（規格 §5-5；結論仍受規則 A 約束）。"""
    rows = []
    for ma in (200, 240):
        k = variant_key(ma, buf)
        d = sweep_agg[(sweep_agg["market"] == "TX_FUT") & (sweep_agg["strategy"] == k)]
        for _, r in d.iterrows():
            rows.append({"ma": ma, "period_years": int(r["period_years"]),
                         "med_sharpe": r["med_sharpe"], "med_cagr": r["med_cagr"],
                         "med_mdd": r["med_mdd"], "med_avg_lev": r["med_avg_lev"],
                         "med_time_derisked": r["med_time_derisked"]})
    return pd.DataFrame(rows).sort_values(["period_years", "ma"])


def spot_check_l05(mkts, calibs) -> pd.DataFrame:
    """L_low=0.5 抽查（規格 §2）：MA 掃描的方向一致性，不進主判定。"""
    agg = sweep.run_sweep(mkts, calibs, l_low=L_LOW_SPOT, buffers=[BOOT_BUFFER],
                          markets=RANKING_MARKETS)
    keys = [variant_key(ma, BOOT_BUFFER) for ma in MA_GRID]
    cp = sweep.variant_cell_percentiles(agg, keys)
    mm = sweep.variant_maximin(cp)
    mm["ma"] = mm["strategy"].apply(lambda s: int(s.split("MA")[1].split("_")[0]))
    return mm.sort_values("ma")


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    charts4.setup_style()

    # ── 1. 資料 + 校準（沿用第三輪）─────────────────────────────────────────
    print("== 載入市場資料 ==")
    tx_cs = build_continuous(load_tx_contracts(), root_params.ROLL_OFFSET_DAYS)
    mkts = load_markets(tx_cs)
    calibs = {k: st.calibrate_market(m.ret, m.spot) for k, m in mkts.items()}
    for k, m in mkts.items():
        print(f"  {k:7s} {m.meta['start']} ~ {m.meta['end']}  {m.T} 日")

    # ── 2. 主掃描（9 參照 + 30 變體 × 5 市場 × 4 期間）──────────────────────
    print("== MA×buffer 掃描回測 ==")
    t1 = time.time()
    sweep_agg = sweep.run_sweep(mkts, calibs)          # 含 TX_IDX（參照）
    print(f"  {time.time()-t1:.0f}s")
    var_keys = [variant_key(ma, buf) for ma in MA_GRID for buf in BUFFER_GRID]
    cellpct = sweep.variant_cell_percentiles(sweep_agg, var_keys)
    maximin = sweep.variant_maximin(cellpct)
    maximin["ma"] = maximin["strategy"].apply(lambda s: int(s.split("MA")[1].split("_")[0]))
    maximin["buffer"] = maximin["strategy"].apply(lambda s: int(s.split("_b")[1]) / 1000)
    maximin = maximin.sort_values(["ma", "buffer"]).reset_index(drop=True)

    # ── 3. 規則 A / B 判定（機械化）─────────────────────────────────────────
    decision = sweep.decide(maximin, cellpct)
    print(f"  規則A：{decision['verdict_a']}（百分位極差={decision['spread_pct']:.3f}, "
          f"原始Sharpe極差 中位={decision['raw_spread_med']:.3f}/最大={decision['raw_spread_max']:.3f}）")
    print(f"  規則B：{decision['verdict_b']}")

    # ── 4. regime 統計 + 窗口正規化 + buffer 交互 ──────────────────────────
    regime = sweep.regime_table(mkts)
    lev_norm = leverage_normalized_table(maximin)
    buf_inter = buffer_interaction_table(maximin)
    tx_duel = tx_200_vs_240(sweep_agg)

    # ── 5. 壓力窗口 H1/H2 ───────────────────────────────────────────────────
    print("== 壓力窗口 ==")
    stress_tbl, lev_paths = stress_mod.run_stress(mkts)
    hypo = stress_mod.test_hypotheses(stress_tbl)
    print(f"  H1(長均線×長陰跌)={hypo['H1'].get('hold')}, "
          f"H2(短均線×V反)={hypo['H2'].get('hold')}, 兩者皆成立={hypo['both_hold']}")

    # ── 6. bootstrap（buffer=0.01 的 10 MA × 4 市場）────────────────────────
    print("== bootstrap ==")
    boot_frames = []
    for i, k in enumerate(RANKING_MARKETS):
        t1 = time.time()
        b = bootstrap4.run_bootstrap_ma(mkts[k], seed=root_params_seed(i))
        boot_frames.append(b)
        print(f"  {k}: {time.time()-t1:.0f}s")
    boot = pd.concat(boot_frames, ignore_index=True)
    boot_order = bootstrap4.hist_vs_boot_ordering(maximin, boot)

    # ── 7. L_low=0.5 抽查 ───────────────────────────────────────────────────
    print("== L_low=0.5 抽查 ==")
    spot05 = spot_check_l05(mkts, calibs)
    # 方向一致性：與 L_low=1.0（buffer=0.01）的 mean_pct 排序比對
    base10 = maximin[(maximin["buffer"] == BOOT_BUFFER)][["ma", "mean_pct_rank"]] \
        .set_index("ma")["mean_pct_rank"]
    s05 = spot05.set_index("ma")["mean_pct_rank"]
    spearman = float(pd.Series(base10).corr(s05, method="spearman"))

    # ── 8. 圖表 ─────────────────────────────────────────────────────────────
    print("== 圖表 ==")
    grids = topography_grids(cellpct, maximin)
    charts4.topography_fig(grids, OUTPUT_DIR / "r4_topography.png")
    charts4.leverage_vs_window_fig(sweep_agg, OUTPUT_DIR / "r4_window_vs_leverage.png")
    charts4.switches_vs_window_fig(regime, OUTPUT_DIR / "r4_window_vs_switches.png")
    charts4.nikkei_window_fig(lev_paths, OUTPUT_DIR / "r4_nikkei1990_windows.png")

    # ── 9. CSV ──────────────────────────────────────────────────────────────
    var_agg = sweep_agg[sweep_agg["strategy"].str.startswith("D_MA")]
    for df, name in [(var_agg, "r4_ma_sweep_metrics"), (maximin, "r4_ma_maximin"),
                     (cellpct, "r4_ma_cellpct"), (stress_tbl, "r4_ma_stress_windows"),
                     (regime, "r4_ma_regime_stats"), (lev_norm, "r4_ma_leverage_norm"),
                     (buf_inter, "r4_ma_buffer_interaction"), (tx_duel, "r4_tx_200_vs_240"),
                     (boot, "r4_ma_bootstrap"), (boot_order, "r4_ma_hist_vs_boot"),
                     (spot05, "r4_ma_lowlow05_spotcheck"),
                     (decision["within_cell_range"], "r4_ma_within_cell_range"),
                     (decision["neighbor_table"], "r4_ma_ruleB_neighbors")]:
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    # ── 10. 快取 + 報告 ─────────────────────────────────────────────────────
    payload = {"sweep_agg": sweep_agg, "maximin": maximin, "cellpct": cellpct,
               "decision": decision, "regime": regime, "lev_norm": lev_norm,
               "buf_inter": buf_inter, "tx_duel": tx_duel, "stress": stress_tbl,
               "hypo": hypo, "boot": boot, "boot_order": boot_order,
               "spot05": spot05, "spearman05": spearman,
               "audit": {k: m.meta for k, m in mkts.items()}}
    with open(CACHE_DIR / "results4.pkl", "wb") as f:
        pickle.dump(payload, f)

    from backtest_ma_robustness import report4
    report4.write_report(payload, OUTPUT_DIR / "report.md")
    print(f"完成，全部 {time.time()-t0:.0f}s → {OUTPUT_DIR}")


def root_params_seed(i: int) -> int:
    from backtest_multimarket.params import BOOT_SEED
    return BOOT_SEED + 4000 + i


if __name__ == "__main__":
    main()
