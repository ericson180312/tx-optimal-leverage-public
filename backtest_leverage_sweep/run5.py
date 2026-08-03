"""第五輪主流程：資料 → L_high 掃描 → frontier/邊際 → 保證金/跌停 → 壓力窗口
→ bootstrap 歸零機率 → 圖表 → 報告。

執行：conda run -n ML_env python -m backtest_leverage_sweep.run5
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

import backtest_threshold.params as root_params
from backtest_threshold.continuous import build_continuous
from backtest_threshold.data import load_tx_contracts

from backtest_multimarket import evaluate as ev
from backtest_multimarket.data_mkts import load_markets
from backtest_multimarket.params import STRESS_WINDOWS

from backtest_leverage_sweep import bootstrap5, charts5, frontier, margin, sweep5
from backtest_leverage_sweep.params import (
    BOOT_SEED, CACHE_DIR, L_HIGH_GRID, MAINT_MARGIN_RATE, OUTPUT_DIR,
    RANKING_MARKETS, RUIN_VETO_PROB, variant_key,
)


def run_stress(mkts, *, maint: float = MAINT_MARGIN_RATE):
    """具名壓力窗口 × 各 L_high（規格 §2-4）。"""
    names = [variant_key(lh) for lh in L_HIGH_GRID]
    rows, paths = [], {}
    for label, mkt_keys, d0, d1 in STRESS_WINDOWS:
        keys = mkt_keys if mkt_keys else list(mkts)
        for k in keys:
            if k not in mkts:
                continue
            m = mkts[k]
            s = int(m.dates.searchsorted(pd.Timestamp(d0)))
            e = int(m.dates.searchsorted(pd.Timestamp(d1), side="right"))
            if e - s < 40 or (s == 0 and m.dates[0] > pd.Timestamp(d0) + pd.Timedelta(days=370)):
                continue
            L_exec = sweep5.build_variant_signals(m.spot)
            starts = np.array([s], dtype=np.int64)
            metrics, sims = sweep5.run_windows_margin(m, L_exec, starts, e - s,
                                                      names, maint=maint,
                                                      keep_sims=True)
            for lh in L_HIGH_GRID:
                v = variant_key(lh)
                r = metrics[v].iloc[0]
                rows.append({
                    "window": label, "market": k, "l_high": lh,
                    "start": str(m.dates[s].date()), "end": str(m.dates[e - 1].date()),
                    "cumret": float(r["cumret"]), "mdd": float(r["mdd"]),
                    "avg_lev": float(r["avg_lev"]),
                    "lev_at_trough": float(r["lev_at_trough"]),
                    "liquidated": bool(r["liquidated"]),
                    "bankrupt": bool(r["bankrupt"]),
                })
                paths[(label, k, lh)] = (m.dates[s:e], sims[v].eq[:, 0], sims[v].lev[:, 0])
    return pd.DataFrame(rows), paths


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    charts5.setup_style()

    print("== 載入市場資料 ==")
    tx_cs = build_continuous(load_tx_contracts(), root_params.ROLL_OFFSET_DAYS)
    mkts = load_markets(tx_cs)
    for k, m in mkts.items():
        print(f"  {k:7s} {m.meta['start']} ~ {m.meta['end']}  {m.T} 日")

    # ── 1. 保證金／跌停解析表（規格 §3-1、§3-2）─────────────────────────────
    marg_tbl = margin.margin_table()
    marg_sens = margin.maint_rate_sensitivity()
    print("== 保證金臨界跌幅 ==")
    for _, r in marg_tbl.iterrows():
        print(f"  L={r['l_high']:.1f}: x_liq(近似)={r['x_liq_approx']:.1%} "
              f"x_liq(精確)={r['x_liq_exact']:.1%} 跌停可承受={int(r['limit_down_days_exact'])}日")

    # ── 2. 主掃描（存活者口徑）──────────────────────────────────────────────
    print("== L_high 掃描 ==")
    t1 = time.time()
    sweep = sweep5.run_sweep(mkts)
    print(f"  {time.time() - t1:.0f}s")
    sweep_scaled = sweep5.run_sweep(mkts, scaled=True, markets=RANKING_MARKETS)
    inception, curves = sweep5.run_inception(mkts)
    lev_real = sweep5.leverage_realized_table(sweep[sweep["market"].isin(RANKING_MARKETS)])

    # ── 3. frontier / 邊際 / 效率 / 凸性 ────────────────────────────────────
    front_all = frontier.frontier_points(sweep)
    front_mkt = frontier.frontier_points(sweep, by_market=True)
    bend = frontier.detect_backward_bend(front_all)
    marg = frontier.marginal_table(front_all)
    eff = frontier.efficiency_table(front_all)
    conv = frontier.mdd_convexity(front_all)
    print(f"  CAGR 2.0→2.5 變化 {bend['cagr_total_change']:+.4f}；"
          f"MDD 惡化 {bend['mdd_total_change']:+.4f}；彎折點 {bend['bend_at']}")

    # ── 4. 壓力窗口 ─────────────────────────────────────────────────────────
    print("== 壓力窗口 ==")
    stress, stress_paths = run_stress(mkts)

    # ── 5. bootstrap 歸零機率（本輪最重要的單一數字）────────────────────────
    print("== bootstrap ==")
    boots = []
    for i, k in enumerate(RANKING_MARKETS):
        t1 = time.time()
        boots.append(bootstrap5.run_bootstrap_leverage(mkts[k], seed=BOOT_SEED + i))
        print(f"  {k}: {time.time() - t1:.0f}s")
    boot = pd.concat(boots, ignore_index=True)
    ruin = bootstrap5.ruin_summary(boot, RUIN_VETO_PROB)
    print(ruin[["l_high", "max_ruin_prob", "disqualified"]].to_string(index=False))

    # ── 6. 圖表 ─────────────────────────────────────────────────────────────
    print("== 圖表 ==")
    charts5.frontier_fig(front_all, front_mkt, OUTPUT_DIR / "r5_frontier.png")
    charts5.three_curves_fig(front_all, front_mkt, OUTPUT_DIR / "r5_three_curves.png")
    charts5.convexity_fig(marg, eff, OUTPUT_DIR / "r5_mdd_convexity.png")
    charts5.nikkei_fig(stress_paths, OUTPUT_DIR / "r5_nikkei1990_leverage.png")

    # ── 7. CSV ──────────────────────────────────────────────────────────────
    for df, name in [(sweep, "r5_leverage_sweep_metrics"), (front_all, "r5_frontier"),
                     (front_mkt, "r5_frontier_by_market"),
                     (marg_tbl, "r5_margin_survival"),
                     (marg_sens, "r5_margin_rate_sensitivity"),
                     (boot, "r5_bootstrap_ruin"), (ruin, "r5_ruin_summary"),
                     (marg, "r5_marginal_tradeoff"), (eff, "r5_efficiency"),
                     (conv, "r5_mdd_convexity"), (stress, "r5_stress_windows"),
                     (inception, "r5_inception"), (lev_real, "r5_realized_leverage"),
                     (sweep_scaled, "r5_scaled_lowlow_check")]:
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    payload = {"sweep": sweep, "sweep_scaled": sweep_scaled, "inception": inception,
               "front_all": front_all, "front_mkt": front_mkt, "bend": bend,
               "marg": marg, "eff": eff, "conv": conv, "lev_real": lev_real,
               "marg_tbl": marg_tbl, "marg_sens": marg_sens,
               "stress": stress, "boot": boot, "ruin": ruin,
               "audit": {k: m.meta for k, m in mkts.items()}}
    with open(CACHE_DIR / "results5.pkl", "wb") as f:
        pickle.dump(payload, f)

    from backtest_leverage_sweep import report5
    report5.write_report(payload, OUTPUT_DIR / "report.md")
    print(f"完成，全部 {time.time() - t0:.0f}s → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
