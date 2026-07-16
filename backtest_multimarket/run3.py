"""第三輪主流程：資料 → 槓桿中性化校準 → 歷史評估 → 壓力窗口 → bootstrap
→ 圖表 → 報告。

執行：conda run -n ML_env python -m backtest_multimarket.run3
輸出：backtest_multimarket/output/ 下的 CSV、PNG 與 report.md；中間結果 backtest_multimarket/output/cache/。
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from backtest_threshold import params as root_params
from backtest_threshold.continuous import build_continuous
from backtest_threshold.data import load_tx_contracts

from backtest_multimarket import charts3, evaluate as ev, strategies as st
from backtest_multimarket.bootstrap import run_bootstrap
from backtest_multimarket.data_mkts import load_markets
from backtest_multimarket.params import (
    BOOT_BLOCK_SENS, BOOT_SEED, BOOT_SENS_PATHS, CACHE_DIR, CALIB_TARGET_LEV,
    CALIB_TOL, IDX_COST_PER_SIDE_SENS, MARKETS, OUTPUT_DIR, RANKING_MARKETS,
    SENS_STRATEGIES, STRATEGIES, TRADING_DAYS,
)

ALL_STRATS = STRATEGIES + SENS_STRATEGIES


def calibration_table(mkts, calibs, inception: pd.DataFrame,
                      agg_all: pd.DataFrame) -> pd.DataFrame:
    """§3 校準結果表：兩套 σ_target 與各口徑下的實際平均槓桿。

    - sigma_wf / rolling_avg_lev：walk-forward（最早10年）校準 → 滾動視窗實際值
      （評估段的槓桿漂移由此欄檢驗）。
    - sigma_is / inception_avg_lev：全樣本 in-sample 校準 → 自始以來路徑實際值
      （±0.1 檢驗以此欄為準，§3-3）。
    """
    rows = []
    real = inception.set_index(["market", "strategy"])["avg_lev"]
    roll = agg_all[agg_all["period_years"] == 10] \
        .set_index(["market", "strategy"])["med_avg_lev"]
    for k, cal in calibs.items():
        yrs = cal["calib_days"] / TRADING_DAYS
        for strat in ALL_STRATS:
            calibrated = strat in cal["sigma"] and strat != "F2_VolTrend"
            r_is = float(real.get((k, strat), np.nan))
            r_wf = float(roll.get((k, strat), np.nan))
            rows.append({
                "market": k, "strategy": strat,
                "sigma_wf": cal["sigma"].get(strat, np.nan),
                "sigma_is": cal["sigma_is"].get(strat, np.nan),
                "calib_years": round(yrs, 1),
                "calib_signal_avg_lev": cal["achieved"].get(strat, np.nan),
                "rolling_avg_lev_10y": r_wf,
                "inception_avg_lev": r_is,
                "within_tol": (abs(r_is - CALIB_TARGET_LEV) <= CALIB_TOL)
                              if r_is == r_is else None,
                "calibrated": calibrated,
                "note": ("walk-forward＋全樣本兩套（§3-3）" if calibrated else
                         ("σ 沿用 C_EWMA（組合定義，§3-4 照實跑）"
                          if strat == "F2_VolTrend" else "無校準變數（照實跑，§3-4）")),
            })
    return pd.DataFrame(rows)


def carry_check(mkts, L_execs) -> pd.DataFrame:
    """TX 期貨管線 vs 指數＋融資模型：共同期間逐策略對照（= carry＋成本模型差異）。"""
    fut, idx_m = mkts["TX_FUT"], mkts["TX_IDX"]
    d0 = max(fut.dates[0], idx_m.dates[0])
    d1 = min(fut.dates[-1], idx_m.dates[-1])
    rows = []
    res = {}
    for key in ["TX_FUT", "TX_IDX"]:
        mkt = mkts[key]
        s = int(mkt.dates.searchsorted(d0))
        e = int(mkt.dates.searchsorted(d1, side="right"))
        starts = np.array([s], dtype=np.int64)
        metrics, _ = ev.run_windows(mkt, L_execs[key], starts, e - s, STRATEGIES)
        res[key] = {n: m.iloc[0] for n, m in metrics.items()}
    for n in STRATEGIES:
        f, i = res["TX_FUT"][n], res["TX_IDX"][n]
        rows.append({"strategy": n, "start": str(d0.date()), "end": str(d1.date()),
                     "cagr_fut": f["cagr"], "cagr_idx": i["cagr"],
                     "cagr_diff": f["cagr"] - i["cagr"],
                     "sharpe_fut": f["sharpe"], "sharpe_idx": i["sharpe"],
                     "mdd_fut": f["mdd"], "mdd_idx": i["mdd"],
                     "avg_lev_fut": f["avg_lev"], "avg_lev_idx": i["avg_lev"],
                     "cost_ratio_fut": f["cost_ratio"], "cost_ratio_idx": i["cost_ratio"]})
    return pd.DataFrame(rows)


def hist_vs_boot(agg_all: pd.DataFrame, boot_dists: dict) -> pd.DataFrame:
    """歷史 20 年滾動 vs bootstrap 的中位數 Sharpe 排名對照（1=最好）。"""
    rows = []
    for mk in RANKING_MARKETS:
        h = agg_all[(agg_all["market"] == mk) & (agg_all["period_years"] == 20)
                    & agg_all["strategy"].isin(STRATEGIES)]
        if h.empty or mk not in boot_dists:
            continue
        h = h.set_index("strategy")["med_sharpe"].rank(ascending=False)
        b = boot_dists[mk].set_index("strategy")["sharpe_p50"].rank(ascending=False)
        for s_name in STRATEGIES:
            rows.append({"market": mk, "strategy": s_name,
                         "hist_rank": float(h[s_name]), "boot_rank": float(b[s_name]),
                         "rank_gap": float(b[s_name] - h[s_name])})
    return pd.DataFrame(rows)


def main() -> None:
    t_all = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    charts3.setup_style()

    # ── 1. 資料 ────────────────────────────────────────────────────────────
    print("== 載入市場資料 ==")
    tx_cs = build_continuous(load_tx_contracts(), root_params.ROLL_OFFSET_DAYS)
    mkts = load_markets(tx_cs)
    audit = {k: m.meta for k, m in mkts.items()}
    for k, m in mkts.items():
        print(f"  {k:7s} {m.meta['start']} ~ {m.meta['end']}  {m.T:6d} 日  "
              f"{m.meta['tr_source']}")

    # ── 2. 槓桿中性化校準（walk-forward：最早 10 年）───────────────────────
    print("== σ_target 校準 ==")
    calibs = {k: st.calibrate_market(m.ret, m.spot) for k, m in mkts.items()}
    L_execs = {k: st.exec_signals(calibs[k], "wf") for k in mkts}      # 滾動/壓力窗口
    L_execs_is = {k: st.exec_signals(calibs[k], "is") for k in mkts}   # 自始以來/carry
    for k, c in calibs.items():
        print(f"  {k}: σ_wf(C_EWMA)={c['sigma']['C_EWMA']:.4f}, "
              f"σ_is={c['sigma_is']['C_EWMA']:.4f}")

    # ── 3. 滾動多路徑（全部市場 × 期間）────────────────────────────────────
    print("== 歷史滾動視窗 ==")
    aggs, aggs_eval = [], []
    for k, m in mkts.items():
        t0 = time.time()
        n_cal = calibs[k]["calib_days"]
        a, a_ev = ev.run_market_all_periods(m, L_execs[k], n_cal, ALL_STRATS)
        aggs.append(a)
        if not a_ev.empty:
            aggs_eval.append(a_ev)
        print(f"  {k}: {time.time() - t0:.0f}s")
    agg_all = pd.concat(aggs, ignore_index=True)
    agg_eval = pd.concat(aggs_eval, ignore_index=True) if aggs_eval else pd.DataFrame()

    # ── 4. 自始以來單路徑 + 資金曲線 ───────────────────────────────────────
    print("== 自始以來路徑（σ_target 全樣本 in-sample 版）==")
    inception_rows, curves_all = [], {}
    for k, m in mkts.items():
        tbl, curves = ev.run_full_path(m, L_execs_is[k], ALL_STRATS)
        inception_rows.append(tbl)
        curves_all[k] = curves
        df = pd.DataFrame(index=m.dates)
        for n, (eqc, levc) in curves.items():
            df[f"eq_{n}"] = eqc
            df[f"lev_{n}"] = levc
        df.index.name = "date"
        df.to_csv(OUTPUT_DIR / f"r3_equity_{k}.csv", encoding="utf-8-sig")
    inception = pd.concat(inception_rows, ignore_index=True)

    calib_tbl = calibration_table(mkts, calibs, inception, agg_all)

    # ── 5. TX carry 校準參照 ───────────────────────────────────────────────
    carry = carry_check(mkts, L_execs_is)

    # ── 6. 壓力窗口 ────────────────────────────────────────────────────────
    print("== 壓力窗口 ==")
    stress, stress_paths = ev.run_stress(mkts, L_execs, STRATEGIES)

    # ── 7. 評比：maximin、報酬排名、retention ──────────────────────────────
    maximin = ev.maximin_ranking(agg_all, "med_sharpe")
    maximin_eval = (ev.maximin_ranking(agg_eval, "med_sharpe")
                    if not agg_eval.empty else pd.DataFrame())
    ret_rank = ev.maximin_ranking(agg_all, "med_cagr")
    retention = ev.retention_table(agg_all)

    # ── 8. 成本敏感度：海外調整成本 10bp ───────────────────────────────────
    print("== 成本敏感度（10bp）==")
    aggs_c = []
    for k, m in mkts.items():
        if m.mode == "futures":
            continue
        a, _ = ev.run_market_all_periods(m, L_execs[k], calibs[k]["calib_days"],
                                         STRATEGIES, cost_rate=IDX_COST_PER_SIDE_SENS)
        aggs_c.append(a)
    agg_cost = pd.concat(aggs_c, ignore_index=True)
    cost_mkts = [m for m in RANKING_MARKETS if mkts[m].mode != "futures"]
    mm_bases = ev.maximin_ranking(agg_all[agg_all["market"].isin(cost_mkts)],
                                  "med_sharpe") if cost_mkts else pd.DataFrame()
    mm_cost = ev.maximin_ranking(agg_cost[agg_cost["market"].isin(cost_mkts)],
                                 "med_sharpe") if cost_mkts else pd.DataFrame()

    # ── 9. Bootstrap ───────────────────────────────────────────────────────
    print("== Bootstrap（1,000 條 × 20 年 × 4 市場）==")
    boot, boot_sens = {}, []
    for i, k in enumerate(RANKING_MARKETS):
        t0 = time.time()
        boot[k] = run_bootstrap(mkts[k], calibs[k], STRATEGIES, seed=BOOT_SEED + i)
        boot[k]["dist"].to_csv(OUTPUT_DIR / f"r3_bootstrap_dist_{k}.csv",
                               index=False, encoding="utf-8-sig")
        long = pd.concat([d.assign(strategy=n) for n, d in boot[k]["per_path"].items()],
                         ignore_index=True)
        long.insert(0, "market", k)
        long.to_csv(OUTPUT_DIR / f"r3_bootstrap_paths_{k}.csv",
                    index=False, encoding="utf-8-sig")
        print(f"  {k} main: {time.time() - t0:.0f}s")
        for blk in BOOT_BLOCK_SENS:
            r = run_bootstrap(mkts[k], calibs[k], STRATEGIES,
                              n_paths=BOOT_SENS_PATHS, mean_block=blk,
                              seed=BOOT_SEED + 100 * blk + i)
            boot_sens.append(r["dist"])
    boot_dist = pd.concat([b["dist"] for b in boot.values()], ignore_index=True)
    boot_sens = pd.concat(boot_sens, ignore_index=True)
    rank_cmp = hist_vs_boot(agg_all, {k: b["dist"] for k, b in boot.items()})

    # ── 10. 圖表 ───────────────────────────────────────────────────────────
    print("== 圖表 ==")
    for k, m in mkts.items():
        charts3.equity_overlay_fig(m.dates, curves_all[k], m.label,
                                   OUTPUT_DIR / f"r3_equity_{k}.png")
    charts3.nikkei_lev_fig(stress_paths, OUTPUT_DIR / "r3_nikkei1990_leverage.png")
    for k in RANKING_MARKETS:
        charts3.violin_fig(boot[k]["per_path"], mkts[k].label, STRATEGIES,
                           OUTPUT_DIR / f"r3_violin_{k}.png")
    charts3.rank_scatter_fig(rank_cmp, OUTPUT_DIR / "r3_rank_scatter.png")

    # ── 11. CSV ────────────────────────────────────────────────────────────
    agg_all.to_csv(OUTPUT_DIR / "r3_market_period_metrics.csv", index=False,
                   encoding="utf-8-sig")
    if not agg_eval.empty:
        agg_eval.to_csv(OUTPUT_DIR / "r3_market_period_metrics_evalonly.csv",
                        index=False, encoding="utf-8-sig")
    for df, name in [(maximin, "r3_maximin"), (ret_rank, "r3_return_ranking"),
                     (calib_tbl, "r3_calibration"), (stress, "r3_stress_windows"),
                     (carry, "r3_tx_carry_check"), (retention, "r3_retention"),
                     (inception, "r3_inception"), (boot_dist, "r3_bootstrap_dist"),
                     (boot_sens, "r3_bootstrap_block_sens"),
                     (rank_cmp, "r3_hist_vs_boot_rank")]:
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    # ── 12. 快取（報告產生器使用）──────────────────────────────────────────
    payload = {
        "audit": audit, "calib_tbl": calib_tbl,
        "sigma": {k: calibs[k]["sigma"] for k in mkts},
        "agg_all": agg_all, "agg_eval": agg_eval,
        "inception": inception, "carry": carry,
        "stress": stress, "maximin": maximin, "maximin_eval": maximin_eval,
        "ret_rank": ret_rank, "retention": retention,
        "mm_cost_base": mm_bases, "mm_cost_10bp": mm_cost,
        "boot_dist": boot_dist, "boot_sens": boot_sens, "rank_cmp": rank_cmp,
    }
    with open(CACHE_DIR / "results3.pkl", "wb") as f:
        pickle.dump(payload, f)

    # ── 13. 報告 ───────────────────────────────────────────────────────────
    from backtest_multimarket import report3
    report3.write_report(payload, OUTPUT_DIR / "report.md")
    print(f"完成，全部 {time.time() - t_all:.0f}s → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
