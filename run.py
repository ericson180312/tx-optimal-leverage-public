"""主流程：資料 → 連續合約（互驗）→ 網格回測 → 敏感度 → 圖表 → 報告。

執行：conda run -n ML_env python run.py
輸出：output/ 下的 CSV、PNG 與 report.md；中間結果 pickle 於 output/cache/。
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from params import (
    BENCHMARKS, INITIAL_EQUITY, LOOKBACK_YEARS, OUTPUT_DIR, CACHE_DIR,
    PLOT_BENCHMARKS, REBAL_FREQ_DAYS, REBAL_PHASE_SENS, ROLL_OFFSET_DAYS,
    ROLL_OFFSET_DAYS_SENS, SLIPPAGE_TICKS, SLIPPAGE_TICKS_SENS,
    TARGET_LEVERAGE, X_GRID, Y_GRID,
)
from data import load_tx_contracts
from continuous import build_continuous
from engine import simulate
from grid import neighborhood_stats, run_period, select_best
from metrics import path_metrics
import charts


def _combo_idx(agg: pd.DataFrame, x: float, y: float) -> int:
    """網格組合在 agg / SimResult 欄位中的位置。"""
    m = agg[(~agg["is_benchmark"]) & (agg["X"] == x) & (agg["Y"] == y)]
    return int(m.index[0])


def _bench_idx(agg: pd.DataFrame, name: str) -> int:
    return int(agg[agg["name"] == name].index[0])


def _equity_curves_csv(pr, sel, path):
    """基準路徑上：兩個最佳解 + 各基準的權益與槓桿。"""
    agg, sim = pr.agg, pr.baseline_sim
    i_s = _combo_idx(agg, sel["med_by_sharpe"]["X"], sel["med_by_sharpe"]["Y"])
    i_r = _combo_idx(agg, sel["med_by_return"]["X"], sel["med_by_return"]["Y"])
    data = {
        "eq_best_sharpe": sim.eq[:, i_s],
        "eq_best_return": sim.eq[:, i_r],
        "lev_best_sharpe": sim.lev[:, i_s],
        "lev_best_return": sim.lev[:, i_r],
    }
    for name in BENCHMARKS:
        data[f"eq_{name}"] = sim.eq[:, _bench_idx(agg, name)]
    data["lev_Periodic5d_2x"] = sim.lev[:, _bench_idx(agg, "Periodic5d_2x")]
    df = pd.DataFrame(index=pr.baseline_index, data=data)
    df.index.name = "date"
    df.to_csv(path, encoding="utf-8-sig")
    return df, i_s, i_r


def _freq_table(results: dict) -> pd.DataFrame:
    """再平衡頻率掃描：BH（∞）→ 季 → 月 → 雙週 → 5日 → 每日。"""
    order = [("BH_2x", "永不（買進持有）"), ("Periodic63d_2x", "每 63 日（季）"),
             ("Periodic21d_2x", "每 21 日（月）"), ("Periodic10d_2x", "每 10 日（雙週）"),
             ("Periodic5d_2x", "每 5 日 ★新增"), ("Daily_2x", "每 1 日（正2倍ETF）")]
    rows = []
    for name, label in order:
        for y in LOOKBACK_YEARS:
            agg = results[y].agg if hasattr(results[y], "agg") else results[y]["agg"]
            r = agg[agg["name"] == name].iloc[0]
            rows.append({
                "strategy": label, "name": name, "period_years": y,
                "med_cumret": r["med_cumret"], "med_cagr": r["med_cagr"],
                "med_sharpe": r["med_sharpe"], "med_mdd": r["med_mdd"],
                "worst_cumret": r["worst_cumret"],
                "n_bankrupt_paths": int(r["n_bankrupt_paths"]),
                "n_paths": int(r["n_paths"]),
                "base_cumret": r["base_cumret"], "base_cagr": r["base_cagr"],
                "base_mdd": r["base_mdd"], "base_sharpe": r["base_sharpe"],
                "base_uw_months": r["base_uw_months"],
                "base_n_rebal": int(r["base_n_up"] + r["base_n_down"]),
                "base_cost_ratio": r["base_cost_ratio"],
                "base_avg_lev": r["base_avg_lev"],
            })
    return pd.DataFrame(rows)


def _phase_sensitivity(cs, years: int) -> pd.DataFrame:
    """5 日再平衡的相位敏感度：建倉日落在週期第 0..4 天，跑同一條基準路徑。

    滾動路徑本已隨機化相位（每條路徑各自從建倉日起算），此表用來量化
    「單一路徑上換個起跑日，結果會差多少」。
    """
    panel = cs.panel
    idx = panel.index
    s = int(idx.searchsorted(idx[-1] - pd.DateOffset(years=years)))
    sl = slice(s, len(idx))
    n = len(REBAL_PHASE_SENS)
    res = simulate(
        panel["ret"].to_numpy()[sl], panel["price"].to_numpy()[sl],
        panel["is_roll"].to_numpy()[sl], panel["p_near"].to_numpy()[sl],
        panel["p_next"].to_numpy()[sl],
        X=np.full(n, np.inf), Y=np.full(n, -np.inf),
        target=np.full(n, TARGET_LEVERAGE),
        entry_price=float(panel["price"].to_numpy()[s - 1]),
        rebal_every=np.full(n, 5, dtype=np.int64),
        rebal_phase=np.array(REBAL_PHASE_SENS, dtype=np.int64),
    )
    m = path_metrics(res)
    m.insert(0, "phase", REBAL_PHASE_SENS)
    m.insert(0, "period_years", years)
    return m


def _rebalance_log(pr, x: float, y: float) -> pd.DataFrame:
    """基準路徑上某組 [X,Y] 的再平衡日誌。"""
    agg, sim = pr.agg, pr.baseline_sim
    i = _combo_idx(agg, x, y)
    t = np.flatnonzero(sim.trig[:, i] != 0)
    return pd.DataFrame({
        "period_years": pr.years,
        "date": pr.baseline_index[t],
        "X": x, "Y": y,
        "action": np.where(sim.trig[t, i] > 0, "加碼", "減碼"),
        "lev_trigger": sim.lev_pre[t, i],       # 觸價槓桿（結算/轉倉後、再平衡前）
        "delta_notional": sim.dN[t, i],         # 調整名目金額
        "day_cost": sim.cost[t, i],             # 當日總成本（含同日轉倉）
        "equity_after": sim.eq[t, i],
        "lev_after": sim.lev[t, i],
    })


def _sens_compare(results_main: dict, results_alt: dict, label: str) -> pd.DataFrame:
    """敏感度比較：各期間最佳解是否移動、主最佳組在替代情境的指標。"""
    rows = []
    for y in LOOKBACK_YEARS:
        sel_m = select_best(results_main[y].agg)
        sel_a = select_best(results_alt[y].agg)
        bm, ba = sel_m["med_by_sharpe"], sel_a["med_by_sharpe"]
        alt_agg = results_alt[y].agg
        same_row = alt_agg.iloc[_combo_idx(alt_agg, bm["X"], bm["Y"])]
        rows.append({
            "period_years": y, "scenario": label,
            "best_main": f"[{bm['X']:g}, {bm['Y']:g}]",
            "best_alt": f"[{ba['X']:g}, {ba['Y']:g}]",
            "same_cell": (bm["X"] == ba["X"]) and (bm["Y"] == ba["Y"]),
            "med_sharpe_main": bm["med_sharpe"],
            "med_sharpe_alt_bestalt": ba["med_sharpe"],
            "med_sharpe_alt_mainbest": same_row["med_sharpe"],
            "med_cumret_alt_mainbest": same_row["med_cumret"],
        })
    return pd.DataFrame(rows)


def _refine_search(cs, years: int, x0: float, y0: float) -> dict:
    """二次精細搜尋（步長 0.05）：以粗網格 Sharpe 最佳解為中心的鄰域。"""
    xs = [round(v, 2) for v in np.arange(max(2.05, x0 - 0.25), x0 + 0.251, 0.05)]
    ys = [round(v, 2) for v in np.arange(max(0.8, y0 - 0.15), min(1.951, y0 + 0.151), 0.05)]
    pr = run_period(cs, years, x_grid=xs, y_grid=ys)
    sel = select_best(pr.agg)["med_by_sharpe"]
    return {"period_years": years, "coarse": f"[{x0:g}, {y0:g}]",
            "refined": f"[{sel['X']:g}, {sel['Y']:g}]",
            "coarse_med_sharpe": None,   # 主表已有，報告端補
            "refined_med_sharpe": sel["med_sharpe"]}


def _robust_pick(results: dict) -> pd.DataFrame:
    """跨期間穩健組合：med_sharpe 百分位排名的「跨期間最小值」最大者（maximin）。

    只考慮在所有期間皆無任何破產路徑的組合。
    """
    ranks = {}
    for y, pr in results.items():
        g = pr.agg[~pr.agg["is_benchmark"]].copy()
        g["rank"] = g["med_sharpe"].rank(pct=True)
        ranks[y] = g.set_index(["X", "Y"])[["rank", "any_bankrupt"]]
    keys = ranks[LOOKBACK_YEARS[0]].index
    rows = []
    for xy in keys:
        rs = [ranks[y].loc[xy, "rank"] for y in LOOKBACK_YEARS]
        bk = any(ranks[y].loc[xy, "any_bankrupt"] for y in LOOKBACK_YEARS)
        if not bk:
            rows.append({"X": xy[0], "Y": xy[1], "min_rank": min(rs),
                         "mean_rank": float(np.mean(rs))})
    df = pd.DataFrame(rows).sort_values(["min_rank", "mean_rank"], ascending=False)
    return df.reset_index(drop=True)


def main() -> None:
    t_all = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    charts.setup_style()

    # ── 1. 資料與連續合約（含兩法互驗）─────────────────────────────────────
    print("== 載入 TX 逐合約資料 ==")
    cdf = load_tx_contracts()
    cs = build_continuous(cdf, ROLL_OFFSET_DAYS)
    ec = cs.expiry_check
    audit = {
        "data_start": str(cs.panel.index[0].date()),
        "data_end": str(cs.panel.index[-1].date()),
        "n_days": len(cs.panel),
        "n_rolls": len(cs.rolls),
        "method_max_diff": cs.method_max_diff,
        "n_price_gaps": cs.n_price_gaps,
        "expiry_match_third_wed": float((ec["diff_days"] == 0).mean()),
        "expiry_shift_counts": ec["diff_days"].value_counts().to_dict(),
        "naive_annual_inflation": cs.naive_annual_inflation,
        "spread_pct_mean": float(cs.rolls["spread_pct"].mean()),
        "spread_pct_median": float(cs.rolls["spread_pct"].median()),
    }
    print(f"  {audit['data_start']} ~ {audit['data_end']}, {audit['n_days']} 日, "
          f"{audit['n_rolls']} 次轉倉, 互驗 max diff = {audit['method_max_diff']:.2e}")
    cs.rolls.to_csv(OUTPUT_DIR / "roll_events.csv", index=False, encoding="utf-8-sig")

    # ── 2. 主網格回測（各期間）────────────────────────────────────────────
    results: dict[int, object] = {}
    selections: dict[int, dict] = {}
    for y in LOOKBACK_YEARS:
        t0 = time.time()
        pr = run_period(cs, y)
        results[y] = pr
        selections[y] = select_best(pr.agg)
        pr.agg.to_csv(OUTPUT_DIR / f"grid_results_{y}y.csv", index=False,
                      encoding="utf-8-sig")
        b = selections[y]["med_by_sharpe"]
        print(f"  {y:>2}y: {pr.n_paths} 條路徑 {time.time()-t0:.1f}s | "
              f"Sharpe最佳 [X={b['X']:g}, Y={b['Y']:g}] med_sharpe={b['med_sharpe']:.3f}")

    # ── 3. 基準路徑輸出：資金曲線 CSV、再平衡日誌、圖表 ────────────────────
    logs = []
    for y in LOOKBACK_YEARS:
        pr, sel = results[y], selections[y]
        _equity_curves_csv(pr, sel, OUTPUT_DIR / f"equity_curve_best_{y}y.csv")
        logs.append(_rebalance_log(pr, sel["med_by_sharpe"]["X"], sel["med_by_sharpe"]["Y"]))

        bs, br = sel["med_by_sharpe"], sel["med_by_return"]
        charts.heatmap_fig(pr.agg, y, (bs["X"], bs["Y"]), (br["X"], br["Y"]),
                           OUTPUT_DIR / f"heatmap_{y}y.png")
        agg, sim = pr.agg, pr.baseline_sim
        curves = {
            "best_sharpe": sim.eq[:, _combo_idx(agg, bs["X"], bs["Y"])],
            "best_ret": sim.eq[:, _combo_idx(agg, br["X"], br["Y"])],
        }
        curves.update({name: sim.eq[:, _bench_idx(agg, name)] for name in PLOT_BENCHMARKS})
        charts.equity_overlay_fig(pr.baseline_index, curves, y,
                                  OUTPUT_DIR / f"equity_overlay_{y}y.png", INITIAL_EQUITY)
        charts.leverage_fig(pr.baseline_index,
                            sim.lev[:, _combo_idx(agg, bs["X"], bs["Y"])],
                            bs["X"], bs["Y"], TARGET_LEVERAGE, y,
                            f"X={bs['X']:g}, Y={bs['Y']:g}",
                            OUTPUT_DIR / f"leverage_best_{y}y.png")
    pd.concat(logs, ignore_index=True).to_csv(
        OUTPUT_DIR / "rebalance_log_best.csv", index=False, encoding="utf-8-sig")

    # ── 4. 平原/尖峰、對稱性、穩健交叉表、跨期間穩健組合 ───────────────────
    plateau = {}
    for y in LOOKBACK_YEARS:
        bs = selections[y]["med_by_sharpe"]; br = selections[y]["med_by_return"]
        plateau[y] = {
            "sharpe": neighborhood_stats(results[y].agg, "med_sharpe", bs["X"], bs["Y"]),
            "return": neighborhood_stats(results[y].agg, "med_cumret", br["X"], br["Y"]),
        }

    b20 = selections[20]["med_by_sharpe"]
    b20r = selections[20]["med_by_return"]
    cross_rows = []
    for y in LOOKBACK_YEARS:
        agg = results[y].agg
        for tag, b in [("20y_Sharpe最佳", b20), ("20y_報酬最佳", b20r)]:
            row = agg.iloc[_combo_idx(agg, b["X"], b["Y"])]
            elig = agg[~agg["is_benchmark"] & ~agg["any_bankrupt"]]
            cross_rows.append({
                "combo": f"[{b['X']:g}, {b['Y']:g}]（{tag}）", "period_years": y,
                "med_cumret": row["med_cumret"], "med_sharpe": row["med_sharpe"],
                "worst_cumret": row["worst_cumret"],
                "win_vs_daily2x": row["win_vs_daily2x"],
                "sharpe_rank_pct": float((elig["med_sharpe"] <= row["med_sharpe"]).mean()),
            })
    crosstab = pd.DataFrame(cross_rows)
    crosstab.to_csv(OUTPUT_DIR / "crosstab_20y_best.csv", index=False, encoding="utf-8-sig")

    robust = _robust_pick(results)
    robust.to_csv(OUTPUT_DIR / "robust_ranking.csv", index=False, encoding="utf-8-sig")

    # ── 4b. 再平衡頻率掃描 + 5 日再平衡的相位敏感度 ────────────────────────
    freq = _freq_table(results)
    freq.to_csv(OUTPUT_DIR / "rebal_frequency.csv", index=False, encoding="utf-8-sig")
    phase = pd.concat([_phase_sensitivity(cs, y) for y in LOOKBACK_YEARS],
                      ignore_index=True)
    phase.to_csv(OUTPUT_DIR / "rebal_phase_sens.csv", index=False, encoding="utf-8-sig")
    charts.freq_fig(freq, OUTPUT_DIR / "rebal_frequency.png")

    # ── 5. 敏感度：滑價 0.5 檔、轉倉 offset 0 ──────────────────────────────
    print("== 敏感度：滑價 0.5 檔 ==")
    results_slip = {y: run_period(cs, y, slippage_ticks=SLIPPAGE_TICKS_SENS)
                    for y in LOOKBACK_YEARS}
    sens_slip = _sens_compare(results, results_slip, f"slippage={SLIPPAGE_TICKS_SENS}")
    sens_slip.to_csv(OUTPUT_DIR / "sens_slippage.csv", index=False, encoding="utf-8-sig")

    print("== 敏感度：轉倉 offset 0 ==")
    cs0 = build_continuous(cdf, ROLL_OFFSET_DAYS_SENS)
    results_off0 = {y: run_period(cs0, y) for y in LOOKBACK_YEARS}
    sens_off = _sens_compare(results, results_off0, f"roll_offset={ROLL_OFFSET_DAYS_SENS}")
    sens_off.to_csv(OUTPUT_DIR / "sens_rolloffset.csv", index=False, encoding="utf-8-sig")

    # ── 6. 二次精細搜尋（步長 0.05，以各期間 Sharpe 最佳為中心）────────────
    print("== 二次精細搜尋 ==")
    refine = []
    for y in LOOKBACK_YEARS:
        bs = selections[y]["med_by_sharpe"]
        r = _refine_search(cs, y, float(bs["X"]), float(bs["Y"]))
        r["coarse_med_sharpe"] = float(bs["med_sharpe"])
        refine.append(r)
    refine = pd.DataFrame(refine)
    refine.to_csv(OUTPUT_DIR / "refine_search.csv", index=False, encoding="utf-8-sig")

    # ── 7. 快取（報告產生器使用）──────────────────────────────────────────
    payload = {
        "audit": audit,
        "results": {y: {"agg": results[y].agg, "n_paths": results[y].n_paths,
                        "path_starts": results[y].path_starts}
                    for y in LOOKBACK_YEARS},
        "selections": selections,
        "plateau": plateau,
        "crosstab": crosstab,
        "robust": robust,
        "freq": freq,
        "phase": phase,
        "sens_slip": sens_slip,
        "sens_off": sens_off,
        "refine": refine,
        "sens_slip_agg": {y: results_slip[y].agg for y in LOOKBACK_YEARS},
        "sens_off_agg": {y: results_off0[y].agg for y in LOOKBACK_YEARS},
    }
    with open(CACHE_DIR / "results.pkl", "wb") as f:
        pickle.dump(payload, f)

    # ── 8. 主報告 ──────────────────────────────────────────────────────────
    import report
    report.write_report(payload, OUTPUT_DIR / "report.md")
    print(f"完成，全部 {time.time()-t_all:.0f}s → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
