"""第六輪主流程：資料（凍結）→ 復現檢核 → 123 變體掃描 → bootstrap → 規則 A–D → 報告。

執行：conda run -n ML_env python -m backtest_signal_timing.run6
輸出：backtest_signal_timing/output/ 下的 r6_*.csv、PNG 與 report.md。
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import backtest_threshold.data as tx_data
import backtest_threshold.params as root_params
from backtest_threshold.continuous import build_continuous

from backtest_multimarket import data_mkts, evaluate as ev, strategies as st
from backtest_multimarket.params import OUTPUT_DIR as R3_OUTPUT

from backtest_signal_timing import bootstrap6, sweep6
from backtest_signal_timing.params import (
    BOOT_SEED6, CACHE_DIR, OUTPUT_DIR, PERIOD_YEARS, RANKING_MARKETS, STRESS_WINDOWS,
    variant_key,
)

REPRO_TOL = 1e-9


def load_frozen() -> tuple[dict, dict]:
    """資料凍結在 repo 快取（與前五輪相同）：一律不觸網、不覆寫受追蹤的 parquet。"""
    tx_data._cache_is_fresh = lambda path, today: path.exists()
    data_mkts._cache_ok = lambda path, max_age_days: path.exists()
    tx_cs = build_continuous(tx_data.load_tx_contracts(), root_params.ROLL_OFFSET_DAYS)
    mkts = {k: m for k, m in data_mkts.load_markets(tx_cs).items() if k in RANKING_MARKETS}
    calibs = {k: st.calibrate_market(m.ret, m.spot) for k, m in mkts.items()}
    return mkts, calibs


def repro_check(mkts: dict, calibs: dict) -> pd.DataFrame:
    """整合檢核：K=5、lag=1、φ=0 必須複現第三輪凍結輸出的 D_10（16 格）。"""
    r3 = pd.read_csv(R3_OUTPUT / "r3_market_period_metrics.csv")
    r3 = r3[(r3["strategy"] == "D_10") & r3["market"].isin(RANKING_MARKETS)] \
        .set_index(["market", "period_years"])["med_sharpe"]
    rows = []
    key = variant_key(5, 1, 0)
    for k, m in mkts.items():
        sig = sweep6.trend_leverage_phased(m.spot, check_every=5, phase=0)
        agg, _ = ev.run_market_all_periods(m, {key: sweep6.apply_lag(sig, 1)},
                                           calibs[k]["calib_days"], [key])
        for _, r in agg.iterrows():
            rows.append({"market": k, "period_years": int(r["period_years"]),
                         "med_sharpe": r["med_sharpe"],
                         "r3": float(r3.loc[(k, r["period_years"])])})
    out = pd.DataFrame(rows)
    out["diff"] = out["med_sharpe"] - out["r3"]
    return out


def run_stress(mkts: dict) -> pd.DataFrame:
    """壓力窗口（描述性，不進判定）：對比變體跨相位取中位數。"""
    contrasts = bootstrap6.CONTRASTS
    rows = []
    for label, keys, d0, d1 in STRESS_WINDOWS:
        for k in (keys or RANKING_MARKETS):
            if k not in mkts:
                continue
            m = mkts[k]
            s = int(m.dates.searchsorted(pd.Timestamp(d0)))
            e = int(m.dates.searchsorted(pd.Timestamp(d1), side="right"))
            if e - s < 40 or (s == 0 and m.dates[0] > pd.Timestamp(d0) + pd.Timedelta(days=370)):
                continue
            var_L = sweep6.build_variant_signals(m.spot)
            starts = np.array([s], dtype=np.int64)
            for kk, lag in contrasts:
                names = [variant_key(kk, lag, ph) for ph in range(kk)]
                metrics, _ = ev.run_windows(m, {n: var_L[n] for n in names}, starts,
                                            e - s, names)
                per = pd.concat([metrics[n] for n in names], ignore_index=True)
                rows.append({"window": label, "market": k, "K": kk, "lag": lag,
                             "cumret": float(per["cumret"].median()),
                             "mdd": float(per["mdd"].median()),
                             "avg_lev": float(per["avg_lev"].median())})
    return pd.DataFrame(rows)


def _boot_job(args):
    mkt, seed = args
    return bootstrap6.run_paired(mkt, seed=seed)


def _sweep_job(args):
    mkt, calib = args
    return sweep6.run_market(mkt, calib)


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("== 載入市場資料（凍結）==", flush=True)
    mkts, calibs = load_frozen()
    for k, m in mkts.items():
        print(f"  {k:7s} {m.meta['start']} ~ {m.meta['end']}  {m.T} 日", flush=True)

    print("== 復現檢核：K=5、lag=1、φ=0 vs 第三輪 D_10 ==", flush=True)
    repro = repro_check(mkts, calibs)
    repro.to_csv(OUTPUT_DIR / "r6_repro_check.csv", index=False, encoding="utf-8-sig")
    worst = float(repro["diff"].abs().max())
    print(f"  16 格最大差異 = {worst:.2e}", flush=True)
    if worst > REPRO_TOL:
        raise SystemExit(f"復現失敗（差異 {worst:.2e}）：停止，不跑任何變體。")

    print("== 123 變體掃描（4 市場 × 4 期間）==", flush=True)
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(_sweep_job, [(mkts[k], calibs[k]) for k in RANKING_MARKETS]))
    agg = pd.concat(frames, ignore_index=True)
    print(f"  {time.time()-t1:.0f}s", flush=True)

    variants = sweep6.tag_variants(agg)
    neutral = sweep6.phase_neutral(variants)
    deltas = sweep6.delta_table(neutral)
    summary = sweep6.summarize(deltas)
    noise, noise_tbl = sweep6.phase_noise(variants)
    delta_p = sweep6.live_phase_delta(variants)

    print("== 配對 bootstrap ==", flush=True)
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(_boot_job, [(mkts[k], BOOT_SEED6 + i)
                                           for i, k in enumerate(RANKING_MARKETS)]))
    boot = pd.concat(frames, ignore_index=True)
    print(f"  {time.time()-t1:.0f}s", flush=True)

    decision = sweep6.decide(summary, noise, delta_p, bootstrap6.same_sign_counter(boot))

    print("== 描述性輸出 ==", flush=True)
    regime = sweep6.regime_table(mkts)
    stress = run_stress(mkts)
    maximin = sweep6.maximin_crosscheck(agg, neutral)
    handcheck = sweep6.next_day_effect(mkts)
    handcheck["engine_sharpe"] = [sweep6.engine_full_sample_diff(mkts[m], k)
                                  for m, k in zip(handcheck["market"], handcheck["K"])]
    tx_t0_phases = [sweep6.engine_full_sample_diff(mkts["TX_FUT"], 5, ph) for ph in range(5)]

    for df, name in [(variants, "r6_variants"), (deltas, "r6_deltas"), (summary, "r6_summary"),
                     (noise_tbl, "r6_phase_noise"), (boot, "r6_bootstrap"),
                     (regime, "r6_regime"), (stress, "r6_stress"),
                     (maximin, "r6_maximin_crosscheck"), (handcheck, "r6_handcheck"),
                     (decision["candidates"], "r6_ruleA_candidates")]:
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    payload = {"summary": summary, "deltas": deltas, "variants": variants,
               "noise_tbl": noise_tbl, "boot": boot, "regime": regime, "stress": stress,
               "maximin": maximin, "decision": decision, "repro": repro,
               "handcheck": handcheck, "tx_t0_phases": tx_t0_phases,
               "audit": {k: m.meta for k, m in mkts.items()},
               "first_date": {k: str(m.dates[0].date()) for k, m in mkts.items()},
               "periods": PERIOD_YEARS}
    with open(CACHE_DIR / "results6.pkl", "wb") as f:
        pickle.dump(payload, f)

    from backtest_signal_timing import report6
    report6.write_from_cache()

    print(f"規則 A：{decision['verdict_a']}（R_A={decision['r_a']:.3f}，τ={decision['tau']:.3f}）",
          flush=True)
    print(f"規則 B：{decision['verdict_b']}（U={decision['u']:.3f}，v*={decision['v_star']}）",
          flush=True)
    print(f"規則 C：spike={decision['spike']}（gap={decision['spike_gap']:.3f}）", flush=True)
    print(f"規則 D：flag={decision['phase_flag']}（ΔP={decision['delta_p']:.3f}）", flush=True)
    print(f"完成，全部 {time.time()-t0:.0f}s → {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
