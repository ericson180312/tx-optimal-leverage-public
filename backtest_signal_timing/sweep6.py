"""檢查頻率 × 執行延遲 × 相位的掃描與判定（機械化執行 PRE_REGISTRATION）。

沿用 backtest_multimarket 的引擎、評估、資料層——本模組只：
  1. 建訊號變體（trend_leverage 加相位參數，phase=0 與第三輪逐位元相同）；
  2. 把變體丟進第三輪的滾動視窗評估；
  3. 相位中性化後算判定量 D、M、N、τ；
  4. 機械化計算規則 A–D。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import evaluate as ev, signals as sg, strategies as st
from backtest_multimarket.data_mkts import Market
from backtest_multimarket.params import TREND_BUFFER, TREND_MA, TARGET_LEV

from backtest_ma_robustness.sweep import regime_stats

from backtest_signal_timing.params import (
    BASELINE_K, BASELINE_LAG, FALSE_SIG_WINDOW, K_GRID, L_LOW, LAG_GRID, LIVE_PHASE,
    MIN_MARKETS, PLATEAU_TOL, RANKING_MARKETS, REF_STRATEGIES, TRADING_DAYS,
    UPPER_BOUND, variant_key,
)

METRICS = ["med_sharpe", "med_cagr", "med_mdd", "worst_mdd", "med_avg_lev",
           "med_time_derisked", "med_cost_ratio", "med_n_adj", "n_bankrupt"]


# ── 訊號變體 ────────────────────────────────────────────────────────────────

def trend_leverage_phased(spot: np.ndarray, l_low: float = L_LOW, *, check_every: int,
                          phase: int = 0, ma_win: int = TREND_MA,
                          buf: float = TREND_BUFFER,
                          l_high: float = TARGET_LEV) -> np.ndarray:
    """signals.trend_leverage 加相位：檢查日改為 (t − phase) % check_every == 0。

    其餘運算順序與 signals.py 逐位元相同（phase=0 時完全一致，test_sweep6 鎖定）。
    另寫一份而不改 signals.py：實盤黃金測試耦合該檔，第六輪不得動共用層。
    """
    s, squeeze = sg._as2d(spot)
    T, P = s.shape
    csum = np.vstack([np.zeros((1, P)), np.cumsum(s, axis=0)])
    state = np.ones(P, dtype=bool)
    out = np.empty((T, P))
    for t in range(T):
        if t >= ma_win - 1 and (t - phase) % check_every == 0:
            ma = (csum[t + 1] - csum[t + 1 - ma_win]) / ma_win
            state = np.where(s[t] > ma * (1.0 + buf), True,
                             np.where(s[t] < ma * (1.0 - buf), False, state))
        out[t] = np.where(state, l_high, l_low)
    return out[:, 0] if squeeze else out


def apply_lag(signal_lev: np.ndarray, lag: int) -> np.ndarray:
    """訊號 → 執行槓桿。lag=1 即 to_exec；lag=0 同收盤執行（含 look-ahead，僅當上界）。"""
    out = signal_lev
    for _ in range(lag):
        out = sg.to_exec(out)
    return out


def variant_grid() -> list[tuple[int, int, int, str]]:
    return [(k, lag, ph, variant_key(k, lag, ph))
            for k in K_GRID for ph in range(k) for lag in LAG_GRID]


def build_variant_signals(spot: np.ndarray) -> dict[str, np.ndarray]:
    """全部 123 個變體的執行槓桿。"""
    out = {}
    for k in K_GRID:
        for ph in range(k):
            sig = trend_leverage_phased(spot, check_every=k, phase=ph)
            for lag in LAG_GRID:
                out[variant_key(k, lag, ph)] = apply_lag(sig, lag)
    return out


# ── 掃描回測（沿用第三輪 evaluate）──────────────────────────────────────────

def run_market(mkt: Market, calib: dict) -> pd.DataFrame:
    """單一市場：9 參照策略 + 123 變體的滾動視窗聚合（全部視窗）。"""
    var_L = build_variant_signals(mkt.spot)
    L_all = {**st.exec_signals(calib, "wf"), **var_L}
    names = REF_STRATEGIES + list(var_L)
    agg_all, _ = ev.run_market_all_periods(mkt, L_all, calib["calib_days"], names)
    return agg_all


def tag_variants(agg: pd.DataFrame) -> pd.DataFrame:
    """長表只留變體列，並拆出 K、lag、phase。"""
    d = agg[agg["strategy"].str.fullmatch(r"D_K\d\d_L\d_P\d\d")].copy()
    d["K"] = d["strategy"].str[3:5].astype(int)
    d["lag"] = d["strategy"].str[7].astype(int)
    d["phase"] = d["strategy"].str[10:12].astype(int)
    return d


def phase_neutral(variants: pd.DataFrame) -> pd.DataFrame:
    """每 (K, lag, 市場, 期間)：各指標跨相位取中位數。med_sharpe 即判定量 S。"""
    keys = ["K", "lag", "market", "period_years"]
    return variants.groupby(keys)[METRICS].median().reset_index()


def regime_table(mkts: dict[str, Market]) -> pd.DataFrame:
    """每市場 × K 的全序列切換統計（跨相位取中位數）。"""
    rows = []
    for name, m in mkts.items():
        for k in K_GRID:
            per = []
            for ph in range(k):
                sig = trend_leverage_phased(m.spot, check_every=k, phase=ph)
                per.append(regime_stats(sig, false_window=FALSE_SIG_WINDOW))
            df = pd.DataFrame(per)
            rows.append({"market": name, "K": k,
                         "switches_per_year": df["n_switches"].median() / (m.T / TRADING_DAYS),
                         "false_rate": df["false_rate"].median(),
                         "share_derisked": df["share_derisked"].median(),
                         "avg_regime_days": df["avg_regime_days"].median()})
    return pd.DataFrame(rows)


def next_day_effect(mkts: dict[str, Market]) -> pd.DataFrame:
    """獨立手算（不經引擎）：T+0 與 T+1 的曝險只在切換日隔天不同，差＝ΔL × 隔天報酬。

    切換日 t 收盤：T+0 已換成新槓桿、T+1 還是舊槓桿，所以第 t+1 日的報酬差 (L_t − L_{t−1})·r_{t+1}；
    其餘日子曝險相同。把這些差加總、年化，再除以策略波動，就是 T+0 的 Sharpe 優勢量級。
    """
    rows = []
    for name, m in mkts.items():
        for k in (BASELINE_K, 1):
            sig = trend_leverage_phased(m.spot, check_every=k, phase=0)
            change = np.diff(sig, prepend=sig[0])
            sw = np.flatnonzero(change[:-1] != 0)
            gain = change[sw] * m.ret[sw + 1]
            years = m.T / TRADING_DAYS
            vol = float(np.std(sig[:-1] * m.ret[1:]) * np.sqrt(TRADING_DAYS))
            down = change[sw] < 0
            rows.append({"market": name, "K": k, "n_switch": len(sw),
                         "next_ret_after_down": float(m.ret[sw[down] + 1].mean()),
                         "next_ret_after_up": float(m.ret[sw[~down] + 1].mean()),
                         "gain_per_year": float(gain.sum() / years),
                         "strategy_vol": vol,
                         "implied_sharpe": float(gain.sum() / years / vol)})
    return pd.DataFrame(rows)


def engine_full_sample_diff(mkt: Market, k: int, phase: int = 0) -> float:
    """手算的對照組：同口徑（全樣本單一路徑）下引擎的 Sharpe(T+0) − Sharpe(T+1)。"""
    sig = trend_leverage_phased(mkt.spot, check_every=k, phase=phase)
    keys = {lag: variant_key(k, lag, phase) for lag in (0, 1)}
    L = {keys[lag]: apply_lag(sig, lag) for lag in (0, 1)}
    metrics, _ = ev.run_windows(mkt, L, np.array([0], dtype=np.int64), mkt.T, list(L))
    return float(metrics[keys[0]]["sharpe"].iloc[0] - metrics[keys[1]]["sharpe"].iloc[0])


def maximin_crosscheck(agg: pd.DataFrame, neutral: pd.DataFrame) -> pd.DataFrame:
    """第三輪 cohort 的 maximin 百分位（僅交叉檢查，不進判定；10 成員 → 0.1 級距）。"""
    ref = agg[agg["strategy"].isin(REF_STRATEGIES)]
    rows = []
    for (k, lag), g in neutral.groupby(["K", "lag"]):
        pct = []
        for _, r in g.iterrows():
            cell = ref[(ref["market"] == r["market"])
                       & (ref["period_years"] == r["period_years"])]
            pool = pd.concat([cell["med_sharpe"], pd.Series([r["med_sharpe"]])],
                             ignore_index=True)
            pct.append(float(pool.rank(pct=True, na_option="bottom").iloc[-1]))
        rows.append({"K": k, "lag": lag, "worst_pct_rank": min(pct),
                     "mean_pct_rank": float(np.mean(pct))})
    return pd.DataFrame(rows)


# ── 判定量 ──────────────────────────────────────────────────────────────────

def delta_table(neutral: pd.DataFrame) -> pd.DataFrame:
    """每 (K, lag, 格) 的 Δ = S(K,lag,c) − S(5,1,c)。"""
    base = neutral[(neutral["K"] == BASELINE_K) & (neutral["lag"] == BASELINE_LAG)] \
        .set_index(["market", "period_years"])
    d = neutral.copy()
    cell = pd.MultiIndex.from_frame(d[["market", "period_years"]])
    d["delta"] = d["med_sharpe"] - cell.map(base["med_sharpe"])
    d["dmdd"] = d["med_mdd"] - cell.map(base["med_mdd"])
    d["dcagr"] = d["med_cagr"] - cell.map(base["med_cagr"])
    return d


def summarize(deltas: pd.DataFrame) -> pd.DataFrame:
    """每 (K, lag)：D（Δ 跨格中位數）、M（逐市場中位數為正的市場數）與描述欄。"""
    rows = []
    for (k, lag), g in deltas.groupby(["K", "lag"]):
        by_mkt = g.groupby("market")["delta"].median()
        rows.append({"K": k, "lag": lag,
                     "D": float(g["delta"].median()),
                     "M": int((by_mkt > 0).sum()),
                     "delta_min": float(g["delta"].min()),
                     "delta_max": float(g["delta"].max()),
                     "med_avg_lev": float(g["med_avg_lev"].median()),
                     "med_cagr": float(g["med_cagr"].median()),
                     "med_mdd": float(g["med_mdd"].median()),
                     # 配對差：16 格的 MDD 中位數落在排序斷層時，兩個中位數相減會誤導
                     "dcagr": float(g["dcagr"].median()),
                     "dmdd": float(g["dmdd"].median()),
                     "med_cost_ratio": float(g["med_cost_ratio"].median()),
                     "n_bankrupt": float(g["n_bankrupt"].sum()),
                     **{f"d_{m}": float(by_mkt.get(m, np.nan)) for m in RANKING_MARKETS}})
    out = pd.DataFrame(rows)
    out["cagr_per_lev"] = out["med_cagr"] / out["med_avg_lev"]
    return out


def phase_noise(variants: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """N：K=5、lag=1 時每格 5 個相位 Sharpe 的全距，跨格取中位數。"""
    d = variants[(variants["K"] == BASELINE_K) & (variants["lag"] == BASELINE_LAG)]
    rng = d.groupby(["market", "period_years"])["med_sharpe"].agg(
        sharpe_range=lambda x: x.max() - x.min(), sharpe_min="min", sharpe_max="max")
    return float(rng["sharpe_range"].median()), rng.reset_index()


def live_phase_delta(variants: pd.DataFrame) -> float:
    """ΔP：TX_FUT 四期間上，實盤相位 Sharpe − 該格 5 相位中位數，再取中位數。"""
    d = variants[(variants["K"] == BASELINE_K) & (variants["lag"] == BASELINE_LAG)
                 & (variants["market"] == "TX_FUT")]
    diffs = []
    for _, g in d.groupby("period_years"):
        live = float(g.loc[g["phase"] == LIVE_PHASE, "med_sharpe"].iloc[0])
        diffs.append(live - float(g["med_sharpe"].median()))
    return float(np.median(diffs))


# ── 規則 A–D（機械化）───────────────────────────────────────────────────────

def decide(summary: pd.DataFrame, noise: float, delta_p: float,
           boot_same_sign) -> dict:
    """機械化執行預先承諾的規則，回傳判定字典（含所有中間量）。

    boot_same_sign(K, lag) -> int：配對 bootstrap 與「D>0」同向的市場數。
    """
    s = summary.set_index(["K", "lag"])
    D = lambda k, lag: float(s.loc[(k, lag), "D"])          # noqa: E731
    M = lambda k, lag: int(s.loc[(k, lag), "M"])            # noqa: E731
    tau = max(PLATEAU_TOL, noise)

    # ── 規則 A：檢查頻率（lag=1）────────────────────────────────────────────
    d_k = {k: D(k, BASELINE_LAG) for k in K_GRID}
    r_a = max(d_k.values()) - min(d_k.values())
    order = sorted(K_GRID, key=lambda k: d_k[k], reverse=True)
    rank5 = order.index(BASELINE_K) + 1
    cand_rows = []
    if r_a < tau:
        verdict_a = "高原"
        text_a = "R_A < τ → 高原，維持 K=5。結論：對檢查頻率不敏感。"
    elif rank5 <= len(K_GRID) / 2:
        verdict_a = "傾斜高原"
        text_a = "R_A ≥ τ 且 K=5 排名前 3 → 傾斜高原，維持 K=5。"
    else:
        for k in K_GRID:
            if k == BASELINE_K:
                continue
            i = K_GRID.index(k)
            nb = [K_GRID[j] for j in (i - 1, i + 1) if 0 <= j < len(K_GRID)]
            row = {"K": k, "D": d_k[k], "M": M(k, BASELINE_LAG),
                   "cond_i": d_k[k] >= tau,
                   "cond_ii": M(k, BASELINE_LAG) >= MIN_MARKETS,
                   "cond_iii": any(d_k[x] >= 0 for x in nb),
                   "cond_iv": boot_same_sign(k, BASELINE_LAG) >= MIN_MARKETS,
                   "cond_v": s.loc[(k, BASELINE_LAG), "n_bankrupt"]
                   <= s.loc[(BASELINE_K, BASELINE_LAG), "n_bankrupt"]}
            row["ok"] = all(row[f"cond_{c}"] for c in ("i", "ii", "iii", "iv", "v"))
            cand_rows.append(row)
        ok = [r["K"] for r in cand_rows if r["ok"]]
        if ok:
            verdict_a = "5日為劣位"
            text_a = f"R_A ≥ τ 且 K=5 排名後 3，且 K={ok} 滿足全部替代條件。"
        else:
            verdict_a = "傾斜但無可行替代"
            text_a = "R_A ≥ τ 且 K=5 排名後 3，但無人滿足全部替代條件 → 維持 K=5。"

    # ── 規則 B：執行時點（上界）─────────────────────────────────────────────
    ub = {v: D(*v) for v in UPPER_BOUND}
    v_star = max(ub, key=ub.get)
    u = ub[v_star]
    if u < tau:
        verdict_b = "排除"
        text_b = "U < τ → 連含 look-ahead 的上界都沒有實質改善，可行的當天做法一併排除，維持 T+1。"
    elif M(*v_star) >= MIN_MARKETS and boot_same_sign(*v_star) >= MIN_MARKETS:
        verdict_b = "上界有空間"
        text_b = "U ≥ τ、方向與 bootstrap 一致 → 上界有空間。不得建議改動，只能另開一輪用可執行資料驗證。"
    else:
        verdict_b = "不一致"
        text_b = "U ≥ τ 但方向或 bootstrap 不一致 → 維持 T+1，報告寫出矛盾。"
    d52 = D(BASELINE_K, 2)

    # ── 規則 C：尖峰否決 ────────────────────────────────────────────────────
    spike_gap = D(5, 1) - max(D(3, 1), D(10, 1))

    return {
        "tau": tau, "noise": noise, "d_k": d_k, "order": order, "rank5": rank5,
        "r_a": r_a, "verdict_a": verdict_a, "text_a": text_a,
        "candidates": pd.DataFrame(cand_rows),
        "ub": ub, "v_star": v_star, "u": u, "verdict_b": verdict_b, "text_b": text_b,
        "d52": d52, "delay_sensitive": d52 <= -tau,
        "spike_gap": spike_gap, "spike": spike_gap > tau,
        "delta_p": delta_p, "phase_flag": abs(delta_p) >= tau,
    }
