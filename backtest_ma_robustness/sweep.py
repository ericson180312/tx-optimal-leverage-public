"""MA 窗口掃描的核心分析（規格 §2–§5）。

沿用 backtest_multimarket 的引擎、評估、資料層——本模組只：
  1. 用不同 (MA, buffer) 建 D 趨勢變體訊號（名稱以 D_ 開頭 → 沿用第三輪 signal 路由）；
  2. 計算 regime 統計（切換次數、平均持續、假訊號＝切換後 20 日內回切）；
  3. 把每個變體丟進「第三輪九策略參照 cohort」同格排名 → maximin 最差格百分位
     （與第三輪同口徑；MA=200／buf=0.01 應複現 D_10 的 worst cell）；
  4. 機械化計算規則 A（[120,250] 極差 vs 0.08）與規則 B（200 vs ±20 鄰居尖峰）。

判定量的兩個讀數（都對照 0.08）：
  - pre-registered：maximin 最差格「百分位」極差（10 成員 cohort → 量化到 0.1）。
  - robust 交叉檢查：同格內核心 MA 變體的「原始中位 Sharpe」全距（市場內可比、不量化，
    即前三輪 0.08 噪音量級的原意）。兩者一致 → 結論穩固；相左則據實揭露。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import evaluate as ev, signals as sg, strategies as st
from backtest_multimarket.data_mkts import Market
from backtest_multimarket.params import TARGET_LEV, TRADING_DAYS

from backtest_ma_robustness.params import (
    BASELINE_MA, BUFFER_GRID, FALSE_SIG_WINDOW, L_LOW_MAIN, MA_CORE_HI,
    MA_CORE_LO, MA_GRID, NEIGHBOR_DELTA, PERIOD_YEARS, PLATEAU_TOL,
    RANKING_MARKETS, REF_STRATEGIES, variant_key, variant_label,
)


# ── 變體訊號 ────────────────────────────────────────────────────────────────

def build_variant_signals(spot: np.ndarray, l_low: float = L_LOW_MAIN,
                          mas=MA_GRID, buffers=BUFFER_GRID) -> dict[str, np.ndarray]:
    """所有 (MA, buffer) 變體的**執行槓桿**（訊號日算、shift 一日 → 次日執行）。"""
    out = {}
    for ma in mas:
        for buf in buffers:
            lev = sg.trend_leverage(spot, l_low, ma_win=ma, buf=buf, l_high=TARGET_LEV)
            out[variant_key(ma, buf)] = sg.to_exec(lev)
    return out


def variant_grid(mas=MA_GRID, buffers=BUFFER_GRID) -> list[tuple[int, float, str]]:
    return [(ma, buf, variant_key(ma, buf)) for ma in mas for buf in buffers]


# ── regime 統計（規格 §5-3）──────────────────────────────────────────────────

def regime_stats(signal_lev: np.ndarray, *, l_high: float = TARGET_LEV,
                 false_window: int = FALSE_SIG_WINDOW) -> dict:
    """單一（未 shift 的）訊號槓桿序列的 regime 統計。

    state = 滿載(l_high) / 降載；切換 = 狀態改變；
    假訊號 = 「被兩次切換夾住、且持續 ≤ false_window 個交易日」的短命 regime
    （切換後很快又切回 → 巴洗）。首尾未被兩次切換夾住的段不計入假訊號。
    """
    state = signal_lev >= l_high - 1e-9
    T = len(state)
    change = np.flatnonzero(np.diff(state.astype(np.int8)) != 0) + 1
    n_sw = int(len(change))
    bounds = np.concatenate([[0], change, [T]])
    seg = np.diff(bounds)                        # 各 regime 段長度（含首尾截斷段）
    internal = seg[1:-1] if len(seg) > 2 else np.array([], dtype=int)
    n_false = int(np.sum(internal <= false_window))
    return {
        "n_switches": n_sw,
        "avg_regime_days": float(np.mean(seg)) if len(seg) else float(T),
        "n_regimes": int(len(seg)),
        "n_false_signals": n_false,
        "false_rate": (n_false / n_sw) if n_sw else 0.0,
        "share_derisked": float(np.mean(~state)),
    }


def regime_table(mkts: dict[str, Market], *, l_low: float = L_LOW_MAIN,
                 buffers=BUFFER_GRID) -> pd.DataFrame:
    """每市場 × 每 (MA, buffer) 的全序列 regime 統計。"""
    rows = []
    for k, m in mkts.items():
        for ma in MA_GRID:
            for buf in buffers:
                sig = sg.trend_leverage(m.spot, l_low, ma_win=ma, buf=buf,
                                        l_high=TARGET_LEV)
                st_ = regime_stats(sig)
                st_.update(market=k, ma=ma, buffer=buf,
                           switches_per_year=st_["n_switches"] / (m.T / TRADING_DAYS))
                rows.append(st_)
    return pd.DataFrame(rows)


# ── 掃描回測（沿用第三輪 evaluate）──────────────────────────────────────────

def run_sweep(mkts: dict[str, Market], calibs: dict, *,
              l_low: float = L_LOW_MAIN, buffers=BUFFER_GRID,
              markets=None) -> pd.DataFrame:
    """對每市場跑「9 參照策略 + 全部 MA 變體」的滾動視窗聚合。

    參照策略沿用第三輪 walk-forward σ_target；MA 變體無校準自由度。
    回傳長表：每 (strategy, market, period_years) 一列（med_sharpe/med_cagr/
    med_avg_lev/... 沿用 evaluate.aggregate 全套欄位）。
    """
    markets = markets or list(mkts)
    names = REF_STRATEGIES + [k for _, _, k in variant_grid(buffers=buffers)]
    frames = []
    for k in markets:
        m = mkts[k]
        ref_L = st.exec_signals(calibs[k], "wf")
        var_L = build_variant_signals(m.spot, l_low, buffers=buffers)
        L_all = {**ref_L, **var_L}
        agg_all, _ = ev.run_market_all_periods(m, L_all, calibs[k]["calib_days"], names)
        frames.append(agg_all)
    return pd.concat(frames, ignore_index=True)


# ── maximin：變體 vs 參照 cohort（規格 §3-4）────────────────────────────────

def variant_cell_percentiles(agg: pd.DataFrame, variant_keys: list[str], *,
                             ref=REF_STRATEGIES, markets=RANKING_MARKETS,
                             periods=PERIOD_YEARS) -> pd.DataFrame:
    """每 (市場×期間) 格：每個 MA 變體在「9 參照 + 該變體」中的 Sharpe 百分位。

    與第三輪 cell_percentiles 同口徑（rank(pct=True, na_option='bottom')），
    但參照集固定為 9 策略、每個變體單獨插入 → 不因掃描網格大小而稀釋。
    另回傳同格核心 MA 變體的原始中位 Sharpe（供 robust 全距檢查）。
    """
    rows = []
    sub = agg[agg["market"].isin(markets) & agg["period_years"].isin(periods)]
    for (mk, yr), cell in sub.groupby(["market", "period_years"]):
        s = cell.set_index("strategy")["med_sharpe"]
        lev = cell.set_index("strategy")["med_avg_lev"]
        cagr = cell.set_index("strategy")["med_cagr"]
        ref_s = s[[r for r in ref if r in s.index]]
        for v in variant_keys:
            if v not in s.index:
                continue
            pool = pd.concat([ref_s, pd.Series({v: s[v]})])
            pct = float(pool.rank(pct=True, na_option="bottom")[v])
            rows.append({"strategy": v, "market": mk, "period_years": yr,
                         "med_sharpe": float(s[v]), "pct_rank": pct,
                         "med_avg_lev": float(lev[v]), "med_cagr": float(cagr[v])})
    return pd.DataFrame(rows)


def variant_maximin(cellpct: pd.DataFrame) -> pd.DataFrame:
    """每變體：最差格百分位（maximin 分數）、平均百分位、最差格、平均槓桿中位。"""
    g = cellpct.groupby("strategy")
    worst = cellpct.loc[g["pct_rank"].idxmin()].set_index("strategy")
    out = pd.DataFrame({
        "worst_pct_rank": g["pct_rank"].min(),
        "mean_pct_rank": g["pct_rank"].mean(),
        "worst_cell": worst["market"] + "×" + worst["period_years"].astype(str) + "y",
        "med_avg_lev": g["med_avg_lev"].median(),
        "med_cagr": g["med_cagr"].median(),
    })
    return out.reset_index()


def _parse_variant(name: str) -> tuple[int, float]:
    """'D_MA200_b010' → (200, 0.010)。"""
    ma = int(name.split("MA")[1].split("_")[0])
    buf = int(name.split("_b")[1]) / 1000.0
    return ma, buf


def within_cell_sharpe_range(cellpct: pd.DataFrame, core_keys: list[str]) -> pd.DataFrame:
    """robust 交叉檢查：每格核心 MA 變體的原始中位 Sharpe 全距（市場內可比、不量化）。"""
    d = cellpct[cellpct["strategy"].isin(core_keys)]
    rng = d.groupby(["market", "period_years"])["med_sharpe"].agg(
        sharpe_range=lambda x: x.max() - x.min(),
        sharpe_min="min", sharpe_max="max")
    return rng.reset_index()


# ── 規則 A / B 判定（規格 §1，機械化）───────────────────────────────────────

def core_variants(buffers=BUFFER_GRID) -> list[str]:
    """MA ∈ [120,250] 的核心變體（規則 A 判定範圍）。"""
    return [variant_key(ma, buf) for ma in MA_GRID
            for buf in buffers if MA_CORE_LO <= ma <= MA_CORE_HI]


def decide(maximin: pd.DataFrame, cellpct: pd.DataFrame, *,
           buffers=BUFFER_GRID) -> dict:
    """機械化執行規則 A 與規則 B，回傳判定字典（含兩個讀數與所有中間量）。"""
    mm = maximin.set_index("strategy")
    core = core_variants(buffers)
    core = [c for c in core if c in mm.index]

    # ── 規則 A：主判定 ──────────────────────────────────────────────────────
    wpr = mm.loc[core, "worst_pct_rank"]
    spread_pct = float(wpr.max() - wpr.min())

    # robust 交叉檢查：核心變體同格原始 Sharpe 全距
    rng = within_cell_sharpe_range(cellpct, core)
    raw_spread_med = float(rng["sharpe_range"].median())
    raw_spread_max = float(rng["sharpe_range"].max())

    # 200 在核心區間內的位置（以 maximin 由高到低排序；1=最好）
    ranked = mm.loc[core, "worst_pct_rank"].sort_values(ascending=False)
    order = list(ranked.index)
    n = len(order)
    pos200 = [order.index(variant_key(BASELINE_MA, b)) + 1
              for b in buffers if variant_key(BASELINE_MA, b) in order]
    median_pos200 = float(np.median(pos200)) if pos200 else np.nan
    front_half = median_pos200 <= n / 2.0

    if spread_pct < PLATEAU_TOL:
        verdict_a = "高原"
        verdict_a_text = ("極差 < 0.08 → 判定為高原。維持 MA=200 不變。"
                          "趨勢機制對窗口長度不敏感，D_10 的冠軍地位獲得強化。")
    elif front_half:
        verdict_a = "傾斜高原（維持200）"
        verdict_a_text = ("極差 ≥ 0.08 且 200 落在區間前 50% → 傾斜高原，維持 200。")
    else:
        verdict_a = "200為劣位"
        verdict_a_text = ("極差 ≥ 0.08 且 200 落在區間後 50% → 200 為劣位選擇，"
                          "此時方討論替代窗口（須為平原中心，非 argmax）。")

    # ── 規則 B：尖峰否決 ────────────────────────────────────────────────────
    neigh = [BASELINE_MA - NEIGHBOR_DELTA, BASELINE_MA + NEIGHBOR_DELTA]  # 180, 220
    b_rows, spike_any = [], False
    for buf in buffers:
        k200 = variant_key(BASELINE_MA, buf)
        kn = [variant_key(x, buf) for x in neigh]
        if k200 not in mm.index or not all(x in mm.index for x in kn):
            continue
        m200 = float(mm.loc[k200, "worst_pct_rank"])
        mn = float(mm.loc[kn, "worst_pct_rank"].max())
        delta = m200 - mn
        spike = delta > PLATEAU_TOL
        spike_any |= spike
        b_rows.append({"buffer": buf, "maximin_200": m200,
                       "maximin_best_neighbor": mn, "delta": delta, "spike": spike})
    verdict_b = ("觸發：200 顯著優於 ±20 鄰居 → D_10 結論作廢，需重新評估"
                 if spike_any else "未觸發：200 未孤立高於相鄰窗口，D_10 結論不受否決")

    return {
        "core_keys": core,
        "spread_pct": spread_pct,
        "raw_spread_med": raw_spread_med,
        "raw_spread_max": raw_spread_max,
        "within_cell_range": rng,
        "n_core": n, "median_pos200": median_pos200, "pos200": pos200,
        "front_half": bool(front_half),
        "verdict_a": verdict_a, "verdict_a_text": verdict_a_text,
        "neighbor_table": pd.DataFrame(b_rows),
        "spike": bool(spike_any), "verdict_b": verdict_b,
        "plateau_by_pct": spread_pct < PLATEAU_TOL,
        # robust 讀數以「代表性的中位數格」為準（max 只反映單一最噪的短期格，過嚴）
        "plateau_by_raw": raw_spread_med < PLATEAU_TOL,
        "plateau_by_raw_max": raw_spread_max < PLATEAU_TOL,
    }
