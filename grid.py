"""網格搜尋＋滾動起點檢驗（規格 §4、§5）。

- 每個期間長度：基準路徑（以資料最後一日為終點）＋ 季頻平移起點的多條重疊路徑。
- 每條路徑一次模擬 K = 100 網格組 + 3 基準策略（同一引擎、參數化表達）。
- 最佳解以「跨路徑中位數」判定；任何一條路徑破產的參數組一律不得列入最佳解。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from params import (
    BENCHMARKS, INITIAL_EQUITY, ROLLING_START_FREQ, SLIPPAGE_TICKS,
    TARGET_LEVERAGE, X_GRID, Y_GRID,
)
from continuous import ContinuousSeries
from engine import SimResult, simulate
from metrics import path_metrics


def combo_table(x_grid=None, y_grid=None) -> pd.DataFrame:
    """K 組參數表：門檻網格組合（rebal_every=0）+ 各基準（is_benchmark=True）。"""
    x_grid = X_GRID if x_grid is None else x_grid
    y_grid = Y_GRID if y_grid is None else y_grid
    rows = [{"name": f"X{x}_Y{y}", "X": x, "Y": y, "target": TARGET_LEVERAGE,
             "rebal_every": 0, "is_benchmark": False}
            for x in x_grid for y in y_grid]
    rows += [{"name": name, "X": bm["X"], "Y": bm["Y"], "target": bm["target"],
              "rebal_every": bm["rebal_every"], "is_benchmark": True}
             for name, bm in BENCHMARKS.items()]
    return pd.DataFrame(rows)


@dataclass
class PeriodResult:
    """單一期間長度的完整結果。"""
    years: int
    agg: pd.DataFrame                 # K rows：跨路徑聚合 + 基準路徑完整指標（base_*）
    n_paths: int
    baseline_sim: SimResult           # 基準路徑（終點=資料尾）的完整模擬紀錄
    baseline_index: pd.DatetimeIndex  # 基準路徑的日期
    path_starts: list[pd.Timestamp] = field(default_factory=list)


def _window_positions(idx: pd.DatetimeIndex, years: int) -> list[tuple[int, int, bool]]:
    """(start_pos, end_pos, is_baseline) 清單：基準路徑 + 季頻滾動起點。"""
    end_date = idx[-1]
    wins: list[tuple[int, int, bool]] = []

    # 基準路徑：終點 = 資料最後一日，起點 = 往回推 years 年
    base_start = idx.searchsorted(end_date - pd.DateOffset(years=years))
    wins.append((int(base_start), len(idx) - 1, True))

    # 滾動起點：季初日期，start + years 需落在資料範圍內
    for qs in pd.date_range(idx[0], end_date - pd.DateOffset(years=years),
                            freq=ROLLING_START_FREQ):
        s = int(idx.searchsorted(qs))
        e = int(idx.searchsorted(qs + pd.DateOffset(years=years), side="right")) - 1
        if e > s and e < len(idx):
            wins.append((s, e, False))
    return wins


def run_period(cs: ContinuousSeries, years: int, *,
               slippage_ticks: float = SLIPPAGE_TICKS,
               e0: float = INITIAL_EQUITY,
               x_grid=None, y_grid=None) -> PeriodResult:
    """單一期間長度：跑基準路徑＋所有滾動路徑，聚合跨路徑統計。"""
    combos = combo_table(x_grid, y_grid)
    X = combos["X"].to_numpy(float)
    Y = combos["Y"].to_numpy(float)
    target = combos["target"].to_numpy(float)
    rebal_every = combos["rebal_every"].to_numpy(np.int64)
    K = len(combos)

    panel = cs.panel
    idx = panel.index
    ret = panel["ret"].to_numpy(float)
    price = panel["price"].to_numpy(float)
    is_roll = panel["is_roll"].to_numpy(bool)
    p_near = panel["p_near"].to_numpy(float)
    p_next = panel["p_next"].to_numpy(float)

    wins = _window_positions(idx, years)
    cum_paths, sharpe_paths, mdd_paths, cagr_paths, bank_paths = [], [], [], [], []
    baseline_sim = None
    baseline_metrics = None
    starts: list[pd.Timestamp] = []

    for s, e, is_base in wins:
        # 建倉價 = 視窗第一個報酬日前一日收盤的持倉合約價
        entry_price = price[s - 1] if s >= 1 else price[0] / (1.0 + ret[0])
        sl = slice(s, e + 1)
        res = simulate(ret[sl], price[sl], is_roll[sl], p_near[sl], p_next[sl],
                       X, Y, target, entry_price, rebal_every=rebal_every,
                       slippage_ticks=slippage_ticks, e0=e0)
        m = path_metrics(res, e0=e0)
        cum_paths.append(m["cumret"].to_numpy())
        sharpe_paths.append(m["sharpe"].to_numpy())
        mdd_paths.append(m["mdd"].to_numpy())
        cagr_paths.append(m["cagr"].to_numpy())
        bank_paths.append(m["bankrupt"].to_numpy())
        starts.append(idx[s])
        if is_base:
            baseline_sim = res
            baseline_metrics = m
            baseline_index = idx[sl]

    cum = np.vstack(cum_paths)        # (P, K)
    shp = np.vstack(sharpe_paths)
    mdd = np.vstack(mdd_paths)
    cagr = np.vstack(cagr_paths)
    bank = np.vstack(bank_paths)
    P = cum.shape[0]

    i_bh2 = int(combos.index[combos["name"] == "BH_2x"][0])
    i_dl2 = int(combos.index[combos["name"] == "Daily_2x"][0])

    with np.errstate(all="ignore"):
        agg = pd.DataFrame({
            "name": combos["name"], "X": combos["X"], "Y": combos["Y"],
            "rebal_every": combos["rebal_every"],
            "is_benchmark": combos["is_benchmark"],
            "n_paths": P,
            "med_cumret": np.median(cum, axis=0),
            "med_cagr": np.median(cagr, axis=0),
            "med_sharpe": np.nanmedian(shp, axis=0),
            "sharpe_p25": np.nanpercentile(shp, 25, axis=0),
            "sharpe_p75": np.nanpercentile(shp, 75, axis=0),
            "med_mdd": np.median(mdd, axis=0),
            "worst_cumret": cum.min(axis=0),
            "win_vs_bh2x": (cum > cum[:, [i_bh2]]).mean(axis=0),
            "win_vs_daily2x": (cum > cum[:, [i_dl2]]).mean(axis=0),
            "n_bankrupt_paths": bank.sum(axis=0),
            "any_bankrupt": bank.any(axis=0),
        })
    agg = agg.join(baseline_metrics.add_prefix("base_"))

    return PeriodResult(years=years, agg=agg, n_paths=P,
                        baseline_sim=baseline_sim, baseline_index=baseline_index,
                        path_starts=starts)


def select_best(agg: pd.DataFrame) -> dict:
    """最佳解選取（破產組與基準組一律排除）。

    回傳：med（跨路徑中位數為準，正式最佳解）與 single（基準單一路徑最大值，
    僅供過擬合幅度對照）兩組，各含 by_return / by_sharpe。
    """
    elig = agg[~agg["is_benchmark"] & ~agg["any_bankrupt"]]
    out = {
        "med_by_return": elig.loc[elig["med_cumret"].idxmax()],
        "med_by_sharpe": elig.loc[elig["med_sharpe"].idxmax()],
        "single_by_return": elig.loc[elig["base_cumret"].idxmax()],
        "single_by_sharpe": elig.loc[elig["base_sharpe"].idxmax()],
        "n_excluded_bankrupt": int((~agg["is_benchmark"] & agg["any_bankrupt"]).sum()),
    }
    return out


def neighborhood_stats(agg: pd.DataFrame, metric: str, x_best: float, y_best: float) -> dict:
    """最佳解 3×3 鄰域統計：判定平原（plateau）或尖峰（spike）。

    retention = 鄰域均值在「全網格分布」中的相對位置（1=與峰值同高、0=網格中位數）。
    """
    g = agg[~agg["is_benchmark"]].pivot(index="Y", columns="X", values=metric)
    xi = list(g.columns).index(x_best)
    yi = list(g.index).index(y_best)
    best = g.iloc[yi, xi]
    nbr = g.iloc[max(0, yi - 1): yi + 2, max(0, xi - 1): xi + 2].to_numpy().ravel()
    nbr = nbr[~np.isnan(nbr)]
    nbr = nbr[nbr != best] if (nbr == best).sum() == 1 else nbr
    grid_med = float(np.nanmedian(g.to_numpy()))
    span = best - grid_med
    retention = float((np.mean(nbr) - grid_med) / span) if span > 0 else np.nan
    return {"best": float(best), "nbr_mean": float(np.mean(nbr)),
            "nbr_min": float(np.min(nbr)), "grid_median": grid_med,
            "retention": retention}
