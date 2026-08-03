"""取捨曲線、邊際交換率與每單位槓桿效率（規格 §5-1、§5-3、§5-4）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_leverage_sweep.params import L_HIGH_GRID, RANKING_MARKETS


def _with_lhigh(sweep: pd.DataFrame) -> pd.DataFrame:
    d = sweep.copy()
    d["l_high"] = d["strategy"].str.extract(r"L(\d+)").astype(int) / 100.0
    return d


def frontier_points(sweep: pd.DataFrame, *, markets=RANKING_MARKETS,
                    by_market: bool = False) -> pd.DataFrame:
    """每個 L_high 的 (MDD, CAGR) 代表點——跨格中位數（規格 §5-1）。"""
    d = _with_lhigh(sweep)
    d = d[d["market"].isin(markets)]
    keys = ["market", "l_high"] if by_market else ["l_high"]
    g = d.groupby(keys).agg(
        cagr=("med_cagr", "median"), mdd=("med_mdd", "median"),
        sharpe=("med_sharpe", "median"), avg_lev=("med_avg_lev", "median"),
        survival=("survival_rate", "min"),
        n_liquidated=("n_liquidated", "sum")).reset_index()
    return g.sort_values(keys).reset_index(drop=True)


def detect_backward_bend(front: pd.DataFrame) -> dict:
    """偵測「向後彎折」：MDD 持續惡化但 CAGR 不再上升的點（規格 §5-1）。

    自最低 L_high 起逐段檢查：若某段 ΔCAGR ≤ 0 而 Δ|MDD| > 0，則該段起點之後
    的所有 L_high 被**嚴格支配**（更深回撤、無更多報酬）。
    """
    f = front.sort_values("l_high").reset_index(drop=True)
    bend_at, dominated = None, []
    for i in range(len(f) - 1):
        d_cagr = f.loc[i + 1, "cagr"] - f.loc[i, "cagr"]
        d_mdd = abs(f.loc[i + 1, "mdd"]) - abs(f.loc[i, "mdd"])
        if d_cagr <= 0 and d_mdd > 0:
            if bend_at is None:
                bend_at = float(f.loc[i, "l_high"])
            dominated.append(float(f.loc[i + 1, "l_high"]))
    return {"bend_at": bend_at, "dominated": dominated,
            "cagr_still_rising": bend_at is None,
            "cagr_total_change": float(f["cagr"].iloc[-1] - f["cagr"].iloc[0]),
            "mdd_total_change": float(abs(f["mdd"].iloc[-1]) - abs(f["mdd"].iloc[0]))}


def marginal_table(front: pd.DataFrame) -> pd.DataFrame:
    """邊際分析（規格 §5-4）：每 +0.1 槓桿換得多少 CAGR、付出多少 MDD。"""
    f = front.sort_values("l_high").reset_index(drop=True)
    rows = []
    for i in range(len(f) - 1):
        d_cagr = f.loc[i + 1, "cagr"] - f.loc[i, "cagr"]
        d_mdd = abs(f.loc[i + 1, "mdd"]) - abs(f.loc[i, "mdd"])
        d_shp = f.loc[i + 1, "sharpe"] - f.loc[i, "sharpe"]
        rows.append({
            "from_l": f.loc[i, "l_high"], "to_l": f.loc[i + 1, "l_high"],
            "d_cagr": d_cagr, "d_mdd_deeper": d_mdd, "d_sharpe": d_shp,
            "trade_ratio": (d_cagr / d_mdd) if d_mdd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def efficiency_table(front: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """每單位平均槓桿的超額 CAGR（規格 §5-3）：報酬增量是否只是曝險的線性結果。"""
    f = front.sort_values("l_high").copy()
    f["excess_cagr"] = f["cagr"] - rf
    f["cagr_per_lev"] = f["excess_cagr"] / f["avg_lev"]
    base = f["cagr_per_lev"].iloc[0]
    f["eff_vs_base"] = f["cagr_per_lev"] / base - 1.0
    return f[["l_high", "cagr", "avg_lev", "cagr_per_lev", "eff_vs_base"]].reset_index(drop=True)


def mdd_convexity(front: pd.DataFrame) -> pd.DataFrame:
    """MDD 惡化是線性還是凸性（規格 §5-2、結論 Q3）：逐段惡化速率是否遞增。"""
    f = front.sort_values("l_high").reset_index(drop=True)
    rows = []
    for i in range(len(f) - 1):
        rows.append({"from_l": f.loc[i, "l_high"], "to_l": f.loc[i + 1, "l_high"],
                     "mdd_deepening": abs(f.loc[i + 1, "mdd"]) - abs(f.loc[i, "mdd"])})
    t = pd.DataFrame(rows)
    if len(t) > 1:
        t["accelerating"] = t["mdd_deepening"].diff() > 0
    return t
