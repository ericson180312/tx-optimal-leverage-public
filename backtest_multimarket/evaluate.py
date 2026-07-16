"""歷史路徑評估（規格 §4、§7、§8）：滾動多路徑、自始以來、壓力窗口、maximin。

滾動視窗口徑：視窗長度 = years×252 個交易日，起點以 63 個交易日（≈季頻）平移，
另含「終點=資料尾」的基準視窗。訊號一律取自全序列（視窗起點前的市場資訊在
當時已可得，屬合法）；CPPI 的 HWM 與保險絲狀態於視窗起點重置（策略新開倉）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import strategies as st
from backtest_multimarket.data_mkts import Market, ann_to_daily
from backtest_multimarket.engine import path_metrics, simulate
from backtest_multimarket.params import (
    CALIB_YEARS, FINANCING_SPREAD, IDX_COST_PER_SIDE, PERIOD_YEARS,
    RANKING_MARKETS, STRATEGIES, STRESS_WINDOWS, TRADING_DAYS, WINDOW_STEP,
)


def gather(a: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """(T,) → (n, P)：P 個視窗的切片矩陣。"""
    return np.asarray(a)[starts[None, :] + np.arange(n)[:, None]]


def window_starts(T: int, n: int, step: int = WINDOW_STEP) -> np.ndarray:
    """滾動起點（0, step, 2·step, …）＋ 基準視窗（終點=資料尾），去重排序。"""
    if n > T:
        return np.array([], dtype=np.int64)
    s = np.arange(0, T - n + 1, step, dtype=np.int64)
    return np.unique(np.append(s, T - n))


def _sim_kwargs(mkt: Market, starts: np.ndarray, n: int,
                cost_rate: float = IDX_COST_PER_SIDE) -> dict:
    """組出 simulate() 的市場端參數（視窗 gather 後）。"""
    kw = {"ret": gather(mkt.ret, starts, n),
          "rf_dep": gather(mkt.rf_daily, starts, n)}
    if mkt.mode == "futures":
        kw.update(financing="futures", costs="lot",
                  price=gather(mkt.price, starts, n),
                  is_roll=gather(mkt.is_roll, starts, n),
                  p_near=gather(mkt.p_near, starts, n),
                  p_next=gather(mkt.p_next, starts, n),
                  entry_price=np.where(starts >= 1, mkt.price[np.maximum(starts - 1, 0)],
                                       mkt.price[0] / (1.0 + mkt.ret[0])))
    else:
        rf_bor = ann_to_daily(mkt.rf_ann + FINANCING_SPREAD)
        kw.update(financing="index", costs="bp", cost_rate=cost_rate,
                  rf_bor=gather(rf_bor, starts, n))
    return kw


def run_windows(mkt: Market, L_exec: dict[str, np.ndarray], starts: np.ndarray,
                n: int, names: list[str], *, cost_rate: float = IDX_COST_PER_SIDE,
                keep_sims: bool = False) -> tuple[dict[str, pd.DataFrame], dict]:
    """對一組視窗跑多個策略。回傳 {策略: 每視窗指標表}（P rows）。"""
    kw = _sim_kwargs(mkt, starts, n, cost_rate)
    Lw = {k: gather(v, starts, n) for k, v in L_exec.items()}
    metrics, sims = {}, {}
    for name in names:
        spec = st.strategy_spec(name, Lw)
        res = simulate(spec, **kw)
        metrics[name] = path_metrics(res, kw["rf_dep"])
        if keep_sims:
            sims[name] = res
    return metrics, sims


def aggregate(metrics: dict[str, pd.DataFrame], mkt_key: str, years,
              n_cal: int, starts: np.ndarray) -> pd.DataFrame:
    """每視窗指標 → 跨視窗聚合表（每策略一列）。"""
    rows = []
    in_cal = float((starts < n_cal).mean()) if len(starts) else np.nan
    for name, m in metrics.items():
        rows.append({
            "market": mkt_key, "period_years": years, "strategy": name,
            "n_paths": len(m),
            "med_sharpe": float(np.nanmedian(m["sharpe"])),
            "sharpe_p25": float(np.nanpercentile(m["sharpe"], 25)),
            "sharpe_p75": float(np.nanpercentile(m["sharpe"], 75)),
            "med_cumret": float(m["cumret"].median()),
            "med_cagr": float(m["cagr"].median()),
            "med_mdd": float(m["mdd"].median()),
            "worst_mdd": float(m["mdd"].min()),
            "worst_cumret": float(m["cumret"].min()),
            "med_avg_lev": float(m["avg_lev"].median()),
            "med_lev_at_trough": float(m["lev_at_trough"].median()),
            "med_time_derisked": float(m["time_derisked"].median()),
            "med_uw_months": float(m["uw_months"].median()),
            "med_cost_ratio": float(m["cost_ratio"].median()),
            "med_n_adj": float(m["n_adj"].median()),
            "med_fuse_trig": float(m["fuse_trig"].median()),
            "max_fuse_trig": int(m["fuse_trig"].max()),
            "n_bankrupt": int(m["bankrupt"].sum()),
            "any_bankrupt": bool(m["bankrupt"].any()),
            "in_calib_share": in_cal,
        })
    return pd.DataFrame(rows)


def run_market_all_periods(mkt: Market, L_exec: dict, n_cal: int,
                           names: list[str], *,
                           cost_rate: float = IDX_COST_PER_SIDE
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """單一市場 × 全部期間長度的滾動視窗聚合。

    回傳 (全部視窗, 純評估段視窗)：後者只含起點落在校準段之後的視窗
    （§3 walk-forward 的嚴格版；不足 8 條路徑的格子不列）。
    """
    frames_all, frames_eval = [], []
    for y in PERIOD_YEARS:
        n = y * TRADING_DAYS
        starts = window_starts(mkt.T, n)
        if len(starts) == 0:
            continue
        metrics, _ = run_windows(mkt, L_exec, starts, n, names, cost_rate=cost_rate)
        frames_all.append(aggregate(metrics, mkt.key, y, n_cal, starts))
        keep = starts >= n_cal
        if keep.sum() >= 8:
            sub = {k: v[keep].reset_index(drop=True) for k, v in metrics.items()}
            frames_eval.append(aggregate(sub, mkt.key, y, n_cal, starts[keep]))
    empty = pd.DataFrame()
    return (pd.concat(frames_all, ignore_index=True) if frames_all else empty,
            pd.concat(frames_eval, ignore_index=True) if frames_eval else empty)


def run_full_path(mkt: Market, L_exec: dict, names: list[str]) -> tuple[pd.DataFrame, dict]:
    """自始以來單一路徑（敘述性質）。回傳（指標表, {策略: (eq, lev)}）。"""
    starts = np.array([0], dtype=np.int64)
    metrics, sims = run_windows(mkt, L_exec, starts, mkt.T, names, keep_sims=True)
    rows = []
    curves = {}
    for name in names:
        m = metrics[name].iloc[0].to_dict()
        m.update(market=mkt.key, strategy=name, period="inception")
        rows.append(m)
        curves[name] = (sims[name].eq[:, 0], sims[name].lev[:, 0])
    return pd.DataFrame(rows), curves


def run_stress(mkts: dict[str, Market], all_L: dict[str, dict],
               names: list[str]) -> tuple[pd.DataFrame, dict]:
    """具名壓力窗口（規格 §4）。回傳（指標表, {(窗口,市場,策略): (dates, eq, lev)}）。"""
    rows, paths = [], {}
    for label, mkt_keys, d0, d1 in STRESS_WINDOWS:
        keys = mkt_keys if mkt_keys else list(mkts)
        for k in keys:
            mkt = mkts[k]
            s = int(mkt.dates.searchsorted(pd.Timestamp(d0)))
            e = int(mkt.dates.searchsorted(pd.Timestamp(d1), side="right"))
            if e - s < 40 or s == 0 and mkt.dates[0] > pd.Timestamp(d0) + pd.Timedelta(days=370):
                continue                      # 該市場資料不涵蓋此窗口
            starts = np.array([s], dtype=np.int64)
            n = e - s
            metrics, sims = run_windows(mkt, all_L[k], starts, n, names, keep_sims=True)
            for name in names:
                m = metrics[name].iloc[0]
                lev_path = sims[name].lev[:, 0]
                rows.append({
                    "window": label, "market": k, "strategy": name,
                    "start": str(mkt.dates[s].date()), "end": str(mkt.dates[e - 1].date()),
                    "cumret": m["cumret"], "mdd": m["mdd"],
                    "avg_lev": m["avg_lev"], "lev_start": float(lev_path[0]),
                    "lev_end": float(lev_path[-1]), "lev_min": m["min_lev"],
                    "lev_max": m["max_lev"], "lev_at_trough": m["lev_at_trough"],
                    "time_derisked": m["time_derisked"],
                    "fuse_trig": int(m["fuse_trig"]), "bankrupt": bool(m["bankrupt"]),
                })
                paths[(label, k, name)] = (mkt.dates[s:e], sims[name].eq[:, 0], lev_path)
    return pd.DataFrame(rows), paths


# ── 評比協定（規格 §8）──────────────────────────────────────────────────────

def cell_percentiles(agg: pd.DataFrame, metric: str,
                     names: list[str] = STRATEGIES,
                     markets: list[str] = RANKING_MARKETS) -> pd.DataFrame:
    """每（市場×期間）格：策略 metric 的百分位排名（0~1，越高越好）。"""
    df = agg[agg["strategy"].isin(names) & agg["market"].isin(markets)].copy()
    df["pct_rank"] = df.groupby(["market", "period_years"])[metric] \
                       .rank(pct=True, na_option="bottom")
    return df


def maximin_ranking(agg: pd.DataFrame, metric: str = "med_sharpe",
                    names: list[str] = STRATEGIES) -> pd.DataFrame:
    """主排名：每策略取所有格子百分位的最差值，由高至低排序。"""
    pct = cell_percentiles(agg, metric, names)
    g = pct.groupby("strategy")
    worst = pct.loc[g["pct_rank"].idxmin()].set_index("strategy")
    out = pd.DataFrame({
        "worst_pct_rank": g["pct_rank"].min(),
        "mean_pct_rank": g["pct_rank"].mean(),
        "worst_cell": worst["market"] + "×" + worst["period_years"].astype(str) + "y",
        "med_avg_lev": g["med_avg_lev"].median(),
        "any_bankrupt": g["any_bankrupt"].any(),
    }).sort_values(["worst_pct_rank", "mean_pct_rank"], ascending=False)
    return out.reset_index()


def retention_table(agg: pd.DataFrame) -> pd.DataFrame:
    """參數敏感度（平原度）：家族內變體的 med_sharpe 極差（每格），跨格中位數/最大。"""
    fams = {"C（σ̂ 估計法）": ["C_EWMA", "C_SD20", "C_SD60"],
            "D（L_low）": ["D_05", "D_10"],
            "E（D_max）": ["E_40", "E_50"]}
    rows = []
    for fam, members in fams.items():
        d = agg[agg["strategy"].isin(members) & agg["market"].isin(RANKING_MARKETS)]
        spread = d.groupby(["market", "period_years"])["med_sharpe"] \
                  .agg(lambda x: x.max() - x.min())
        wc = spread.idxmax()
        rows.append({"family": fam, "n_variants": len(members),
                     "med_spread": float(spread.median()),
                     "max_spread": float(spread.max()),
                     "worst_cell": f"{wc[0]}×{int(wc[1])}y"})
    return pd.DataFrame(rows)
