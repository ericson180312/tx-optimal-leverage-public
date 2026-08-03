"""L_high 掃描與「存活者口徑」聚合（規格 §1、§4、§8）。

🔴 存活者紀律（規格 §8）：破產／強制平倉的路徑**一律排除於報酬統計之外**，
但計入生存率欄位——否則會出現「歸零前漲很多」的假優勝者（第二輪踩過的坑）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import engine as eng, evaluate as ev, signals as sg
from backtest_multimarket.data_mkts import Market

from backtest_leverage_sweep.params import (
    L_HIGH_GRID, L_LOW, MAINT_MARGIN_RATE, PERIOD_YEARS, SCALED_L_LOW,
    TRADING_DAYS, TREND_BUFFER, TREND_CHECK, TREND_MA, variant_key,
)


def build_variant_signals(spot: np.ndarray, l_grid=L_HIGH_GRID,
                          scaled: bool = False) -> dict[str, np.ndarray]:
    """各 L_high 的 D_10 執行槓桿（MA/buffer/週頻/T+1 全部凍結，只換 L_high）。"""
    out = {}
    for lh in l_grid:
        l_low = SCALED_L_LOW[lh] if scaled else L_LOW
        lev = sg.trend_leverage(spot, l_low, ma_win=TREND_MA, buf=TREND_BUFFER,
                                check_every=TREND_CHECK, l_high=lh)
        out[variant_key(lh, scaled)] = sg.to_exec(lev)
    return out


def run_windows_margin(mkt: Market, L_exec: dict[str, np.ndarray],
                       starts: np.ndarray, n: int, names: list[str], *,
                       maint: float = MAINT_MARGIN_RATE,
                       keep_sims: bool = False) -> tuple[dict, dict]:
    """沿用第三輪視窗切片與成本／融資設定，另開啟維持保證金強制平倉。"""
    kw = ev._sim_kwargs(mkt, starts, n)
    Lw = {k: ev.gather(v, starts, n) for k, v in L_exec.items()}
    metrics, sims = {}, {}
    for name in names:
        res = eng.simulate({"kind": "signal", "L_exec": Lw[name]},
                           maint_margin_rate=maint, **kw)
        metrics[name] = eng.path_metrics(res, kw["rf_dep"])
        if keep_sims:
            sims[name] = res
    return metrics, sims


def aggregate_survivor(metrics: dict[str, pd.DataFrame], mkt_key: str,
                       years) -> pd.DataFrame:
    """存活者口徑聚合：報酬／風險統計僅計存活路徑，另列生存率（規格 §4、§8）。"""
    rows = []
    for name, m in metrics.items():
        died = m["bankrupt"].to_numpy() | m["liquidated"].to_numpy()
        s = m[~died]
        n_all = len(m)
        row = {
            "market": mkt_key, "period_years": years, "strategy": name,
            "n_paths": n_all,
            "n_liquidated": int(m["liquidated"].sum()),
            "n_bankrupt": int(m["bankrupt"].sum()),
            "n_dead": int(died.sum()),
            "survival_rate": float((~died).mean()),
        }
        if len(s) == 0:                      # 全滅 → 報酬統計無意義
            row.update({k: np.nan for k in (
                "med_cagr", "med_cumret", "worst_cumret", "med_sharpe",
                "med_sortino", "med_calmar", "med_mdd", "mdd_p25", "mdd_p75",
                "med_uw_months", "med_avg_lev", "med_lev_at_trough",
                "med_max_lev", "med_cost_ratio", "med_n_adj")})
            rows.append(row)
            continue
        row.update({
            "med_cagr": float(s["cagr"].median()),
            "med_cumret": float(s["cumret"].median()),
            "worst_cumret": float(s["cumret"].min()),
            "med_sharpe": float(np.nanmedian(s["sharpe"])),
            "med_sortino": float(np.nanmedian(s["sortino"])),
            "med_calmar": float(np.nanmedian(s["calmar"])),
            "med_mdd": float(s["mdd"].median()),
            "mdd_p25": float(np.percentile(s["mdd"], 25)),   # 較深的一側
            "mdd_p75": float(np.percentile(s["mdd"], 75)),
            "med_uw_months": float(s["uw_months"].median()),
            "med_avg_lev": float(s["avg_lev"].median()),
            "med_lev_at_trough": float(s["lev_at_trough"].median()),
            "med_max_lev": float(s["max_lev"].median()),
            "med_cost_ratio": float(s["cost_ratio"].median()),
            "med_n_adj": float(s["n_adj"].median()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_sweep(mkts: dict[str, Market], *, scaled: bool = False,
              maint: float = MAINT_MARGIN_RATE,
              markets=None, periods=PERIOD_YEARS) -> pd.DataFrame:
    """全市場 × 全期間 × 全 L_high 的滾動視窗掃描（存活者口徑）。"""
    markets = markets or list(mkts)
    names = [variant_key(lh, scaled) for lh in L_HIGH_GRID]
    frames = []
    for k in markets:
        m = mkts[k]
        L_exec = build_variant_signals(m.spot, scaled=scaled)
        for y in periods:
            n = y * TRADING_DAYS
            starts = ev.window_starts(m.T, n)
            if len(starts) == 0:
                continue
            metrics, _ = run_windows_margin(m, L_exec, starts, n, names, maint=maint)
            frames.append(aggregate_survivor(metrics, k, y))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_inception(mkts: dict[str, Market], *, maint: float = MAINT_MARGIN_RATE
                  ) -> tuple[pd.DataFrame, dict]:
    """自始以來單一路徑（敘述性質）＋權益/槓桿曲線。"""
    names = [variant_key(lh) for lh in L_HIGH_GRID]
    rows, curves = [], {}
    for k, m in mkts.items():
        L_exec = build_variant_signals(m.spot)
        starts = np.array([0], dtype=np.int64)
        metrics, sims = run_windows_margin(m, L_exec, starts, m.T, names,
                                           maint=maint, keep_sims=True)
        for name in names:
            r = metrics[name].iloc[0].to_dict()
            r.update(market=k, strategy=name, period="inception")
            rows.append(r)
            curves[(k, name)] = (m.dates, sims[name].eq[:, 0], sims[name].lev[:, 0])
    return pd.DataFrame(rows), curves


def leverage_realized_table(sweep: pd.DataFrame) -> pd.DataFrame:
    """L_high → 實際時間加權平均槓桿（規格 §4 必列：L_high=2.5 ≠ 平均 2.5）。"""
    d = sweep.copy()
    d["l_high"] = d["strategy"].str.extract(r"L(\d+)").astype(int) / 100.0
    g = d.groupby("l_high").agg(
        med_avg_lev=("med_avg_lev", "median"),
        min_avg_lev=("med_avg_lev", "min"),
        max_avg_lev=("med_avg_lev", "max"),
        med_max_lev=("med_max_lev", "median"),
        med_lev_at_trough=("med_lev_at_trough", "median"))
    return g.reset_index()
