"""MA 變體的 stationary bootstrap（規格 §3-6）。

沿用第三輪設定（期望區塊 60 日、1000 條 × 20 年、參數凍結）。只跑 buffer=0.01 的
10 個 MA 變體。🔴 保留對趨勢策略的**向下偏誤**警語：bootstrap 打斷跨區塊序列相關，
趨勢 alpha 活在長週期自相關裡——本結果**不得用於窗口間淘汰**，僅檢查「窗口間相對
排序是否與歷史路徑一致」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import signals as sg
from backtest_multimarket.bootstrap import stationary_idx
from backtest_multimarket.data_mkts import Market, ann_to_daily
from backtest_multimarket.engine import path_metrics, simulate
from backtest_multimarket.params import (
    BOOT_MEAN_BLOCK, BOOT_N_PATHS, BOOT_YEARS, FINANCING_SPREAD,
    IDX_COST_PER_SIDE, INITIAL_EQUITY, TARGET_LEV, TRADING_DAYS,
)

from backtest_ma_robustness.params import BOOT_BUFFER, MA_GRID

PCTS = [5, 25, 50, 75, 95]


def run_bootstrap_ma(mkt: Market, *, buf: float = BOOT_BUFFER,
                     n_paths: int = BOOT_N_PATHS, mean_block: float = BOOT_MEAN_BLOCK,
                     seed: int = 0) -> pd.DataFrame:
    """單一市場：10 個 MA 變體（L_low=1.0, buffer=buf）在合成路徑上的分布。"""
    rng = np.random.default_rng(seed)
    n = BOOT_YEARS * TRADING_DAYS
    idx = stationary_idx(mkt.T, n, n_paths, mean_block, rng)
    ret = mkt.ret[idx]                                   # (n, P)
    spot = np.cumprod(1.0 + ret, axis=0)

    rf_ann = float(np.mean(mkt.rf_ann))
    kw = {"ret": ret, "rf_dep": ann_to_daily(rf_ann), "cost_rate": IDX_COST_PER_SIDE}
    if mkt.mode == "futures":
        kw.update(financing="futures", costs="bp")
    else:
        kw.update(financing="index", costs="bp",
                  rf_bor=ann_to_daily(rf_ann + FINANCING_SPREAD))

    rows = []
    for ma in MA_GRID:
        lev = sg.trend_leverage(spot, 1.0, ma_win=ma, buf=buf, l_high=TARGET_LEV)
        spec = {"kind": "signal", "L_exec": sg.to_exec(lev)}
        res = simulate(spec, **kw)
        m = path_metrics(res, kw["rf_dep"])
        tw = res.eq[-1] / INITIAL_EQUITY
        row = {"market": mkt.key, "ma": ma, "buffer": buf,
               "ruin_prob": float(m["bankrupt"].mean()),
               "med_avg_lev": float(m["avg_lev"].median())}
        for p in PCTS:
            row[f"sharpe_p{p}"] = float(np.nanpercentile(m["sharpe"], p))
            row[f"mdd_p{p}"] = float(np.percentile(m["mdd"], p))
            row[f"tw_p{p}"] = float(np.percentile(tw, p))
        rows.append(row)
    return pd.DataFrame(rows)


def hist_vs_boot_ordering(maximin: pd.DataFrame, boot: pd.DataFrame, *,
                          buf: float = BOOT_BUFFER) -> pd.DataFrame:
    """窗口間排序一致性：歷史 maximin（buffer=buf 的 MA 變體）vs bootstrap Sharpe p50。"""
    from backtest_ma_robustness.params import variant_key
    hist = maximin.set_index("strategy")
    rows = []
    for ma in MA_GRID:
        v = variant_key(ma, buf)
        if v not in hist.index:
            continue
        rows.append({"ma": ma, "hist_worst_pct": float(hist.loc[v, "worst_pct_rank"]),
                     "hist_mean_pct": float(hist.loc[v, "mean_pct_rank"])})
    h = pd.DataFrame(rows)
    b = boot.groupby("ma")["sharpe_p50"].mean().reset_index()
    m = h.merge(b, on="ma")
    m["hist_rank"] = m["hist_mean_pct"].rank(ascending=False)
    m["boot_rank"] = m["sharpe_p50"].rank(ascending=False)
    return m
