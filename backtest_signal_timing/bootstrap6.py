"""配對 stationary bootstrap：同一組合成路徑上比較時序變體（預先承諾 §3）。

沿用第三輪設定（期望區塊 60 日、1000 條 × 20 年）。相位固定 0：區塊重抽本身已打亂相位。
🔴 bootstrap 打斷長週期自相關，對趨勢策略有向下偏誤：本結果**只用方向、只能降級不能升級**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket.bootstrap import stationary_idx
from backtest_multimarket.data_mkts import Market, ann_to_daily
from backtest_multimarket.engine import path_metrics, simulate
from backtest_multimarket.params import (
    BOOT_MEAN_BLOCK, BOOT_N_PATHS, BOOT_YEARS, FINANCING_SPREAD,
    IDX_COST_PER_SIDE, TRADING_DAYS,
)

from backtest_signal_timing.params import BASELINE_K, BASELINE_LAG, K_GRID
from backtest_signal_timing.sweep6 import apply_lag, trend_leverage_phased

CONTRASTS = [(k, 1) for k in K_GRID] + [(5, 0), (1, 0), (5, 2)]


def run_paired(mkt: Market, *, n_paths: int = BOOT_N_PATHS,
               mean_block: float = BOOT_MEAN_BLOCK, seed: int = 0) -> pd.DataFrame:
    """單一市場：每個對比變體相對基準 (5,1) 的配對 Sharpe 差分布。"""
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

    sharpe, ruin, lev = {}, {}, {}
    for k, lag in CONTRASTS:
        sig = trend_leverage_phased(spot, check_every=k, phase=0)
        res = simulate({"kind": "signal", "L_exec": apply_lag(sig, lag)}, **kw)
        m = path_metrics(res, kw["rf_dep"])
        sharpe[(k, lag)] = m["sharpe"].to_numpy()
        ruin[(k, lag)] = float(m["bankrupt"].mean())
        lev[(k, lag)] = float(m["avg_lev"].median())

    base = sharpe[(BASELINE_K, BASELINE_LAG)]
    rows = []
    for v in CONTRASTS:
        diff = sharpe[v] - base
        rows.append({"market": mkt.key, "K": v[0], "lag": v[1],
                     "med_diff": float(np.nanmedian(diff)),
                     "share_better": float(np.nanmean(diff > 0)),
                     "med_sharpe": float(np.nanmedian(sharpe[v])),
                     "ruin_prob": ruin[v], "med_avg_lev": lev[v]})
    return pd.DataFrame(rows)


def same_sign_counter(boot: pd.DataFrame):
    """回傳 f(K, lag) → 配對 Sharpe 差中位數 > 0 的市場數（僅限對比變體，其餘回 0）。"""
    t = boot.groupby(["K", "lag"])["med_diff"].apply(lambda x: int((x > 0).sum()))

    def f(k: int, lag: int) -> int:
        return int(t.get((k, lag), 0))
    return f
