"""每個 L_high 的 bootstrap 歸零／強制平倉機率（規格 §3-3，本輪最重要的單一數字）。

沿用第三／四輪 stationary bootstrap 設定（期望區塊 60 日、1000 條 × 20 年、
參數凍結不重校準），並開啟維持保證金強制平倉。

🔴 警語（沿用）：bootstrap 打斷跨區塊序列相關，趨勢策略的 alpha 活在長週期
自相關裡 → 對 D_10 全體有結構性向下偏誤。本表僅用於 **L_high 之間的相對比較**，
不作絕對水準的判斷（規格 §3-3）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import engine as eng, signals as sg
from backtest_multimarket.bootstrap import stationary_idx
from backtest_multimarket.data_mkts import Market, ann_to_daily
from backtest_multimarket.params import (
    FINANCING_SPREAD, IDX_COST_PER_SIDE, INITIAL_EQUITY,
)

from backtest_leverage_sweep.params import (
    BOOT_BLOCK, BOOT_N_PATHS, BOOT_YEARS, L_HIGH_GRID, L_LOW,
    MAINT_MARGIN_RATE, SCALED_L_LOW, TRADING_DAYS, TREND_BUFFER, TREND_CHECK,
    TREND_MA, variant_key,
)

PCTS = [5, 25, 50, 75, 95]


def run_bootstrap_leverage(mkt: Market, *, n_paths: int = BOOT_N_PATHS,
                           mean_block: float = BOOT_BLOCK,
                           maint: float = MAINT_MARGIN_RATE,
                           scaled: bool = False, seed: int = 0) -> pd.DataFrame:
    """單一市場：各 L_high 在合成路徑上的分布與歸零／強平機率。"""
    rng = np.random.default_rng(seed)
    n = BOOT_YEARS * TRADING_DAYS
    idx = stationary_idx(mkt.T, n, n_paths, mean_block, rng)
    ret = mkt.ret[idx]
    spot = np.cumprod(1.0 + ret, axis=0)

    rf_ann = float(np.mean(mkt.rf_ann))
    kw = {"ret": ret, "rf_dep": ann_to_daily(rf_ann), "cost_rate": IDX_COST_PER_SIDE}
    if mkt.mode == "futures":
        kw.update(financing="futures", costs="bp")
    else:
        kw.update(financing="index", costs="bp",
                  rf_bor=ann_to_daily(rf_ann + FINANCING_SPREAD))

    rows = []
    for lh in L_HIGH_GRID:
        l_low = SCALED_L_LOW[lh] if scaled else L_LOW
        lev = sg.trend_leverage(spot, l_low, ma_win=TREND_MA, buf=TREND_BUFFER,
                                check_every=TREND_CHECK, l_high=lh)
        res = eng.simulate({"kind": "signal", "L_exec": sg.to_exec(lev)},
                           maint_margin_rate=maint, **kw)
        m = eng.path_metrics(res, kw["rf_dep"])
        died = m["bankrupt"].to_numpy() | m["liquidated"].to_numpy()
        tw = res.eq[-1] / INITIAL_EQUITY
        s = m[~died]
        row = {"market": mkt.key, "l_high": lh, "n_paths": n_paths,
               "ruin_prob": float(died.mean()),
               "liq_prob": float(m["liquidated"].mean()),
               "bankrupt_prob": float(m["bankrupt"].mean()),
               "med_avg_lev": float(m["avg_lev"].median())}
        # 分布統計同樣採存活者口徑（規格 §8）
        for p in PCTS:
            row[f"sharpe_p{p}"] = (float(np.nanpercentile(s["sharpe"], p))
                                   if len(s) else np.nan)
            row[f"mdd_p{p}"] = (float(np.percentile(s["mdd"], p))
                                if len(s) else np.nan)
            row[f"tw_p{p}"] = (float(np.percentile(tw[~died], p))
                               if died.sum() < len(tw) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def ruin_summary(boot: pd.DataFrame, veto: float) -> pd.DataFrame:
    """跨市場歸零機率彙總 + 一票否決判定（規格 §3-3）。"""
    g = boot.groupby("l_high").agg(
        max_ruin_prob=("ruin_prob", "max"),
        mean_ruin_prob=("ruin_prob", "mean"),
        max_liq_prob=("liq_prob", "max"),
        worst_market=("ruin_prob", lambda s: boot.loc[s.idxmax(), "market"]),
    ).reset_index()
    g["disqualified"] = g["max_ruin_prob"] > veto
    return g
