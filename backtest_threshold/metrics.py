"""績效指標（規格 §6）— 由權益曲線 (T,K) 一次算出 K 組指標。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_threshold.params import DAILY_RF, IDLE_CASH_RATE, INITIAL_EQUITY, TRADING_DAYS_PER_YEAR

from backtest_threshold.engine import SimResult


def _longest_true_run(mask: np.ndarray) -> int:
    """一維布林陣列中最長連續 True 的長度。"""
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return int(np.max(edges[1::2] - edges[0::2]))


def path_metrics(res: SimResult, *, e0: float = INITIAL_EQUITY,
                 rf_daily: float = DAILY_RF) -> pd.DataFrame:
    """單一路徑上 K 組參數的完整指標表（K rows）。

    破產組的 Sharpe/Sortino/Calmar 設為 NaN（權益歸零後統計無意義），
    cumret / CAGR / MDD 保留其真實值（-100%）。
    """
    eq, lev = res.eq, res.lev
    T, K = eq.shape
    years = T / TRADING_DAYS_PER_YEAR

    # 日報酬（首日相對 e0；破產後權益 0/0 → 報酬記 0）
    prev = np.vstack([np.full((1, K), e0), eq[:-1]])
    r = np.divide(eq - prev, prev, out=np.zeros_like(eq), where=prev > 0)
    ex = r - rf_daily

    cumret = eq[-1] / e0 - 1.0
    cagr = np.where(eq[-1] > 0, (eq[-1] / e0) ** (1.0 / years) - 1.0, -1.0)

    mean_ex = ex.mean(axis=0)
    std_ex = ex.std(axis=0, ddof=1)
    vol_ann = r.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = np.divide(mean_ex, std_ex, out=np.full(K, np.nan), where=std_ex > 0) \
        * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sortino：下檔偏差（相對無風險利率），全樣本分母
    neg = np.minimum(ex, 0.0)
    dd_ann = np.sqrt((neg ** 2).mean(axis=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = np.divide(mean_ex * TRADING_DAYS_PER_YEAR, dd_ann,
                        out=np.full(K, np.nan), where=dd_ann > 0)

    # 最大回撤（峰值含起始權益 e0）
    peak = np.maximum.accumulate(np.vstack([np.full((1, K), e0), eq]), axis=0)[1:]
    dd = eq / peak - 1.0
    mdd = dd.min(axis=0)
    calmar = np.divide(cagr, -mdd, out=np.full(K, np.nan), where=mdd < 0)

    # 最長水下期間（月 ≈ 天/21）
    under = dd < -1e-12
    uw_months = np.array([_longest_true_run(under[:, k]) for k in range(K)]) / 21.0

    out = pd.DataFrame({
        "cumret": cumret,
        "cagr": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "sortino": sortino,
        "n_up": res.n_up,
        "n_down": res.n_down,
        "cost_ratio": res.cost_cum / e0,
        "avg_lev": lev.mean(axis=0),
        "max_lev": res.max_lev,
        "uw_months": uw_months,
        "bankrupt": res.bankrupt,
    })
    # 破產組：風險調整指標無意義 → NaN
    out.loc[out["bankrupt"], ["sharpe", "sortino", "calmar"]] = np.nan
    return out
