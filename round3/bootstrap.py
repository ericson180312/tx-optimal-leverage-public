"""區塊拔靴合成路徑（規格 §5）：stationary bootstrap（Politis–Romano）。

- 對每市場的日報酬序列重抽（期望區塊長度 60 交易日，幾何分布；敏感度 20/120），
  合成 n_paths 條 × 20 年（5040 交易日）路徑。
- 校準參數（σ_target、策略參數）**凍結**，不得對合成路徑重校準。
- 訊號在合成路徑上重新計算：σ̂ 的 EWMA 初始值、趨勢 MA 的暖機（前 199 日
  預設滿載）沿用歷史口徑；趨勢訊號的「現貨」以合成報酬的 cumprod 代理。
- 成本／融資簡化（報告揭露）：所有市場一律 5bp 名目調整成本；TX 用期貨口徑
  （權益全額計息、無融資成本；合成路徑無合約行事曆 → 不另計轉倉成本，
  實測 TX 每邊成本 ≈2.5–3bp，5bp 屬保守）；海外用融資模型、短率取該市場
  歷史平均（常數，凍結）。
- 🔴 解讀警語（報告必列）：區塊拔靴打斷跨區塊序列相關 —— 趨勢策略（D/F2）的
  alpha 活在長週期自相關裡，其 bootstrap 績效有結構性向下偏誤，排名不得用於
  淘汰；對 vol targeting／CPPI／保險絲則屬公平測試（波動聚集在區塊內保留）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from round3 import signals as sg
from round3 import strategies as st
from round3.data_mkts import Market, ann_to_daily
from round3.engine import path_metrics, simulate
from round3.params import (
    BOOT_MEAN_BLOCK, BOOT_N_PATHS, BOOT_YEARS, EWMA_LAMBDA, FINANCING_SPREAD,
    IDX_COST_PER_SIDE, INITIAL_EQUITY, TRADING_DAYS, TREND_L_LOW,
    VT_SD_WINDOWS,
)

PCTS = [5, 25, 50, 75, 95]


def stationary_idx(t_src: int, n: int, p: int, mean_block: float,
                   rng: np.random.Generator) -> np.ndarray:
    """(n,P) 重抽索引：每日以 1/mean_block 機率跳到新的均勻起點，否則順延（環繞）。"""
    idx = np.empty((n, p), dtype=np.int64)
    cur = rng.integers(0, t_src, p)
    p_new = 1.0 / mean_block
    for t in range(n):
        idx[t] = cur
        jump = rng.random(p) < p_new
        nxt = np.where(jump, rng.integers(0, t_src, p), (cur + 1) % t_src)
        cur = nxt
    return idx


def run_bootstrap(mkt: Market, calib: dict, names: list[str], *,
                  n_paths: int = BOOT_N_PATHS, mean_block: float = BOOT_MEAN_BLOCK,
                  seed: int = 0) -> dict:
    """單一市場的 bootstrap：全部策略、凍結參數。回傳分布與逐路徑指標。"""
    rng = np.random.default_rng(seed)
    n = BOOT_YEARS * TRADING_DAYS
    idx = stationary_idx(mkt.T, n, n_paths, mean_block, rng)
    ret = mkt.ret[idx]                                    # (n,P)

    # 訊號（凍結參數；σ̂ 初始值 = 歷史全樣本日變異數）
    init_var = float(np.var(mkt.ret))
    vols = {"EWMA": sg.ewma_vol(ret, EWMA_LAMBDA, init_var)}
    for w in VT_SD_WINDOWS:
        vols[f"SD{w}"] = sg.rolling_vol(ret, w, np.sqrt(init_var * TRADING_DAYS))
    spot = np.cumprod(1.0 + ret, axis=0)
    trends = {ll: sg.trend_leverage(spot, ll) for ll in TREND_L_LOW}
    # 合成路徑重抽自全樣本分布 → σ_target 用全樣本校準值（in-sample，凍結）
    boot_calib = {"vols": vols, "trends": trends, "sigma": calib["sigma_is"]}
    L_exec = st.exec_signals(boot_calib, which="wf")

    rf_ann = float(np.mean(mkt.rf_ann))                   # 常數短率（凍結）
    kw = {"ret": ret, "rf_dep": ann_to_daily(rf_ann), "cost_rate": IDX_COST_PER_SIDE}
    if mkt.mode == "futures":
        kw.update(financing="futures", costs="bp")
    else:
        kw.update(financing="index", costs="bp",
                  rf_bor=ann_to_daily(rf_ann + FINANCING_SPREAD))

    dist_rows, per_path = [], {}
    for name in names:
        spec = st.strategy_spec(name, L_exec)
        res = simulate(spec, **kw)
        m = path_metrics(res, kw["rf_dep"])
        tw = res.eq[-1] / INITIAL_EQUITY                  # 終端財富倍數
        row = {"market": mkt.key, "strategy": name, "mean_block": mean_block,
               "n_paths": n_paths,
               "ruin_prob": float(m["bankrupt"].mean()),
               "med_avg_lev": float(m["avg_lev"].median())}
        for pct in PCTS:
            row[f"sharpe_p{pct}"] = float(np.nanpercentile(m["sharpe"], pct))
            row[f"mdd_p{pct}"] = float(np.percentile(m["mdd"], pct))
            row[f"tw_p{pct}"] = float(np.percentile(tw, pct))
        dist_rows.append(row)
        per_path[name] = pd.DataFrame({"sharpe": m["sharpe"], "mdd": m["mdd"],
                                       "terminal_wealth": tw,
                                       "bankrupt": m["bankrupt"]})
    return {"dist": pd.DataFrame(dist_rows), "per_path": per_path}
