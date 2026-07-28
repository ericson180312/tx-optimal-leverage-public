"""壓力窗口的 regime 交互作用（規格 §4）：H1（長均線×長陰跌）、H2（短均線×V反）。

每窗口 × 每 MA 變體（buffer=0.01）輸出：期間報酬、MDD、平均槓桿、切換次數、
假訊號次數。切換/假訊號用「全序列訊號切到窗口」計（保留窗前 MA 歷史，才是真實
訊號行為）；報酬/MDD/槓桿用第三輪引擎（融資與成本一致）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_multimarket import evaluate as ev, signals as sg
from backtest_multimarket.data_mkts import Market
from backtest_multimarket.params import STRESS_WINDOWS, TARGET_LEV

from backtest_ma_robustness.params import (
    BOOT_BUFFER, MA_GRID, MA_LONG, MA_SHORT, variant_key,
)
from backtest_ma_robustness.sweep import build_variant_signals, regime_stats


def run_stress(mkts: dict[str, Market], *, buf: float = BOOT_BUFFER) -> tuple[pd.DataFrame, dict]:
    """每 (窗口, 市場, MA) 的壓力表 + 槓桿路徑（供日經圖）。"""
    variant_keys = [variant_key(ma, buf) for ma in MA_GRID]
    rows, lev_paths = [], {}
    for label, mkt_keys, d0, d1 in STRESS_WINDOWS:
        keys = mkt_keys if mkt_keys else list(mkts)
        for k in keys:
            if k not in mkts:
                continue
            m = mkts[k]
            s = int(m.dates.searchsorted(pd.Timestamp(d0)))
            e = int(m.dates.searchsorted(pd.Timestamp(d1), side="right"))
            if e - s < 40 or (s == 0 and m.dates[0] > pd.Timestamp(d0) + pd.Timedelta(days=370)):
                continue
            var_L = build_variant_signals(m.spot, 1.0, buffers=[buf])
            starts = np.array([s], dtype=np.int64)
            metrics, sims = ev.run_windows(m, var_L, starts, e - s, variant_keys,
                                           keep_sims=True)
            for ma in MA_GRID:
                v = variant_key(ma, buf)
                met = metrics[v].iloc[0]
                full_sig = sg.trend_leverage(m.spot, 1.0, ma_win=ma, buf=buf,
                                             l_high=TARGET_LEV)
                rg = regime_stats(full_sig[s:e])
                rows.append({
                    "window": label, "market": k, "ma": ma,
                    "start": str(m.dates[s].date()), "end": str(m.dates[e - 1].date()),
                    "cumret": float(met["cumret"]), "mdd": float(met["mdd"]),
                    "avg_lev": float(met["avg_lev"]),
                    "lev_at_trough": float(met["lev_at_trough"]),
                    "n_switches": rg["n_switches"], "n_false": rg["n_false_signals"],
                    "share_derisked": rg["share_derisked"],
                })
                lev_paths[(label, k, ma)] = (m.dates[s:e], sims[v].lev[:, 0])
    return pd.DataFrame(rows), lev_paths


def test_hypotheses(stress: pd.DataFrame) -> dict:
    """檢驗 H1/H2（規格 §4 預先寫定）。回傳布林與量化取捨幅度。"""
    # H1：長均線在日經 1990–2003 長陰跌中較優（報酬較高／假訊號較少）
    nk = stress[(stress["window"].str.startswith("日經1990")) & (stress["market"] == "NKX")]
    h1 = {}
    if not nk.empty:
        long_r = nk[nk["ma"].isin(MA_LONG)]["cumret"].mean()
        short_r = nk[nk["ma"].isin(MA_SHORT)]["cumret"].mean()
        long_f = nk[nk["ma"].isin(MA_LONG)]["n_false"].mean()
        short_f = nk[nk["ma"].isin(MA_SHORT)]["n_false"].mean()
        h1 = {"long_cumret": long_r, "short_cumret": short_r,
              "cumret_gap": long_r - short_r,
              "long_false": long_f, "short_false": short_f,
              "hold": bool(long_r > short_r)}

    # H2：短均線在 V 型反彈（2020 COVID，全市場）中較優（報酬較高）
    cov = stress[stress["window"].str.startswith("2020")]
    h2 = {}
    if not cov.empty:
        by_ma = cov.groupby("ma")["cumret"].mean()
        long_r = by_ma[[x for x in MA_LONG if x in by_ma.index]].mean()
        short_r = by_ma[[x for x in MA_SHORT if x in by_ma.index]].mean()
        h2 = {"long_cumret": long_r, "short_cumret": short_r,
              "cumret_gap": short_r - long_r, "hold": bool(short_r > long_r)}

    both = bool(h1.get("hold") and h2.get("hold"))
    return {"H1": h1, "H2": h2, "both_hold": both,
            "regime_bet": both,
            "text": ("H1 與 H2 同時成立 → 窗口長度是又一個 regime 賭注，"
                     "不存在單一「最佳窗口」，兩端各有取捨。" if both else
                     "H1、H2 未同時成立 → 窗口長度不是純 regime 賭注（見數值）。")}
