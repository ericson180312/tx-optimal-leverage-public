"""D_10 趨勢過濾訊號狀態機——自 round3/signals.py trend_leverage 逐位元移植。

🔴 黃金測試（規格 §3.2）鎖定本模組：把第三輪回測的還原連續序列餵入，
regime 切換日期必須與回測輸出完全一致。因此：
- 數值運算順序（cumsum 建 MA、比較式）與 round3 完全相同，不得「等價改寫」。
- 檢查日規則 = 序列索引 t % SIGNAL_CHECK_EVERY == 0（t 自序列首日起算），
  非固定星期幾（見 config.SIGNAL_CHECK_EVERY 的口徑說明）。
- MA 可得前（前 199 日）預設 BULL（=滿載），與回測一致。
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from live import config

BULL, BEAR = "BULL", "BEAR"


def trend_leverage(spot: np.ndarray, l_low: float = config.L_LOW, *,
                   ma_win: int = config.MA_WINDOW,
                   buf: float = config.BUFFER,
                   check_every: int = config.SIGNAL_CHECK_EVERY,
                   l_high: float = config.L_HIGH) -> np.ndarray:
    """spot > 200MA(±1% 緩衝) → L_high，否則 L_low；每 5 個序列日檢查一次。

    緩衝帶遲滯：on→off 需收盤 < MA×(1−buf)；off→on 需 > MA×(1+buf)；帶內維持。
    實作與 round3.signals.trend_leverage 的 1D 路徑逐位元相同。
    """
    s = np.asarray(spot, dtype=float)[:, None]
    T, P = s.shape
    csum = np.vstack([np.zeros((1, P)), np.cumsum(s, axis=0)])
    state = np.ones(P, dtype=bool)
    out = np.empty((T, P))
    for t in range(T):
        if t >= ma_win - 1 and t % check_every == 0:
            ma = (csum[t + 1] - csum[t + 1 - ma_win]) / ma_win
            state = np.where(s[t] > ma * (1.0 + buf), True,
                             np.where(s[t] < ma * (1.0 - buf), False, state))
        out[t] = np.where(state, l_high, l_low)
    return out[:, 0]


def compute_regime(spot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (訊號槓桿陣列, regime 布林陣列 True=BULL)。"""
    lev = trend_leverage(spot)
    return lev, lev == config.L_HIGH


def regime_switches(dates: list[dt.date],
                    lev: np.ndarray) -> list[tuple[dt.date, str, str]]:
    """訊號槓桿序列 → (切換日, 前 regime, 後 regime) 清單。切換日=訊號日。"""
    out = []
    for i in range(1, len(lev)):
        if lev[i] != lev[i - 1]:
            frm = BULL if lev[i - 1] == config.L_HIGH else BEAR
            to = BULL if lev[i] == config.L_HIGH else BEAR
            out.append((dates[i], frm, to))
    return out


# T+1 執行索引與今日預覽索引由 live.data.series.SignalSeries 提供
# （exec_index / today_index），確保「嚴格早於今日」的口徑集中在資料層。
