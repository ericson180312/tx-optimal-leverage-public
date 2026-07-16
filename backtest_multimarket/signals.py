"""條件式曝險的「可預計算訊號」：波動率估計、趨勢濾網、vol-target 槓桿。

口徑（規格 §2，避免 look-ahead）：
- 訊號在第 t 日收盤以「含當日」的資訊計算，於第 t+1 日收盤執行。
  本模組輸出的訊號陣列皆為「訊號值」；引擎端一律讀取 shift 一日後的
  desired-leverage（見 to_exec()），視窗切片時沿用全序列訊號 —— 視窗起點
  之前的市場資訊在當時已可得，屬合法資訊。
- 所有函式都同時支援 (T,) 單路徑與 (T,P) 多路徑（bootstrap）矩陣。
"""
from __future__ import annotations

import numpy as np

from backtest_multimarket.params import (
    TARGET_LEV, TRADING_DAYS, TREND_BUFFER, TREND_CHECK, TREND_MA, VT_CLIP,
)


def _as2d(a: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a[:, None], True
    return a, False


def ewma_vol(ret: np.ndarray, lam: float, init_var: float) -> np.ndarray:
    """EWMA 年化波動率 σ̂_t（含第 t 日報酬）。init_var＝日報酬變異數起始值。"""
    r, squeeze = _as2d(ret)
    T, P = r.shape
    out = np.empty((T, P))
    var = np.full(P, float(init_var))
    for t in range(T):
        var = lam * var + (1.0 - lam) * r[t] ** 2
        out[t] = var
    out = np.sqrt(out * TRADING_DAYS)
    return out[:, 0] if squeeze else out


def rolling_vol(ret: np.ndarray, window: int, init_vol_ann: float) -> np.ndarray:
    """滾動標準差年化波動率；不足 window 日的前段以無條件波動率補。"""
    r, squeeze = _as2d(ret)
    T, P = r.shape
    c1 = np.vstack([np.zeros((1, P)), np.cumsum(r, axis=0)])
    c2 = np.vstack([np.zeros((1, P)), np.cumsum(r ** 2, axis=0)])
    out = np.full((T, P), float(init_vol_ann))
    if T >= window:
        n = float(window)
        s1 = c1[window:] - c1[:-window]
        s2 = c2[window:] - c2[:-window]
        var = np.maximum((s2 - s1 ** 2 / n) / (n - 1.0), 0.0)
        out[window - 1:] = np.sqrt(var * TRADING_DAYS)
    return out[:, 0] if squeeze else out


def vol_target_leverage(sigma_hat: np.ndarray, sigma_target: float) -> np.ndarray:
    """C：L = clip(σ_target / σ̂, 0.5, 3.0)。"""
    with np.errstate(divide="ignore"):
        L = sigma_target / sigma_hat
    return np.clip(L, VT_CLIP[0], VT_CLIP[1])


def trend_leverage(spot: np.ndarray, l_low: float, *, ma_win: int = TREND_MA,
                   buf: float = TREND_BUFFER, check_every: int = TREND_CHECK,
                   l_high: float = TARGET_LEV) -> np.ndarray:
    """D：現貨 > 200MA(±1% 緩衝) → L_high，否則 L_low；訊號週頻檢查（每 5 交易日）。

    緩衝帶遲滯：狀態 on→off 需收盤 < MA×(1−buf)；off→on 需 > MA×(1+buf)；
    帶內維持原狀態。MA 可得前（前 199 日）預設 on（=滿載），報告揭露。
    """
    s, squeeze = _as2d(spot)
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
    return out[:, 0] if squeeze else out


def to_exec(signal_lev: np.ndarray) -> np.ndarray:
    """訊號 → 執行槓桿：shift 一日（第 t 列＝第 t−1 日收盤訊號；首列沿用首值）。"""
    return np.concatenate([signal_lev[:1], signal_lev[:-1]], axis=0)


def calibrate_sigma_target(mean_lev_fn, lo: float, hi: float,
                           target: float, *, tol: float = 1e-4,
                           max_iter: int = 60) -> tuple[float, float]:
    """bisection 求 σ_target 使 mean_lev_fn(σ)=target（單調遞增）。

    回傳 (σ_target, 達成的平均槓桿)。若邊界都達不到 target，取最接近的邊界
    （§3-4：無法精確校準者照實跑並揭露平均槓桿）。
    """
    f_lo, f_hi = mean_lev_fn(lo), mean_lev_fn(hi)
    if f_lo >= target:
        return lo, f_lo
    if f_hi <= target:
        return hi, f_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = mean_lev_fn(mid)
        if abs(f - target) < tol:
            return mid, f
        if f < target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, mean_lev_fn(mid)
