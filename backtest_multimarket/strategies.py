"""策略家族的訊號建構與槓桿中性化校準（規格 §2、§3）。

σ_target 校準（🔴 §3 方法論核心）：
- 目標：每策略×每市場的時間加權平均槓桿 = 2.0 ± 0.1。
- 校準段 = 各市場**最早 10 年**（walk-forward）；之後凍結，不得對評估段或
  合成路徑重校準。校準以「訊號槓桿」的時間平均為準（確定性、無路徑相依），
  實際模擬的平均槓桿另行驗證並列於報告（含 deadband 漂移的差異）。
- D／E 無校準變數（由 regime 佔比／m 錨定），照實跑並在所有比較表並列平均槓桿。
- F1 沿用 C 的 σ_target（保險絲不改變訊號平均）；F2 以 min(L_C, L_D10) 的
  訊號平均獨立校準（可能觸及 clip 上限而無法達 2.0 → 取邊界並揭露）。
"""
from __future__ import annotations

import numpy as np

from backtest_multimarket import signals as sg
from backtest_multimarket.params import (
    B1_REBAL_EVERY, CALIB_TARGET_LEV, CALIB_YEARS, CPPI_CAP, CPPI_DMAX,
    EWMA_LAMBDA, FUSE_X, FUSE_Y, SENS_STRATEGIES, SIGMA_TARGET_BOUNDS,
    STRATEGIES, TARGET_LEV, TRADING_DAYS, TREND_L_LOW, VT_SD_WINDOWS,
)

ALL_STRATEGIES = STRATEGIES + SENS_STRATEGIES


def build_vol_estimates(ret: np.ndarray) -> dict[str, np.ndarray]:
    """三種 σ̂（年化）：EWMA(λ=0.94) 主、SD20/SD60 敏感度。

    初始值取最早 60 日的變異數（僅影響最初數十日的暖機，報告揭露）。
    """
    r = np.asarray(ret, dtype=float)
    warm = r[:60] if r.ndim == 1 else r[:60]
    init_var = float(np.var(warm)) if r.ndim == 1 else np.var(warm, axis=0)
    init_vol = np.sqrt(init_var * TRADING_DAYS)
    out = {"EWMA": sg.ewma_vol(r, EWMA_LAMBDA, init_var)}
    for w in VT_SD_WINDOWS:
        out[f"SD{w}"] = sg.rolling_vol(r, w, init_vol)
    return out


def build_trend_signals(spot: np.ndarray) -> dict[float, np.ndarray]:
    """D 家族訊號槓桿：L_low ∈ {0.5, 1.0}。"""
    return {ll: sg.trend_leverage(spot, ll) for ll in TREND_L_LOW}


def calibrate_market(ret: np.ndarray, spot: np.ndarray) -> dict:
    """單一市場的 σ_target 校準。

    兩套校準值（規格 §3-3）：
    - sigma（walk-forward）：以最早 CALIB_YEARS 年為校準段 —— 滾動視窗與
      壓力窗口的主口徑；評估段的實際平均槓桿可能因波動 regime 變遷而漂移
      （這本身是重要發現，報告揭露）。
    - sigma_is（全樣本 in-sample）：「自始以來」路徑與 bootstrap（合成路徑
      重抽自全樣本分布）使用 —— 依 §3-3「無法外樣本校準者以全樣本校準並
      標註 in-sample」。
    F2 沿用 C_EWMA 的 σ_target（規格 §2「L = min(L_C, L_D)」的組合定義；
    若讓 F2 自行灌大 σ_target 追平均槓桿 2.0，會退化成 D_10）。其平均槓桿
    受 regime 佔比支配，依 §3-4 照實跑並於所有比較表並列平均槓桿。
    """
    T = len(ret)
    n_cal = min(CALIB_YEARS * TRADING_DAYS, T)
    vols = build_vol_estimates(ret)
    trends = build_trend_signals(spot)
    lo, hi = SIGMA_TARGET_BOUNDS

    sigma: dict[str, float] = {}
    achieved: dict[str, float] = {}
    sigma_is: dict[str, float] = {}
    achieved_is: dict[str, float] = {}

    for est, key in [("EWMA", "C_EWMA"), ("SD20", "C_SD20"), ("SD60", "C_SD60")]:
        for sig_arr, s_dict, a_dict in [(vols[est][:n_cal], sigma, achieved),
                                        (vols[est], sigma_is, achieved_is)]:
            s, a = sg.calibrate_sigma_target(
                lambda x, v=sig_arr: float(np.mean(sg.vol_target_leverage(v, x))),
                lo, hi, CALIB_TARGET_LEV)
            s_dict[key], a_dict[key] = s, a
    for s_dict, a_dict in [(sigma, achieved), (sigma_is, achieved_is)]:
        s_dict["F1_VolFuse"] = s_dict["C_EWMA"]     # 同訊號，保險絲不動平均
        a_dict["F1_VolFuse"] = a_dict["C_EWMA"]
        s_dict["F2_VolTrend"] = s_dict["C_EWMA"]    # 組合定義：沿用 C 的 σ_target
        a_dict["F2_VolTrend"] = np.nan              # 平均槓桿無校準目標（§3-4）

    return {"sigma": sigma, "achieved": achieved,
            "sigma_is": sigma_is, "achieved_is": achieved_is,
            "calib_days": n_cal, "vols": vols, "trends": trends}


def leverage_signals(calib: dict, which: str = "wf") -> dict[str, np.ndarray]:
    """由校準結果組出各「訊號型」策略的訊號槓桿（未 shift）。

    which="wf"：walk-forward σ_target（滾動視窗主口徑）；
    which="is"：全樣本 in-sample σ_target（自始以來、bootstrap）。
    """
    vols, trends = calib["vols"], calib["trends"]
    s = calib["sigma"] if which == "wf" else calib["sigma_is"]
    L: dict[str, np.ndarray] = {}
    L["C_EWMA"] = sg.vol_target_leverage(vols["EWMA"], s["C_EWMA"])
    L["C_SD20"] = sg.vol_target_leverage(vols["SD20"], s["C_SD20"])
    L["C_SD60"] = sg.vol_target_leverage(vols["SD60"], s["C_SD60"])
    L["D_05"] = trends[0.5]
    L["D_10"] = trends[1.0]
    L["F1_VolFuse"] = sg.vol_target_leverage(vols["EWMA"], s["F1_VolFuse"])
    L["F2_VolTrend"] = np.minimum(
        sg.vol_target_leverage(vols["EWMA"], s["F2_VolTrend"]), trends[1.0])
    return L


def strategy_spec(name: str, L_exec: dict[str, np.ndarray]) -> dict:
    """策略名 → 引擎 spec（L_exec 為已 shift 的執行槓桿 dict）。"""
    if name == "B1_Const2x":
        return {"kind": "periodic", "target": TARGET_LEV, "every": B1_REBAL_EVERY}
    if name == "B2_Fuse":
        return {"kind": "fuse", "X": FUSE_X, "Y": FUSE_Y, "target": TARGET_LEV}
    if name == "B3_BH1x":
        return {"kind": "bh", "target": 1.0}
    if name.startswith(("C_",)) or name == "F2_VolTrend":
        return {"kind": "signal", "L_exec": L_exec[name]}
    if name.startswith("D_"):
        return {"kind": "signal", "L_exec": L_exec[name]}
    if name == "F1_VolFuse":
        return {"kind": "signal_fuse", "L_exec": L_exec[name],
                "X": FUSE_X, "target": TARGET_LEV}
    if name == "E_40":
        return {"kind": "cppi", "m": 2.0 / CPPI_DMAX[0], "dmax": CPPI_DMAX[0],
                "cap": CPPI_CAP}
    if name == "E_50":
        return {"kind": "cppi", "m": 2.0 / CPPI_DMAX[1], "dmax": CPPI_DMAX[1],
                "cap": CPPI_CAP}
    raise ValueError(name)


def exec_signals(calib: dict, which: str = "wf") -> dict[str, np.ndarray]:
    """訊號槓桿 shift 一日 → 執行槓桿（引擎直接使用）。"""
    return {k: sg.to_exec(v) for k, v in leverage_signals(calib, which).items()}
