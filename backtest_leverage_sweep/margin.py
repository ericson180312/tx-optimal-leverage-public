"""保證金、強制平倉臨界跌幅與台股連續跌停生存窗口（規格 §3-1、§3-2）。

推導（單筆持倉、期間不再平衡——跌停時本來就無流動性可調整）：
  期初權益 E₀、槓桿 L → 名目 N₀ = L·E₀。標的下跌 x 後：
      E(x) = E₀(1 − L·x)          N(x) = L·E₀(1 − x)
  強制平倉條件 E < m·N：
      E₀(1 − Lx) < m·L·E₀(1 − x)
      ⇒ x_liq(精確) = (1 − mL) / (L(1 − m))
  規格 §3-1 給的是省略 (1−m) 的近似式 x_liq ≈ (1 − mL)/L（其預期值
  L=2.0→46%、L=2.5→36% 即由此得出）。兩者本模組都輸出：近似式用於與規格
  對照，精確式為回測引擎實際使用的條件（引擎逐日檢查 E < m·|N|）。
  權益歸零（破產）門檻為 x_ruin = 1/L，恆大於 x_liq。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_leverage_sweep.params import (
    L_HIGH_GRID, LIMIT_DOWN_PCT, MAINT_MARGIN_RATE, MAINT_RATE_SENS,
    MAX_LIMIT_DAYS,
)


def x_liq_approx(L: float, m: float = MAINT_MARGIN_RATE) -> float:
    """規格 §3-1 的近似臨界跌幅 (1 − mL)/L。"""
    return (1.0 - m * L) / L


def x_liq_exact(L: float, m: float = MAINT_MARGIN_RATE) -> float:
    """精確臨界跌幅 (1 − mL)/(L(1−m))——引擎條件 E < m·N 的解析解。"""
    return (1.0 - m * L) / (L * (1.0 - m))


def x_ruin(L: float) -> float:
    """權益歸零門檻 1/L（不含利息與成本）。"""
    return 1.0 / L


def limit_down_days(L: float, m: float = MAINT_MARGIN_RATE, *,
                    limit: float = LIMIT_DOWN_PCT,
                    max_days: int = MAX_LIMIT_DAYS) -> dict:
    """連續跌停生存窗口（TX 專屬，§3-2）。

    連續 k 日跌停的累積跌幅 x_k = 1 − (1−limit)^k（跌停期間無流動性、不再平衡）。
    回傳可承受（未觸發強平）的最大 k，以及觸發當日的累積跌幅。
    """
    xl_e, xl_a, xr = x_liq_exact(L, m), x_liq_approx(L, m), x_ruin(L)
    surv_e = surv_a = surv_r = 0
    trig_x = np.nan
    for k in range(1, max_days + 1):
        x_k = 1.0 - (1.0 - limit) ** k
        if x_k < xl_e:
            surv_e = k
        if x_k < xl_a:
            surv_a = k
        if x_k < xr:
            surv_r = k
        if np.isnan(trig_x) and x_k >= xl_e:
            trig_x = x_k
    return {"survive_days_exact": surv_e, "survive_days_approx": surv_a,
            "survive_days_to_ruin": surv_r, "trigger_cum_drop": trig_x}


def margin_table(l_grid=L_HIGH_GRID, m: float = MAINT_MARGIN_RATE) -> pd.DataFrame:
    """L_high × 臨界跌幅 × 連續跌停可承受日數（規格 §3-1、§3-2 的主表）。"""
    rows = []
    for L in l_grid:
        d = limit_down_days(L, m)
        rows.append({
            "l_high": L, "maint_rate": m,
            "x_liq_approx": x_liq_approx(L, m),      # 規格 §3-1 近似式
            "x_liq_exact": x_liq_exact(L, m),        # 引擎實際條件的解析解
            "x_ruin": x_ruin(L),
            "limit_down_days_exact": d["survive_days_exact"],
            "limit_down_days_approx": d["survive_days_approx"],
            "limit_down_days_to_ruin": d["survive_days_to_ruin"],
            "trigger_cum_drop": d["trigger_cum_drop"],
        })
    return pd.DataFrame(rows)


def maint_rate_sensitivity(l_grid=L_HIGH_GRID, rates=MAINT_RATE_SENS) -> pd.DataFrame:
    """維持保證金率敏感度：約束在哪個 m 才真正早於「權益歸零」咬到。

    m 越低 → x_liq 越接近 x_ruin=1/L → 保證金約束幾乎不獨立存在（強平≈破產）。
    以 gap = x_ruin − x_liq_exact 量化「保證金約束提前多少」。
    """
    rows = []
    for m in rates:
        for L in l_grid:
            xe, xr = x_liq_exact(L, m), x_ruin(L)
            rows.append({"maint_rate": m, "l_high": L, "x_liq_exact": xe,
                         "x_ruin": xr, "gap_vs_ruin": xr - xe,
                         "limit_down_days": limit_down_days(L, m)["survive_days_exact"]})
    return pd.DataFrame(rows)
