"""第六輪圖表：K × lag 的 Sharpe 差值熱力圖、切換頻率 vs 檢查頻率。沿用前幾輪的調色盤 tokens。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from backtest_multimarket.charts3 import setup_style       # noqa: F401 （含中文字型 fallback）
from backtest_threshold.charts import BASELINE, INK, INK_2, MUTED

from backtest_ma_robustness.charts4 import MK_COLOR

from backtest_signal_timing.params import (
    BASELINE_K, BASELINE_LAG, K_GRID, LAG_GRID, MARKET_LABEL, RANKING_MARKETS,
)

# 雙色發散、中點中性灰：藍＝優於基準，紅＝劣於基準
_CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#e34948", "#f3b8b4", "#e8e7e2", "#a9c9f2", "#2a78d6"])


def heatmap_fig(summary: pd.DataFrame, tau: float, path_out) -> None:
    """綜合 D 與逐市場 Δ：列＝K、欄＝lag，格內標數字，基準格加框。色階對稱於 0。"""
    panels = [("D", "綜合（跨 16 格中位數）")] + \
             [(f"d_{m}", MARKET_LABEL[m]) for m in RANKING_MARKETS]
    s = summary.set_index(["K", "lag"])
    vmax = max(1.5 * tau, float(np.nanmax(np.abs(summary[[c for c, _ in panels]].to_numpy()))))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 4.3), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        g = np.array([[s.loc[(k, lag), col] for lag in LAG_GRID] for k in K_GRID])
        ax.imshow(g, cmap=_CMAP_DIV, vmin=-vmax, vmax=vmax, aspect="auto")
        for i, k in enumerate(K_GRID):
            for j, lag in enumerate(LAG_GRID):
                ax.text(j, i, f"{g[i, j]:+.3f}", ha="center", va="center",
                        fontsize=9, color=INK)
                if (k, lag) == (BASELINE_K, BASELINE_LAG):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor=INK, linewidth=2))
        ax.set_xticks(range(len(LAG_GRID)),
                      ["T+0（上界）", "T+1", "T+2"], fontsize=9)
        ax.set_yticks(range(len(K_GRID)), [f"每 {k} 日" for k in K_GRID], fontsize=9)
        ax.set_title(title, fontsize=11, color=INK)
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"Sharpe 相對基準（每 5 日、T+1，黑框）的差值；材料性門檻 τ = {tau:.3f}",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path_out, dpi=150)
    plt.close(fig)


def switches_fig(regime: pd.DataFrame, path_out) -> None:
    """每年切換次數 vs 檢查頻率（機制來源）。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = range(len(K_GRID))
    for m in RANKING_MARKETS:
        g = regime[regime["market"] == m].set_index("K").loc[K_GRID]
        ax.plot(xs, g["switches_per_year"], color=MK_COLOR[m], linewidth=2, marker="o",
                markersize=8, label=MARKET_LABEL[m])
        # 標在左端：K=1 時四市場分得最開，右端會互相重疊
        ax.annotate(MARKET_LABEL[m], (0, g["switches_per_year"].iloc[0]),
                    xytext=(-10, 0), textcoords="offset points", va="center", ha="right",
                    fontsize=9, color=INK_2)
    ax.set_xticks(list(xs), [f"每 {k} 日" for k in K_GRID])
    ax.set_ylabel("每年切換次數（跨相位中位數）", color=INK_2)
    ax.set_ylim(0, 4.0)
    ax.set_xlim(-1.5, len(K_GRID) - 0.7)
    ax.axvline(K_GRID.index(BASELINE_K), color=BASELINE, linewidth=1)
    ax.text(K_GRID.index(BASELINE_K), 3.95, " 實盤", va="top", fontsize=9, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
