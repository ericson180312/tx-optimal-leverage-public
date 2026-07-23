"""第四輪圖表（規格 §5、§6）：平原地形熱力圖、窗口 vs 平均槓桿、窗口 vs 切換頻率、
日經 1990 各窗口槓桿路徑疊圖。沿用第三輪／前輪的調色盤 tokens。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

from backtest_multimarket.charts3 import setup_style       # noqa: F401 （含中文字型 fallback）
from backtest_threshold.charts import (
    BASELINE, INK, INK_2, MUTED, _CMAP_SEQ,
)
from backtest_ma_robustness.params import (
    BASELINE_MA, BUFFER_GRID, MA_GRID, variant_key,
)

MARKET_SEQ = ["TX_FUT", "SPX", "NKX", "HSI"]
# 各市場一色（線圖用）
MK_COLOR = {"TX_FUT": "#e34948", "SPX": "#2a78d6", "NKX": "#1baf7a", "HSI": "#eda100",
            "TX_IDX": "#898781"}
# 窗口疊圖用（由短到長：冷→暖）
MA_LINE = ["#256abf", "#3987e5", "#6da7ec", "#898781", "#eda100", "#e07b39", "#e34948"]


def _plain_log(ax) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def topography_fig(grids: dict[str, np.ndarray], path_out) -> None:
    """平原地形圖：MA(x) × buffer(y) 的 maximin/最差百分位熱力圖。

    grids: {面板標題: (nbuf, nma) 陣列}；含 4 市場 + 綜合。值越高越好（藍愈深）。
    """
    keys = list(grids)
    ncol = len(keys)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 3.6), squeeze=False)
    axes = axes[0]
    vmin = min(np.nanmin(g) for g in grids.values())
    vmax = max(np.nanmax(g) for g in grids.values())
    for ax, title in zip(axes, keys):
        g = grids[title]
        im = ax.imshow(g, cmap=_CMAP_SEQ, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(MA_GRID)))
        ax.set_xticklabels(MA_GRID, fontsize=7, rotation=90)
        ax.set_yticks(range(len(BUFFER_GRID)))
        ax.set_yticklabels([f"{b*100:g}%" for b in BUFFER_GRID], fontsize=7)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("MA 窗口", fontsize=8)
        # 標出 MA=200 欄
        ax.axvline(MA_GRID.index(BASELINE_MA), color="#e34948", lw=1.0, ls="--")
        for yi in range(g.shape[0]):
            for xi in range(g.shape[1]):
                if np.isfinite(g[yi, xi]):
                    ax.text(xi, yi, f"{g[yi, xi]:.2f}", ha="center", va="center",
                            fontsize=6, color=INK_2)
    axes[0].set_ylabel("緩衝帶 buffer", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01,
                 label="maximin 最差格 Sharpe 百分位（越高越好）")
    fig.suptitle("平原地形圖：MA × buffer 的 maximin 百分位（紅虛線＝MA200）",
                 fontsize=11, color=INK)
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def leverage_vs_window_fig(sweep_agg: pd.DataFrame, path_out, *,
                           buf: float = 0.01, period: int = 20) -> None:
    """窗口 vs 平均槓桿：長窗口→降載較慢→平均槓桿較高（本輪主混淆變數）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in MARKET_SEQ:
        keys = [variant_key(ma, buf) for ma in MA_GRID]
        d = sweep_agg[(sweep_agg["market"] == mk) & (sweep_agg["period_years"] == period)
                      & sweep_agg["strategy"].isin(keys)].copy()
        if d.empty:
            continue
        d["ma"] = d["strategy"].str.extract(r"MA(\d+)_").astype(int)
        d = d.sort_values("ma")
        ax.plot(d["ma"], d["med_avg_lev"], "o-", color=MK_COLOR[mk], lw=1.3,
                markersize=4, label=mk)
    ax.axvline(BASELINE_MA, color=BASELINE, lw=0.9, ls="--")
    ax.axhline(2.0, color=BASELINE, lw=0.9, ls=":")
    ax.set_xlabel("MA 窗口長度")
    ax.set_ylabel(f"平均槓桿中位數（{period}年滾動，buffer={buf*100:g}%）")
    ax.set_title("窗口長度 vs 平均槓桿——長窗口的曝險偏高（Q4 混淆變數）",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8, edgecolor=BASELINE)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)


def switches_vs_window_fig(regime: pd.DataFrame, path_out, *, buf: float = 0.01) -> None:
    """窗口 vs 切換頻率／假訊號率：解釋績效差異的機制來源。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    d0 = regime[regime["buffer"] == buf]
    for mk in MARKET_SEQ:
        d = d0[d0["market"] == mk].sort_values("ma")
        if d.empty:
            continue
        axes[0].plot(d["ma"], d["switches_per_year"], "o-", color=MK_COLOR[mk],
                     lw=1.3, markersize=4, label=mk)
        axes[1].plot(d["ma"], d["false_rate"], "o-", color=MK_COLOR[mk],
                     lw=1.3, markersize=4, label=mk)
    for ax in axes:
        ax.axvline(BASELINE_MA, color=BASELINE, lw=0.9, ls="--")
        ax.set_xlabel("MA 窗口長度")
    axes[0].set_ylabel("每年切換次數")
    axes[0].set_title("切換頻率", fontsize=10, color=INK)
    axes[1].set_ylabel("假訊號率（≤20日回切/切換）")
    axes[1].set_title("假訊號率（巴洗）", fontsize=10, color=INK)
    axes[0].legend(fontsize=8, edgecolor=BASELINE)
    fig.suptitle(f"窗口長度 vs regime 行為（buffer={buf*100:g}%）", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)


def nikkei_window_fig(lev_paths: dict, path_out, *, buf: float = 0.01,
                      show_mas=(120, 150, 200, 240, 300)) -> None:
    """日經 1990–2003：各 MA 窗口的槓桿路徑疊圖（H1 的視覺化）。"""
    label = "日經1990–2003 長陰跌"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = dict(zip(show_mas, MA_LINE))
    for ma in show_mas:
        key = (label, "NKX", ma)
        if key not in lev_paths:
            continue
        dates, lev = lev_paths[key]
        ax.plot(dates, lev, lw=1.0, color=colors.get(ma, MUTED), label=f"MA{ma}")
    ax.axhline(2.0, color=BASELINE, lw=0.8, ls="--")
    ax.set_ylabel("槓桿")
    ax.set_title("日經 225・1990–2003：各 MA 窗口的槓桿路徑（長窗口→降載較慢）",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8, ncol=len(show_mas), edgecolor=BASELINE)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)
