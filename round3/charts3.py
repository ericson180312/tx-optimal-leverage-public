"""第三輪圖表（規格 §9）：資金曲線疊圖（線性＋對數）、日經 1990 槓桿路徑、
bootstrap 小提琴圖、歷史 vs bootstrap 排名散點。沿用前輪調色盤 tokens。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from charts import (  # 前輪調色盤與樣式
    BASELINE, GRID, INK, INK_2, MUTED, SURFACE, setup_style as _setup_base,
)


def setup_style() -> None:
    _setup_base()
    # JhengHei 缺 U+2212（對數軸負指數的減號）→ 以 DejaVu Sans 作字型 fallback
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans", "sans-serif"]
from round3.params import INITIAL_EQUITY, STRATEGY_LABEL

# 類別色：依實體固定指派（B 基準組暖色/灰、測試組冷色）
STRAT_COLOR = {
    "B1_Const2x":  "#e34948",
    "B2_Fuse":     "#eda100",
    "B3_BH1x":     "#898781",
    "C_EWMA":      "#2a78d6",
    "C_SD20":      "#86b6ef",
    "C_SD60":      "#9ec5f4",
    "D_05":        "#79c7a9",
    "D_10":        "#1baf7a",
    "E_40":        "#a08ad0",
    "E_50":        "#4a3aa7",
    "F1_VolFuse":  "#12a5b5",
    "F2_VolTrend": "#d2649a",
}
PLOT_STRATS = ["B1_Const2x", "B2_Fuse", "B3_BH1x", "C_EWMA", "D_10",
               "E_50", "F1_VolFuse", "F2_VolTrend"]
NKX_LEV_STRATS = ["B2_Fuse", "C_EWMA", "D_10", "E_50", "F2_VolTrend"]


def _legend(ax, ncol=2):
    ax.legend(loc="upper left", fontsize=8, ncol=ncol, framealpha=0.9,
              edgecolor=BASELINE)


def _plain_log(ax) -> None:
    """對數軸改用純數字刻度（JhengHei 的 mathtext 缺 U+2212，10⁻¹ 會變豆腐字）。"""
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def equity_overlay_fig(dates: pd.DatetimeIndex, curves: dict[str, np.ndarray],
                       market_label: str, path) -> None:
    """自始以來資金曲線疊圖：上=線性、下=對數。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name in PLOT_STRATS:
        if name not in curves:
            continue
        eq = curves[name][0] / INITIAL_EQUITY
        for ax in axes:
            ax.plot(dates, eq, lw=1.1, color=STRAT_COLOR[name],
                    label=STRATEGY_LABEL[name])
    axes[0].set_title(f"{market_label}：策略資金曲線（自始以來，期初=1）",
                      fontsize=11, color=INK)
    axes[0].set_ylabel("權益（線性）")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("權益（對數）")
    _plain_log(axes[1])
    _legend(axes[0])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def nikkei_lev_fig(paths: dict, path_out) -> None:
    """日經 1990–2003 窗口：上=權益（對數）、下=槓桿路徑（本輪最重要壓力樣本）。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    for name in NKX_LEV_STRATS:
        key = ("日經1990–2003 長陰跌", "NKX", name)
        if key not in paths:
            continue
        dates, eq, lev = paths[key]
        axes[0].plot(dates, eq / INITIAL_EQUITY, lw=1.2, color=STRAT_COLOR[name],
                     label=STRATEGY_LABEL[name])
        axes[1].plot(dates, lev, lw=1.0, color=STRAT_COLOR[name],
                     label=STRATEGY_LABEL[name])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("權益（對數，期初=1）")
    _plain_log(axes[0])
    axes[0].set_title("日經 225・1990–2003 長陰跌：權益與槓桿路徑",
                      fontsize=11, color=INK)
    axes[1].axhline(2.0, color=BASELINE, lw=0.8, ls="--")
    axes[1].set_ylabel("槓桿")
    _legend(axes[0], ncol=1)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)


def violin_fig(per_path: dict[str, pd.DataFrame], market_label: str,
               names: list[str], path_out) -> None:
    """bootstrap Sharpe / MDD 分布小提琴圖（單一市場）。"""
    names = [n for n in names if n in per_path]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, col, ttl in [(axes[0], "sharpe", "Sharpe"), (axes[1], "mdd", "最大回撤")]:
        data = [per_path[n][col].dropna().to_numpy() for n in names]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(STRAT_COLOR[names[i]])
            b.set_alpha(0.55)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[k].set_color(INK_2)
            parts[k].set_linewidth(0.8)
        ax.set_ylabel(ttl)
    n_paths = len(per_path[names[0]]) if names else 0
    axes[0].set_title(f"{market_label}：bootstrap 合成路徑分布（{n_paths:,} 條 × 20 年）",
                      fontsize=11, color=INK)
    axes[1].set_xticks(np.arange(1, len(names) + 1))
    axes[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)


def rank_scatter_fig(cmp: pd.DataFrame, path_out) -> None:
    """歷史路徑排名 vs bootstrap 排名散點（偏離 45 度線＝依賴歷史序列結構）。

    cmp 欄位：market, strategy, hist_rank, boot_rank（1=最好）。
    """
    mkts = cmp["market"].unique()
    fig, axes = plt.subplots(1, len(mkts), figsize=(4.2 * len(mkts), 4.6),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, mk in zip(axes, mkts):
        d = cmp[cmp["market"] == mk]
        lim = max(d["hist_rank"].max(), d["boot_rank"].max()) + 0.5
        ax.plot([0.5, lim], [0.5, lim], color=BASELINE, lw=0.8, ls="--")
        for _, r in d.iterrows():
            ax.scatter(r["hist_rank"], r["boot_rank"],
                       color=STRAT_COLOR[r["strategy"]], s=42, zorder=3)
            ax.annotate(r["strategy"].split("_")[0], (r["hist_rank"], r["boot_rank"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7,
                        color=INK_2)
        ax.set_title(mk, fontsize=10, color=INK)
        ax.set_xlabel("歷史排名（1=最好）")
    axes[0].set_ylabel("bootstrap 排名（1=最好）")
    fig.suptitle("歷史 vs bootstrap 中位數 Sharpe 排名（偏離對角線＝依賴序列結構）",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)
