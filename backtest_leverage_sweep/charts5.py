"""第五輪圖表（規格 §7）：取捨曲線、三條主曲線、MDD 凸性、日經1990 槓桿。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

from backtest_multimarket.charts3 import setup_style          # noqa: F401 中文字型 fallback
from backtest_threshold.charts import BASELINE, INK, INK_2, MUTED

from backtest_leverage_sweep.params import (
    L_HIGH_GRID, MARKET_LABEL, RANKING_MARKETS,
)

MK_COLOR = {"TX_FUT": "#e34948", "SPX": "#2a78d6", "NKX": "#1baf7a",
            "HSI": "#eda100", "TX_IDX": "#898781"}
# L_high 由低到高：淺藍 → 深紅（曝險漸增）
L_COLOR = ["#256abf", "#5598e7", "#9ec5f4", "#eda100", "#e07b39", "#e34948"]
LC = dict(zip(L_HIGH_GRID, L_COLOR))


def _plain_log(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def frontier_fig(front_all: pd.DataFrame, front_mkt: pd.DataFrame, path_out) -> None:
    """取捨曲線（規格 §5-1）：x=MDD（越右越深）、y=CAGR；標註向後彎折。"""
    n = len(RANKING_MARKETS)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.5 * (n + 1), 4.2), squeeze=False)
    axes = axes[0]
    for ax, mk in zip(axes[:n], RANKING_MARKETS):
        d = front_mkt[front_mkt["market"] == mk].sort_values("l_high")
        if d.empty:
            continue
        ax.plot(-d["mdd"] * 100, d["cagr"] * 100, "-", color=MUTED, lw=1.0, zorder=1)
        for _, r in d.iterrows():
            ax.scatter(-r["mdd"] * 100, r["cagr"] * 100, s=46,
                       color=LC.get(r["l_high"], INK), zorder=3)
            ax.annotate(f"{r['l_high']:.1f}", (-r["mdd"] * 100, r["cagr"] * 100),
                        textcoords="offset points", xytext=(5, 3), fontsize=7,
                        color=INK_2)
        ax.set_title(MARKET_LABEL.get(mk, mk), fontsize=9, color=INK)
        ax.set_xlabel("MDD（%，越右越深）", fontsize=8)
    a = axes[n]
    d = front_all.sort_values("l_high")
    a.plot(-d["mdd"] * 100, d["cagr"] * 100, "-", color=INK_2, lw=1.4, zorder=1)
    for _, r in d.iterrows():
        a.scatter(-r["mdd"] * 100, r["cagr"] * 100, s=70,
                  color=LC.get(r["l_high"], INK), zorder=3)
        a.annotate(f"L={r['l_high']:.1f}", (-r["mdd"] * 100, r["cagr"] * 100),
                   textcoords="offset points", xytext=(6, 3), fontsize=8, color=INK)
    a.set_title("綜合（跨市場中位）", fontsize=10, color=INK)
    a.set_xlabel("MDD（%，越右越深）", fontsize=8)
    axes[0].set_ylabel("CAGR（%）", fontsize=8)
    fig.suptitle("取捨曲線：MDD ↔ CAGR（向右上=有補償；向右平/下=被嚴格支配）",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)


def three_curves_fig(front_all: pd.DataFrame, front_mkt: pd.DataFrame, path_out) -> None:
    """L_high vs CAGR / Sharpe / MDD（規格 §5-2）。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = [("cagr", "CAGR（%）", 100.0), ("sharpe", "Sharpe", 1.0),
             ("mdd", "MDD（%，越低越深）", 100.0)]
    for ax, (col, lab, mul) in zip(axes, specs):
        for mk in RANKING_MARKETS:
            d = front_mkt[front_mkt["market"] == mk].sort_values("l_high")
            if d.empty:
                continue
            ax.plot(d["l_high"], d[col] * mul, "o-", ms=3.5, lw=1.0,
                    color=MK_COLOR.get(mk, MUTED), alpha=0.75, label=mk)
        d = front_all.sort_values("l_high")
        ax.plot(d["l_high"], d[col] * mul, "o-", ms=6, lw=2.2, color=INK,
                label="綜合", zorder=5)
        ax.set_xlabel("L_high")
        ax.set_ylabel(lab)
        ax.set_xticks(L_HIGH_GRID)
    axes[0].legend(fontsize=7, ncol=2, edgecolor=BASELINE)
    fig.suptitle("三條主曲線：目標槓桿 vs 報酬／風險調整／回撤", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)


def convexity_fig(marg: pd.DataFrame, eff: pd.DataFrame, path_out) -> None:
    """左：每 +0.1 槓桿的 MDD 惡化（凸性）；右：每單位平均槓桿的 CAGR 效率。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lbl = [f"{r['from_l']:.1f}→{r['to_l']:.1f}" for _, r in marg.iterrows()]
    axes[0].bar(range(len(marg)), marg["d_mdd_deeper"] * 100, color="#e34948", alpha=0.85)
    ax2 = axes[0].twinx()
    ax2.plot(range(len(marg)), marg["d_cagr"] * 100, "o-", color="#2a78d6", lw=1.4)
    ax2.set_ylabel("ΔCAGR（%，藍線）", fontsize=8, color="#2a78d6")
    ax2.axhline(0, color=BASELINE, lw=0.8, ls="--")
    axes[0].set_xticks(range(len(marg)))
    axes[0].set_xticklabels(lbl, fontsize=7)
    axes[0].set_ylabel("ΔMDD 惡化（百分點，紅柱）", fontsize=8)
    axes[0].set_title("邊際：每 +0.1 槓桿的代價與收益", fontsize=10, color=INK)

    axes[1].plot(eff["l_high"], eff["cagr_per_lev"] * 100, "o-", color="#1baf7a", lw=1.6)
    axes[1].set_xlabel("L_high")
    axes[1].set_ylabel("每單位平均槓桿的 CAGR（%）", fontsize=8)
    axes[1].set_xticks(L_HIGH_GRID)
    axes[1].set_title("曝險效率：報酬增量是否只是曝險的線性結果", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)


def nikkei_fig(stress_paths: dict, path_out) -> None:
    """日經 1990–2003：各 L_high 的權益（對數）與槓桿路徑（規格 §5-5）。"""
    label = "日經1990–2003 長陰跌"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    for lh in L_HIGH_GRID:
        key = (label, "NKX", lh)
        if key not in stress_paths:
            continue
        dates, eq, lev = stress_paths[key]
        axes[0].plot(dates, eq / eq[0], lw=1.1, color=LC[lh], label=f"L={lh:.1f}")
        axes[1].plot(dates, lev, lw=0.9, color=LC[lh], alpha=0.85)
    axes[0].set_yscale("log")
    _plain_log(axes[0])
    axes[0].set_ylabel("權益（對數，期初=1）")
    axes[0].set_title("日經 225・1990–2003 長陰跌：目標槓桿的懲罰放大", fontsize=11, color=INK)
    axes[0].legend(fontsize=8, ncol=6, edgecolor=BASELINE)
    axes[1].set_ylabel("槓桿")
    axes[1].axhline(1.0, color=BASELINE, lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(path_out)
    plt.close(fig)
