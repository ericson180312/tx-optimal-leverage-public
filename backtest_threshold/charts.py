"""圖表輸出（靜態 PNG，light mode）。

色彩依 dataviz 參考調色盤：熱力圖用單一藍色 sequential ramp（跨零時改
藍↔紅 diverging、灰中點）；資金曲線用固定順序類別色；文字/格線用 ink tokens。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ── 調色盤 tokens ───────────────────────────────────────────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
# 類別色（固定順序，依實體指派、不循環）
C_BEST_SHARPE = "#2a78d6"   # slot1 藍：Sharpe 最佳
C_BEST_RET    = "#1baf7a"   # slot2 aqua：報酬最佳
C_BH2         = "#eda100"   # slot3 黃：BH 2x
C_DAILY2      = "#4a3aa7"   # slot5 紫：每日再平衡 2x
C_PERIODIC5D  = "#e34948"   # slot6 紅：每 5 日再平衡 2x
C_BH1         = "#898781"   # 無槓桿基準以 muted 灰呈現（參照線角色）
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_LOW, DIV_MID, DIV_HIGH = "#d03b3b", "#f0efec", "#256abf"

_CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)
_CMAP_DIV = LinearSegmentedColormap.from_list("div_rb", [DIV_LOW, DIV_MID, DIV_HIGH])


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["Microsoft JhengHei", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    })


def _heat_panel(ax, g: pd.DataFrame, title: str, fmt, best_xy: tuple[float, float]):
    """單一熱力圖面板：值標註 + 最佳解框線。跨零 → diverging。"""
    vals = g.to_numpy(float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if vmin < 0 < vmax:
        im = ax.imshow(vals, cmap=_CMAP_DIV, norm=TwoSlopeNorm(0, vmin, vmax),
                       aspect="auto", origin="lower")
    else:
        im = ax.imshow(vals, cmap=_CMAP_SEQ, vmin=vmin, vmax=vmax,
                       aspect="auto", origin="lower")
    ax.set_xticks(range(len(g.columns)), [f"{x:g}" for x in g.columns])
    ax.set_yticks(range(len(g.index)), [f"{y:g}" for y in g.index])
    ax.set_xlabel("X（減碼上限）")
    ax.set_ylabel("Y（加碼下限）")
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(False)
    # 值標註：依底色亮度換字色
    norm_vals = im.norm(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            r, g_, b, _ = im.cmap(norm_vals[i, j])
            lum = 0.299 * r + 0.587 * g_ + 0.114 * b
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6.8,
                    color="#ffffff" if lum < 0.55 else INK)
    xi = list(g.columns).index(best_xy[0])
    yi = list(g.index).index(best_xy[1])
    ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False,
                               edgecolor=INK, linewidth=2.0))


def heatmap_fig(agg: pd.DataFrame, years: int, best_sharpe_xy, best_ret_xy, path):
    """每期間一張：中位數 Sharpe（左）與中位數累積報酬（右）熱力圖。"""
    gs = agg[~agg["is_benchmark"]].pivot(index="Y", columns="X", values="med_sharpe")
    gr = agg[~agg["is_benchmark"]].pivot(index="Y", columns="X", values="med_cumret")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    _heat_panel(axes[0], gs, f"{years} 年期：跨路徑中位數 Sharpe（黑框 = 最佳）",
                lambda v: f"{v:.2f}", best_sharpe_xy)
    _heat_panel(axes[1], gr, f"{years} 年期：跨路徑中位數累積報酬（黑框 = 最佳）",
                lambda v: f"{v * 100:.0f}%", best_ret_xy)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def equity_overlay_fig(dates, curves: dict[str, np.ndarray], years: int, path,
                       e0: float):
    """最佳解 vs 三條 benchmark 的資金曲線疊圖（基準路徑，對數座標）。

    curves 鍵固定：best_sharpe / best_ret / BH_2x / Daily_2x / BH_1x。
    """
    style = {   # (顏色, 線寬)
        "best_sharpe":    (C_BEST_SHARPE, 2.0),
        "best_ret":       (C_BEST_RET, 2.0),
        "BH_2x":          (C_BH2, 1.3),
        "Daily_2x":       (C_DAILY2, 1.3),
        "Periodic5d_2x":  (C_PERIODIC5D, 1.6),
        "BH_1x":          (C_BH1, 1.3),
    }
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for name, curve in curves.items():
        c, lw = style[name]
        label = {
            "best_sharpe": "Sharpe 最佳", "best_ret": "報酬最佳",
            "BH_2x": "買進持有 2x", "Daily_2x": "每日再平衡 2x",
            "Periodic5d_2x": "每 5 日再平衡 2x", "BH_1x": "買進持有 1x",
        }[name]
        plot = np.where(curve <= 0, np.nan, curve)   # 破產(權益0)在對數座標畫不出
        ax.plot(dates, plot, color=c, linewidth=lw, label=label)
    ax.set_yscale("log")
    ax.set_title(f"{years} 年期（基準路徑）：最佳解 vs 基準策略 資金曲線", color=INK)
    ax.set_ylabel("權益（元，對數座標）")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.axhline(e0, color=BASELINE, linewidth=0.8, linestyle=":")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def freq_fig(freq: pd.DataFrame, path):
    """再平衡頻率掃描：各期間的中位數 Sharpe 與中位數報酬 vs 再平衡間隔。

    x 軸為序數（永不 → 季 → 月 → 雙週 → 5日 → 每日），非線性刻度，故用類別軸。
    """
    order = freq["strategy"].drop_duplicates().tolist()
    periods = sorted(freq["period_years"].unique())
    # 期間是有序類別 → 用同一藍色 sequential ramp 的離散步（ordinal，不低於 step250）
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    colors = {p: ramp[i % len(ramp)] for i, p in enumerate(periods)}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for ax, col, title, fmt in [
            (axes[0], "med_sharpe", "跨路徑中位數 Sharpe", lambda v: f"{v:.2f}"),
            (axes[1], "med_cagr", "跨路徑中位數 CAGR", lambda v: f"{v * 100:.0f}%")]:
        for p in periods:
            sub = freq[freq["period_years"] == p].set_index("strategy").loc[order]
            ax.plot(range(len(order)), sub[col], marker="o", markersize=6,
                    linewidth=2, color=colors[p], label=f"{p} 年期")
            ax.annotate(f"{p}y", xy=(len(order) - 1, sub[col].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=colors[p], va="center")
        ax.set_xticks(range(len(order)),
                      [o.replace("（", "\n（").replace(" ★新增", "\n★新增") for o in order],
                      fontsize=8)
        ax.set_title(title, color=INK, fontsize=11)
        # f=fmt 綁定當前值：避免 lambda 延遲查找取到迴圈最後一個 fmt
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, f=fmt: f(v)))
        ax.set_xlabel("再平衡頻率（左：不再平衡 → 右：每日）")
        # 最左點（買進持有）的 Sharpe 有存活者偏誤：破產路徑 Sharpe 記 NaN、不進中位數
        ax.axvspan(-0.35, 0.35, color=GRID, alpha=0.55, zorder=0)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("再平衡頻率對績效的影響（每 5 日再平衡 = 正2倍ETF 週期改為 5 日）",
                 color=INK, fontsize=12)
    bh = freq[freq["name"] == "BH_2x"].set_index("period_years")
    bank = "、".join(f"{p}年期 {int(bh.loc[p, 'n_bankrupt_paths'])}/{int(bh.loc[p, 'n_paths'])}"
                    for p in periods if bh.loc[p, "n_bankrupt_paths"] > 0)
    fig.text(0.5, -0.03,
             f"灰底：買進持有 2x 的中位數有存活者偏誤——破產路徑（{bank}）的 Sharpe 無定義、不計入中位數",
             ha="center", fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def leverage_fig(dates, lev: np.ndarray, x_thr: float, y_thr: float,
                 target: float, years: int, name: str, path):
    """最佳解的槓桿時間序列：看實際在 X/Y 門檻間如何游走。"""
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.plot(dates, lev, color=C_BEST_SHARPE, linewidth=1.1, label=f"槓桿 L（{name}）")
    for v, lbl, ls in [(x_thr, f"X = {x_thr:g}（減碼）", "--"),
                       (target, f"目標 {target:g}", ":"),
                       (y_thr, f"Y = {y_thr:g}（加碼）", "--")]:
        ax.axhline(v, color=INK_2, linewidth=0.9, linestyle=ls)
        ax.annotate(lbl, xy=(dates[0], v), xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=INK_2)
    ax.set_title(f"{years} 年期 Sharpe 最佳解的實際槓桿路徑", color=INK)
    ax.set_ylabel("槓桿倍數")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
