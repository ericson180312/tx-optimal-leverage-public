"""第三輪主報告產生器（規格 §9、§10）。

讀取 run3.py 的 payload（round3/output/cache/results3.pkl），輸出 report.md。
只改報告文字時可直接：
    from round3 import report3; report3.write_report(pickle.load(...), path)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from round3.params import (
    BOOT_MEAN_BLOCK, BOOT_N_PATHS, BOOT_YEARS, CALIB_TARGET_LEV, CALIB_TOL,
    CALIB_YEARS, DEADBAND, DERISK_LEV, FINANCING_SPREAD, IDX_COST_PER_SIDE,
    IDX_COST_PER_SIDE_SENS, MARKET_LABEL, PERIOD_YEARS, RANKING_MARKETS,
    RUIN_VETO_PROB, STRATEGIES, STRATEGY_LABEL, TW_SHORT_RATE, WINDOW_STEP,
)

COND_STRATS = [s for s in STRATEGIES if s[0] in "CDEF"]
MECH_STRATS = ["B1_Const2x", "B2_Fuse"]


# ── 格式化 ──────────────────────────────────────────────────────────────────

def _f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def _pc(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x * 100:.{nd}f}%"


def _lab(s):
    return STRATEGY_LABEL.get(s, s)


def _table(df: pd.DataFrame, spec: list[tuple[str, str, object]]) -> str:
    """spec = [(欄名, 標題, 格式函數)]。"""
    head = "| " + " | ".join(t for _, t, _ in spec) + " |"
    sep = "|" + "|".join("---" for _ in spec) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for col, _, fmt in spec:
            v = r.get(col)
            cells.append(fmt(v) if callable(fmt) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell_table(agg: pd.DataFrame, market: str, years: int) -> str:
    d = agg[(agg["market"] == market) & (agg["period_years"] == years)].copy()
    if d.empty:
        return "（此市場資料長度不足此期間）"
    d = d.sort_values("med_sharpe", ascending=False)
    d["label"] = d["strategy"].map(_lab)
    return _table(d, [
        ("label", "策略", str),
        ("med_sharpe", "中位Sharpe", _f),
        ("med_cagr", "中位CAGR", _pc),
        ("med_mdd", "中位MDD", _pc),
        ("worst_cumret", "最差路徑報酬", _pc),
        ("med_avg_lev", "平均槓桿", _f),
        ("med_lev_at_trough", "MDD谷底槓桿", _f),
        ("med_time_derisked", "降載時間佔比", _pc),
        ("med_cost_ratio", "成本/初始資本", _pc),
        ("n_bankrupt", "破產路徑", lambda v: str(int(v))),
    ])


# ── 結論六問（§10）──────────────────────────────────────────────────────────

def _q1(payload) -> str:
    agg = payload["agg_all"]
    d = agg[agg["market"].isin(RANKING_MARKETS) & agg["strategy"].isin(STRATEGIES)]
    diffs = []
    for (_, _), cell in d.groupby(["market", "period_years"]):
        c = cell.set_index("strategy")["med_sharpe"]
        best_cond = c[[s for s in COND_STRATS if s in c.index]].max()
        best_mech = c[[s for s in MECH_STRATS if s in c.index]].max()
        diffs.append(best_cond - best_mech)
    diffs = np.array(diffs)
    med, lo, hi = np.median(diffs), diffs.min(), diffs.max()
    frac_pos = (diffs > 0).mean()
    big = (np.abs(diffs) > 0.08).mean()
    verdict = ("**有系統性差距**" if (frac_pos >= 0.75 and med > 0.08) or
               (frac_pos <= 0.25 and med < -0.08) else
               "**沒有系統性優勢**——差距多半落在前輪認定的平坦噪音量級內")
    return (f"每個「市場×期間」格子中，最佳條件式策略與最佳機械策略（B1/B2）的"
            f"中位 Sharpe 差：中位數 **{med:+.3f}**（範圍 {lo:+.3f} ~ {hi:+.3f}；"
            f"條件式較高的格子佔 {frac_pos:.0%}；|差| > 0.08 的格子佔 {big:.0%}）。"
            f"結論：{verdict}（前輪平坦噪音量級 Sharpe ≈ 0.08）。")


def _q2(payload) -> str:
    stx = payload["stress"]
    d = stx[(stx["window"].str.startswith("日經1990")) & (stx["market"] == "NKX")]
    if d.empty:
        return "（日經窗口無資料）"
    d = d.set_index("strategy")
    rows = []
    for s in ["B2_Fuse", "D_05", "D_10", "C_EWMA", "E_50", "F2_VolTrend", "B3_BH1x"]:
        if s in d.index:
            r = d.loc[s]
            rows.append(f"| {_lab(s)} | {_pc(r['cumret'])} | {_pc(r['mdd'])} | "
                        f"{_f(r['avg_lev'])} | {_pc(r['time_derisked'])} |")
    tbl = ("| 策略 | 期間報酬 | MDD | 平均槓桿 | 降載時間佔比 |\n|---|---|---|---|---|\n"
           + "\n".join(rows))
    b2, d10, d05 = d.loc["B2_Fuse"], d.loc["D_10"], d.loc["D_05"]
    trend_wins = (d10["cumret"] > b2["cumret"]) and (d10["mdd"] > b2["mdd"])
    concl = ("趨勢濾網在報酬與 MDD 上**同時**優於保險絲——長陰跌確實推翻了台股"
             "樣本上「寬保險絲已足夠」的結論" if trend_wins else
             "趨勢濾網並未在報酬與 MDD 上同時勝出，前輪結論未被完全推翻，"
             "但兩者的差異結構（見下表）值得注意")
    return (f"{tbl}\n\n保險絲 B2 期間報酬 {_pc(b2['cumret'])}／MDD {_pc(b2['mdd'])}；"
            f"趨勢 D_10 為 {_pc(d10['cumret'])}／{_pc(d10['mdd'])}、"
            f"D_05 為 {_pc(d05['cumret'])}／{_pc(d05['mdd'])}。{concl}。")


def _q3(payload) -> str:
    agg = payload["agg_all"]
    inc = payload["inception"]
    f1 = agg[(agg["strategy"] == "F1_VolFuse") & agg["market"].isin(RANKING_MARKETS)]
    med_t, max_t = f1["med_fuse_trig"].median(), f1["max_fuse_trig"].max()
    inc_f1 = inc[inc["strategy"] == "F1_VolFuse"]
    inc_txt = "、".join(f"{r['market']} {int(r['fuse_trig'])} 次"
                        for _, r in inc_f1.iterrows())
    if max_t == 0:
        verdict = ("保險絲**從未觸發**——vol targeting 的連續降載完全取代了它，"
                   "掛上保險絲純屬冗餘（但作為尾端保險的成本也是零）")
    else:
        verdict = (f"保險絲仍有觸發（滾動視窗最多 {int(max_t)} 次），"
                   "vol targeting 的降載不夠快時它仍有殘餘價值")
    return (f"F1 的 [3.5] 減碼保險絲觸發次數：滾動視窗中位數 {med_t:.1f} 次、"
            f"最大 {int(max_t)} 次；自始以來路徑：{inc_txt}。{verdict}。")


def _q4(payload) -> str:
    mm = payload["maximin"]
    top = mm.iloc[0]
    lev_gap = top["med_avg_lev"] - CALIB_TARGET_LEV
    if top["med_avg_lev"] < CALIB_TARGET_LEV - CALIB_TOL:
        lev_txt = (f"其平均槓桿 {_f(top['med_avg_lev'])} **低於** 2.0 校準帶"
                   f"（−{abs(lev_gap):.2f}），Sharpe 排名雖幾乎不受平均槓桿影響，"
                   "報酬排名的解讀必須扣除這個低曝險紅利（§3-4 對照表）")
    else:
        lev_txt = (f"其平均槓桿 {_f(top['med_avg_lev'])} 落在 2.0±{CALIB_TOL} "
                   "校準帶內——優勢來自**機制**（曝險的時點配置），"
                   "不是來自平均開得比較小")
    return (f"maximin 最差格最高者：**{_lab(top['strategy'])}**"
            f"（最差格 {top['worst_cell']}，最差百分位 {top['worst_pct_rank']:.2f}，"
            f"平均百分位 {top['mean_pct_rank']:.2f}）。{lev_txt}。")


def _q5(payload) -> str:
    boot = payload["boot_dist"]
    agg = payload["agg_all"]
    rows, texts = [], []
    for mk in RANKING_MARKETS:
        b = boot[boot["market"] == mk].set_index("strategy")
        h = agg[(agg["market"] == mk) & (agg["period_years"] == 20)
                & agg["strategy"].isin(STRATEGIES)].set_index("strategy")
        if h.empty:
            continue
        common = [s for s in STRATEGIES if s in b.index and s in h.index]
        ruin = b.loc[common, "ruin_prob"]
        mddh = h.loc[common, "worst_mdd"]
        if ruin.max() == 0:
            texts.append(f"{mk}：所有策略歸零機率皆為 0")
            continue
        rr = ruin.rank(ascending=False)
        mr = mddh.rank(ascending=True)      # 越深(越負)越危險
        rho = rr.corr(mr, method="spearman")
        rows.append(rho)
        texts.append(f"{mk}：Spearman ρ = {rho:.2f}"
                     f"（非零歸零機率的策略：{', '.join(ruin[ruin > 0].index)}）")
    summary = ("整體而言兩個排序方向一致" if rows and np.nanmedian(rows) > 0.5
               else "多數市場所有策略的歸零機率為 0，兩個排序的可比資訊有限"
               if not rows else "兩個排序僅部分一致")
    return "；".join(texts) + f"。{summary}。"


# ── 主報告 ──────────────────────────────────────────────────────────────────

def write_report(payload: dict, path) -> None:
    audit = payload["audit"]
    agg = payload["agg_all"]
    mm, ret_rank = payload["maximin"], payload["ret_rank"]
    boot = payload["boot_dist"]
    sec = []

    # ══ 標題與摘要 ══════════════════════════════════════════════════════════
    top = mm.iloc[0]
    veto = boot.groupby("strategy")["ruin_prob"].max()
    veto = veto[veto > RUIN_VETO_PROB].index.tolist()
    sec.append(f"""# 條件式曝險 2x 槓桿策略・跨市場回測（第三輪）

**核心問題**：哪一種 2x 槓桿的曝險控制機制，在不同市場、不同 regime、合成路徑上最穩健？

**主要結論（摘要）**
- maximin 主排名冠軍：**{_lab(top['strategy'])}**（所有市場×期間格子的最差 Sharpe 百分位 {top['worst_pct_rank']:.2f}）。
- 歸零機率 > {RUIN_VETO_PROB:.0%} 的不合格策略：{('、'.join(_lab(s) for s in veto)) if veto else '無'}。
- 詳細六問結論見 §10。
""")

    # ══ §1 方法與假設 ═══════════════════════════════════════════════════════
    sec.append(f"""## 1. 方法與假設

**執行口徑（規格 §2）**：所有條件式策略（C/D/E/F）訊號於第 t 日收盤計算、第 t+1 日
收盤執行（避免 look-ahead）；曝險日頻檢查、|ΔL| > {DEADBAND} 才執行（減少碎單）。
B2 保險絲的觸發變數是自身組合槓桿（收盤已知、無 look-ahead），沿用前輪「觸發當日
收盤執行」口徑；B1 每 {5} 個交易日無條件調回 2.0。

**融資成本模型（指數市場）**：超過 1 倍的部分 `(L−1)×E` 收取「當地短率 + {FINANCING_SPREAD:.1%}」
年化成本、日頻計提；L<1 時閒置現金賺當地短率。台股期貨版（TX_FUT）沿用前輪口徑：
權益全額按 {TW_SHORT_RATE:.1%} 計息、不對名目借款計息（保證金交易）。

**成本**：TX 期貨沿用前輪明細模型（手續費 20 元/口/邊、期交稅 2×10⁻⁵、滑價 1 檔、
每月轉倉兩腳）；海外指數模擬為單邊 {IDX_COST_PER_SIDE * 1e4:.0f}bp 名目變動量成本
（敏感度 {IDX_COST_PER_SIDE_SENS * 1e4:.0f}bp，§8）。

**滾動視窗**：長度 = 期間年數×252 交易日、起點以 {WINDOW_STEP} 交易日（≈季頻）平移、
另含終點=資料尾的基準視窗；期間 {PERIOD_YEARS} 年。訊號取自全序列（視窗起點前的
市場資訊當時已可得）；CPPI 的 HWM 與策略持倉於視窗起點重置。

**參數紀律（規格 §2）**：全部策略參數（λ=0.94、200MA±1%、D_max、保險絲 [3.5,1.1]、
deadband 0.1）四市場凍結一致；唯一的市場別量為 σ_target（§3 槓桿中性化的校準變數，
以各市場最早 {CALIB_YEARS} 年 walk-forward 校準後凍結）。

**破產口徑**：權益 ≤ 0 即歸零、不可反彈（含融資利息把權益吃穿的情形）。
""")

    # ══ §2 資料揭露 ═════════════════════════════════════════════════════════
    rows = []
    for k in ["TX_FUT", "TX_IDX", "SPX", "NKX", "HSI"]:
        a = audit[k]
        rows.append(f"| {MARKET_LABEL[k]} | {a['start']} | {a['end']} | {a['n_days']:,} | "
                    f"{a['tr_source']} | {a['rf_source']} | "
                    f"{a['min_day_ret'] * 100:.1f}% / {a['max_day_ret'] * 100:.1f}% |")
    sec.append(f"""## 2. 各市場資料揭露（規格 §1）

| 市場 | 起 | 迄 | 交易日 | 報酬序列來源 | 短率來源 | 單日極值(min/max) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

- **日經 225 涵蓋 1989-12-29 高點**（資料自 {audit['NKX']['start']} 起）——本輪最重要的
  壓力樣本可完整檢驗。✔
- S&P 500 自 {audit['SPX']['start']} 起（涵蓋 1929 大蕭條）；1988-01 之後使用
  ^SP500TR 真實總報酬指數，之前以「價格指數＋年代平均股息率」近似（§11 揭露）。
- 日經／恆生無免費可得的長總報酬序列，全程以「價格＋年代平均股息率」近似；
  日本／香港短率為分段常數近似（FRED 於本機網路不可達）。這些近似對**同市場內
  策略互比**影響是二階的（所有策略吃同一條報酬序列），但絕對報酬水準有偏差。
- TX 期貨版趨勢訊號用「比值還原連續期貨價」（含 carry 漂移）；指數版用 TAIEX
  現貨價格指數。
""")

    # ══ §3 槓桿中性化 ══════════════════════════════════════════════════════
    ct = payload["calib_tbl"]
    cal = ct[ct["calibrated"]].copy()
    cal["mlab"] = cal["market"].map(MARKET_LABEL)
    cal["slab"] = cal["strategy"].map(_lab)
    tbl_cal = _table(cal, [
        ("mlab", "市場", str), ("slab", "策略", str),
        ("sigma_wf", "σ_wf(最早10年)", lambda v: _pc(v, 1)),
        ("sigma_is", "σ_is(全樣本)", lambda v: _pc(v, 1)),
        ("calib_signal_avg_lev", "校準段訊號平均槓桿", _f),
        ("rolling_avg_lev_10y", "滾動10y實際平均槓桿(wf)", _f),
        ("inception_avg_lev", "自始以來實際平均槓桿(is)", _f),
        ("within_tol", "is版±0.1內", lambda v: "✔" if v else ("✘" if v is not None else "–")),
    ])
    unc = ct[~ct["calibrated"]].groupby("strategy")["inception_avg_lev"].median()
    unc_txt = "、".join(f"{_lab(s)} ≈ {_f(v)}" for s, v in unc.items())
    drift = cal[cal["strategy"] == "C_EWMA"]
    drift_txt = "、".join(f"{r['market']} {_f(r['rolling_avg_lev_10y'])}"
                          for _, r in drift.iterrows())
    sec.append(f"""## 3. 槓桿中性化校準結果（🔴 規格 §3）

校準協定（§3-3，兩套並行）：
- **σ_wf（walk-forward）**：以各市場**最早 {CALIB_YEARS} 年**的訊號平均槓桿
  = {CALIB_TARGET_LEV} 校準（bisection）後凍結 → 用於滾動視窗與壓力窗口。
- **σ_is（全樣本，明確標註 in-sample）**：用於「自始以來」路徑與 bootstrap
  （合成路徑重抽自全樣本分布，σ_is 與之內部一致）。

{tbl_cal}

**⚠️ 本輪重要發現——vol targeting 無法用單次校準做到跨 regime 的槓桿中性**：
σ_wf 校準於最早 10 年的波動 regime，之後波動結構改變，C_EWMA 在滾動 10 年視窗的
實際平均槓桿變成 {drift_txt}（目標 2.0）。S&P 尤其嚴重（校準段=1928–38 大蕭條），
評估段幾乎貼著 3.0 上限。因此**滾動視窗表中 C/F1 的報酬與 MDD 讀數含有高曝險
成分**，機制比較請以 Sharpe 與（槓桿中性的）自始以來/bootstrap 口徑為主。

無校準變數的策略（§3-4，照實跑、所有比較表並列平均槓桿）——自始以來平均槓桿
中位數：{unc_txt}。F2 沿用 C 的 σ_target（`min(L_C, L_D)` 的組合定義——若讓 F2
自行灌大 σ_target 追 2.0，vol 腿會恆頂 3.0 上限、策略退化成 D_10），其平均槓桿
受趨勢 regime 佔比支配。**本報告任何「報酬較高」的陳述都必須對照平均槓桿欄**；
Sharpe 幾乎槓桿不變，為機制比較主軸。
""")

    # ══ §4 主排名 ══════════════════════════════════════════════════════════
    mm2 = mm.copy()
    mm2["slab"] = mm2["strategy"].map(_lab)
    mm2["veto"] = mm2["strategy"].map(
        lambda s: "⛔" if s in veto else "")
    tbl_mm = _table(mm2, [
        ("slab", "策略", str),
        ("worst_pct_rank", "最差格百分位", _f),
        ("mean_pct_rank", "平均百分位", _f),
        ("worst_cell", "最差格", str),
        ("med_avg_lev", "平均槓桿(中位)", _f),
        ("veto", "歸零否決", str),
    ])
    rr2 = ret_rank.copy()
    rr2["slab"] = rr2["strategy"].map(_lab)
    tbl_rr = _table(rr2, [
        ("slab", "策略", str),
        ("worst_pct_rank", "最差格百分位", _f),
        ("mean_pct_rank", "平均百分位", _f),
        ("worst_cell", "最差格", str),
        ("med_avg_lev", "**平均槓桿(中位)**", _f),
    ])
    mmev = payload["maximin_eval"]
    if not mmev.empty:
        ev_top3 = "、".join(_lab(s) for s in mmev["strategy"].head(3))
        ev_txt = (f"僅用「起點在校準段之後」視窗重算的 maximin 前三名：{ev_top3}"
                  f"（詳 r3_market_period_metrics_evalonly.csv）。")
    else:
        ev_txt = ""
    sec.append(f"""## 4. 主排名（規格 §8）

**主排名：maximin over {{市場 × 期間}}**——每策略取其在所有「市場×期間」格子
（{len(RANKING_MARKETS)} 市場 × {len(PERIOD_YEARS)} 期間）中位 Sharpe 百分位的最差值。
不求任一格最好，只求處處不差。TX 以期貨版計入排名（指數版為 §7 的校準參照）。

{tbl_mm}

{ev_txt}

**報酬導向第二排名（maximin over 中位 CAGR 百分位）——必附平均槓桿欄（§3-4）**：

{tbl_rr}

⚠️ D／E／F2 的平均槓桿系統性低於 2.0（regime 佔比／棘輪所致），其報酬排名弱勢
有相當部分是**曝險較低**而非機制較差；反之若其報酬排名仍高，代表機制擇時真的
補回了曝險缺口。Sharpe 排名（上表）幾乎不受平均槓桿影響，是機制比較的主軸。
""")

    # ══ 各市場明細表（10y、20y）═══════════════════════════════════════════
    detail = []
    for mk in ["TX_FUT", "SPX", "NKX", "HSI", "TX_IDX"]:
        for y in [10, 20]:
            t = _cell_table(agg, mk, y)
            detail.append(f"### {MARKET_LABEL[mk]}・{y} 年滾動\n\n{t}")
    sec.append("## 4b. 市場×期間明細（10／20 年；3／5 年見 CSV）\n\n"
               + "\n\n".join(detail) + "\n\n完整指標（含 3/5 年、C 的 SD20/60 敏感度變體）："
               "`r3_market_period_metrics.csv`。")

    # ══ §5 壓力窗口 ════════════════════════════════════════════════════════
    stx = payload["stress"]
    nk = _q2(payload)
    other_tbls = []
    for w in stx["window"].unique():
        if w.startswith("日經1990"):
            continue
        d = stx[stx["window"] == w].copy()
        d["slab"] = d["strategy"].map(_lab)
        d = d.sort_values(["market", "cumret"], ascending=[True, False])
        t = _table(d, [
            ("market", "市場", str), ("slab", "策略", str),
            ("cumret", "期間報酬", _pc), ("mdd", "MDD", _pc),
            ("avg_lev", "平均槓桿", _f), ("lev_at_trough", "谷底槓桿", _f),
            ("lev_end", "期末槓桿", _f), ("fuse_trig", "保險絲觸發", lambda v: str(int(v))),
            ("bankrupt", "歸零", lambda v: "是" if v else "否"),
        ])
        other_tbls.append(f"### {w}（{d['start'].iloc[0]} ~ {d['end'].iloc[0]}）\n\n{t}")
    sec.append(f"""## 5. 具名壓力窗口（規格 §4）

### 🔴 日經 1990–2003 長陰跌（本輪最重要的單一測試，獨立成節）

{nk}

槓桿路徑圖：`r3_nikkei1990_leverage.png`。

{chr(10).join(other_tbls)}

完整欄位（期末/期初槓桿、降載佔比、歸零旗標）見 `r3_stress_windows.csv`。
""")

    # ══ §6 Bootstrap ═══════════════════════════════════════════════════════
    bt = boot.copy()
    bt["mlab"] = bt["market"].map(MARKET_LABEL)
    bt["slab"] = bt["strategy"].map(_lab)
    bt = bt.sort_values(["market", "sharpe_p50"], ascending=[True, False])
    tbl_bt = _table(bt, [
        ("mlab", "市場", str), ("slab", "策略", str),
        ("sharpe_p5", "Sharpe p5", _f), ("sharpe_p50", "p50", _f),
        ("sharpe_p95", "p95", _f),
        ("mdd_p5", "MDD p5(最深)", _pc), ("mdd_p50", "MDD p50", _pc),
        ("tw_p5", "終值 p5", _f), ("tw_p50", "終值 p50", _f),
        ("ruin_prob", "歸零機率", lambda v: _pc(v, 2)),
    ])
    bs = payload["boot_sens"]
    blk_med = bs.groupby(["market", "mean_block"])["sharpe_p50"].median().unstack()
    blk_txt = "；".join(
        f"{mk}: " + " / ".join(f"L={int(c)}→{v:.2f}" for c, v in r.items())
        for mk, r in blk_med.iterrows())
    rc = payload["rank_cmp"]
    worst_gap = rc.loc[rc["rank_gap"].abs().idxmax()] if not rc.empty else None
    gap_txt = (f"歷史 vs bootstrap 排名偏離最大者：**{_lab(worst_gap['strategy'])}**"
               f"（{worst_gap['market']}，歷史第 {worst_gap['hist_rank']:.0f} → "
               f"bootstrap 第 {worst_gap['boot_rank']:.0f}）——最依賴歷史序列結構"
               "（這是資訊，不是罪狀）。" if worst_gap is not None else "")
    sec.append(f"""## 6. 區塊拔靴合成路徑（規格 §5）

方法：stationary bootstrap（Politis–Romano），期望區塊 {BOOT_MEAN_BLOCK} 交易日，
每市場 {BOOT_N_PATHS:,} 條 × {BOOT_YEARS} 年；校準參數凍結、不對合成路徑重校準
（σ_target 用全樣本 in-sample 版，與重抽母體一致，§3）。
分布小提琴圖：`r3_violin_{{市場}}.png`；原始逐路徑值：`r3_bootstrap_paths_*.csv`。

> **🔴 解讀警語（必讀）**：區塊拔靴會打斷跨區塊的序列相關，而**趨勢策略（D、F2）
> 的 alpha 正是活在長週期自相關裡**——趨勢濾網在合成路徑上的績效有**結構性向下
> 偏誤**，其 bootstrap 排名**不得**用於淘汰它。bootstrap 對趨勢策略只回答
> 「若市場毫無趨勢結構，它會多糟」（下界）；對 vol targeting／CPPI／保險絲
> 則是公平測試（其機制依賴的波動聚集在 60 日區塊內獲得保留）。

{tbl_bt}

**歸零一票否決（規格 §8-4）**：歸零機率 > {RUIN_VETO_PROB:.0%} 者不合格——
{('、'.join(_lab(s) for s in veto)) if veto else '本輪無策略觸發'}。

**區塊長度敏感度**（20／120 日，各市場 bootstrap Sharpe p50 中位）：{blk_txt}。

**歷史路徑排名 vs bootstrap 排名**（`r3_hist_vs_boot_rank.csv`、散點圖
`r3_rank_scatter.png`）：{gap_txt}
""")

    # ══ §7 TX carry 校準參照 ═══════════════════════════════════════════════
    cy = payload["carry"].copy()
    cy["slab"] = cy["strategy"].map(_lab)
    tbl_cy = _table(cy, [
        ("slab", "策略", str),
        ("cagr_fut", "CAGR 期貨版", _pc), ("cagr_idx", "CAGR 指數+融資版", _pc),
        ("cagr_diff", "差（≈carry+成本模型差）", lambda v: f"{v * 100:+.2f}%"),
        ("sharpe_fut", "Sharpe 期貨", _f), ("sharpe_idx", "Sharpe 指數", _f),
        ("mdd_fut", "MDD 期貨", _pc), ("mdd_idx", "MDD 指數", _pc),
    ])
    sec.append(f"""## 7. TX 期貨管線 vs 指數＋融資模型（carry 校準參照）

共同期間 {cy['start'].iloc[0]} ~ {cy['end'].iloc[0]}。兩版差異 = 期貨 carry（內生於
價格）＋成本模型差（期貨明細成本 vs 5bp 名目 + 融資利差）。此差即海外市場
「指數＋融資」簡化法相對真實期貨交易的偏差量級。

{tbl_cy}

差異的組成（以 2x 恆定槓桿為例）：期貨口徑對**全額權益**計息（+{TW_SHORT_RATE:.1%}）、
不付融資成本；指數＋融資口徑對超過 1 倍部分付 {TW_SHORT_RATE:.1%}+{FINANCING_SPREAD:.1%}
（L≈2 時 ≈ −2%）；再加上 TX 歷史逆價差的正 carry（前輪實測約 +1.3%/年 × 槓桿倍數）。
故海外市場的「指數＋融資」數字系統性**低估**同機制的期貨實作報酬，量級見上表
——但同市場內策略互比不受影響（所有策略吃同一套口徑）。
""")

    # ══ §8 敏感度 ══════════════════════════════════════════════════════════
    mb, mc = payload["mm_cost_base"], payload["mm_cost_10bp"]
    if not mb.empty and not mc.empty:
        order_b = list(mb["strategy"])
        order_c = list(mc["strategy"])
        same = order_b == order_c
        cost_txt = (f"海外調整成本 5bp→10bp：maximin 排名"
                    f"{'完全不變' if same else '有變動'}。"
                    f"5bp 序：{'、'.join(order_b[:3])}…；10bp 序：{'、'.join(order_c[:3])}…")
    else:
        cost_txt = "（無指數市場敏感度資料）"
    rt = payload["retention"]
    tbl_rt = _table(rt, [
        ("family", "家族（變體）", str), ("n_variants", "變體數", lambda v: str(int(v))),
        ("med_spread", "格內Sharpe極差(中位)", lambda v: _f(v, 3)),
        ("max_spread", "最大極差", lambda v: _f(v, 3)),
        ("worst_cell", "最敏感格", str),
    ])
    sec.append(f"""## 8. 成本與參數敏感度

**成本敏感度**：{cost_txt}（僅比較指數市場的格子；TX 期貨成本模型不變。）

**參數敏感度／平原度（規格 §2-2 retention）**——家族內變體在同一格子的
中位 Sharpe 極差（小＝平原、對自身參數不敏感）：

{tbl_rt}

若某家族極差大，其績效依賴特定參數，依規格屬扣分項。
""")

    # ══ §10 結論六問 ═══════════════════════════════════════════════════════
    sec.append(f"""## 10. 結論（規格 §10 六問）

**Q1 條件式曝險是否系統性優於機械家族？**
{_q1(payload)}

**Q2 日經 1990：保險絲 vs 趨勢濾網，是否推翻前輪結論？**
{_q2(payload)}

**Q3 F1 的保險絲是否被 vol targeting 取代而冗餘？**
{_q3(payload)}

**Q4 maximin 冠軍贏在機制還是平均槓桿較低？**
{_q4(payload)}

**Q5 bootstrap 歸零機率排序與歷史 MDD 排序是否一致？**
{_q5(payload)}

**Q6 誠實揭露（資料限制、in-sample 範圍、方法偏誤、未答問題）**：見 §11。
""")

    # ══ §11 誠實揭露 ═══════════════════════════════════════════════════════
    sec.append(f"""## 11. 誠實揭露

1. **資料限制**：日經自 {audit['NKX']['start']}（涵蓋 1989 高點✔）、恆生自
   {audit['HSI']['start']}、S&P 自 {audit['SPX']['start']}、TX 期貨自
   {audit['TX_FUT']['start']}、TAIEX 報酬指數自 {audit['TX_IDX']['start']}。
   日經／恆生**全程**、S&P **1988 前**沒有真實總報酬指數，以年代平均股息率
   近似——絕對報酬水準有偏，同市場內策略互比受影響為二階。
2. **短率近似**：日本／香港短率為分段常數（政策利率的粗略描摹）；融資成本在
   高利率年代（如日本 1990、香港 1998）的日內變化未反映。美國用 Ken French
   真實日頻 RF、台灣用 1.7% 常數（前輪口徑）。
3. **in-sample 範圍**：σ_target 於各市場最早 {CALIB_YEARS} 年校準——起點落在該段
   的滾動視窗與「自始以來」路徑，對 C/F 家族是部分 in-sample（一個自由度）；
   純評估段對照見 §4。B/D/E 無任何校準自由度，不受此影響。
4. **bootstrap 偏誤**：對趨勢策略（D、F2）結構性向下偏誤（§6 警語）；
   對 vol targeting／CPPI／保險絲公平。TX 合成路徑以 5bp 名目成本近似期貨
   明細成本（實測 ≈2.5–3bp/邊，屬保守），且無轉倉事件。
5. **訊號暖機**：σ̂ 初始值取最早 60 日變異數、200MA 可得前預設滿載；
   僅影響各序列最前段。趨勢訊號週頻檢查的相位固定於序列起點（未掃相位）。
6. **本輪仍無法回答的問題**：
   - 條件式策略的參數（λ、MA 長度、D_max）是否在**未來** regime 下仍然平原——
     四條歷史路徑＋合成路徑仍不足以回答結構轉變（利率制度、市場微結構）。
   - 日經 1990 型長陰跌在**有 carry 的期貨實作**下的結果（本輪日經僅指數+融資
     近似；日本期貨的逆價差結構未建模）。
   - 匯率避險成本：各市場以本幣計，台灣投資人的實際可得報酬含匯險，未建模。
   - 稅制：海外期貨/ETF 的股息稅、資本利得稅差異未建模。

## 附錄：輸出檔案清單

| 檔案 | 內容 |
|---|---|
| `r3_market_period_metrics.csv` | 每「策略×市場×期間」完整指標（含 C 的 SD 變體） |
| `r3_market_period_metrics_evalonly.csv` | 純評估段（起點在校準段後）的對照 |
| `r3_maximin.csv` / `r3_return_ranking.csv` | 主排名／報酬排名（附平均槓桿） |
| `r3_calibration.csv` | σ_target 校準表 |
| `r3_stress_windows.csv` | 壓力窗口逐格結果 |
| `r3_bootstrap_dist*.csv` / `r3_bootstrap_paths_*.csv` | bootstrap 分布與原始值 |
| `r3_bootstrap_block_sens.csv` | 區塊長度 20/120 敏感度 |
| `r3_hist_vs_boot_rank.csv` | 歷史 vs bootstrap 排名對照 |
| `r3_tx_carry_check.csv` | TX 期貨 vs 指數+融資對照 |
| `r3_inception.csv` / `r3_equity_*.csv` | 自始以來指標與資金曲線 |
| `r3_equity_*.png` | 各市場資金曲線疊圖（線性+對數） |
| `r3_nikkei1990_leverage.png` | 日經 1990 窗口槓桿路徑 |
| `r3_violin_*.png` | bootstrap Sharpe/MDD 小提琴圖 |
| `r3_rank_scatter.png` | 歷史 vs bootstrap 排名散點 |
""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sec))
    print(f"  報告 → {path}")
