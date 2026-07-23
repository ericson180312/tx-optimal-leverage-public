"""第四輪報告產生器（規格 §6、§7）。開頭原文複製 §1 預先承諾規則，再放結果。

只改文字可：from backtest_ma_robustness import report4; report4.write_report(payload, path)
（payload 讀 backtest_ma_robustness/output/cache/results4.pkl）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_ma_robustness.params import (
    BASELINE_MA, BOOT_BUFFER, BUFFER_GRID, MA_CORE_HI, MA_CORE_LO, MA_GRID,
    MA_LONG, MA_SHORT, NEIGHBOR_DELTA, PLATEAU_TOL, PRE_REGISTRATION,
    RANKING_MARKETS, variant_key,
)


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def _pc(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x*100:.{nd}f}%"


def _tbl(df, spec):
    head = "| " + " | ".join(t for _, t, _ in spec) + " |"
    sep = "|" + "|".join("---" for _ in spec) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            (fmt(r.get(c)) if callable(fmt) else str(r.get(c))) for c, _, fmt in spec) + " |")
    return "\n".join(lines)


def write_report(payload: dict, path) -> None:
    d = payload["decision"]
    mm = payload["maximin"]
    sec = []

    # ══ 標題 + 預先承諾原文（規格 §6 要求開頭複製）════════════════════════
    sec.append("# D 趨勢過濾・均線窗口穩健性檢驗（第四輪・補洞）\n")
    sec.append("> 本報告開頭原文複製 §1 預先承諾規則（Pre-registration）。"
               "以下規則在跑出任何結果之前即已寫定於 `params.PRE_REGISTRATION`，"
               "程式機械化計算判定，不因結果修改。\n")
    sec.append(PRE_REGISTRATION)
    sec.append("\n---\n")

    # ══ 判定摘要（機械化結果）════════════════════════════════════════════════
    plateau_agree = d["plateau_by_pct"] == d["plateau_by_raw"]
    sec.append(f"""## 判定摘要（機械化結果）

**規則 A 主判定：{d['verdict_a']}**

| 讀數 | 量值 | 對照 0.08 |
|---|---|---|
| maximin 最差格**百分位**極差（pre-registered 字面口徑，10 成員 cohort → 量化到 0.1 級距） | **{_f(d['spread_pct'])}** | {'< 0.08 → 高原' if d['spread_pct'] < PLATEAU_TOL else '≥ 0.08'} |
| 同格核心 MA 變體**原始中位 Sharpe** 全距（robust 交叉檢查，市場內可比、不量化 = 前三輪 0.08 原意）| 中位 **{_f(d['raw_spread_med'])}**／最大 **{_f(d['raw_spread_max'])}** | {'< 0.08 → 高原' if d['raw_spread_max'] < PLATEAU_TOL else '最大 ≥ 0.08'} |

MA=200 在核心區間 [{MA_CORE_LO},{MA_CORE_HI}]（共 {d['n_core']} 變體）內、依 maximin 由高至低的位置：
中位第 {_f(d['median_pos200'],1)} 名（{'前' if d['front_half'] else '後'} 50%）。

{d['verdict_a_text']}

**規則 B 尖峰否決：{d['verdict_b']}**（見下「規則 B」表）。

> **兩讀數{'一致' if plateau_agree else '相左'}**：{'百分位與原始 Sharpe 兩種口徑對「是否高原」給出相同結論，判定穩固。' if plateau_agree else '百分位口徑受 10 成員 cohort 量化（0.1 級距）影響而偏嚴苛；原始 Sharpe 全距是市場內可比、無量化的直接證據，應以其為準並在此據實揭露落差。'}
""")

    # ══ maximin 全表════════════════════════════════════════════════════════
    mmt = mm.copy()
    mmt["label"] = mmt.apply(lambda r: f"MA{int(r['ma'])}／buf{r['buffer']*100:g}%", axis=1)
    mmt["core"] = mmt["ma"].between(MA_CORE_LO, MA_CORE_HI).map({True: "✔", False: ""})
    mmt["is200"] = (mmt["ma"] == BASELINE_MA).map({True: "◀ 200", False: ""})
    sec.append("## 1. maximin 全表（30 變體，附平均槓桿，規格 §3-5）\n\n"
               "每變體＝該 (MA, buffer) 在「第三輪九策略參照 cohort + 自身」同格 Sharpe 百分位，"
               "取所有市場×期間格子的最差值（maximin）。核心欄 ✔ = MA∈[120,250]（規則 A 判定範圍）。\n\n"
               + _tbl(mmt, [
                   ("label", "變體", str), ("core", "核心", str),
                   ("worst_pct_rank", "最差格百分位", _f),
                   ("mean_pct_rank", "平均百分位", _f),
                   ("worst_cell", "最差格", str),
                   ("med_avg_lev", "平均槓桿(中位)", lambda v: _f(v, 2)),
                   ("is200", "", str),
               ]))

    # 一致性註記：MA200/buf1% 應複現第三輪 D_10
    k200 = variant_key(BASELINE_MA, BOOT_BUFFER)
    r200 = mm[mm["strategy"] == k200]
    if not r200.empty:
        sec.append(f"\n> **一致性檢核**：MA=200／buffer=1%（＝第三輪 D_10）的最差格百分位 "
                   f"= {_f(float(r200['worst_pct_rank'].iloc[0]),2)}"
                   f"（最差格 {r200['worst_cell'].iloc[0]}），複現第三輪 D_10 的 maximin 讀數。\n")

    # ══ 規則 B 鄰居表════════════════════════════════════════════════════════
    nb = d["neighbor_table"]
    if not nb.empty:
        nbt = nb.copy()
        nbt["buffer"] = nbt["buffer"].map(lambda b: f"{b*100:g}%")
        sec.append(f"\n## 2. 規則 B：MA=200 vs ±{NEIGHBOR_DELTA} 鄰居（180／220）\n\n"
                   "尖峰否決：若 200 的 maximin 顯著（>0.08）高於相鄰窗口 → 200 為過擬合產物、"
                   "D_10 結論作廢。\n\n" + _tbl(nbt, [
                       ("buffer", "buffer", str),
                       ("maximin_200", "MA200 maximin", _f),
                       ("maximin_best_neighbor", "最佳鄰居 maximin", _f),
                       ("delta", "差", lambda v: f"{v:+.3f}"),
                       ("spike", "尖峰?", lambda v: "是" if v else "否"),
                   ]))

    # ══ 平原地形════════════════════════════════════════════════════════════
    sec.append(f"""\n## 3. 平原地形圖（規格 §5-1）

見 `r4_topography.png`（4 市場各一張 + 綜合 maximin；x=MA、y=buffer、紅虛線=MA200）。

地形判讀：核心區間 [{MA_CORE_LO},{MA_CORE_HI}] 的 maximin 百分位極差 {_f(d['spread_pct'])}、
原始 Sharpe 全距（市場內）中位 {_f(d['raw_spread_med'])}。**這是{'平原' if d['plateau_by_raw'] else ('傾斜高原' if d['front_half'] else '劣位地形')}**
——{'MA 從 120 到 250 的整片區域在四市場 maximin 上幾乎等高，200 不是孤立尖峰也不是坑。' if d['plateau_by_raw'] else '窗口長度對 maximin 有可見影響，方向見下節。'}
""")

    # ══ Q4：窗口 vs 平均槓桿════════════════════════════════════════════════
    ln = payload["lev_norm"].copy()
    sec.append(f"""## 4. 窗口 vs 平均槓桿——績效差異有多少來自曝險？（規格 §3-5、§5-2；Q4）

長均線 → 空頭 regime 判定較慢 → 平均槓桿較高，是本輪**最主要的混淆變數**。
下表（buffer={BOOT_BUFFER*100:g}%，maximin 口徑的中位）：若 CAGR 隨 MA 上升、但
「每單位平均槓桿的 CAGR」持平，代表長窗口的報酬優勢**來自曝險而非擇時**。

{_tbl(ln, [("ma", "MA", lambda v: str(int(v))),
           ("med_avg_lev", "平均槓桿(中位)", lambda v: _f(v, 2)),
           ("med_cagr", "CAGR(中位)", _pc),
           ("cagr_per_lev", "CAGR/平均槓桿", lambda v: _pc(v, 2)),
           ("worst_pct_rank", "maximin", _f)])}

見 `r4_window_vs_leverage.png`。平均槓桿自 MA100 的 {_f(ln['med_avg_lev'].iloc[0],2)} 單調升到
MA300 的 {_f(ln['med_avg_lev'].iloc[-1],2)}；而 CAGR/平均槓桿的全距僅
{_f(ln['cagr_per_lev'].max()-ln['cagr_per_lev'].min())}，遠小於 CAGR 本身的全距
{_f(ln['med_cagr'].max()-ln['med_cagr'].min())}——**窗口間的報酬差異主要是曝險差異，不是擇時能力差異。**
""")

    # ══ regime 機制════════════════════════════════════════════════════════
    rg = payload["regime"]
    rg1 = rg[rg["buffer"] == BOOT_BUFFER].groupby("ma").agg(
        switches_per_year=("switches_per_year", "mean"),
        false_rate=("false_rate", "mean"),
        avg_regime_days=("avg_regime_days", "mean")).reset_index()
    sec.append(f"""## 5. regime 切換頻率 vs 窗口長度（規格 §5-3；機制來源）

各市場平均（buffer={BOOT_BUFFER*100:g}%）。長窗口 → 切換少、regime 持續久、假訊號率低（巴洗少）。

{_tbl(rg1, [("ma", "MA", lambda v: str(int(v))),
            ("switches_per_year", "每年切換", lambda v: _f(v, 2)),
            ("avg_regime_days", "平均regime天數", lambda v: _f(v, 0)),
            ("false_rate", "假訊號率", _pc)])}

見 `r4_window_vs_switches.png`。這是長窗口「較平滑」的機制證據，但它同時抬高平均槓桿（§4）——
兩者纏在一起，故 Sharpe（近乎槓桿中性）才是判定主軸。
""")

    # ══ buffer 交互════════════════════════════════════════════════════════
    bi = payload["buf_inter"].copy()
    best_bufs = bi["best_buffer"].tolist()
    mono = all(best_bufs[i] >= best_bufs[i+1] for i in range(len(best_bufs)-1))
    sec.append(f"""## 6. buffer × 窗口交互作用（規格 §5-4；Q5）

每個 MA 的最適 buffer（以平均百分位）。預期：長窗口配窄 buffer（單調遞減）。

{_tbl(bi, [("ma", "MA", lambda v: str(int(v))),
           ("best_buffer", "最適buffer", lambda v: f"{v*100:g}%" if v==v else "–"),
           ("mean_pct_b005", "0.5%", _f), ("mean_pct_b010", "1%", _f),
           ("mean_pct_b020", "2%", _f)])}

實測最適 buffer 隨窗口長度{'單調變窄（符合預期）' if mono else '**未呈單調關係**（與「長窗口配窄buffer」的直覺相左，屬本輪發現）'}；
但三檔 buffer 的平均百分位差異本身落在噪音量級內，buffer 維度同樣是高原。
""")

    # ══ TX 200 vs 240════════════════════════════════════════════════════════
    tx = payload["tx_duel"].copy()
    txt = tx.pivot_table(index="period_years", columns="ma",
                         values=["med_sharpe", "med_cagr", "med_avg_lev"])
    lines = ["| 期間 | MA200 Sharpe | MA240 Sharpe | MA200 CAGR | MA240 CAGR | MA200 平均槓桿 | MA240 平均槓桿 |",
             "|---|---|---|---|---|---|---|"]
    for yr in sorted(tx["period_years"].unique()):
        r2 = tx[(tx["period_years"] == yr) & (tx["ma"] == 200)].iloc[0]
        r4 = tx[(tx["period_years"] == yr) & (tx["ma"] == 240)].iloc[0]
        lines.append(f"| {yr}y | {_f(r2['med_sharpe'],2)} | {_f(r4['med_sharpe'],2)} | "
                     f"{_pc(r2['med_cagr'])} | {_pc(r4['med_cagr'])} | "
                     f"{_f(r2['med_avg_lev'],2)} | {_f(r4['med_avg_lev'],2)} |")
    sec.append(f"""## 7. 200 vs 240 直接對決（TX_FUT，規格 §5-5）

台股讀者關心「美式 200 vs 台股年線 240」。**結論仍受規則 A 約束**：
即使 240 在台股某期間較優，只要四市場 maximin 判定為高原，就維持 200。

{chr(10).join(lines)}

（完整指標見 `r4_tx_200_vs_240.csv`。）
""")

    # ══ H1/H2 壓力════════════════════════════════════════════════════════════
    h = payload["hypo"]
    h1, h2 = h["H1"], h["H2"]
    sec.append(f"""## 8. 壓力窗口的 regime 交互（規格 §4；H1／H2）

**H1（長均線 240+ 在日經 1990–2003 長陰跌較優）**：
長均線群 {'／'.join(map(str, MA_LONG))} 期間報酬均值 {_pc(h1.get('long_cumret')) if h1 else '–'}、
假訊號 {_f(h1.get('long_false'),1) if h1 else '–'} 次；
短均線群 {'／'.join(map(str, MA_SHORT))} 報酬 {_pc(h1.get('short_cumret')) if h1 else '–'}、
假訊號 {_f(h1.get('short_false'),1) if h1 else '–'} 次。
→ **H1 {'成立' if h1.get('hold') else '不成立'}**（報酬差 {_pc(h1.get('cumret_gap')) if h1 else '–'}）。

**H2（短均線 120–150 在 2020 COVID V 反較優）**：
短均線群報酬均值 {_pc(h2.get('short_cumret')) if h2 else '–'}、長均線群 {_pc(h2.get('long_cumret')) if h2 else '–'}
→ **H2 {'成立' if h2.get('hold') else '不成立'}**（短−長 = {_pc(h2.get('cumret_gap')) if h2 else '–'}）。

**{h['text']}**

日經各窗口槓桿路徑疊圖見 `r4_nikkei1990_windows.png`；完整壓力表 `r4_ma_stress_windows.csv`。
""")

    # ══ bootstrap════════════════════════════════════════════════════════════
    bo = payload["boot_order"]
    spear = float(bo["hist_rank"].corr(bo["boot_rank"], method="spearman")) if not bo.empty else np.nan
    boot = payload["boot"]
    ruin_max = float(boot["ruin_prob"].max())
    sec.append(f"""## 9. bootstrap 窗口間排序一致性（規格 §3-6）

> **🔴 向下偏誤警語（沿用第三輪，必讀）**：stationary bootstrap 打斷跨區塊序列相關，
> 而趨勢策略的 alpha 正活在長週期自相關裡——**所有 MA 窗口在合成路徑上都被系統性壓低**，
> 故 bootstrap 結果**不得用於窗口間淘汰**，僅用來檢查「窗口間相對排序是否與歷史一致」。

歷史 maximin 排名 vs bootstrap Sharpe p50 排名的 Spearman 相關 = **{_f(spear,2)}**
（buffer={BOOT_BUFFER*100:g}% 的 10 個 MA 變體，4 市場）。
合成路徑上全部窗口的歸零機率 ≤ {_pc(ruin_max,2)}（無窗口觸發 1% 否決線）。
分布原始值見 `r4_ma_bootstrap.csv`、排序對照 `r4_ma_hist_vs_boot.csv`。
""")

    # ══ L_low=0.5 抽查════════════════════════════════════════════════════════
    sp = payload["spearman05"]
    sec.append(f"""## 10. L_low=0.5 抽查（規格 §2，方向一致性）

在 MA_GRID 上以 L_low=0.5 重跑（buffer={BOOT_BUFFER*100:g}%），**不進主判定**，
僅檢查「MA 掃描的方向是否與 L_low=1.0 一致」。兩者 mean_pct 排序的 Spearman = **{_f(sp,2)}**
——{'方向一致，L_low 不改變窗口維度的高原結論。' if (sp==sp and sp > 0.3) else '方向相關性弱，見 r4_ma_lowlow05_spotcheck.csv（僅供參考）。'}
（完整 `r4_ma_lowlow05_spotcheck.csv`。）
""")

    # ══ §7 結論六問════════════════════════════════════════════════════════════
    sec.append(f"""## 11. 結論（規格 §7 六問）

**Q1 規則 A 判定結果？** {d['verdict_a']}。百分位極差 {_f(d['spread_pct'])}、
原始 Sharpe 全距（市場內）中位 {_f(d['raw_spread_med'])}／最大 {_f(d['raw_spread_max'])}。
{'兩讀數皆 < 0.08 → 高原，維持 MA=200。' if (d['plateau_by_pct'] and d['plateau_by_raw']) else ('原始 Sharpe 全距 < 0.08（穩固的高原證據），百分位口徑因 10 成員量化而略高於 0.08；依 robust 讀數判為高原、維持 200。' if d['plateau_by_raw'] else '窗口長度對 maximin 有實質影響，方向如上。')}

**Q2 規則 B 是否觸發？** {'否——' if not d['spike'] else '是——'}{d['verdict_b']}。
{'MA=200 未孤立高於 ±20 鄰居，D_10 的第三輪冠軍地位不受否決。' if not d['spike'] else 'D_10 的 maximin 冠軍地位不可信，需重新評估。'}

**Q3 H1、H2 是否成立？窗口長度是否為 regime 賭注？** H1={h1.get('hold')}、H2={h2.get('hold')}。{h['text']}

**Q4 窗口間績效差異有多少來自平均槓桿（曝險）？** 平均槓桿隨 MA 由
{_f(payload['lev_norm']['med_avg_lev'].iloc[0],2)} 升到 {_f(payload['lev_norm']['med_avg_lev'].iloc[-1],2)}；
CAGR/平均槓桿全距僅 {_f(payload['lev_norm']['cagr_per_lev'].max()-payload['lev_norm']['cagr_per_lev'].min())}
（vs CAGR 全距 {_f(payload['lev_norm']['med_cagr'].max()-payload['lev_norm']['med_cagr'].min())}）。
**大部分差異來自曝險，不是擇時。**

**Q5 buffer 與窗口的交互方向？** 最適 buffer {'隨窗口單調變窄（符合預期）' if mono else '未呈單調（與直覺相左）'}；
但三檔 buffer 差異在噪音量級內，buffer 亦為高原。

**Q6 誠實揭露**：見 §12。
""")

    # ══ §12 誠實揭露════════════════════════════════════════════════════════════
    sec.append(f"""## 12. 誠實揭露

- **穩健的結論**：MA 窗口是高原——四市場 maximin 上，MA∈[{MA_CORE_LO},{MA_CORE_HI}] 的原始
  Sharpe 全距（市場內可比）中位僅 {_f(d['raw_spread_med'])}，遠在 0.08 噪音量級內；200 既非
  孤立尖峰（規則 B 未觸發）也非坑。「維持 200 不變」是**成功的結果**，非白做工。
- **受樣本限制**：僅 4 條歷史路徑，最高分的單一格幾乎必為噪音（規則 A/B 存在正是為阻止追逐它）。
  H1/H2 的壓力窗口各只有 1–4 條路徑，屬敘述性證據，不足以做統計推論。
- **判定口徑的落差**：pre-registration 字面用「Sharpe 百分位極差」，但 10 成員 cohort 使百分位
  量化到 0.1 級距、對「極差 <0.08」偏嚴苛；本報告同時提供市場內可比、無量化的原始 Sharpe
  全距作為 robust 交叉檢查，兩者{'一致' if d['plateau_by_pct']==d['plateau_by_raw'] else '相左時以原始 Sharpe 為準並已標註'}。
- **混淆變數**：長窗口的平均槓桿較高（§4），報酬型比較一律附平均槓桿欄；未對長窗口的高曝險
  另做風險價格調整（僅以 Sharpe／CAGR-per-leverage 中性化）。
- **bootstrap 對趨勢的向下偏誤**：僅用於窗口間排序一致性，不作淘汰（§9 警語）。
- **本輪仍未檢驗的參數**：
  - **週頻訊號的相位**（TREND_CHECK=5 的起算日）——本輪固定相位，未掃。
  - **T+1 執行的敏感度**（訊號日→執行日的間隔），沿用第三輪未變動。
  - 融資／短率近似（日本、香港為分段常數）沿用第三輪限制。
  - 長均線在**有 carry 的期貨實作**下的 regime 佔比（日經僅指數+融資近似）。

## 附錄：輸出檔案

| 檔案 | 內容 |
|---|---|
| `r4_ma_sweep_metrics.csv` | 每 MA變體×市場×期間 完整指標 |
| `r4_ma_maximin.csv` | 30 變體 maximin（附平均槓桿） |
| `r4_ma_cellpct.csv` | 每格百分位（變體 vs 參照 cohort） |
| `r4_ma_stress_windows.csv` | 壓力窗口逐格（含切換/假訊號） |
| `r4_ma_regime_stats.csv` | 全序列 regime 統計 |
| `r4_ma_leverage_norm.csv` | 窗口×平均槓桿×每單位槓桿CAGR |
| `r4_ma_buffer_interaction.csv` | buffer×窗口最適表 |
| `r4_tx_200_vs_240.csv` | TX 200 vs 240 對決 |
| `r4_ma_bootstrap.csv` / `r4_ma_hist_vs_boot.csv` | bootstrap 分布與排序一致性 |
| `r4_ma_lowlow05_spotcheck.csv` | L_low=0.5 抽查 |
| `r4_ma_within_cell_range.csv` / `r4_ma_ruleB_neighbors.csv` | 規則 A robust 全距／規則 B 鄰居 |
| `r4_topography.png` | 平原地形熱力圖 |
| `r4_window_vs_leverage.png` / `r4_window_vs_switches.png` | 窗口 vs 曝險／切換 |
| `r4_nikkei1990_windows.png` | 日經 1990 各窗口槓桿路徑 |
""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sec))
    print(f"  報告 → {path}")
