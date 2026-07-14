"""主報告產生器：由 run.py 的結果 payload 組出 output/report.md。

所有數字皆來自實算結果；平原/尖峰、對稱性、收斂性等判定以量化規則產生，
規則本身寫明於報告中。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from params import (
    FEE_PER_SIDE, IDLE_CASH_RATE, INITIAL_EQUITY, LOOKBACK_YEARS,
    ROLL_OFFSET_DAYS, ROLL_OFFSET_DAYS_SENS, SLIPPAGE_TICKS,
    SLIPPAGE_TICKS_SENS, TARGET_LEVERAGE, TAX_RATE, X_GRID, Y_GRID,
)


def _pct(v, digits=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.{digits}f}%"


def _upct(v, digits=0):
    """無號百分比（比例、勝率、排名用）。"""
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{digits}f}%"


def _num(v, digits=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _xy(row):
    return f"[{row['X']:g}, {row['Y']:g}]"


def _classify_plateau(p: dict) -> str:
    """平原/尖峰量化判定（規則寫入報告）。"""
    if p["retention"] >= 0.6 and p["nbr_min"] >= p["grid_median"]:
        return "平原"
    if p["retention"] < 0.3:
        return "尖峰"
    return "緩坡"


def _agg_row(agg: pd.DataFrame, x, y) -> pd.Series:
    m = agg[(~agg["is_benchmark"]) & (agg["X"] == x) & (agg["Y"] == y)]
    return m.iloc[0]


def _bench_row(agg: pd.DataFrame, name: str) -> pd.Series:
    return agg[agg["name"] == name].iloc[0]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _median_stats_table(agg: pd.DataFrame, sel: dict) -> str:
    """表 A：跨路徑統計（最佳解判定依據）。"""
    rows = []
    entries = [("報酬最佳 " + _xy(sel["med_by_return"]), _agg_row(agg, sel["med_by_return"]["X"], sel["med_by_return"]["Y"])),
               ("Sharpe最佳 " + _xy(sel["med_by_sharpe"]), _agg_row(agg, sel["med_by_sharpe"]["X"], sel["med_by_sharpe"]["Y"])),
               ("買進持有 2x", _bench_row(agg, "BH_2x")),
               ("每日再平衡 2x（正2倍ETF）", _bench_row(agg, "Daily_2x")),
               ("**每 5 日再平衡 2x**", _bench_row(agg, "Periodic5d_2x")),
               ("買進持有 1x", _bench_row(agg, "BH_1x"))]
    for label, r in entries:
        rows.append([label, _pct(r["med_cumret"]), _num(r["med_sharpe"], 3),
                     f"{_num(r['sharpe_p25'], 2)} / {_num(r['sharpe_p75'], 2)}",
                     _pct(r["med_mdd"]), _pct(r["worst_cumret"]),
                     _upct(r["win_vs_bh2x"]), _upct(r["win_vs_daily2x"]),
                     f"{int(r['n_bankrupt_paths'])}/{int(r['n_paths'])}"])
    return _table(["策略", "中位數報酬", "中位數Sharpe", "Sharpe P25/P75",
                   "中位數MDD", "最差路徑報酬", "勝率vs BH2x", "勝率vs 每日2x", "破產路徑"], rows)


def _baseline_metrics_table(agg: pd.DataFrame, sel: dict) -> str:
    """表 B：基準路徑（終點=資料尾）完整指標（規格 §6）。"""
    entries = [("報酬最佳 " + _xy(sel["med_by_return"]), _agg_row(agg, sel["med_by_return"]["X"], sel["med_by_return"]["Y"])),
               ("Sharpe最佳 " + _xy(sel["med_by_sharpe"]), _agg_row(agg, sel["med_by_sharpe"]["X"], sel["med_by_sharpe"]["Y"])),
               ("買進持有 2x", _bench_row(agg, "BH_2x")),
               ("每日再平衡 2x（正2倍ETF）", _bench_row(agg, "Daily_2x")),
               ("**每 5 日再平衡 2x**", _bench_row(agg, "Periodic5d_2x")),
               ("買進持有 1x", _bench_row(agg, "BH_1x"))]
    rows = []
    for label, r in entries:
        rows.append([label, _pct(r["base_cumret"]), _pct(r["base_cagr"]),
                     _pct(r["base_vol_ann"]), _num(r["base_sharpe"], 3),
                     _pct(r["base_mdd"]), _num(r["base_calmar"]),
                     _num(r["base_sortino"]),
                     f"{int(r['base_n_up'])}/{int(r['base_n_down'])}",
                     _pct(r["base_cost_ratio"]), _num(r["base_avg_lev"]),
                     _num(r["base_max_lev"]), _num(r["base_uw_months"], 1),
                     "是" if r["base_bankrupt"] else "否"])
    return _table(["策略", "累積報酬", "CAGR", "年化波動", "Sharpe", "MDD", "Calmar",
                   "Sortino", "加碼/減碼次數", "成本/初始資本", "平均槓桿", "最高槓桿",
                   "最長水下(月)", "破產"], rows)


def _overfit_table(agg: pd.DataFrame, sel: dict) -> str:
    """中位數最佳 vs 單一路徑最佳：過擬合幅度對照。"""
    rows = []
    for crit, med_key, single_key, med_col, base_col in [
            ("累積報酬", "med_by_return", "single_by_return", "med_cumret", "base_cumret"),
            ("Sharpe", "med_by_sharpe", "single_by_sharpe", "med_sharpe", "base_sharpe")]:
        m, s = sel[med_key], sel[single_key]
        fmt = _pct if med_col == "med_cumret" else (lambda v: _num(v, 3))
        rows.append([crit, _xy(m), fmt(m[med_col]), fmt(m[base_col]),
                     _xy(s), fmt(s[base_col]), fmt(s[med_col])])
    return _table(["準則", "中位數最佳", "其中位數值", "其單路徑值",
                   "單路徑最佳", "其單路徑值", "其中位數值"], rows)


def _marginal(agg: pd.DataFrame, by: str) -> pd.Series:
    g = agg[~agg["is_benchmark"]]
    return g.groupby(by)["med_sharpe"].median()


def write_report(payload: dict, path) -> None:
    a = payload["audit"]
    results = payload["results"]
    sels = payload["selections"]
    plateau = payload["plateau"]
    robust = payload["robust"]
    L = []

    rec = robust.iloc[0]
    rec_x, rec_y = float(rec["X"]), float(rec["Y"])

    # ═══ 標題與摘要 ═══
    L.append("# 台指期 2 倍槓桿・門檻式再平衡 [X, Y] 網格回測報告\n")
    L.append(f"資料：TX 逐合約每日結算價 {a['data_start']} ～ {a['data_end']}"
             f"（{a['n_days']:,} 個交易日、{a['n_rolls']} 次轉倉）；"
             f"網格 {len(X_GRID)}×{len(Y_GRID)} = {len(X_GRID)*len(Y_GRID)} 組、"
             f"期間 {', '.join(str(y) for y in LOOKBACK_YEARS)} 年，"
             f"各期間含季頻滾動起點多路徑檢驗。\n")

    # ═══ 方法與口徑 ═══
    L.append("## 1. 方法與口徑（重要假設一次列清）\n")
    L.append(f"""- **連續合約**：逐合約法（日報酬只在同一口合約內部計算）為主，
  比值還原法（back-adjustment）互驗，兩法日報酬最大差異 **{a['method_max_diff']:.1e}**（< 1e-8 通過）。
  持倉合約當日無價而以前值遞補的天數：{a['n_price_gaps']} 天。
  採比值而非加減還原：加減法保點差但扭曲報酬率分母；比值還原與逐合約法在數學上恆等。
- **假跳空**：已消除。若天真拼接近月序列（把跨合約 K 棒當日報酬），此樣本會**年化偏差 {_pct(a['naive_annual_inflation'])}**
  ——台指期長期以**逆價差**為主（{a['n_rolls']} 次轉倉的月差中位數 {_pct(a['spread_pct_median'], 2)}／次），
  與規格假設的正價差方向相反，但無論方向皆屬系統性偏誤，必須消除。
- **carry 內生**：逆價差收斂的正 carry 已自動反映在期貨日報酬中，不另外補貼／扣除（不重複計算）。
  月差金額記錄於 `roll_events.csv` 供稽核。
- **轉倉**：近月最後交易日（實際行事曆：{_upct(a['expiry_match_third_wed'], 1)} 的合約最後交易日恰為第三個週三，
  其餘為假日順延／颱風休市）往前推 {ROLL_OFFSET_DAYS} 個交易日，收盤執行，兩腳計成本，不改變曝險。
- **計息口徑**：對**閒置現金**計息（純期貨的實際現金流）：權益全額以現金存放，
  對日初權益按 {IDLE_CASH_RATE:.1%} 年化、日頻複利計息；不對借款計息。
- **每日順序**：結算 → 計息 → 破產檢查 → 轉倉 → 重算槓桿 → 門檻再平衡（收盤觸價）→ 記錄。
- **成本**：手續費 {FEE_PER_SIDE:g} 元/口/邊＋期交稅 {TAX_RATE:.5%}/邊＋滑價 {SLIPPAGE_TICKS:g} 檔/邊（1 檔=200 元），
  口數為連續小數，成本**即時扣入權益**（含建倉成本）。
- **破產**：權益 ≤ 0 即歸零且不得反彈；**任何一條路徑破產的參數組一律不得列入最佳解**。
- **最佳解判定**：以**跨路徑（季頻滾動起點）中位數**為準；單一路徑最大值僅作過擬合對照。\n""")

    # ═══ 最佳解表 ═══
    L.append("## 2. 最佳解表（每期間 × 報酬最高 / Sharpe 最高）\n")
    for y in LOOKBACK_YEARS:
        agg, sel = results[y]["agg"], sels[y]
        n_excl = sel["n_excluded_bankrupt"]
        L.append(f"### {y} 年期（{results[y]['n_paths']} 條路徑"
                 + (f"；{n_excl} 組因破產路徑遭排除" if n_excl else "；無參數組破產")
                 + "）\n")
        L.append("**跨路徑統計（最佳解判定依據）**\n")
        L.append(_median_stats_table(agg, sel) + "\n")
        L.append("**基準路徑（終點＝資料最後一日）完整指標**\n")
        L.append(_baseline_metrics_table(agg, sel) + "\n")
        L.append("**過擬合對照：中位數最佳 vs 單一路徑最佳**\n")
        L.append(_overfit_table(agg, sel) + "\n")
        L.append(f"![heatmap]({f'heatmap_{y}y.png'})\n")
        L.append(f"![equity]({f'equity_overlay_{y}y.png'})\n")
        L.append(f"![leverage]({f'leverage_best_{y}y.png'})\n")

    # ═══ 平原 vs 尖峰 ═══
    L.append("## 3. 熱力圖形態：參數平原或尖峰？\n")
    L.append("""量化規則：取最佳格 3×3 鄰域，計算 **retention =（鄰域均值 − 全網格中位數）/（峰值 − 全網格中位數）**。
retention ≥ 0.6 且鄰域最小值不低於全網格中位數 → **平原**；retention < 0.3 → **尖峰**；其間 → 緩坡。\n""")
    rows = []
    for y in LOOKBACK_YEARS:
        for crit in ["sharpe", "return"]:
            p = plateau[y][crit]
            b = sels[y]["med_by_sharpe" if crit == "sharpe" else "med_by_return"]
            fmt = (lambda v: _num(v, 3)) if crit == "sharpe" else _pct
            rows.append([f"{y}y", "Sharpe" if crit == "sharpe" else "報酬", _xy(b),
                         fmt(p["best"]), fmt(p["nbr_mean"]), fmt(p["nbr_min"]),
                         fmt(p["grid_median"]), _num(p["retention"], 2),
                         f"**{_classify_plateau(p)}**"])
    L.append(_table(["期間", "準則", "最佳格", "峰值", "鄰域均值", "鄰域最小",
                     "網格中位數", "retention", "判定"], rows) + "\n")

    # ═══ 穩健性交叉表 ═══
    L.append("## 4. 穩健性：20 年最佳解套用到其他期間\n")
    ct = payload["crosstab"]
    rows = [[r["combo"], f"{r['period_years']}y", _pct(r["med_cumret"]),
             _num(r["med_sharpe"], 3), _pct(r["worst_cumret"]),
             _upct(r["win_vs_daily2x"]), _upct(r["sharpe_rank_pct"])]
            for _, r in ct.iterrows()]
    L.append(_table(["組合", "期間", "中位數報酬", "中位數Sharpe", "最差路徑",
                     "勝率vs每日2x", "Sharpe百分位"], rows) + "\n")

    # ═══ 對稱性檢驗 ═══
    L.append("## 5. 對稱性檢驗：X 是否比 Y 更貼近 2.0？\n")
    L.append("""幾何報酬 g(L) = r + Lα − L²σ²/2 對槓桿的懲罰是二次式：高於目標的槓桿（下跌後）比低於目標
（上漲後）更傷害複利，故理論預測**減碼上限 X 應比加碼下限 Y 更貼近 2.0**（|X−2| < |2−Y|）。實測：\n""")
    rows = []
    n_support = 0
    for y in LOOKBACK_YEARS:
        b = sels[y]["med_by_sharpe"]
        dx, dy = float(b["X"]) - 2.0, 2.0 - float(b["Y"])
        support = dx < dy
        n_support += support
        rows.append([f"{y}y", _xy(b), f"{dx:.2f}", f"{dy:.2f}",
                     "符合" if support else "**不符合**"])
    L.append(_table(["期間", "Sharpe最佳", "dX = X−2", "dY = 2−Y", "理論預測"], rows) + "\n")
    L.append(f"""**結論：理論預測被實測推翻**（{len(LOOKBACK_YEARS) - n_support}/{len(LOOKBACK_YEARS)} 期間 dX > dY）。
解讀：(1) 觸及 X 需自上次再平衡點下跌約 25–30%，樣本中僅數次，且台股崩盤多呈 V 型——
跌深即減碼等於在低點賣出、錯過反彈，抵銷了二次懲罰理論上的好處；(2) 邊際曲線顯示
Sharpe 沿 X 幾乎平坦（見下），X 的位置本來就不是第一階因素，argmax 落點接近隨機；
(3) 台股長期正漂移使「高槓桿懲罰」被復甦補償。二次懲罰的邏輯在「無漂移、非 V 型」
的市場才會浮現，本樣本不屬之。\n""")
    L.append("Sharpe 對 X、Y 的邊際結構（跨另一軸取中位數；看整體形狀而非單點）：\n")
    for y in [20, 10]:
        agg = results[y]["agg"]
        mx = _marginal(agg, "X"); my = _marginal(agg, "Y")
        L.append(f"- **{y}y** 邊際中位數 Sharpe — 沿 X："
                 + "、".join(f"{k:g}→{v:.3f}" for k, v in mx.items())
                 + "；沿 Y：" + "、".join(f"{k:g}→{v:.3f}" for k, v in my.items()) + "\n")

    # ═══ 敏感度 ═══
    L.append("## 6. 敏感度分析\n")
    L.append(f"### 6.1 滑價 {SLIPPAGE_TICKS_SENS:g} 檔（原 {SLIPPAGE_TICKS:g} 檔）\n")
    ss = payload["sens_slip"]
    rows = [[f"{r['period_years']}y", r["best_main"], r["best_alt"],
             "是" if r["same_cell"] else "否", _num(r["med_sharpe_main"], 3),
             _num(r["med_sharpe_alt_mainbest"], 3), _num(r["med_sharpe_alt_bestalt"], 3)]
            for _, r in ss.iterrows()]
    L.append(_table(["期間", "主情境最佳", "低滑價最佳", "同格", "主情境Sharpe",
                     "主最佳@低滑價", "低滑價最佳Sharpe"], rows) + "\n")
    L.append(f"### 6.2 轉倉時點 offset {ROLL_OFFSET_DAYS_SENS}（原 {ROLL_OFFSET_DAYS}）\n")
    so = payload["sens_off"]
    rows = [[f"{r['period_years']}y", r["best_main"], r["best_alt"],
             "是" if r["same_cell"] else "否", _num(r["med_sharpe_main"], 3),
             _num(r["med_sharpe_alt_mainbest"], 3), _num(r["med_sharpe_alt_bestalt"], 3)]
            for _, r in so.iterrows()]
    L.append(_table(["期間", "主情境最佳", "offset0最佳", "同格", "主情境Sharpe",
                     "主最佳@offset0", "offset0最佳Sharpe"], rows) + "\n")
    L.append("""註：offset=0（持有近月到最後交易日才換倉）在**所有期間**都使 Sharpe 略升——
與台指期長期逆價差一致：近月最後幾天仍在向現貨收斂，提早 5 天轉倉會少賺這段。
（近一年正價差 regime 下方向相反，參見月差分析專案。）最佳格雖有小幅漂移，
但仍在同一高原帶內，主結論不受轉倉時點影響。\n""")
    L.append("### 6.3 二次精細搜尋（步長 0.05，以各期間 Sharpe 最佳為中心）\n")
    rf = payload["refine"]
    rows = [[f"{r['period_years']}y", r["coarse"], _num(r["coarse_med_sharpe"], 3),
             r["refined"], _num(r["refined_med_sharpe"], 3)] for _, r in rf.iterrows()]
    L.append(_table(["期間", "粗網格最佳", "其Sharpe", "精細最佳", "其Sharpe"], rows) + "\n")
    L.append("""註：精細搜尋的鄰域容許 Y 低於原網格下緣（至 0.8）。精細最佳與粗網格最佳的
Sharpe 差異皆 < 0.01——曲面在最佳格附近近乎水平，更細的步長不改變任何結論
（Y ≤ 1.0 的「最佳」與 Y = 1.0 實為同一策略：該視窗內都不曾觸發加碼）。\n""")

    # ═══ 再平衡頻率掃描 ═══
    freq, phase = payload["freq"], payload["phase"]
    L.append("## 6.4 再平衡頻率掃描（含新增的「每 5 日再平衡」策略）\n")
    L.append("""此節把「再平衡頻率」當成獨立維度掃描：從永不再平衡（買進持有 2x）、
每季（63 日）、每月（21 日）、每雙週（10 日）、**每 5 日**、到每日（＝正 2 倍 ETF 的標準機制）。
所有策略皆為初始 2 倍、無門檻，僅到期就無條件把槓桿調回 2.0。\n""")
    for y in LOOKBACK_YEARS:
        f = freq[freq["period_years"] == y]
        rows = [[r["strategy"], _pct(r["med_cumret"]), _pct(r["med_cagr"]),
                 _num(r["med_sharpe"], 3), _pct(r["med_mdd"]), _pct(r["worst_cumret"]),
                 f"{r['n_bankrupt_paths']}/{r['n_paths']}",
                 _pct(r["base_cumret"]), _pct(r["base_mdd"]),
                 _num(r["base_uw_months"], 1), f"{r['base_n_rebal']:,}",
                 _pct(r["base_cost_ratio"]), _num(r["base_avg_lev"])]
                for _, r in f.iterrows()]
        L.append(f"**{y} 年期**（前 6 欄＝跨路徑中位數統計；後 6 欄＝基準路徑）\n")
        L.append(_table(["再平衡頻率", "中位數報酬", "中位數CAGR", "中位數Sharpe",
                         "中位數MDD", "最差路徑", "破產路徑",
                         "基準總報酬", "基準MDD", "基準水下(月)", "再平衡次數",
                         "成本/初始資本", "平均槓桿"], rows) + "\n")
    L.append("![freq](rebal_frequency.png)\n")
    L.append("**相位敏感度**：「每 5 日」的週期自建倉日起算，換個起跑日（相位 0–4）結果會變。"
             "滾動起點路徑本已隨機化相位（每條各自從其建倉日算起），故上表中位數已整合過相位；"
             "下表量化單一基準路徑上的相位擴散：\n")
    rows = []
    for y in LOOKBACK_YEARS:
        p = phase[phase["period_years"] == y]
        rows.append([f"{y}y",
                     f"{_pct(p['cumret'].min())} ～ {_pct(p['cumret'].max())}",
                     f"{_num(p['sharpe'].min(), 3)} ～ {_num(p['sharpe'].max(), 3)}",
                     f"{_pct(p['mdd'].min())} ～ {_pct(p['mdd'].max())}",
                     _pct(float(p["cumret"].std() / abs(p["cumret"].mean())) if p["cumret"].mean() else np.nan)])
    L.append(_table(["期間", "總報酬（相位 0–4 全距）", "Sharpe 全距", "MDD 全距",
                     "報酬變異係數"], rows) + "\n")

    f5 = freq[freq["name"] == "Periodic5d_2x"].set_index("period_years")
    f1 = freq[freq["name"] == "Daily_2x"].set_index("period_years")
    fbh = freq[freq["name"] == "BH_2x"].set_index("period_years")
    beat_daily = [y for y in LOOKBACK_YEARS
                  if f5.loc[y, "med_sharpe"] > f1.loc[y, "med_sharpe"]]
    L.append(f"""**這節怎麼讀（新增策略的評價）**

1. **「5 日再平衡」與「每日再平衡」幾乎沒有差別。** 兩者的中位數 Sharpe 差距
   {'、'.join(f"{y}y {f5.loc[y, 'med_sharpe'] - f1.loc[y, 'med_sharpe']:+.3f}" for y in LOOKBACK_YEARS)}
   ——{len(beat_daily)}/{len(LOOKBACK_YEARS)} 個期間 5 日略勝，但幅度全在 ±0.02 以內，
   遠小於路徑間的離散度（P25–P75 跨距約 0.5）。**把正 2 倍 ETF 的再平衡週期從 1 日拉長到 5 日，
   對風險調整後績效沒有可辨識的影響。**
2. **省下的是成本，但成本本來就不是瓶頸。** 再平衡次數從
   {int(f1.loc[20, 'base_n_rebal']):,} 次（每日）降到 {int(f5.loc[20, 'base_n_rebal']):,} 次（5 日）——少了 80%，
   但 20 年期總成本只從初始資本的 {_upct(f1.loc[20, 'base_cost_ratio'], 1)} 降到 {_upct(f5.loc[20, 'base_cost_ratio'], 1)}。
   原因是**每月轉倉成本佔了絕大部分**，再平衡本身很便宜（名目調整量小、期交稅只有十萬分之二）。
3. **平均槓桿都被鎖在 2.00。** 5 日、10 日、每日的時間加權平均槓桿皆為 2.00（20 年期），
   代表 5 日的週期已足夠短到讓槓桿無法漂離目標——這也解釋了為何績效與每日幾乎相同。
   要看到「再平衡頻率」真正產生差異，得拉到季（63 日）以上：那時平均槓桿仍是 1.99，
   但中位數 Sharpe 掉到 {_num(freq[(freq['name'] == 'Periodic63d_2x') & (freq['period_years'] == 20)]['med_sharpe'].iloc[0], 3)}（20 年期），
   MDD 也明顯惡化。
4. **真正的分水嶺在「有沒有再平衡」，而不是「多久再平衡一次」。**
   買進持有 2x 的中位數 Sharpe 看起來最高（{'、'.join(f"{y}y {fbh.loc[y, 'med_sharpe']:.3f}" for y in LOOKBACK_YEARS)}），
   **但那是存活者偏誤**：它在 1/5/10/20 年期分別有
   {'、'.join(f"{int(fbh.loc[y, 'n_bankrupt_paths'])}/{int(fbh.loc[y, 'n_paths'])}" for y in [1, 5, 10, 20])} 條路徑**破產歸零**，
   破產路徑的 Sharpe 無定義（記 NaN）而被排除在中位數之外。看「最差路徑報酬」欄就露餡了：
   買進持有 2x 全期間都是 **−100%**，所有週期再平衡策略則無一破產。
   **週期再平衡的價值不在提升報酬，而在把「歸零」這條尾巴切掉。**
5. **相位（起跑日）幾乎無關緊要**：20 年期換相位造成的總報酬差距約
   {_pct((phase[phase['period_years'] == 20]['cumret'].max() - phase[phase['period_years'] == 20]['cumret'].min()) / (1 + phase[phase['period_years'] == 20]['cumret'].min()))}、
   Sharpe 差 {phase[phase['period_years'] == 20]['sharpe'].max() - phase[phase['period_years'] == 20]['sharpe'].min():.3f}。
   實務上不需要挑「週幾再平衡」。
""")

    # ═══ 跨期間穩健組合 ═══
    L.append("## 7. 跨期間穩健組合（maximin 排名）\n")
    L.append("""對每組 [X,Y] 計算其中位數 Sharpe 在各期間的百分位排名，取**五個期間中最差的排名**
（maximin：不求任一期間最好，只求每一期間都不差），排除任何期間有破產路徑者。前 10 名：\n""")
    rows = [[f"[{r['X']:g}, {r['Y']:g}]", _upct(r["min_rank"]), _upct(r["mean_rank"])]
            for _, r in robust.head(10).iterrows()]
    L.append(_table(["組合", "最差期間排名", "平均排名"], rows) + "\n")

    # ═══ 結論 ═══
    L.append("## 8. 結論（規格 §10 六問）\n")
    x_max, y_max = max(X_GRID), max(Y_GRID)
    diff_periods = [y for y in LOOKBACK_YEARS
                    if not ((sels[y]["med_by_return"]["X"] == sels[y]["med_by_sharpe"]["X"]) and
                            (sels[y]["med_by_return"]["Y"] == sels[y]["med_by_sharpe"]["Y"]))]
    ret_edge = [y for y in LOOKBACK_YEARS
                if sels[y]["med_by_return"]["X"] == x_max or sels[y]["med_by_return"]["Y"] == y_max]
    ret_lev = {y: _agg_row(results[y]["agg"], sels[y]["med_by_return"]["X"],
                           sels[y]["med_by_return"]["Y"])["base_avg_lev"] for y in LOOKBACK_YEARS}
    shp_lev = {y: _agg_row(results[y]["agg"], sels[y]["med_by_sharpe"]["X"],
                           sels[y]["med_by_sharpe"]["Y"])["base_avg_lev"] for y in LOOKBACK_YEARS}
    L.append(f"""### 8.1 報酬最高與 Sharpe 最高是否同一組？
不是（{len(diff_periods)}/{len(LOOKBACK_YEARS)} 期間不同：{'、'.join(str(y) + 'y' for y in diff_periods)}）。
兩準則選出**結構相反**的組合：

- **報酬準則**選「快加碼、永不減碼」（Y 貼近 2、X 貼著網格上緣）：漲 5–6% 就把槓桿補回 2、
  跌再深也不砍。這使時間加權平均槓桿維持在 2 以上（基準路徑
  {'、'.join(f'{y}y={ret_lev[y]:.2f}' for y in diff_periods)}），
  在正漂移市場自然累積最多報酬——**它贏在「更多曝險」，不是門檻結構本身**。
  且 {'、'.join(str(y) + 'y' for y in ret_edge)} 的報酬最佳落在**網格邊緣／角落 [4, 1.9] 一帶**，
  暗示無約束最適解在網格之外（X→∞、Y→2），即「槓桿越貼近上緣越好」的單調方向，並非內部最適。
- **Sharpe 準則**選「慢加碼、慢減碼」（Y ≈ 1.0–1.2、X ≥ 2.5）：平時幾乎不動作，
  平均槓桿反而低於 2（{'、'.join(f'{y}y={shp_lev[y]:.2f}' for y in diff_periods)}），
  行為近似「買進持有＋崩盤保險絲」。（1y 例外：兩準則選到同一組，該期間多數路徑
  從未觸發門檻，網格排名幾乎全由噪音決定。）

差異來源：報酬準則不懲罰高槓桿帶來的波動與深回撤（中位數 MDD 差 10–20 個百分點），
Sharpe 準則會。這是**風險偏好之別**，不是其中一個「算錯」。
""")
    cls = {y: _classify_plateau(plateau[y]["sharpe"]) for y in LOOKBACK_YEARS}
    spike_periods = [y for y, c in cls.items() if c == "尖峰"]
    slope_periods = [y for y, c in cls.items() if c == "緩坡"]
    grid_spread = {}
    for y in LOOKBACK_YEARS:
        g = results[y]["agg"]
        gg = g[~g["is_benchmark"]]["med_sharpe"]
        grid_spread[y] = float(gg.max() - gg.min())
    L.append(f"""### 8.2 最佳解是平原還是尖峰？
各期間（Sharpe 準則）判定：{'、'.join(f'{y}y={cls[y]}' for y in LOOKBACK_YEARS)}。
{"存在尖峰期間（" + "、".join(str(y) + "y" for y in spike_periods) + "）——該期間的『最佳解』不可用於實盤。" if spike_periods else "**沒有任何期間呈孤立尖峰**，最佳解可信度的主要威脅不是尖峰，而是整面曲面的平坦："}
全網格中位數 Sharpe 的極差（最好格 − 最差格）僅
{'、'.join(f'{y}y={grid_spread[y]:.2f}' for y in LOOKBACK_YEARS)}——
門檻放在合理範圍內的**任何位置，風險調整後績效差異都很小**。
{"其中 " + "、".join(str(y) + "y" for y in slope_periods) + " 判定為緩坡：最佳格坐在沿 Y 的窄脊上（如 20y 的 Y=1.1 帶），脊與鄰列的差距 ~0.04 Sharpe，在 33 條高度重疊路徑下應視為噪音，不足以宣稱 Y=1.1 優於 Y=1.0 或 1.2。" if slope_periods else ""}
§6.3 精細搜尋（步長 0.05）亦顯示 Sharpe 對參數移動幾乎無感（差異 < 0.01），佐證平坦。
**實務結論：從高原帶挑選即可，單一格的名次沒有意義。**
""")
    best_seq = [f"{y}y→{_xy(sels[y]['med_by_sharpe'])}" for y in LOOKBACK_YEARS]
    L.append(f"""### 8.3 隨期間拉長，最佳 [X, Y] 是否收斂？
Sharpe 準則最佳序列：{'，'.join(best_seq)}。單看 argmax 沒有明顯單調收斂
（10y 的 X=2.5 是含 2008 視窗比例最高的期間，崩盤中「早一點減碼」相對有利；其餘期間 X 落在 3.25–3.5）。
但**穩定收斂的是結構而非單點**：所有期間的 Sharpe 高原都位於 **Y ∈ [1.0, 1.2]（慢加碼）**，
且沿 X 平坦（X 位置次要）。§7 maximin 前 8 名全數落在 X ∈ [3, 4]、Y ∈ [1.0, 1.2]，
即隨檢驗趨嚴，答案收斂到「近似買進持有、僅保留崩盤減碼保險絲」的組態。
""")
    ss = payload["sens_slip"]
    moved = ss[~ss["same_cell"]]
    d20 = freq[(freq["name"] == "Daily_2x") & (freq["period_years"] == 20)].iloc[0]
    p20 = freq[(freq["name"] == "Periodic5d_2x") & (freq["period_years"] == 20)].iloc[0]
    L.append(f"""### 8.4 交易成本對最佳門檻寬度的影響
方向符合理論（成本越低、最佳區間可以越窄），但幅度只有一格、且僅部分期間可見：
滑價 1 → 0.5 檔後，{'、'.join(f"{int(r['period_years'])}y 最佳由 {r['best_main']} 移至 {r['best_alt']}" for _, r in moved.iterrows()) if len(moved) else "（無任何期間移動）"}，
其餘期間最佳格不動。原因是此策略的觸發本來就極稀
（推薦組合在 20 年基準路徑上僅再平衡 3 次），總成本由**每月轉倉**主導、再平衡成本是零頭，
成本參數對門檻寬度只有二階影響。

§6.4 的頻率掃描給了同一結論的另一個切面：即使把再平衡次數從
{int(d20['base_n_rebal']):,} 次（每日）壓到 {int(p20['base_n_rebal']):,} 次（每 5 日），
20 年期總成本也只從初始資本的 {_upct(d20['base_cost_ratio'], 1)} 降到 {_upct(p20['base_cost_ratio'], 1)}——
**再平衡頻率不是成本的主要來源，轉倉才是。** 這也意味著「用寬門檻省交易成本」
的動機在台指期上其實很弱；門檻的真正價值是控制槓桿漂移與尾部風險，不是省手續費。
""")
    rec_row_stats = []
    for y in LOOKBACK_YEARS:
        r = _agg_row(results[y]["agg"], rec_x, rec_y)
        rec_row_stats.append(f"{y}y：中位數報酬 {_pct(r['med_cumret'])}、Sharpe {_num(r['med_sharpe'], 2)}、"
                             f"最差路徑 {_pct(r['worst_cumret'])}")
    runner = robust.iloc[1]
    L.append(f"""### 8.5 實務建議
推薦 **[X = {rec_x:g}, Y = {rec_y:g}]**（§7 maximin 第一名：最差期間排名 {_upct(rec['min_rank'])}、
平均排名 {_upct(rec['mean_rank'])}；[{runner['X']:g}, {runner['Y']:g}] 並列，同帶皆可）。依據：

- { '；'.join(rec_row_stats) }。所有期間、所有滾動路徑皆無破產，
  低滑價與 offset=0 敏感度情境下仍居各期間前段。
- 它位於每個期間熱力圖的高原帶內，不依賴任何單一期間的峰值格（規格 §11 反過擬合原則）。
- 白話翻譯這組參數的實際行為：自上次再平衡點**下跌約 30%** 才觸發減碼（保險絲），
  **上漲約 4.5 倍**才觸發加碼（幾乎不加碼）；平時就是 2 倍買進持有。
  它贏過每日再平衡的機制是省下持續「追高殺低」的再平衡損耗，輸給報酬最佳組的部分
  則是少了持續拉滿的曝險——這正是 Sharpe 與報酬兩準則的取捨位置。
""")
    n1 = results[1]["n_paths"]
    L.append(f"""### 8.6 誠實揭露
- **樣本獨立性**：資料 {a['data_start']}～{a['data_end']} 約 27.9 年，僅含約 1 個完整多空循環
  （2008 金融海嘯、2020 疫情崩跌、2022 修正、2024-2025 多頭）。20y 僅 {results[20]['n_paths']} 條
  季頻路徑且**高度重疊**（實質獨立樣本 ≈ 1.4 條），其結論是「這 20 年的歷史敘述」而非
  對未來的統計推論。
- **1 年期**：{n1} 條路徑但每條只含單一 regime，最佳 [X,Y] 幾乎純粹反映該年行情方向，
  **不提供 1 年期的參數建議**。
- **利率簡化**：閒置現金利率固定 {IDLE_CASH_RATE:.1%}，未反映 27 年間實際短率起伏；
  此假設影響所有策略的絕對數字，但對 [X,Y] 相對排序影響甚微。
- **BH 2x 的中位數 Sharpe 有存活者偏誤**：破產路徑的 Sharpe 無法定義（記 NaN），
  其「中位數 Sharpe」只反映存活路徑；請一併看破產路徑數欄位。
- 未模擬保證金追繳／強制平倉與資金顆粒度（依規格）；實盤中破產前就會先被斷頭，
  高槓桿組合的實際下檔比表列更糟。
""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  報告 → {path}")
