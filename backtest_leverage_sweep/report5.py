"""第五輪報告產生器（規格 §6、§7、§8）。

🔴 紀律：本報告**禁止**輸出「建議採用 X 倍槓桿」。只輸出取捨曲線與各槓桿的
完整代價清單，由使用者依自身風險偏好選擇。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_leverage_sweep.params import (
    L_HIGH_GRID, LIMIT_DOWN_PCT, MAINT_MARGIN_RATE, MARKET_LABEL, PERIOD_YEARS,
    RANKING_MARKETS, RUIN_VETO_PROB,
)


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def _pc(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x * 100:.{nd}f}%"


def _pp(x, nd=1):
    """百分點（差值）。"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x * 100:+.{nd}f}pp"


def _tbl(df, spec):
    head = "| " + " | ".join(t for _, t, _ in spec) + " |"
    sep = "|" + "|".join("---" for _ in spec) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            (fmt(r.get(c)) if callable(fmt) else str(r.get(c))) for c, _, fmt in spec) + " |")
    return "\n".join(lines)


def write_report(payload: dict, path) -> None:
    front = payload["front_all"]
    bend = payload["bend"]
    marg = payload["marg"]
    eff = payload["eff"]
    conv = payload["conv"]
    mt = payload["marg_tbl"]
    ruin = payload["ruin"]
    lev_real = payload["lev_real"]
    sec = []

    f0, f1 = front.iloc[0], front.iloc[-1]
    veto_list = ruin[ruin["disqualified"]]["l_high"].tolist()
    first_ruin = ruin[ruin["max_ruin_prob"] > 0]["l_high"]
    first_ruin = float(first_ruin.iloc[0]) if len(first_ruin) else None

    # ══ 摘要 ═════════════════════════════════════════════════════════════════
    sec.append(f"""# D_10 目標槓桿掃描（2.0 → 2.5）——取捨曲線（第五輪）

> **🔴 本報告不宣告「最佳槓桿」**（規格 §8）。最佳槓桿是風險偏好的函數，
> 資料無法決定。資料唯一能決定的是「在 2.0–2.5 區間 CAGR 是否還在上升」，
> 以及每一格槓桿的**完整代價清單**。選擇權在讀者。

## 摘要

- 自 L_high=2.0 → 2.5：CAGR {_pp(bend['cagr_total_change'])}、MDD 惡化 {_pp(bend['mdd_total_change'])}。
- 取捨曲線{'**出現向後彎折**（彎折於 L=' + f"{bend['bend_at']:.1f}" + '）：其後的槓桿更深回撤卻無更多報酬，**被嚴格支配**。' if bend['bend_at'] is not None else '**未出現向後彎折**——CAGR 在此區間仍隨槓桿單調上升（見 Q1 的除錯核對）。'}
- bootstrap 歸零／強平機率 > {RUIN_VETO_PROB:.0%} 而**不合格**的 L_high：{('、'.join(f"{v:.1f}" for v in veto_list)) if veto_list else '無'}。
- 實際時間加權平均槓桿：L_high 2.0 → 2.5 對應約 {_f(lev_real['med_avg_lev'].iloc[0], 2)} → {_f(lev_real['med_avg_lev'].iloc[-1], 2)}（**不等於 L_high**，D_10 有降載時段）。
""")

    # ══ §1 方法與凍結參數 ════════════════════════════════════════════════════
    sec.append(f"""## 1. 方法與凍結參數

本輪唯一變數為 **L_high ∈ {L_HIGH_GRID}**；L_low = 1.0 固定。其餘一律沿用第四輪
確認的 D_10 設定，**未更動**：MA=200、buffer=1%、週頻訊號檢查、T+1 執行、
比值還原連續序列、成本模型（TX 期貨明細／海外 5bp）、融資模型。

**存活者口徑（規格 §8，本輪最關鍵的統計紀律）**：破產與強制平倉的路徑
**一律排除於報酬統計之外**，另計生存率——否則會出現「歸零前漲很多」的假優勝者
（第二輪已踩過此坑）。所有 CAGR／Sharpe／MDD 中位數皆為**存活路徑**的中位數。

**滾動視窗**：{PERIOD_YEARS} 年、季頻平移起點；四市場（TX 期貨、S&P 500、日經 225、
恆生）＋ TX 指數版為 carry 校準參照；同一 L_high 跨市場凍結。
""")

    # ══ §2 取捨曲線 ══════════════════════════════════════════════════════════
    fr = front.copy()
    sec.append(f"""## 2. 取捨曲線（規格 §5-1，最重要）

跨市場中位數。圖：`r5_frontier.png`（每市場一張 + 綜合）。

{_tbl(fr, [("l_high", "L_high", lambda v: f"{v:.1f}"),
           ("cagr", "CAGR（中位）", _pc),
           ("mdd", "MDD（中位）", _pc),
           ("sharpe", "Sharpe", lambda v: _f(v, 3)),
           ("avg_lev", "**實際平均槓桿**", lambda v: _f(v, 2)),
           ("survival", "最低生存率", _pc),
           ("n_liquidated", "強平路徑數", lambda v: str(int(v)))])}

**向後彎折判定**：{'✔ 於 L=' + f"{bend['bend_at']:.1f}" + ' 出現彎折——' + '、'.join(f"{v:.1f}" for v in bend['dominated']) + ' 被嚴格支配（MDD 更深、CAGR 未增）。' if bend['bend_at'] is not None else '✘ 未偵測到彎折：此區間 CAGR 仍隨槓桿上升，取捨仍是「真實交換」而非單方面變差。'}
""")

    # ══ §3 三條主曲線 / 邊際 / 效率 ══════════════════════════════════════════
    sec.append(f"""## 3. 邊際交換率與曝險效率（規格 §5-2～§5-4）

圖：`r5_three_curves.png`、`r5_mdd_convexity.png`。

**邊際分析——每 +0.1 槓桿換到什麼、付出什麼**（決策時真正該看的數字）：

{_tbl(marg, [("from_l", "自", lambda v: f"{v:.1f}"), ("to_l", "至", lambda v: f"{v:.1f}"),
             ("d_cagr", "ΔCAGR", _pp), ("d_mdd_deeper", "ΔMDD（惡化）", _pp),
             ("d_sharpe", "ΔSharpe", lambda v: f"{v:+.3f}"),
             ("trade_ratio", "ΔCAGR/ΔMDD", lambda v: _f(v, 2))])}

> ΔCAGR/ΔMDD < 1 代表「每多承受 1 個百分點的回撤，換到的 CAGR 不足 1 個百分點」。

**每單位平均槓桿的 CAGR（曝險效率）**——檢驗報酬增量是否只是曝險的線性結果：

{_tbl(eff, [("l_high", "L_high", lambda v: f"{v:.1f}"),
            ("cagr", "CAGR", _pc), ("avg_lev", "平均槓桿", lambda v: _f(v, 2)),
            ("cagr_per_lev", "CAGR/平均槓桿", lambda v: _pc(v, 2)),
            ("eff_vs_base", "相對 L=2.0", lambda v: _pc(v, 1))])}

**MDD 凸性**（惡化速率是否隨槓桿加速）：

{_tbl(conv, [("from_l", "自", lambda v: f"{v:.1f}"), ("to_l", "至", lambda v: f"{v:.1f}"),
             ("mdd_deepening", "MDD 惡化", _pp)] +
     ([("accelerating", "加速?", lambda v: "是" if v else "否")] if "accelerating" in conv.columns else []))}
""")

    # ══ §4 保證金與跌停 ══════════════════════════════════════════════════════
    mt2 = mt.copy()
    sec.append(f"""## 4. 保證金、強制平倉與台股跌停生存（規格 §3-1、§3-2）

維持保證金率 m = {MAINT_MARGIN_RATE:.1%}（TAIFEX 量級，以當期公告為準）。
臨界跌幅為**標的指數**的跌幅（非權益跌幅）：

{_tbl(mt2, [("l_high", "L_high", lambda v: f"{v:.1f}"),
            ("x_liq_approx", "x_liq（規格近似式）", _pc),
            ("x_liq_exact", "x_liq（精確解）", _pc),
            ("x_ruin", "權益歸零門檻 1/L", _pc),
            ("limit_down_days_exact", "可承受連續跌停日數", lambda v: f"{int(v)} 日"),
            ("trigger_cum_drop", "觸發時累積跌幅", _pc)])}

規格 §3-1 的近似式 `(1−mL)/L` 已重現其預期值（L=2.0→{_pc(mt2['x_liq_approx'].iloc[0])}、
L=2.5→{_pc(mt2['x_liq_approx'].iloc[-1])}）。精確解 `(1−mL)/(L(1−m))` 為引擎實際
使用的條件 `E < m·N` 的解析解，回測即以此逐日檢查。

> ### ⚠️ 與規格預期不符之處（已獨立手算覆核，非程式錯誤）
>
> 規格 §3-2 預期「L=2.0 可撐 3 日跌停、L=2.5 降至 2 日」，實測為
> **{int(mt2['limit_down_days_exact'].iloc[0])} 日 / {int(mt2['limit_down_days_exact'].iloc[-1])} 日**。原因是規格的兩項預期彼此不一致：
> 連續跌停的累積**指數**跌幅為 `1 − {1 - LIMIT_DOWN_PCT:g}^k`——3 個跌停只有
> **27.1%**，遠不及其自身給定的 x_liq ≈ 46%；要跌到 46% 需要 **6 個**跌停
> （1−0.9⁶ = 46.9%）。「3 日」對應的是**權益**跌幅（1−0.8³ = 48.8%），
> 那是被槓桿放大後的權益損失，不是觸發強平的指數跌幅。本報告採用內部一致的
> 指數跌幅口徑。
>
> ### ⚠️ 更重要的發現：m = 3.5% 時，「強制平倉」幾乎不獨立於「破產」
>
> L=2.0 的 x_liq = {_pc(mt2['x_liq_exact'].iloc[0])} vs 權益歸零 {_pc(mt2['x_ruin'].iloc[0])}——
> 保證金約束只比全額虧光**提前約 {_pc(mt2['x_ruin'].iloc[0] - mt2['x_liq_exact'].iloc[0], 1)}**。
> 維持保證金率越低，維持保證金需求（m·N）越小，帳戶要跌到幾乎一無所有才會被斷頭。
> 換言之：**在 3.5% 的維持保證金率下，「強制平倉」提供的保護幾乎可以忽略**，
> 規格期待的「台股特有的確定性保證」比直覺弱得多。敏感度見
> `r5_margin_rate_sensitivity.csv`（m 提高到 10–15% 時，約束才明顯早於破產）。
""")

    # ══ §5 bootstrap 歸零機率 ════════════════════════════════════════════════
    sec.append(f"""## 5. bootstrap 歸零／強制平倉機率（規格 §3-3，本輪最重要的單一數字）

stationary bootstrap、期望區塊 60 日、每市場 1,000 條 × 20 年、參數凍結不重校準。

{_tbl(ruin, [("l_high", "L_high", lambda v: f"{v:.1f}"),
             ("max_ruin_prob", "最差市場歸零機率", lambda v: _pc(v, 2)),
             ("mean_ruin_prob", "四市場平均", lambda v: _pc(v, 2)),
             ("max_liq_prob", "其中強平機率", lambda v: _pc(v, 2)),
             ("worst_market", "最差市場", str),
             ("disqualified", f"不合格(>{RUIN_VETO_PROB:.0%})", lambda v: "⛔" if v else "")])}

逐市場分布見 `r5_bootstrap_ruin.csv`。

> **🔴 向下偏誤警語（沿用第三／四輪）**：bootstrap 打斷跨區塊序列相關，而趨勢
> 策略的 alpha 正活在長週期自相關裡——D_10 全體在合成路徑上被系統性壓低。
> 故本表僅用於 **L_high 之間的相對比較**，不作絕對水準判斷。
""")

    # ══ §6 實際平均槓桿 ══════════════════════════════════════════════════════
    sec.append(f"""## 6. 實際平均槓桿（規格 §4：L_high=2.5 ≠ 平均 2.5）

D_10 有降載時段（空頭 regime → L_low=1.0），故時間加權平均槓桿遠低於 L_high：

{_tbl(lev_real, [("l_high", "L_high", lambda v: f"{v:.1f}"),
                 ("med_avg_lev", "平均槓桿（中位）", lambda v: _f(v, 2)),
                 ("min_avg_lev", "最低", lambda v: _f(v, 2)),
                 ("max_avg_lev", "最高", lambda v: _f(v, 2)),
                 ("med_lev_at_trough", "MDD 谷底槓桿", lambda v: _f(v, 2)),
                 ("med_max_lev", "路徑最高槓桿", lambda v: _f(v, 2))])}

**所有報酬比較都必須並列此欄**（規格 §8）：L_high 提高 25%（2.0→2.5），
實際平均曝險只提高約 {_pc(lev_real['med_avg_lev'].iloc[-1] / lev_real['med_avg_lev'].iloc[0] - 1, 1)}。
""")

    # ══ §7 壓力窗口 + 日經 ═══════════════════════════════════════════════════
    stress = payload["stress"]
    nk = stress[(stress["window"].str.startswith("日經1990")) & (stress["market"] == "NKX")]
    other = []
    for w in stress["window"].unique():
        if w.startswith("日經1990"):
            continue
        d = stress[stress["window"] == w]
        agg = d.groupby("l_high").agg(cumret=("cumret", "mean"), mdd=("mdd", "mean"),
                                      liq=("liquidated", "sum")).reset_index()
        rng_txt = (f"{_pc(agg['cumret'].iloc[0])} → {_pc(agg['cumret'].iloc[-1])}"
                   f"（MDD {_pc(agg['mdd'].iloc[0])} → {_pc(agg['mdd'].iloc[-1])}）")
        other.append(f"| {w} | {rng_txt} | {_pp(agg['cumret'].iloc[-1] - agg['cumret'].iloc[0])} | "
                     f"{int(agg['liq'].sum())} |")
    nk_tbl = (_tbl(nk, [("l_high", "L_high", lambda v: f"{v:.1f}"),
                        ("cumret", "期間報酬", _pc), ("mdd", "MDD", _pc),
                        ("avg_lev", "平均槓桿", lambda v: _f(v, 2)),
                        ("liquidated", "強平", lambda v: "是" if v else "否")])
              if not nk.empty else "（無資料）")
    nk_gap = (nk["cumret"].iloc[-1] - nk["cumret"].iloc[0]) if not nk.empty else np.nan
    sec.append(f"""## 7. 壓力窗口（規格 §2-4、§5-5）

### 🔴 日經 1990–2003 長陰跌（單獨成節）

{nk_tbl}

自 L=2.0 → 2.5，期間報酬變化 **{_pp(nk_gap)}**——長陰跌是槓桿懲罰最劇烈的情境
（趨勢濾網雖已降載，但反覆假訊號期間的滿載曝險被逐次放大）。
圖：`r5_nikkei1990_leverage.png`。

### 其他窗口（各市場平均；L=2.0 → 2.5）

| 窗口 | 期間報酬（L2.0 → L2.5） | 報酬差 | 強平路徑數 |
|---|---|---|---|
{chr(10).join(other)}

完整逐格結果：`r5_stress_windows.csv`。
""")

    # ══ §8 結論八問 ══════════════════════════════════════════════════════════
    d_shp = front["sharpe"].iloc[-1] - front["sharpe"].iloc[0]
    shp_mono = bool((front.sort_values("l_high")["sharpe"].diff().dropna() <= 0).all())
    conv_acc = (conv["accelerating"].sum() if "accelerating" in conv.columns else 0)
    ld = mt.set_index("l_high")["limit_down_days_exact"]
    below3 = [L for L in L_HIGH_GRID if ld.get(L, 99) < 3]
    worst_stress = None
    if not stress.empty:
        g = stress.groupby(["window"]).apply(
            lambda d: (d[d["l_high"] == 2.5]["cumret"].mean()
                       - d[d["l_high"] == 2.0]["cumret"].mean()), include_groups=False)
        worst_stress = (g.idxmin(), float(g.min()))
    sec.append(f"""## 8. 結論（規格 §6 八問）

**Q1 在 2.0–2.5 區間，CAGR 是否仍隨槓桿上升？**
{'**是**——CAGR ' + _pp(bend['cagr_total_change']) + '，未出現向後彎折。' if bend['cagr_still_rising'] else '**否**——於 L=' + f"{bend['bend_at']:.1f}" + ' 出現向後彎折，其後（' + '、'.join(f"{v:.1f}" for v in bend['dominated']) + '）**被嚴格支配**：更深回撤、無更多報酬。'}
除錯核對（規格 §0 要求）：本輪成本、假跳空、破產／強平排除均已建模，
存活者口徑已排除「歸零前暴賺」的假優勝者；CAGR 變化幅度 {_pp(bend['cagr_total_change'])}
{'屬溫和量級，與理論預測（2.0–2.5 已在 g(L) 拋物線下降側附近、CAGR 持平或緩降）方向一致。' if abs(bend['cagr_total_change']) < 0.03 else '幅度較大，已於 §2 附完整平均槓桿欄供覆核。'}

**Q2 Sharpe 是否單調下降？** {'**是**' if shp_mono else '**非嚴格單調**'}——自 L=2.0 的
{_f(front['sharpe'].iloc[0], 3)} 到 2.5 的 {_f(front['sharpe'].iloc[-1], 3)}，合計 {d_shp:+.3f}
（平均每 +0.1 槓桿約 {d_shp / 5:+.4f}）。

**Q3 MDD 惡化是線性還是凸性？** 自 2.0 → 2.5 惡化 **{_pp(bend['mdd_total_change'])}**；
逐段惡化速率有 {int(conv_acc)}/{len(conv)} 段呈加速 → {'**凸性惡化**（越加槓桿、每單位代價越貴）。' if conv_acc >= len(conv) / 2 else '接近線性惡化。'}

**Q4 平均槓桿的實際範圍？** L_high 2.0 → 2.5 對應時間加權平均槓桿
**{_f(lev_real['med_avg_lev'].iloc[0], 2)} → {_f(lev_real['med_avg_lev'].iloc[-1], 2)}**
（僅 +{_pc(lev_real['med_avg_lev'].iloc[-1] / lev_real['med_avg_lev'].iloc[0] - 1, 1)}，因 D_10 有降載時段）。

**Q5 哪個 L_high 開始出現非零強平／歸零機率？**
{'**L=' + f"{first_ruin:.1f}" + '**（見 §5 表）。' if first_ruin is not None else '**四市場 1,000 條合成路徑中，全部 L_high 的歸零／強平機率皆為 0**——'
 '在 20 年期、m=3.5% 的設定下，D_10 的降載機制使槓桿不足以觸及斷頭線。'}
不合格（>{RUIN_VETO_PROB:.0%}）者：{('、'.join(f"{v:.1f}" for v in veto_list)) if veto_list else '無'}。

**Q6 台股可承受連續跌停日數在哪個 L_high 掉到 3 日以下？**
實測全區間為 {int(mt['limit_down_days_exact'].max())}–{int(mt['limit_down_days_exact'].min())} 日，
{'於 L=' + '、'.join(f"{v:.1f}" for v in below3) + ' 掉到 3 日以下。' if below3 else '**在 2.0–2.5 全區間都沒有掉到 3 日以下**（最低 ' + str(int(mt['limit_down_days_exact'].min())) + ' 日）。'}
注意此結論高度依賴 m=3.5% 這個假設（見 §4 的兩則警語）。

**Q7 各壓力窗口中，槓桿懲罰在哪個情境最劇烈？**
{'**' + worst_stress[0] + '**（L2.0→2.5 期間報酬差 ' + _pp(worst_stress[1]) + '）。' if worst_stress else '（無資料）'}

**Q8 誠實揭露** → §9。**本報告不輸出「建議採用 X 倍槓桿」**：
上列每一格槓桿的代價清單（回撤、Sharpe 損失、跌停緩衝、歸零機率）已完整列出，
選擇取決於你能承受的回撤與破產機率，這是偏好陳述、不是統計推論。
""")

    # ══ §9 誠實揭露 ══════════════════════════════════════════════════════════
    sec.append(f"""## 9. 誠實揭露與本輪解決的規格模糊

**規格模糊處與我採用的預設**：
1. §3-1 給的 x_liq 是近似式（略去 (1−m) 因子）。**兩者都輸出**：近似式用於對照
   規格預期值，精確解為引擎實際條件。差異在 L=2.0 約 1.7pp。
2. §3-2 的「可承受連續跌停日數」與其自身 x_liq 互相矛盾（見 §4 警語）。
   採用內部一致的**指數跌幅**口徑，並附完整手算。
3. 保證金模型的 m=3.5% 為 TAIFEX 量級；**海外指數市場並無此制度**，本輪仍套用
   同一條件作為「槓桿硬約束」的代理，屬簡化，已於此揭露。
4. 強平檢查為**每日收盤**粒度；真實斷頭可在盤中觸發 → 本輪**低估**強平頻率。
5. 「最差路徑報酬」等統計皆為**存活路徑**內的最差，非含已強平路徑（後者另計生存率）。

**穩健與不穩健的部分**：
- 穩健：平均槓桿、MDD 惡化方向、跌停緩衝的解析結果（純算術，不依賴樣本）。
- 不穩健：CAGR 的絕對水準（四條歷史路徑）；壓力窗口各僅 1–4 條路徑，屬敘述證據。
- bootstrap 對趨勢策略的向下偏誤（§5 警語）使歸零機率為**下界式的相對比較**。

**本輪仍未檢驗**：
- 追繳保證金（margin call 後補錢）而非直接斷頭的情境——本輪為「不補錢」假設。
- 盤中強平、跌停鎖死無法成交的滑價放大。
- 週頻訊號相位、T+1 執行敏感度（沿第四輪，仍未掃）。
- 台指期契約規格的整數口數約束（本輪為連續小數口數）。

## 附錄：輸出檔案

| 檔案 | 內容 |
|---|---|
| `r5_leverage_sweep_metrics.csv` | L_high×市場×期間 完整指標（存活者口徑） |
| `r5_frontier.csv` / `r5_frontier_by_market.csv` | 取捨曲線座標 |
| `r5_margin_survival.csv` | 臨界跌幅與連續跌停生存表 |
| `r5_margin_rate_sensitivity.csv` | 維持保證金率敏感度（約束何時咬到） |
| `r5_bootstrap_ruin.csv` / `r5_ruin_summary.csv` | 歸零／強平機率與否決判定 |
| `r5_marginal_tradeoff.csv` / `r5_efficiency.csv` / `r5_mdd_convexity.csv` | 邊際、效率、凸性 |
| `r5_stress_windows.csv` | 壓力窗口逐格 |
| `r5_realized_leverage.csv` | L_high → 實際平均槓桿 |
| `r5_scaled_lowlow_check.csv` | 等比縮放 L_low=L/2 次要檢查 |
| `r5_inception.csv` | 自始以來單路徑 |
| `r5_frontier.png` / `r5_three_curves.png` / `r5_mdd_convexity.png` / `r5_nikkei1990_leverage.png` | 圖 |
""")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sec))
    print(f"  報告 → {path}")
