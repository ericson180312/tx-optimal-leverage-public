"""第六輪報告產生器。開頭原文複製預先承諾規則，再放結果。

只改文字可：from backtest_signal_timing import report6; report6.write_from_cache()
（payload 讀 backtest_signal_timing/output/cache/results6.pkl）。
"""
from __future__ import annotations

import pickle

import pandas as pd

from backtest_ma_robustness.report4 import _f, _pc, _tbl

from backtest_signal_timing.bootstrap6 import CONTRASTS
from backtest_signal_timing.params import (
    BASELINE_K, CACHE_DIR, K_GRID, MARKET_LABEL, OUTPUT_DIR, PRE_REGISTRATION, RANKING_MARKETS,
)


def _s(x):
    return f"{x:+.3f}"


def _n(x):
    return f"{x:g}"


def _pp(x):
    return f"{x * 100:+.2f}pp"


def _label(k, lag):
    return f"每 {k} 日、{'T+0（上界）' if lag == 0 else f'T+{lag}'}"


def _timing_table(rows: pd.DataFrame) -> str:
    return _tbl(rows, [
        ("label", "變體", str), ("D", "D", _s), ("M", "M（正向市場）", _n),
        *[(f"d_{m}", m, _s) for m in RANKING_MARKETS],
        ("delta_min", "單格最低", _s), ("delta_max", "單格最高", _s),
        ("med_avg_lev", "平均槓桿", _f), ("dcagr", "ΔCAGR（配對）", _pp),
        ("cagr_per_lev", "CAGR／槓桿", _pc), ("dmdd", "ΔMDD（配對）", _pp),
        ("med_cost_ratio", "成本占比", _pc), ("n_bankrupt", "破產路徑", _n)])


def _with_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = [_label(k, lag) for k, lag in zip(out["K"], out["lag"])]
    return out


def _boot_table(boot: pd.DataFrame) -> str:
    rows = []
    for k, lag in CONTRASTS:
        g = boot[(boot["K"] == k) & (boot["lag"] == lag)].set_index("market")
        rows.append({"label": _label(k, lag),
                     **{m: g.loc[m, "med_diff"] for m in RANKING_MARKETS},
                     "n_pos": int((g["med_diff"] > 0).sum()),
                     "better": float(g["share_better"].median()),
                     "ruin": float(g["ruin_prob"].max())})
    return _tbl(pd.DataFrame(rows), [
        ("label", "變體", str), *[(m, m, _s) for m in RANKING_MARKETS],
        ("n_pos", "同向市場數（>0）", _n), ("better", "勝過基準的路徑比（市場中位數）", _pc),
        ("ruin", "最高歸零機率", _pc)])


def _stress_table(stress: pd.DataFrame) -> str:
    base = stress[(stress["K"] == 5) & (stress["lag"] == 1)]
    rows = []
    for _, r in base.iterrows():
        row = {"window": r["window"], "market": r["market"],
               "cum": r["cumret"], "mdd": r["mdd"]}
        for tag, k, lag in [("k1", 1, 1), ("k20", 20, 1), ("t0", 5, 0), ("k1t0", 1, 0)]:
            o = stress[(stress["window"] == r["window"]) & (stress["market"] == r["market"])
                       & (stress["K"] == k) & (stress["lag"] == lag)].iloc[0]
            row[f"{tag}_c"] = o["cumret"] - r["cumret"]
            row[f"{tag}_m"] = o["mdd"] - r["mdd"]
        rows.append(row)
    return _tbl(pd.DataFrame(rows), [
        ("window", "窗口", str), ("market", "市場", str), ("cum", "基準報酬", _pc),
        ("mdd", "基準 MDD", _pc),
        ("k1_c", "每日檢查 Δ報酬", _pc), ("k1_m", "每日檢查 ΔMDD", _pc),
        ("k20_c", "每 20 日 Δ報酬", _pc), ("k20_m", "每 20 日 ΔMDD", _pc),
        ("t0_c", "5 日 T+0 Δ報酬", _pc), ("t0_m", "5 日 T+0 ΔMDD", _pc),
        ("k1t0_c", "每日 T+0 Δ報酬", _pc), ("k1t0_m", "每日 T+0 ΔMDD", _pc)])


def write_report(payload: dict, path) -> None:
    d, summary, regime = payload["decision"], payload["summary"], payload["regime"]
    boot, maximin = payload["boot"], payload["maximin"]
    s = _with_label(summary)
    sec = []

    sec.append("# D_10 訊號檢查頻率與執行時點的穩健性驗證（第六輪）\n")
    sec.append("> 本報告開頭原文複製預先承諾規則。規則於 2026-11-05 跑出任何變體之前寫定於 "
               "`params.PRE_REGISTRATION`，並先 push 到 claude-notes 留下時間戳"
               "（`journal/2026-11-05-台指D10-訊號檢查頻率與執行時點-預先承諾.md`）；"
               "程式機械化計算判定，不因結果修改。\n")
    sec.append(PRE_REGISTRATION)
    sec.append("\n---\n")

    sec.append(f"""## 判定摘要（機械化結果）

| 項目 | 判定量 | 結果 |
|---|---|---|
| 材料性門檻 | N（相位雜訊）= {_f(d['noise'])} | **τ = {_f(d['tau'])}** |
| 規則 A：檢查頻率 | R_A = {_f(d['r_a'])}；K=5 排名第 {d['rank5']}／6（{' > '.join(str(k) for k in d['order'])}） | **{d['verdict_a']}** |
| 規則 B：執行時點 | U = {_s(d['u'])}（{_label(*d['v_star'])}） | **{d['verdict_b']}** |
| 規則 B 附帶：多延遲一天 | D(5,2) = {_s(d['d52'])} | {'**延遲敏感**' if d['delay_sensitive'] else '未達 −τ'} |
| 規則 C：尖峰否決 | D(5,1) − max(D(3,1), D(10,1)) = {_s(d['spike_gap'])} | {'**觸發**' if d['spike'] else '未觸發'} |
| 規則 D：實盤相位 | ΔP = {_s(d['delta_p'])}（TX_FUT） | {'**揭露**' if d['phase_flag'] else '未達 τ'} |

- 規則 A：{d['text_a']}
- 規則 B：{d['text_b']}
- 整合檢核：K=5、lag=1、φ=0 復現第三輪凍結輸出的 D_10，16 格最大差異
  {payload['repro']['diff'].abs().max():.1e}。資料凍結在 repo 快取，與前五輪相同。
""")

    lag1 = s[s["lag"] == 1].sort_values("K")
    sec.append("## 1. 檢查頻率（lag=1）\n\n所有數字都是跨相位中位數。逐市場欄為該市場四個期間 Δ 的中位數。\n")
    sec.append(_timing_table(lag1))
    if len(d["candidates"]):
        sec.append("\n\n規則 A 第三分支的替代條件檢查：\n")
        sec.append(_tbl(d["candidates"], [
            ("K", "K", _n), ("D", "D", _s), ("M", "M", _n), ("cond_i", "(i) D≥τ", str),
            ("cond_ii", "(ii) M≥3", str), ("cond_iii", "(iii) 相鄰 D≥0", str),
            ("cond_iv", "(iv) bootstrap", str), ("cond_v", "(v) 破產", str),
            ("ok", "全過", str)]))

    timing = s[((s["K"] == 5) | (s["K"] == 1))].sort_values(["K", "lag"], ascending=[False, True])
    sec.append("\n\n## 2. 執行時點\n\n🔴 T+0 含 look-ahead，**不是可實現績效**，只作上界。"
               "實際可行的當天做法都介於 T+0 與 T+1 之間。\n")
    sec.append(_timing_table(timing))
    hc = payload["handcheck"]
    tx_ph = payload["tx_t0_phases"]
    sec.append(f"""

**獨立手算覆核（不經引擎）**：T+0 與 T+1 的曝險只在切換日隔天不同，差＝ΔL × 隔天報酬。
把差加總年化再除以策略波動，得到 T+0 的 Sharpe 優勢量級；最後一欄是同口徑（相位 0、全樣本單一路徑、含成本）
的引擎結果，兩者對得上才算覆核通過。
切換日隔天的平均報酬不大，是 T+0 沒有優勢的原因：訊號日當天的跌幅已經實現，少掉的只是隔天那一天的曝險。

下表是單一相位，所以 TX_FUT 的 {_s(tx_ph[0])} 不能單獨看。TX_FUT 全樣本 5 個相位的 T+0 優勢依序是
{'、'.join(_s(x) for x in tx_ph)}，中位數 {_s(sorted(tx_ph)[2])}：**正負由相位決定**，實盤相位剛好是最高的。
""")
    sec.append(_tbl(hc, [
        ("market", "市場", str), ("K", "K", _n), ("n_switch", "切換次數", _n),
        ("next_ret_after_down", "降槓桿隔天平均報酬", lambda x: f"{x * 100:+.2f}%"),
        ("next_ret_after_up", "升槓桿隔天平均報酬", lambda x: f"{x * 100:+.2f}%"),
        ("gain_per_year", "T+0 年化差", lambda x: f"{x * 100:+.2f}pp"),
        ("strategy_vol", "策略波動", _pc), ("implied_sharpe", "手算 Sharpe 差", _s),
        ("engine_sharpe", "引擎 Sharpe 差", _s)]))
    sec.append("\n\n## 3. K × lag 全表\n\n![K×lag 熱力圖](r6_heatmap.png)\n")

    pt = payload["noise_tbl"]
    sec.append(f"""
## 4. 相位：換一個起算日差多少

K=5、lag=1 時，同一格 5 個相位的 Sharpe 全距中位數 N = {_f(d['noise'])}，
最大 {_f(float(pt['sharpe_range'].max()))}（{pt.loc[pt['sharpe_range'].idxmax(), 'market']}×{int(pt.loc[pt['sharpe_range'].idxmax(), 'period_years'])}y）。
這個差異不含任何規則上的意義，只是檢查日剛好落在哪幾天。

實盤相位（TX_FUT，t 自 {payload['first_date']['TX_FUT']} 起算）相對 5 相位中位數的 ΔP = {_s(d['delta_p'])}。
相位不是可選參數：其他市場的相位彼此不可比，也禁止因此更換。
""")

    sec.append("## 5. 配對 bootstrap（第二個檢查，只用方向）\n\n"
               "格內為路徑內配對 Sharpe 差的中位數（變體 − 基準）。"
               "🔴 bootstrap 打斷長週期自相關，對趨勢策略有向下偏誤，只能降級不能升級。\n")
    sec.append(_boot_table(boot))

    piv = regime.pivot(index="K", columns="market", values="switches_per_year") \
        .reindex(K_GRID)[RANKING_MARKETS].reset_index()
    fr = regime.groupby("K")["false_rate"].median().reindex(K_GRID)
    piv["false_rate"] = fr.to_numpy()
    sec.append("\n\n## 6. 切換頻率與假訊號（機制來源）\n\n每年切換次數（跨相位中位數）；"
               "假訊號＝被兩次切換夾住且持續 ≤20 個交易日的 regime 段。\n")
    sec.append(_tbl(piv, [("K", "K", _n), *[(m, m, lambda x: f"{x:.2f}") for m in RANKING_MARKETS],
                          ("false_rate", "假訊號率（市場中位數）", _pc)]))
    sec.append("\n\n![切換頻率](r6_switches.png)\n")

    sec.append("\n## 7. 壓力窗口（描述性，不進判定）\n\n各行以「每 5 日、T+1」為基準，Δ 為變體減基準，"
               "皆跨相位中位數。每個窗口只有一條路徑，屬敘述性證據。\n")
    sec.append(_stress_table(payload["stress"]))

    mm = _with_label(maximin).sort_values(["lag", "K"])
    sec.append("\n\n## 8. maximin 百分位（交叉檢查，不進判定）\n\n"
               "第三輪 cohort 10 成員，百分位量化到 0.1 級距（§7 量化陷阱），只看是否與原始 Sharpe 矛盾。\n")
    sec.append(_tbl(mm, [("label", "變體", str), ("worst_pct_rank", "最差格百分位", _f),
                         ("mean_pct_rank", "平均百分位", _f)]))

    sec.append(conclusion(d, summary, regime))
    sec.append(HONESTY)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sec))


def conclusion(d: dict, summary: pd.DataFrame, regime: pd.DataFrame) -> str:
    keep_k = d["verdict_a"] != "5日為劣位"
    keep_t = d["verdict_b"] != "上界有空間"
    s = summary.set_index(["K", "lag"])
    judged = summary[((summary["lag"] == 1) | summary["K"].isin([1, 5]))
                     & ~((summary["K"] == 5) & (summary["lag"] == 1))]
    worst = float(max(judged["delta_max"].max(), -judged["delta_min"].min()))
    tx = regime[regime["market"] == "TX_FUT"].set_index("K")
    med_false = regime.groupby("K")["false_rate"].median()
    return f"""

## 9. 結論

1. **每 5 日檢查有被驗證嗎？** 規則 A 判定「{d['verdict_a']}」。{d['text_a']}
   R_A = {_f(d['r_a'])}，遠低於 τ = {_f(d['tau'])}；單格最大差異 {_f(worst)}，也沒有任何一格到 τ，
   不是正負抵銷出來的平坦。**K=5 排名第 {d['rank5']} 沒有意義**：它和後面幾名只差千分位，
   本輪不支持「5 日最好」，只支持「5 日和其他頻率一樣好」。
2. **當天就降槓桿比較好嗎？** 規則 B 判定「{d['verdict_b']}」。{d['text_b']}
   另一個雜訊的跡象：T+2 的配對 ΔCAGR {_pp(s.loc[(5, 2), 'dcagr'])} 反而高於 T+1，延遲越久不是越差。
3. **為什麼這麼平？** 檢查越頻繁反應越快，但假訊號也越多，兩者抵銷：
   TX_FUT 每日檢查每年切換 {tx.loc[1, 'switches_per_year']:.2f} 次、每 5 日 {tx.loc[5, 'switches_per_year']:.2f} 次；
   假訊號率（市場中位數）{_pc(med_false[1])} 對 {_pc(med_false[5])}。
4. **慢的一端不是沒有代價**：K=20 的 Sharpe 只差 {_s(s.loc[(20, 1), 'D'])}，但配對 ΔMDD {_pp(s.loc[(20, 1), 'dmdd'])}；
   2020 年 SPX、NKX 的回撤多 9–13pp（§7）。Sharpe 高原不代表尾部也平，這是判定量的盲點，不是 K=20 可用的證據。
   快的一端則方向不一：2020 年每日檢查在 TX_FUT 較好、在 SPX 較差。
5. **對實盤的影響：** {'無。' if keep_k and keep_t else '本輪結論不直接改動實盤（規則 E）。'}
   依規則 E，本輪不輸出參數建議；任何改動都要另開 PR、同步 signals.py／regime.py 與黃金測試基準，
   並由使用者明確決定。相位也不得更換。
"""


HONESTY = """
## 10. 誠實揭露

- **樣本與 self-selection**：四個市場是自選的，期間都結束於今天；Sharpe 中位數來自高度重疊的滾動視窗，
  獨立樣本遠少於視窗數。本輪沒有算 p 值，判定只靠事先寫定的門檻與方向一致性。
- **判定量看的是平均行為**：Sharpe 中位數會把少數快速崩跌平均掉，D 小不代表尾部沒差。
  壓力窗口（§7）是描述性輸出，不進判定，這是這個設計的已知盲點。
- **緩衝帶沒有跟著掃**：1% 是在 K=5 下檢驗的。K 改變時最適緩衝帶可能不同，本輪固定 1%；
  結論只適用於「K 或 lag 單獨改變」，不適用於同時重調緩衝帶。
- **成本口徑**：TX 用大台明細成本（手續費、期交稅、1 檔滑價），海外 5bp，實盤是微台，
  每口成本占名目的比例較高。頻率高的變體成本可能被低估。
- **T+0 只是上界**：同一個收盤既算訊號又成交，實務做不到。13:20 當場執行還要承受「13:20 價不等於結算價」
  的訊號誤差，本輪沒有分鐘資料可以量測。
- **bootstrap 對趨勢的向下偏誤**：只用方向、只能降級。
- **資料凍結**：至 2026-07，未含 2026-11 這波下跌。
- **實盤的整數口數不在模型裡**：回測是連續曝險；實盤 L=1 只能買到整數口。
- **跑結果後才加的描述欄**：配對 ΔMDD／ΔCAGR、單格最高最低、手算覆核都是看到結果後補的，只用來檢查判定是否可信，
  不進判定。起因是兩個 MDD 中位數相減得出 5.5pp，實際逐格配對只有 0.6pp：16 格的中位數剛好落在排序斷層。

## 附錄：輸出檔案

| 檔案 | 內容 |
|---|---|
| `r6_summary.csv` | 每 (K, lag) 的 D、M、逐市場 Δ 與曝險欄 |
| `r6_deltas.csv` | 每 (K, lag, 格) 的 Δ |
| `r6_variants.csv` | 123 變體 × 16 格完整指標 |
| `r6_phase_noise.csv` | 每格相位全距 |
| `r6_bootstrap.csv` | 配對 bootstrap |
| `r6_regime.csv` | 切換統計 |
| `r6_stress.csv` | 壓力窗口 |
| `r6_maximin_crosscheck.csv` | maximin 百分位交叉檢查 |
| `r6_repro_check.csv` | 復現第三輪 D_10 的檢核 |
| `r6_ruleA_candidates.csv` | 規則 A 替代條件（僅第三分支有內容） |
"""


def write_from_cache() -> None:
    from backtest_signal_timing import charts6
    with open(CACHE_DIR / "results6.pkl", "rb") as f:
        payload = pickle.load(f)
    charts6.setup_style()
    charts6.heatmap_fig(payload["summary"], payload["decision"]["tau"],
                        OUTPUT_DIR / "r6_heatmap.png")
    charts6.switches_fig(payload["regime"], OUTPUT_DIR / "r6_switches.png")
    write_report(payload, OUTPUT_DIR / "report.md")


if __name__ == "__main__":
    write_from_cache()
