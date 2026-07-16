# 台指期 2 倍槓桿・門檻式再平衡 [X, Y] 網格回測

對「TX 2 倍槓桿 + 門檻式再平衡」做 10×10 網格搜尋（X=減碼上限、Y=加碼下限），
分別找累積報酬與 Sharpe 最高的 [X, Y]，含季頻滾動起點的反過擬合檢驗。
完整規格見原始任務文件；主要結論見 `output/report.md`。

## 執行

```bash
conda run -n ML_env python run.py        # 全流程約 90 秒
conda run -n ML_env python test_engine.py  # 引擎手算對照 sanity check
```

## 第三輪：條件式曝險・跨市場（branch `round3-conditional-exposure`）

`backtest_multimarket/` 子套件——條件式曝險策略（vol targeting／趨勢過濾／CPPI／混合）×
四市場（TX 期貨、S&P 500、日經 225、恆生）× stationary bootstrap 合成路徑，
含槓桿中性化校準協定（σ_target walk-forward + 全樣本雙軌）。
主報告 `backtest_multimarket/output/report.md`。

```bash
conda run -n ML_env python -m backtest_multimarket.run3         # 全流程（含 1,000 條 bootstrap）
conda run -n ML_env python -m backtest_multimarket.test_engine  # 引擎手算對照＋與前輪引擎互驗
```

## 模組

| 檔案 | 內容 |
|---|---|
| `params.py` | 所有可調參數（網格、成本、期間、資料路徑） |
| `data.py` | TX 逐合約日行情（FinMind API + parquet 快取，優先沿用姊妹專案快取） |
| `continuous.py` | **連續合約構建**：逐合約法 + 比值還原法互驗（<1e-8），假跳空消除 |
| `engine.py` | **回測引擎**：numpy 向量化（一次日迴圈模擬 K 組參數），支援門檻式與週期式（每 N 日）再平衡，每日順序見檔頭 |
| `metrics.py` | 績效指標（CAGR/Sharpe/Sortino/MDD/Calmar/水下期間/槓桿統計） |
| `grid.py` | 網格 × 滾動起點聚合、最佳解選取、平原/尖峰鄰域統計 |
| `charts.py` | 熱力圖、資金曲線疊圖、槓桿時序圖 |
| `report.py` | 主報告產生器（讀 `output/cache/results.pkl`） |
| `run.py` | 主流程（含滑價 0.5 檔、轉倉 offset 0 敏感度與 0.05 步長精細搜尋） |

## 輸出（output/）

- `report.md` — 主報告（最佳解表、平原/尖峰判定、對稱性、敏感度、結論六問）
- `grid_results_{1,3,5,10,20}y.csv` — 每組 [X,Y] 完整指標（跨路徑統計 + 基準路徑）
- `equity_curve_best_{p}y.csv`、`rebalance_log_best.csv`、`roll_events.csv`
- `heatmap_{p}y.png`、`equity_overlay_{p}y.png`、`leverage_best_{p}y.png`
- `crosstab_20y_best.csv`、`robust_ranking.csv`、`sens_*.csv`、`refine_search.csv`
- `rebal_frequency.csv` / `.png`、`rebal_phase_sens.csv` — 再平衡頻率掃描（買進持有 → 季 → 月 → 雙週 → 5日 → 每日）

## 基準策略（皆以同一引擎參數化表達）

| 策略 | 引擎參數 |
|---|---|
| 買進持有 2x | X=+inf, Y=−inf |
| 每日再平衡 2x（正2倍ETF） | X=Y=2.0（等價於 rebal_every=1） |
| **每 5 日再平衡 2x** | X=+inf, Y=−inf, rebal_every=5 |
| 每 10/21/63 日再平衡 2x | rebal_every=10/21/63（僅進頻率掃描表） |
| 買進持有 1x | target=1.0 |

## 重要口徑（詳見 report.md §1）

- 連續合約：日報酬只在同一口合約內部計算；月差（carry）內生於價格序列，不重複計算。
- 計息：對閒置現金（全額權益）以 1.7% 年化日頻複利；不對借款計息。
- 破產：權益 ≤ 0 歸零不反彈；任何路徑破產的參數組不得列入最佳解。
- 最佳解以跨路徑中位數判定，單一路徑最大值僅作過擬合對照。
