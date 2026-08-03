# 台指最佳槓桿——研究與實盤

單一 repo、三個平級套件；研究結論沿「回測 → 跨市場驗證 → 實盤」溯源鏈遞進，
共用同一份 FinMind 快取（`data/`）。

| 套件 | 內容 | 主要輸出 |
|---|---|---|
| [`backtest_threshold/`](backtest_threshold/README.md) | 第 1+2 輪：TX 2x 門檻式再平衡 [X,Y] 網格回測（單市場） | `backtest_threshold/output/report.md` |
| [`backtest_multimarket/`](backtest_multimarket/) | 第 3 輪：條件式曝險策略 × 4 市場 × stationary bootstrap；**maximin 冠軍 = D_10 趨勢過濾** | `backtest_multimarket/output/report.md` |
| [`backtest_ma_robustness/`](backtest_ma_robustness/) | 第 4 輪・補洞：D_10 的 **200MA 窗口穩健性**（pre-registration）；規則 B 未觸發、原始 Sharpe 全距 0.076<0.08、四市場無更長窗口勝過 200 → **維持 MA=200** | `backtest_ma_robustness/output/report.md` |
| [`backtest_leverage_sweep/`](backtest_leverage_sweep/) | 第 5 輪：D_10 **目標槓桿掃描 2.0→2.5**（取捨曲線；新增維持保證金／強制平倉模型）。**不宣告最佳槓桿**——僅輸出 frontier 與各槓桿代價清單 | `backtest_leverage_sweep/output/report.md` |
| [`d10_trading/`](d10_trading/README.md) | D_10 策略的 Shioaji 自動交易系統（動真錢，分階段閘門 G0–G3） | 每日摘要 `d10_trading/var/logs/` |

```
執行（一律在 repo 根目錄）：
conda run -n ML_env python -m backtest_threshold.run          # 第 1+2 輪全流程
conda run -n ML_env python -m backtest_multimarket.run3       # 第 3 輪全流程
conda run -n ML_env python -m backtest_ma_robustness.run4     # 第 4 輪 MA 窗口穩健性
conda run -n ML_env python -m backtest_leverage_sweep.run5    # 第 5 輪 目標槓桿掃描
conda run -n ML_env python -m d10_trading.main --dry-run      # 實盤日批次（測試）
conda run -n ML_env python -m pytest d10_trading/tests -q     # 實盤系統測試（含黃金測試）
```

相依關係：`backtest_multimarket` 引用 `backtest_threshold` 的資料管線與引擎互驗；
`backtest_ma_robustness` 唯讀引用 `backtest_multimarket` 的引擎／評估／資料層（其
MA200/buf1% 變體複現 D_10 的 maximin，作為整合檢核），只寫入自己的 `output/`，
不觸碰凍結基準；`backtest_leverage_sweep` 同樣唯讀引用，但**additively 擴充了共用
引擎**（`maint_margin_rate` 維持保證金強制平倉，預設 0 = 停用、零行為變更；
d10_trading 未引用 engine.py，其耦合點為 signals.py 與凍結 output CSV，已跑
120 項測試確認未回歸）；
`d10_trading` 的黃金測試（G0 閘門）比對 `backtest_multimarket` 的訊號程式與輸出
CSV——**三者必須同 repo 版本化**，`backtest_multimarket/output/` 的 CSV 是凍結
比對基準，不可重產覆蓋。`data/` 為共用 FinMind 逐年快取（parquet），實盤系統
另有自己的執行期快取（`d10_trading/var/cache/`，gitignored）。
