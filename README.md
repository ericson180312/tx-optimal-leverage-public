# 台指最佳槓桿——研究與實盤

單一 repo、三個平級套件；研究結論沿「回測 → 跨市場驗證 → 實盤」溯源鏈遞進，
共用同一份 FinMind 快取（`data/`）。

| 套件 | 內容 | 主要輸出 |
|---|---|---|
| [`backtest_threshold/`](backtest_threshold/README.md) | 第 1+2 輪：TX 2x 門檻式再平衡 [X,Y] 網格回測（單市場） | `backtest_threshold/output/report.md` |
| [`backtest_multimarket/`](backtest_multimarket/) | 第 3 輪：條件式曝險策略 × 4 市場 × stationary bootstrap；**maximin 冠軍 = D_10 趨勢過濾** | `backtest_multimarket/output/report.md` |
| [`d10_trading/`](d10_trading/README.md) | D_10 策略的 Shioaji 自動交易系統（動真錢，分階段閘門 G0–G3） | 每日摘要 `d10_trading/var/logs/` |

```
執行（一律在 repo 根目錄）：
conda run -n ML_env python -m backtest_threshold.run          # 第 1+2 輪全流程
conda run -n ML_env python -m backtest_multimarket.run3       # 第 3 輪全流程
conda run -n ML_env python -m d10_trading.main --dry-run      # 實盤日批次（測試）
conda run -n ML_env python -m pytest d10_trading/tests -q     # 實盤系統測試（含黃金測試）
```

相依關係：`backtest_multimarket` 引用 `backtest_threshold` 的資料管線與引擎互驗；
`d10_trading` 的黃金測試（G0 閘門）比對 `backtest_multimarket` 的訊號程式與輸出
CSV——**三者必須同 repo 版本化**，`backtest_multimarket/output/` 的 CSV 是凍結
比對基準，不可重產覆蓋。`data/` 為共用 FinMind 逐年快取（parquet），實盤系統
另有自己的執行期快取（`d10_trading/var/cache/`，gitignored）。
