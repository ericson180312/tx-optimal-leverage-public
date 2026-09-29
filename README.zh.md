# 台指最佳槓桿——研究與實盤

> 🇬🇧 English: **[README.en → README.md](README.md)** ｜ 本檔為中文版。
> 公開版**不含**部署 runbook、主機資訊與帳戶金額；市場資料快取亦未隨附（見英文版 Market data）。

單一 repo、五個平級套件。研究沿下列**溯源鏈**
遞進——每一輪只回答一個問題，把結論凍結後交給下一輪：

```
單市場網格 → 跨市場條件式曝險 → 參數穩健性補洞 → 曝險取捨曲線 → 實盤
  (1+2)          (3)                 (4)              (5)          d10
```

> 📄 **[RETROSPECTIVE.md](RETROSPECTIVE.md)** — 專案總結：最初目標、五輪研究如何遞進、
> 實盤系統的設計原則、至今成果與真實運行找到的缺陷、尚未驗證的部分（2026-09-22）。

## 溯源鏈：每輪問了什麼、決定了什麼

| 輪次 | 問的問題 | 結論（流向下一輪的東西） | tag |
|---|---|---|---|
| **1+2** | 2x 槓桿的門檻式再平衡，[X,Y] 該選哪裡？ | 曲面平坦、差異在噪音量級；**一階因子只有「有效平均槓桿」與「有無防歸零機制」** | — |
| **3** | 哪一種曝險控制機制跨市場最穩健？ | **D_10（200MA 趨勢過濾）為 maximin 冠軍** → 成為實盤策略；輸出凍結為黃金測試基準 | `round3-complete` |
| **4** | D_10 的 200MA 是慣例還是過擬合？ | 窗口是高原（原始 Sharpe 全距 0.076 < 0.08）、規則 B 未觸發 → **維持 MA=200** | `round4-complete` |
| **5** | 目標槓桿從 2.0 加到 2.5 值不值得？ | CAGR 僅 +0.8pp、MDD 惡化 +6.6pp、每單位槓桿報酬衰退 12.1%；**依紀律不宣告最佳槓桿**，只交出 frontier 與代價清單 | `round5-complete` |
| **實盤** | — | `d10_trading` 現行 `L_HIGH = 2.0`。⚠️ 這是**風險偏好的選擇，不是第 5 輪的統計結論**——第 5 輪只證明 2.0→2.5 的交換率很差（每 1pp 回撤換不到 0.25pp CAGR），沒有、也無法由資料指定一個最佳值 | — |

## 套件

| 套件 | 內容 | 主要輸出 |
|---|---|---|
| [`backtest_threshold/`](backtest_threshold/README.md) | 第 1+2 輪：TX 2x 門檻式再平衡 [X,Y] 網格回測（單市場） | `backtest_threshold/output/report.md` |
| [`backtest_multimarket/`](backtest_multimarket/) | 第 3 輪：條件式曝險策略 × 4 市場 × stationary bootstrap | `backtest_multimarket/output/report.md` |
| [`backtest_ma_robustness/`](backtest_ma_robustness/) | 第 4 輪・補洞：D_10 的 200MA 窗口穩健性（pre-registration，規則 A/B 機械判定） | `backtest_ma_robustness/output/report.md` |
| [`backtest_leverage_sweep/`](backtest_leverage_sweep/) | 第 5 輪：D_10 目標槓桿掃描 2.0→2.5（取捨曲線；新增維持保證金／強制平倉模型） | `backtest_leverage_sweep/output/report.md` |
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
比對基準，不可重產覆蓋。市場資料快取（FinMind parquet）**未隨公開版散布**——
設好 `FINMIND_TOKEN` 後資料層會自行抓取並快取；黃金測試需要它才能重建序列，
其餘測試全部以合成 fixture 離線執行。
