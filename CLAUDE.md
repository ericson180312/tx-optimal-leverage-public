# CLAUDE.md — 台指最佳槓桿（研究與實盤）

專案結構、各輪結論與執行指令見 [README.md](README.md)。本檔只寫「動手前必須知道的約束」
與「工作流程」。

## 🔴 動手前必讀：三個硬約束

1. **`backtest_multimarket/output/` 是凍結比對基準，不可重產覆蓋。**
   `d10_trading` 的黃金測試（G0 閘門）拿它比對切換日期。要重跑第三輪流程前，
   先確認不會覆蓋這批 CSV。

2. **改共用層前先確認耦合點。** `d10_trading` 只耦合
   `backtest_multimarket/signals.py`（`trend_leverage` 逐位元移植）與上述凍結 CSV
   ——見 `d10_trading/tests/test_golden.py`；**它並未引用 `engine.py`**。
   因此：動 `signals.py` 幾乎必然破壞黃金測試（除非同步改實盤的 `strategy/regime.py`）；
   動 `engine.py` 相對安全，但**一律 additive**——新參數預設關閉、對既有呼叫零行為變更。

3. **任何動到共用層的改動，改完必跑這四組**（第五輪即以此證明引擎擴充未回歸）：
   ```bash
   conda run -n ML_env python -m pytest d10_trading/tests -q                    # 120 項，含黃金測試
   conda run -n ML_env python -m pytest backtest_leverage_sweep/test_sweep5.py -q
   conda run -n ML_env python -m backtest_multimarket.test_engine
   conda run -n ML_env python -m backtest_ma_robustness.test_sweep
   ```

## 完成一輪回測／一項功能後的標準流程

1. **在分支上做**（`roundN-<主題>`、`fix-<主題>`），不在 main 上開發。
2. **atomic commits**：params / 核心邏輯 / 測試 / 流程與報告 / 輸出，各自一個 commit。
   可讀的歷史是交付物的一部分。
3. **更新 `README.md`**：套件表 + **溯源鏈表**（該輪問了什麼、決定了什麼、對應 tag）。
   若該輪的結論不會流進實盤，溯源鏈要寫明白（例：第五輪未指定最佳槓桿，
   實盤 `L_HIGH=2.0` 是偏好選擇而非統計結論）。
4. **開 PR**：`gh pr create`。內文寫**判斷與取捨**，不只列改了什麼——
   PR 內文就是決策記錄（見下節）。
5. **合併**：`gh pr merge <N> --merge`。**用 merge commit，不要 squash**，
   否則 atomic 歷史會被壓平。
6. **同步本地 main**：`git fetch origin main:main`（不必切換分支）。
7. **打 tag**：`git tag -a roundN-complete <merge-sha> -m "<結論摘要>"`
   → `git push origin roundN-complete`。既有慣例是 `-complete` 後綴。
8. **刪分支**：`git branch -d <branch>` + `git push origin --delete <branch>`。
   用 `-d` 而非 `-D`——被拒絕代表還有未合併的 commit，那是保護不是阻礙。
   （合併點已有 tag 與 merge commit 記錄，分支 ref 留著只會被誤用成雜項工作的暫存區。）

**驗收前先確認**：工作區乾淨。跑回測常會順手更新 `data/`、`backtest_multimarket/data/`
的 parquet 快取，那些與本次改動無關，提交前用 `git checkout --` 還原。

## 直接 push main vs 走 PR

**可直接 push 到 main**：純文件（README、註解、本檔），不改任何執行路徑，
且沒有需要留存的判斷。

**一律走 PR**：
- 任何會被執行到的程式（引擎、策略、資料層、實盤系統）
- 任何碰到凍結基準、`signals.py`、`d10_trading/` 的改動
- **一整輪回測**，即使沒有人 review

判準不是「有沒有人要 review」（這是單人 repo），而是
**「半年後會不會想知道當初為什麼這樣決定」**。會，就開 PR。
前五輪的 PR 內文已經是查得到的決策記錄——規格矛盾、判定口徑的取捨都寫在裡面。

## 報告與 PR 的誠實紀律（前五輪累積，不要退化）

- **規格與實測不符時**：先獨立手算覆核，確認不是 bug，再據實寫進報告並解釋為何不符；
  **不要默默改成規格預期的數字**。（第四輪的跌停日數、第五輪的 §3-2 都是這樣處理的。）
- **標註統計口徑陷阱**：例如第四輪 maximin 百分位受 10 成員 cohort 量化到 0.1 級距，
  字面判定會誤導，須並列不量化的原始 Sharpe 全距作為 robust 讀數。
- **報酬比較一律並列「平均槓桿」欄**，避免「高槓桿賺更多」的套套邏輯。
- **破產／強制平倉路徑排除於報酬統計之外**，另計生存率——否則會出現
  「歸零前漲很多」的假優勝者。
- **規格明令不可宣告的結論就不要宣告**（第五輪的「不指定最佳槓桿」）。
- 綜合值與逐市場結論不一致時，**兩者都要寫**；但也要區分「實質惡化」與
  「單段下凹的噪音」，不要往任一方向過度解讀。
