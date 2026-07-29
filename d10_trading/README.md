# D_10 趨勢過濾策略・Shioaji 自動化交易系統

把三輪回測驗證的 **D_10 策略**（200MA ± 1% 遲滯、L_high=2.0／L_low=1.0、
訊號週頻、T+1 執行）實作為每交易日 **13:20** 執行一次的日批次程式，
透過永豐金 Shioaji API 交易台指期貨（預設 **TMF 微型台指**）。

**這是動真錢的系統。** 優先序：正確性 > 安全性 > 可觀測性 > 功能完整 > 效能。
`config.py` 的 `LIVE_TRADING = False` 為預設；在 G2 閘門獲得明確「通過」之前
**絕不**改為 `True`。

---

## 1. 專案結構

```
d10_trading/
├── config.py          全部參數（凍結策略參數 / 執行參數 / 閘門旗標）
├── main.py            日批次編排器（python -m d10_trading.main）
├── data/              資料層
│   ├── finmind.py       FinMind 逐合約行情（移植自回測 data.py）
│   ├── continuous.py    連續合約構建（移植自回測 continuous.py，兩法互驗）
│   ├── series.py        日常重建、新鮮度檢查、雙源核對、當日暫定價
│   └── calendar.py      TAIFEX 行事曆（假日檔 + 實際行情互驗）
├── data_static/taifex_holidays.csv   假日檔（人工逐年維護，見 §7）
├── strategy/          純函數核心（不 import shioaji）
│   ├── regime.py        D_10 訊號狀態機（黃金測試鎖定）
│   ├── sizing.py        口數計算（half-up 捨入、防呆上限、保險絲）
│   └── rollover.py      轉倉時點（regime 自適應：月差>0 → T−5，否則結算日）
├── execution/
│   ├── broker.py        Shioaji 唯一封裝層（查詢 retry、下單不盲重試）
│   └── orders.py        限價單流程（讓 2 檔 → 90s 改價一次 → 撤單告警）、轉倉
├── ops/
│   ├── audit.py         操作稽核軌跡 var/logs/audit.jsonl（見 §2.1）
│   ├── state.py         狀態檔 var/state.json（原子寫入）
│   ├── reconcile.py     開場對帳（不一致 → 告警停止，絕不自動修正）
│   ├── notify.py        Telegram + 日誌
│   └── oplog.py         每日摘要 CSV / 成交 JSONL / 執行價 vs 收盤價稽核
├── tests/             74 項測試（單元 + 黃金 + 整合 + 稽核 + 監控 hook）
├── .env.example       環境變數樣板（複製為 .env，永不提交）
└── var/               執行期產物（狀態/快取/日誌，gitignored）
```

## 2. 每日批次流程（13:20 台北時間）

1. **斷電開關**：`d10_trading/KILL_SWITCH` 檔案存在 → 直接退出。
2. **冪等**：狀態檔 `executed_date == 今日` → 退出（同日重跑不會重複下單）。
3. **對帳**：API 實際部位/權益 vs 狀態檔；不一致 → 告警 + 本日停止。
4. **保險絲**：`L_current ≥ 3.2` → CRITICAL 告警＋立即減碼回 L_target（退出碼 2）。
5. **資料**：FinMind 重抓當年度 → 重建還原連續序列 → 新鮮度檢查
   （序列必須到上一交易日；FinMind 缺漏時以 Shioaji 日盤收盤價暫代並告警；
   兩者皆無 → 停止）→ 雙源核對（差 >0.1% 告警）→ 追加今日暫定價（僅預覽）。
6. **訊號**：全序列重算 regime → 今日執行槓桿 = **上一交易日**的訊號（T+1）。
7. **口數**：`round(L_target × 權益數 ÷ (價格 × 乘數))`（half-up）；
   超過 `MAX_CONTRACTS` → 拒單告警。
8. **轉倉**：近月結算日（Shioaji delivery_date）；月差>0 → T−5 轉倉，否則結算日；
   優先跨月價差單，失敗退兩腳（先次月後近月）。轉倉+再平衡重疊 → 一次淨到位。
9. **下單**：限價 ±2 檔 → 90 秒未成交改價一次 → 仍未成交撤單告警、今日放棄。
10. **落地**：讀回 API 實際部位寫入狀態、每日摘要、Telegram 推播。

退出碼：`0` 正常、`1` 告警停止、`2` 保險絲/嚴重。

### 2.1 紀錄檔案總覽（`var/logs/`，全部保留、不輪刪）

| 檔案 | 內容 |
|---|---|
| **audit.jsonl** | 🔴 **操作稽核軌跡**（G1/G2′/G3 全階段統一留痕）：批次生命週期（start/decision/end/stop）、每筆委託/組合單/改價/撤單與終態、所有通知告警原文、手動工具（adopt-state / flatten_sim / validate_live_path）、保險絲、斷電開關。掛在 broker 封裝層與批次編排器兩個「繞不過去的漏斗」上，任何呼叫端都無從遺漏。每行一個 JSON 事件（ts=台北時間、mode=sim/live、event、其餘依事件） |
| live_YYYYMM.log | 完整程式日誌（月檔） |
| daily_summary.csv | 每日一行摘要（regime/權益/口數/動作/警示） |
| trades.jsonl | 成交明細 |
| exec_vs_close.csv | 執行價 vs 官方收盤價偏差（次日補寫，規格 §2） |
| scheduler.log | 排程器輸出（run_daily.bat 導向） |

稽核查詢範例（PowerShell）：
```powershell
Get-Content d10_trading\var\logs\audit.jsonl | Select-String '"event": "order_'   # 所有委託事件
Get-Content d10_trading\var\logs\audit.jsonl | Select-String '"mode": "live"'    # 只看真實模式
Get-Content d10_trading\var\logs\audit.jsonl -Tail 20                            # 最近 20 筆
```

## 3. 部署步驟

### 3.1 前置（Phase 0，缺一不可）
1. 永豐帳戶＋期貨 API 約定書已簽署（<https://www.sinotrade.com.tw/newweb/signCenter/F_openApi/>）
2. API Key/Secret 已建立（<https://www.sinotrade.com.tw/newweb/PythonAPIKey/>）
3. 已完成模擬環境登入＋下單測試（簽署後的必要審核步驟）
4. G2 前：於同一頁面下載憑證 `Sinopac.pfx`
5. 環境：`ML_env`（已含 `shioaji 1.5.6`、`python-dotenv`）

### 3.2 設定
```
cd live
copy .env.example .env      # 填入 SJ_API_KEY / SJ_SEC_KEY（本機編輯，勿貼到任何對話）
```
Telegram：向 @BotFather 建 bot 取得 token，填 `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`。

### 3.3 排程（每交易日 13:20）

**Windows（本機，工作排程器）** — 已於 2026-07-16 依下列方式註冊（任務名 `D10_daily`）：

入口為 [scripts/run_daily.bat](scripts/run_daily.bat)（輸出導向 `var/logs/scheduler.log`）。
⚠️ **bat 檔內容必須純 ASCII**：cmd.exe 以 OEM 編碼（zh-TW 為 CP950）讀取 .bat
位元組，檔內若有 UTF-8 中文（例如 repo 路徑）會成亂碼導致 cd 失敗——G1 首日
（2026-07-17）即因此未觸發。中文路徑一律以 `%~dp0` 於執行期解析。註冊：
```powershell
$action   = New-ScheduledTaskAction -Execute "C:\Users\User\Desktop\python_台指最佳槓桿\d10_trading\scripts\run_daily.bat" `
            -WorkingDirectory "C:\Users\User\Desktop\python_台指最佳槓桿"
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "13:20"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName "D10_daily" -Action $action -Trigger $trigger -Settings $settings
```
設定含意：電池供電照跑、錯過觸發時間會於開機後補跑（程式的 13:05–13:40
執行時間窗會擋住盤後下單並告警，等於「今日已略過」通知）、可從**睡眠**喚醒
（無法從關機喚醒）。管理指令：
```powershell
Get-ScheduledTask D10_daily | Get-ScheduledTaskInfo   # 查下次執行時間/上次結果
Start-ScheduledTask D10_daily                          # 立即手動觸發（測試）
Unregister-ScheduledTask D10_daily -Confirm:$false     # 移除排程
```
任務以互動式使用者身分執行（`LogonType: Interactive`），13:20 觸發時
電腦各狀態的行為（2026-07-17 依實際任務設定核對）：

| 電腦狀態 | 會觸發？ | 說明 |
|---|---|---|
| 使用中 / **螢幕鎖定** | ✅ | 判定條件是「登入 session 存在」，與螢幕是否解鎖無關 |
| **睡眠** | ✅ | `WakeToRun` 會喚醒電腦執行 |
| **完全登出** | ❌ | Interactive 型任務需要 session；交易日請保持登入（鎖定即可） |
| **關機 / 休眠斷電** | ❌ | 無法喚醒；開機後 `StartWhenAvailable` 會補跑，但已過執行時間窗 → 拒單告警（= 今日已略過通知），次一交易日自動恢復 |

若未來需要「登出也能跑」，須改用 `Password` 登入類型（註冊時存 Windows
密碼）；但該場景更適合直接走 [deploy/README.md](../deploy/README.md) 的
VPS 方案（systemd 不依賴登入 session）。

**Linux / 雲端 VPS**：完整部署 runbook 見 [deploy/README.md](../deploy/README.md)
（systemd oneshot + timer、uv 鎖版環境、healthchecks 心跳、切換程序——
🔴 切換時絕不允許雙機同時排程）。

手動執行 / 測試：
```
python -m d10_trading.main --dry-run                 # 完整流程但不下單、不落地
python -m d10_trading.main --ignore-window           # 略過 13:05–13:40 時窗（測試用）
python -m d10_trading.main --adopt-state             # 人工確認後以 API 部位重置狀態
python -m pytest d10_trading/tests -q                # 74 項測試（含黃金測試）
```

## 4. 上線閘門檢核清單（🔴 每關需要使用者明確說「通過」）

> 2026-07-16 與使用者確認修訂：原 G2「真實 1 口」假設從空手開始，不適用於
> 真倉已持有滿額部位（3 口 TMF 202608）的現況——改為 G2′「不成交驗證＋
> 接管監管期」。風險遞進：模擬轉倉 → 零成交風險的真實委託路徑 → 監管下
> 日常運作 → 監管下真實轉倉 → 無人值守。

| 閘門 | 模式 | 通過條件 | 檢核 |
|---|---|---|---|
| **G0** 開發 | 離線 | 60 項測試全綠；黃金測試與第三輪回測逐日一致 | `pytest d10_trading/tests -q`；審閱 §8「規格判斷紀錄」 |
| **G1** 模擬（延長版） | `LIVE_TRADING=False`（現值） | 模擬環境自 2026-07-17 連跑至 **8/12 含模擬轉倉演練**（自然轉倉日；約 19 個交易日）：每日摘要正確、無誤動作、轉倉正確執行。期間真倉由使用者自管，**8→9 月真實轉倉最後一次手動執行** | 前置：`flatten_sim --yes` 清掉模擬殘留部位；核對 `var/logs/daily_summary.csv` 逐日；8/12 觀察價差單/兩腳行為 |
| **G2′** 真實・接管監管期 | `LIVE_TRADING=True`＋CA＋`MAX_CONTRACTS=4` | ① 不成交驗證通過（見下）② 接管後 **20+ 交易日**對帳/成交/滑價全程乾淨 ③ 含 **9/9 前後**系統執行的第一次真實轉倉（人在螢幕前監看） | 每日核對 Telegram 摘要 vs 券商 App；`exec_vs_close.csv` 偏差在預期內 |
| **G3** 真實全額・無人值守 | 同 G2′ 設定 | — | — |

G0→G3 程式碼相同，只改 `config.py` 設定（`LIVE_TRADING`、`MAX_CONTRACTS`、`SIM_EQUITY`）。

### 4.1 G2′ 接管程序（依序執行）

1. **確認真倉帳戶只有 TMF 部位**——對帳不容忍其他商品（選擇權、其他期貨
   會觸發告警停止），有的話先人工隔離/處理。
2. `config.py` 改 `LIVE_TRADING = True`（`.env` 需已設 SJ_CA_PATH/SJ_CA_PASSWD）。
3. **不成交驗證**（盤中執行；驗證 CA 簽章/下單/回報/撤單的正式環境全路徑，
   掛遠價限價買單 1 口、零成交風險，確認回報後即撤單）：
   ```
   python -m d10_trading.scripts.validate_live_path --yes
   ```
4. `python -m d10_trading.main --adopt-state` 接管現有部位。
5. **預覽第一天動作**：`python -m d10_trading.main --dry-run --ignore-window`。
   權益數此時改用真實 API 值——確認目標口數與預期一致（權益 ≈<equity>時
   目標 3 口＝現有部位，接管日應無交易）再讓排程接手。
6. 改動 commit 並打 tag（`g2-takeover`），推送遠端。

## 5. 故障排除

| 症狀 | 處理程序 |
|---|---|
| **對帳不符告警**（手動干預過/前日部分成交） | 1. 看告警訊息中 API vs 狀態檔差異 2. 人工核對券商回報 3. 確認無誤後 `python -m d10_trading.main --adopt-state` 4. 次日恢復自動 |
| **「帳戶存在非 TMF 部位」告警** | 本系統只代管 `PRODUCT` 部位。模擬環境：`python -m d10_trading.scripts.flatten_sim --yes` 清空（多半是 API 開通測試單殘留）。真實環境：人工在券商端處理該部位 |
| **「無月合約」告警** | 微台無 `Futures["TMF"]` 群組鍵，系統以 `TMFR1`/`TMFR2` 連續月別名（`target_code`）解析近月/次月，備援為群組列舉與全檔掃描。若仍失敗多為商品檔未載入完成，稍後重跑 |
| **資料過期告警**（FinMind 未更新且 Shioaji 補不到） | 通常為資料源延遲：次日自動恢復。若因**臨時停市（颱風）**：把該日補進 `data_static/taifex_holidays.csv` 後重跑 |
| **API 斷線 / 登入失敗** | 查詢類已自動退避重試 3 次；仍失敗 → 本日停止告警。檢查網路與永豐服務狀態，可於 13:40 前手動重跑（冪等保護） |
| **憑證過期**（G2 起） | 至 API 管理頁重新下載 `Sinopac.pfx`，更新 `.env` 的 `SJ_CA_PATH/SJ_CA_PASSWD` |
| **下單 timeout 告警** | 已自動撤單，部位以實際成交為準；明日批次自然補回目標口數，無需介入 |
| **保險絲 CRITICAL** | 程式已自動減碼回 L_target。**人工檢視市場狀況與帳戶**，確認後才恢復排程 |

## 6. 手動接管程序

**安全停機**：在 `d10_trading/` 下建立空檔 `KILL_SWITCH` → 之後每次批次啟動即退出。
（`type nul > d10_trading\KILL_SWITCH`；恢復時刪除該檔。）

**人工平倉後重置**：
1. 建立 `KILL_SWITCH`，於券商端人工處理部位。
2. 處理完畢後刪除 `KILL_SWITCH`，執行 `python -m d10_trading.main --adopt-state`
   （把 API 實際部位採納為新狀態；絕不要手改 `state.json`）。
3. 次一交易日恢復自動運行。

## 7. 假日檔維護（每年一次＋臨時停市）

`data_static/taifex_holidays.csv` 供**未來**交易日推算（轉倉倒數）；
過去的交易日以 FinMind 實際行情為準並每日自動互驗，不一致會告警。

- 每年 12 月：依 TAIFEX 公告（<https://www.taifex.com.tw/cht/5/annAndOpDate>）
  補次年假日，並把 `# covered_years:` 加上該年。未涵蓋 → 程式 fail-closed 拒跑。
- 臨時停市（颱風）：事後把該日補入（互驗告警會提醒）。
- 2026-07-10 已依實際行情互驗補列（疑颱風停市），請與 TAIFEX 公告核對。

## 8. 規格判斷紀錄（🔴 G0 審查重點——實作中遇到的規格模糊處與採用的解法）

1. **SIGNAL_DOW（訊號檢查日）**：規格草稿寫「固定每週某交易日」，但第三輪回測
   實際規則是**連續序列索引 `t % 5 == 0`**（自 1998-07-22 起算，`backtest_multimarket/signals.py`）。
   固定星期幾無法通過黃金測試（假日會使相位漂移），故採回測原始規則。
   下一個檢查日完全可預測（每 5 個 TX 交易日）。
2. **轉倉安全補課（late_roll）**：規格 §6 只定義「今日==轉倉日才轉」。若月差在
   T−5 之後才由負轉正，字面規則會導致**永不轉倉、抱進結算**。已加入 fail-safe：
   T−5 < 今日 ≤ 結算日且月差>0 且仍持近月 → 當日補轉並記錄 `late_roll`。
   這是「不得持倉進入現金結算」的完備化，非新策略規則。
3. **模擬環境權益數（2026-07-30 修訂為市值計價）**：Shioaji 模擬伺服器
   `margin()` 回 0。G1 期間權益 = `SIM_EQUITY`（起始權益，預設 <equity>）
   ＋累計已實現損益＋未實現損益，**逐日市值計價**；真實模式一律用 API 權益數。
   - 為何不能用固定值：權益凍結時，下跌會使目標口數 L×E/(價×乘數) 反而**上升**
     （愈跌愈買），與真實帳戶（權益隨虧損下降→目標下降）相反。實測 7 月下旬
     −5.6% 那波，凍結法在跌破 40,000 時會加碼到 4 口，真實帳戶則維持 3 口。
   - 已實現部分不可省略：每月轉倉會把未實現損益實現掉，只計未實現會讓權益
     在轉倉日跳回起始值。
   - 已實現只計 `SIM_PNL_SINCE`（G1 首日 2026-07-20）起、且屬 `PRODUCT` 的紀錄
     ——模擬帳戶殘留的 API 開通測試單（TXFH6，−235,000）不屬本系統，計入會讓
     權益憑空少 33%。此範圍設定於 `config.py`。
4. **到期週的訊號序列重建**：回測管線以「合約最後出現日」判定到期（ex-post）。
   即時運行時，結算日前最後幾天管線尚不知道到期日，資料越過到期日後會**回溯**
   插入 T−5 轉倉，該窗口的訊號值可能微調。系統每日全量重算並比對前日已定案
   訊號，若有變動會告警（`資料重建後訊號變動`），部位隨當日目標自動校正。
   此為忠實移植回測口徑的固有性質，影響量級 = 月差 × ≤5 日。
5. **訊號預覽 vs 定案**：今日 13:20 即時價算出的訊號僅為預覽；正式訊號一律於
   次日以官方結算價定案（T+1 執行完全對齊回測）。兩者不一致時告警提示。
6. **捨入**：`round()` 採 half-up（1.5 → 2），非 Python 內建銀行家捨入，
   邊界行為確定且已測試；捨入方向與殘餘偏差逐日記錄。
7. **雙源核對口徑**：FinMind「結算價」vs Shioaji 日盤「收盤價」本有微小差異，
   >0.1% 僅告警不停止（規格 §3.3 字面為「告警」）；資料「缺漏」才停止。

## 9. 明確不做（規格 §11）

不做行情串流／tick 邏輯／夜盤；不做回測參數外的任何「優化」；
不自動處理對帳異常；日誌與訊息永不出現金鑰或憑證路徑；
不引入回測未驗證的新規則（§8 所列為模糊處的保守完備化，均待 G0 審查）。
