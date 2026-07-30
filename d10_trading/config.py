"""D_10 自動交易系統——全部參數集中於此檔（規格 §10）。

分層：
1. 凍結策略參數（§3.1）——與第三輪回測逐位元一致，🔴 不得修改。
2. 交易/執行參數（§4–§6）——合約選擇、下單容忍度、轉倉。
3. 營運參數（§7）——路徑、通知、告警。
4. 秘密（API key 等）一律走環境變數 / d10_trading/.env，程式碼與日誌永不出現明文。

🔴 上線閘門（§9）：LIVE_TRADING 預設 False。G2 之前不得改為 True；
   改動此常數視同進入真實下單階段，需使用者明確確認「通過」。
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv 未安裝時仍可運作（僅靠真實環境變數）
    load_dotenv = None

# ── 路徑 ────────────────────────────────────────────────────────────────────
LIVE_DIR = Path(__file__).resolve().parent            # d10_trading/
ROOT_DIR = LIVE_DIR.parent                            # 回測專案根目錄
VAR_DIR = LIVE_DIR / "var"                            # 執行期產物（gitignore）
CACHE_DIR = VAR_DIR / "cache"                         # FinMind parquet 快取
LOG_DIR = VAR_DIR / "logs"
STATE_FILE = VAR_DIR / "state.json"
KILL_SWITCH_FILE = LIVE_DIR / "KILL_SWITCH"           # 存在 → 程式啟動即退出（§7）
HOLIDAY_FILE = LIVE_DIR / "data_static" / "taifex_holidays.csv"

# 既有回測專案的 FinMind parquet 快取（存在就直接複製，不重抓）
SIBLING_CACHE_DIRS = [
    ROOT_DIR / "data",
    Path(r"C:\Users\User\Desktop\python_台指跨月價差回測\analysis\roll_carry\data"),
]

if load_dotenv is not None:
    load_dotenv(LIVE_DIR / ".env")

# ── 凍結策略參數（§3.1，🔴 與回測逐位元一致，不得修改）─────────────────────
MA_WINDOW = 200          # 還原連續期貨價的 200 日均線
BUFFER = 0.01            # ±1% 遲滯緩衝帶
L_HIGH = 2.0
L_LOW = 1.0
T_PLUS_1 = True          # 訊號日決定、次一交易日執行
FUSE_L = 3.2             # 結構保險絲（斷電開關，正常永不觸發）

# 訊號檢查時點（🔴 口徑說明）：規格草稿寫「固定每週某交易日」，但第三輪回測的
# 實際規則是「連續序列索引 t % 5 == 0 的交易日檢查」（backtest_multimarket/signals.py
# trend_leverage，check_every=5，t 自 1998-07-22 起算）。黃金測試（§3.2）要求
# 訊號輸出與回測逐日完全一致，固定星期幾無法滿足（假日順延會使相位漂移），
# 因此採用回測原始規則。序列每日追加後 t 持續遞增，檢查日完全可預測。
SIGNAL_CHECK_EVERY = 5

# ── 連續序列管線（§3.3，沿用回測口徑）───────────────────────────────────────
FUTURES_ID = "TX"            # 訊號序列用 TX 大台（與回測一致）
FUTURES_FIRST_YEAR = 1998    # TX 1998/07/21 上市
ROLL_OFFSET_DAYS = 5         # 訊號序列的轉倉偏移（到期前 5 交易日；回測凍結值）
FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")   # 免費層可空
DUAL_SOURCE_TOL = 0.001      # Shioaji vs FinMind 雙源核對容忍（0.1% → 告警）
# 雙源核對用 Shioaji contract.reference（=前一交易日結算價，與 FinMind 同口徑）。
# ⚠️ reference 只在「日盤收盤前」等於前日結算；日盤收盤（13:45）後 TAIFEX 公布當日
# 結算、reference 隨即滾動成當日值（實測 2026-07-20）。故僅在此時刻前才信任 reference
# 做核對；批次固定 13:20 執行、恆在此窗內。盤後 dry-run 會跳過核對（避免假警報）。
DAY_SESSION_CLOSE = "13:45"

# ── 合約與口數（§4）─────────────────────────────────────────────────────────
PRODUCT = "TMF"              # 預設 TMF 微型台指；可選 "MXF" 小台
MULTIPLIERS = {"TMF": 10.0, "MXF": 50.0, "TXF": 200.0}   # 每點價值（NT$）
MULTIPLIER = MULTIPLIERS[PRODUCT]
TICK_POINTS = 1.0            # 台指期貨系列最小跳動 = 1 點
SIGNAL_PRODUCT = "TXF"       # 訊號序列對應的 Shioaji 商品（TX 大台）

# 防呆上限（§4）：依資金算出的合理上限 + 1 口，寫死於此。
# 例：70 萬資金、指數 46,000、TMF 乘數 10 → 2x ≈ 3 口 → 上限 3 + 1 = 4。
# 🔴 G2 階段（真實 1 口）應手動改為 1；G3 依實際資金重算。
MAX_CONTRACTS = 4

# ── 下單執行（§5）───────────────────────────────────────────────────────────
PRICE_TOLERANCE_TICKS = 2    # 以最佳買/賣價讓 2 檔掛限價（禁用市價單）
ORDER_TIMEOUT_SECS = 90      # 逾時未成交 → 改價一次（再讓 2 檔）→ 仍未成交撤單
ORDER_POLL_SECS = 3          # 成交回報輪詢間隔
EXEC_WINDOW = ("13:05", "13:40")   # 允許下單的時間窗（台北時間；日盤收盤前）
TIMEZONE = "Asia/Taipei"

# ── 轉倉（§6）───────────────────────────────────────────────────────────────
ROLL_LEAD_DAYS = 5           # 月差 > 0 → 結算日前 5 交易日轉倉；否則結算日當天
ROLL_USE_COMBO = True        # 優先用 TAIFEX 跨月價差單；失敗退回兩腳分開下

# ── 上線閘門（§9）🔴 ────────────────────────────────────────────────────────
LIVE_TRADING = False         # G2 之前絕不改 True；True = 真實資金下單
SIMULATION = not LIVE_TRADING

# 模擬環境的 margin() 回傳 0（Shioaji 模擬伺服器不提供權益數）。
# 此常數為模擬帳戶的**起始權益**；實際使用的權益 = SIM_EQUITY ＋累計已實現損益
# ＋未實現損益（逐日市值計價，見 broker.equity），使模擬的曝險調整行為與真實
# 帳戶一致。真實模式一律用 API 權益數，不使用此常數。
SIM_EQUITY = float(os.environ.get("SIM_EQUITY", "700000"))
# 模擬權益的「起算日」：SIM_EQUITY 是此日的權益，僅計入此日起、且屬本商品
# （PRODUCT）的已實現損益。用意是排除本系統之外的模擬帳戶活動——實測
# 2026-07-16 的 API 開通測試單（TXFH6 1 口）留下 −235,000 已實現損益，
# 若一併計入會讓 G1 權益憑空少 33%（2026-07-30 發現）。
SIM_PNL_SINCE = "2026-07-20"   # G1 首個交易日（本系統首次建倉）

# ── 認證（一律環境變數，永不落地程式碼/日誌）────────────────────────────────
SJ_API_KEY = os.environ.get("SJ_API_KEY", "")
SJ_SEC_KEY = os.environ.get("SJ_SEC_KEY", "")
SJ_CA_PATH = os.environ.get("SJ_CA_PATH", "")         # 真實下單必需；模擬不需
SJ_CA_PASSWD = os.environ.get("SJ_CA_PASSWD", "")

# ── 通知（§7）────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# healthchecks.io 心跳（無人值守監控；空 = 略過）。批次結尾 ping：
# 正常 → GET URL；異常 → GET URL/fail。排程缺席由 healthchecks 端發告警。
HEALTHCHECK_URL = os.environ.get("HEALTHCHECK_URL", "")

# CA 憑證到期預警（僅真實模式；需身分證字號查詢，空 = 略過檢查）
SJ_PERSON_ID = os.environ.get("SJ_PERSON_ID", "")
CA_EXPIRY_WARN_DAYS = 30

# ── 到期哨兵（§7 可觀測性）──────────────────────────────────────────────────
# 會讓系統停擺的外部到期事項；日批次每日檢查並提前預警（ops/watchdog.py）。
# 假日檔涵蓋年份自 data_static/taifex_holidays.csv 的 covered_years 自動推導，
# 不必列在此表——它是唯一「到期即 fail-closed、完全不跑」的項目。
# 🔴 CA 憑證：真實模式另有 broker.ca_expiry_days() 向券商查詢（更權威）；
#    但模擬模式查不到（CaError: CA not activated，2026-07-30 實測），
#    故 G1 期間本表是唯一可見性。續期後記得同步更新此處日期。
EXPIRY_DATES: dict[str, dt.date] = {
    "永豐 CA 憑證":     dt.date(2028, 7, 15),
    "Shioaji API 金鑰": dt.date(2027, 7, 15),
    "Vultr 信用卡":     dt.date(2031, 10, 31),   # 卡片於當月月底失效
}
EXPIRY_WARN_DAYS = 60        # ≤60 天 → WARNING（每週一推播，避免告警疲勞）
EXPIRY_ALERT_DAYS = 14       # ≤14 天 → ALERT（每日推播）；已過期 → CRITICAL

# ── API 重試（§10）───────────────────────────────────────────────────────────
RETRY_MAX = 3                # 查詢類 API 重試上限（指數退避）；下單不盲重試
RETRY_BASE_SECS = 1.0
