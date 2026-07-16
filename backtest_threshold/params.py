"""全域參數（任務規格 §8）— 所有可調參數集中於此檔。

任何資金／成本假設都寫成參數；報告開頭會列出實際採用值。
"""
from __future__ import annotations

from pathlib import Path

# ── 策略核心參數（規格 §8）────────────────────────────────────────────────
INITIAL_EQUITY     = 1_000_000
TARGET_LEVERAGE    = 2.0
X_GRID             = [2.2, 2.3, 2.4, 2.5, 2.6, 2.75, 3.0, 3.25, 3.5, 4.0]   # 減碼上限
Y_GRID             = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]     # 加碼下限
LOOKBACK_YEARS     = [1, 3, 5, 10, 20]
ROLL_OFFSET_DAYS   = 5          # 到期日前 N 個「交易日」轉倉；另跑 0 做敏感度
IDLE_CASH_RATE     = 0.017      # 閒置現金利率（年化，日頻複利）；亦作 Sharpe 的無風險利率
FEE_PER_SIDE       = 20.0       # 手續費（元／口／邊）
TAX_RATE           = 0.00002    # 期交稅：名目金額 × 十萬分之二，每邊
SLIPPAGE_TICKS     = 1.0        # 滑價（檔／邊）；另跑 0.5 做敏感度
ROLLING_START_FREQ = "QS"       # 滾動起點間隔（季初，pandas freq）

# ── 敏感度情境 ──────────────────────────────────────────────────────────────
SLIPPAGE_TICKS_SENS   = 0.5     # 成本敏感度：滑價 0.5 檔
ROLL_OFFSET_DAYS_SENS = 0       # 轉倉時點敏感度：最後交易日當天轉倉

# ── 合約規格（TX 台指期）────────────────────────────────────────────────────
MULTIPLIER   = 200.0            # 每點 200 元
TICK_POINTS  = 1.0              # 最小跳動 1 點
TICK_VALUE   = MULTIPLIER * TICK_POINTS   # 1 檔 = 200 元

# ── 年化基準 ────────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
# 日頻複利：(1+年利率)^(1/252) - 1
DAILY_RF = (1.0 + IDLE_CASH_RATE) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

# ── 資料層 ──────────────────────────────────────────────────────────────────
FINMIND_API        = "https://api.finmindtrade.com/api/v4/data"
FUTURES_ID         = "TX"
FUTURES_FIRST_YEAR = 1998       # TX 1998/07/21 上市
BASE_DIR   = Path(__file__).resolve().parent          # backtest_threshold/
DATA_DIR   = BASE_DIR.parent / "data"                 # repo 根目錄的共用 FinMind 快取
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = OUTPUT_DIR / "cache"
# 既有專案的 parquet 快取（存在就直接複製，不重抓）
SIBLING_CACHE_DIR = Path(r"C:\Users\User\Desktop\python_台指跨月價差回測\analysis\roll_carry\data")

# ── 基準組（以引擎參數表達）────────────────────────────────────────────────
# X=+inf / Y=-inf 表示門檻永不觸發；rebal_every=N 表示每 N 個交易日無條件調回目標。
import math
_INF = math.inf
# 週期再平衡的間隔（交易日）：5=正2倍ETF改為5日再平衡（本次新增的主角）；
# 10/21/63 一併跑，用來看「再平衡頻率」這個維度的完整形狀（每日 = Daily_2x）。
REBAL_FREQ_DAYS = [5, 10, 21, 63]
BENCHMARKS = {
    "BH_2x":    {"target": 2.0, "X": _INF, "Y": -_INF, "rebal_every": 0},  # 買進持有 2x，永不再平衡
    "Daily_2x": {"target": 2.0, "X": 2.0,  "Y": 2.0,   "rebal_every": 0},  # 每日再平衡（= 正2倍ETF）
    **{f"Periodic{n}d_2x": {"target": 2.0, "X": _INF, "Y": -_INF, "rebal_every": n}
       for n in REBAL_FREQ_DAYS},                                          # 每 n 個交易日再平衡回 2x
    "BH_1x":    {"target": 1.0, "X": _INF, "Y": -_INF, "rebal_every": 0},  # 買進持有 1x（無槓桿）
}
# 資金曲線疊圖只畫這幾條（其餘頻率僅進頻率掃描表，避免圖表過載）
PLOT_BENCHMARKS = ["BH_2x", "Daily_2x", "Periodic5d_2x", "BH_1x"]
# 5 日再平衡的「相位」敏感度：建倉日落在週期的哪一天（0..4）
REBAL_PHASE_SENS = [0, 1, 2, 3, 4]
