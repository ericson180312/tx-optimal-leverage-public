"""第三輪全域參數（任務規格 §1–§8）— 所有可調假設集中於此檔。

參數紀律（規格 §2）：策略參數網格刻意保持小、四個市場強制同一組參數；
唯一的例外是 σ_target —— 它是「槓桿中性化」的校準變數（規格 §3），
按市場以最早 10 年校準（walk-forward），其後凍結，不屬於績效調參。
"""
from __future__ import annotations

import math
from pathlib import Path

# ── 基本 ────────────────────────────────────────────────────────────────────
INITIAL_EQUITY = 1_000_000.0
TARGET_LEV     = 2.0
DEADBAND       = 0.1          # 曝險調整門檻：|L_des − L_cur| > 0.1 才執行（規格 §2）
TRADING_DAYS   = 252

BASE_DIR   = Path(__file__).resolve().parent          # round3/
ROOT_DIR   = BASE_DIR.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = OUTPUT_DIR / "cache"

# ── 市場（規格 §1）──────────────────────────────────────────────────────────
# TX_FUT：沿用既有逐合約期貨管線（carry 內生）；TX_IDX：TAIEX 報酬指數＋融資模型，
# 兩者差異即 carry 貢獻（海外簡化法的校準參照）。maximin 主排名用 RANKING_MARKETS
# （TX 只算期貨版，避免重複計入台股）。
MARKETS = ["TX_FUT", "TX_IDX", "SPX", "NKX", "HSI"]
RANKING_MARKETS = ["TX_FUT", "SPX", "NKX", "HSI"]
MARKET_LABEL = {
    "TX_FUT": "台股 TX（期貨管線）",
    "TX_IDX": "台股（報酬指數＋融資）",
    "SPX":    "S&P 500",
    "NKX":    "日經 225",
    "HSI":    "恆生指數",
}
YAHOO_SYMBOLS = {          # 價格指數（現貨，供 200MA 訊號與 TR 拼接）
    "SPX": "^GSPC", "NKX": "^N225", "HSI": "^HSI",
}
YAHOO_TR_SYMBOLS = {"SPX": "^SP500TR"}   # 真實總報酬指數（可得者）

# ── 融資成本模型（指數槓桿模擬，規格 §1）───────────────────────────────────
FINANCING_SPREAD = 0.003     # 超過 1 倍部分：當地短率 + 30bp（年化，日頻計提）

# 短率：美國用 Ken French 日頻 RF（1926-07 起，真實序列）；日本／香港無免費長序列
# （FRED 於本機網路不通），以下列「分段常數」近似政策／拆款利率，報告揭露。
# 格式：[(自該年起, 年化利率), ...]
KENFRENCH_FACTORS_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
                         "ftp/F-F_Research_Data_Factors_daily_CSV.zip")
SHORT_RATE_STEPS = {
    "NKX": [(1965, .055), (1973, .080), (1978, .040), (1980, .065), (1983, .050),
            (1987, .025), (1990, .060), (1992, .035), (1995, .005), (1999, .001),
            (2006, .003), (2009, .001), (2016, .000), (2024, .003)],
    "HSI": [(1987, .070), (1997, .075), (1999, .055), (2003, .015), (2006, .040),
            (2009, .004), (2022, .035), (2023, .045)],
}
TW_SHORT_RATE = 0.017        # 台灣：沿用前兩輪 1.7% 常數（閒置現金利率／無風險利率）

# 股息率近似（僅用於「無真實 TR」的區段：價格指數＋股息率，規格 §1）
# S&P 1988 起有 ^SP500TR 真實 TR；之前用 Shiller 口徑的年代平均近似。
DIV_YIELD_STEPS = {
    "SPX": [(1927, .053), (1940, .059), (1950, .047), (1960, .032), (1970, .041),
            (1980, .040)],                             # 1988+ 改用真實 TR
    "NKX": [(1965, .040), (1970, .020), (1980, .010), (1990, .008), (2000, .012),
            (2010, .018), (2020, .022)],
    "HSI": [(1987, .035), (2000, .033), (2010, .035), (2020, .038)],
}

# ── 策略家族（規格 §2）──────────────────────────────────────────────────────
VT_CLIP        = (0.5, 3.0)   # vol targeting 槓桿上下限
EWMA_LAMBDA    = 0.94
VT_SD_WINDOWS  = [20, 60]     # 敏感度：滾動標準差視窗
TREND_MA       = 200
TREND_BUFFER   = 0.01         # ±1% 緩衝帶
TREND_CHECK    = 5            # 訊號週頻檢查（每 5 個交易日）
TREND_L_LOW    = [0.5, 1.0]
CPPI_DMAX      = [0.4, 0.5]   # m = 2 / D_max（HWM 處初始槓桿恰為 2.0）
CPPI_CAP       = 3.0
FUSE_X, FUSE_Y = 3.5, 1.1     # B2 保險絲（前輪 maximin 冠軍）
B1_REBAL_EVERY = 5            # B1 恆定 2x：每 5 日再平衡

# 主排名策略集（C 的 SD 變體僅作參數敏感度／retention，不進 maximin）
STRATEGIES = ["B1_Const2x", "B2_Fuse", "B3_BH1x",
              "C_EWMA", "D_05", "D_10", "E_40", "E_50", "F1_VolFuse", "F2_VolTrend"]
SENS_STRATEGIES = ["C_SD20", "C_SD60"]
STRATEGY_LABEL = {
    "B1_Const2x":  "B1 恆定 2x（每5日再平衡）",
    "B2_Fuse":     "B2 保險絲 [3.5, 1.1]",
    "B3_BH1x":     "B3 買進持有 1x",
    "C_EWMA":      "C 波動率目標（EWMA λ=0.94）",
    "C_SD20":      "C' 波動率目標（SD20）",
    "C_SD60":      "C' 波動率目標（SD60）",
    "D_05":        "D 趨勢過濾 200MA（L_low=0.5）",
    "D_10":        "D 趨勢過濾 200MA（L_low=1.0）",
    "E_40":        "E CPPI（D_max=0.4, m=5）",
    "E_50":        "E CPPI（D_max=0.5, m=4）",
    "F1_VolFuse":  "F1 vol target＋保險絲[3.5]",
    "F2_VolTrend": "F2 min(vol target, 趨勢)",
}

# ── 槓桿中性化（規格 §3）────────────────────────────────────────────────────
CALIB_YEARS      = 10         # 最早 10 年為校準段
CALIB_TARGET_LEV = 2.0        # 校準目標：時間加權平均槓桿
CALIB_TOL        = 0.1        # ±0.1
SIGMA_TARGET_BOUNDS = (0.01, 1.50)   # bisection 搜尋域（年化 σ_target）

# ── 回測期間（規格 §4）──────────────────────────────────────────────────────
PERIOD_YEARS  = [3, 5, 10, 20]
WINDOW_STEP   = 63            # 滾動起點平移（交易日 ≈ 季頻）
# 具名壓力窗口：(名稱, 市場清單或 None=全市場, 起, 迄)
STRESS_WINDOWS = [
    ("日經1990–2003 長陰跌", ["NKX"],        "1990-01-01", "2003-04-30"),
    ("2000–02 網路泡沫",     ["SPX"],        "2000-03-24", "2002-10-09"),
    ("2008 金融海嘯",        None,           "2007-10-01", "2009-03-31"),
    ("2020 COVID 急跌V反",   None,           "2020-01-01", "2020-12-31"),
    ("2022 緩跌",            None,           "2022-01-01", "2022-12-31"),
]

# ── 區塊拔靴（規格 §5）──────────────────────────────────────────────────────
BOOT_N_PATHS    = 1000
BOOT_YEARS      = 20
BOOT_MEAN_BLOCK = 60
BOOT_BLOCK_SENS = [20, 120]
BOOT_SENS_PATHS = 400         # 敏感度情境的路徑數（只看分布中位數是否移動）
BOOT_SEED       = 20260715
RUIN_VETO_PROB  = 0.01        # 歸零機率 > 1% → 一票否決（規格 §8-4）

# ── 成本模型（規格 §6）──────────────────────────────────────────────────────
IDX_COST_PER_SIDE      = 0.0005    # 海外指數模擬：單邊 5bp（含滑價），按名目變動量
IDX_COST_PER_SIDE_SENS = 0.0010    # 敏感度 10bp
# TX 期貨沿用既有明細模型（根目錄 params.py：手續費 20 元/口/邊、期交稅 2e-5、
# 滑價 1 檔、每月轉倉）；bootstrap 合成路徑無合約行事曆，TX 亦以 5bp 名目成本近似
# （實測 TX 每邊成本 ≈ 2.5–3bp，5bp 屬保守），報告揭露。

# ── 指標口徑 ────────────────────────────────────────────────────────────────
DERISK_LEV = TARGET_LEV - DEADBAND    # 「降載狀態」定義：收盤槓桿 < 1.9

_INF = math.inf
