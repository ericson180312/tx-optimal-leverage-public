"""第五輪參數（規格 §1、§3）——除 L_high 外一切凍結，沿用第四輪確認的 D_10 設定。"""
from __future__ import annotations

from pathlib import Path

from backtest_multimarket.params import (           # noqa: F401 沿用既有口徑
    MARKET_LABEL, PERIOD_YEARS, RANKING_MARKETS, TRADING_DAYS, TREND_BUFFER,
    TREND_CHECK, TREND_MA,
)

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = OUTPUT_DIR / "cache"

# ── 掃描網格（規格 §1；本輪唯一變數）──────────────────────────────────────
L_HIGH_GRID = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]   # 多頭 regime 槓桿
L_LOW       = 1.0                               # 固定（D_10 定義）
# 次要檢查（不進主表）：等比縮放版，區分「只加多頭曝險」vs「整體放大」
SCALED_L_LOW = {lh: lh / 2.0 for lh in L_HIGH_GRID}

# 凍結參數（第四輪確認；不得更動）：MA=200、buffer=1%、週頻訊號、T+1、成本模型
# 直接沿用 backtest_multimarket.params 的 TREND_MA / TREND_BUFFER / TREND_CHECK。

# ── 保證金與強制平倉（規格 §3）────────────────────────────────────────────
MAINT_MARGIN_RATE = 0.035        # 維持保證金 ÷ 契約價值（TAIFEX 量級，以當期公告為準）
# 敏感度：維持保證金率是本輪最吃重的假設，掃一組看「約束何時真正咬到」
MAINT_RATE_SENS   = [0.035, 0.07, 0.10, 0.15]
LIMIT_DOWN_PCT    = 0.10         # 台股單日漲跌幅限制（TX 專屬，§3-2）
MAX_LIMIT_DAYS    = 12           # 連續跌停模擬的最大天數
RUIN_VETO_PROB    = 0.01         # §3-3 一票否決線：bootstrap 歸零機率 > 1% → 不合格

# ── bootstrap（規格 §2-5，沿用第三／四輪設定）─────────────────────────────
BOOT_N_PATHS = 1000
BOOT_YEARS   = 20
BOOT_BLOCK   = 60
BOOT_SEED    = 20260728

# ── 命名 ──────────────────────────────────────────────────────────────────

def variant_key(l_high: float, scaled: bool = False) -> str:
    """L_high 變體鍵名（以 D_ 開頭 → 相容既有 signal 路由）。"""
    tag = "S" if scaled else ""
    return f"D_L{int(round(l_high * 100)):03d}{tag}"


def variant_label(l_high: float, scaled: bool = False) -> str:
    return f"L_high={l_high:.1f}" + ("（等比 L_low=L/2）" if scaled else "")
