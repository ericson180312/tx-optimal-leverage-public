"""口數計算（規格 §4）：整數口數四捨五入，同一 regime 內另加遲滯帶。

原設計認為整數口數天然構成 deadband，這只在遠離 N ± 0.5 時成立：raw 貼著邊界時，
價格每穿越一次就加減一口；BEAR（L=1）單向下跌更會隔日加減（README §2.2）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from d10_trading import config


class ContractsCapExceeded(RuntimeError):
    """目標/下單口數超過 MAX_CONTRACTS 防呆上限 → 拒單 + 告警（規格 §4）。"""


@dataclass
class SizingResult:
    target_leverage: float
    equity: float
    price: float
    raw_contracts: float          # L×E / (價格×乘數)，未捨入
    target_contracts: int
    rounding_residual: float      # raw − target（捨入方向與殘餘偏差，落日誌）
    achieved_leverage: float      # target×價格×乘數 / E（實際落點槓桿）
    band_hold: bool = False       # 四捨五入會改口數，但仍在遲滯帶內而維持現有口數


def target_contracts(target_leverage: float, equity: float, price: float,
                     *, multiplier: float = config.MULTIPLIER,
                     held: int | None = None, band: float = 0.0) -> SizingResult:
    """目標口數 = round(L_target × E / (價格 × 乘數))，可選遲滯帶。

    捨入採「四捨五入（half-up）」，非 Python 內建的銀行家捨入——
    0.5 邊界行為必須確定且可測試（規格範例 1.5 口 → 2 口）。

    held／band：給定現有口數 held（>0）與帶寬 band 時，raw 仍落在
    [held − 0.5 − band, held + 0.5 + band) 就維持 held，跨出才照四捨五入重算。
    band=0（預設）或 held 為 None／0 時與純四捨五入完全相同。是否適用
    （同一 regime、非保險絲）由呼叫端決定，本函式不判斷。
    """
    if equity <= 0:
        raise ValueError(f"權益數異常：{equity}")
    if price <= 0:
        raise ValueError(f"價格異常：{price}")
    if band < 0:
        raise ValueError(f"遲滯帶寬不可為負：{band}")
    raw = target_leverage * equity / (price * multiplier)
    n = math.floor(raw + 0.5)
    hold = (held is not None and held > 0 and band > 0 and n != held
            and held - 0.5 - band <= raw < held + 0.5 + band)
    if hold:
        n = held
    return SizingResult(
        target_leverage=target_leverage, equity=equity, price=price,
        raw_contracts=raw, target_contracts=n, rounding_residual=raw - n,
        achieved_leverage=n * price * multiplier / equity, band_hold=hold,
    )


def check_caps(target: int, order_qty: int) -> None:
    """防呆（不是策略）：目標部位或單次下單口數超過上限 → 拒單。"""
    if target > config.MAX_CONTRACTS:
        raise ContractsCapExceeded(
            f"目標口數 {target} > MAX_CONTRACTS {config.MAX_CONTRACTS}")
    if abs(order_qty) > config.MAX_CONTRACTS:
        raise ContractsCapExceeded(
            f"下單口數 {abs(order_qty)} > MAX_CONTRACTS {config.MAX_CONTRACTS}")


def capacity_headroom(equity: float, price: float, *,
                      multiplier: float = config.MULTIPLIER,
                      max_contracts: int = config.MAX_CONTRACTS,
                      l_max: float = config.L_HIGH) -> tuple[float, float]:
    """MAX_CONTRACTS 上限的權益容量與目前餘裕比例。

    上限在 raw ≥ max_contracts + 0.5 時被觸及（half-up 捨入），故
        E_bind = (max_contracts + 0.5) × price × multiplier ÷ l_max
    以 **L_HIGH**（策略可能要求的最高槓桿）計算，而非當下的 l_exec——
    BEAR 期間 L 較低會使餘裕看似寬鬆，但 regime 一翻轉上限就可能立即綁定。

    回傳 (E_bind, headroom)；headroom = 1 − equity/E_bind，可為負（已超過）。
    """
    e_bind = (max_contracts + 0.5) * price * multiplier / l_max
    return e_bind, 1.0 - equity / e_bind


def current_leverage(contracts: int, price: float, equity: float,
                     *, multiplier: float = config.MULTIPLIER) -> float:
    """對帳後的現時槓桿（保險絲檢查用）。"""
    if equity <= 0:
        return math.inf if contracts > 0 else 0.0
    return contracts * price * multiplier / equity


def fuse_triggered(lev_current: float) -> bool:
    """結構保險絲（規格 §7）：L_current ≥ FUSE_L → 立即減碼回 L_target。"""
    return lev_current >= config.FUSE_L
