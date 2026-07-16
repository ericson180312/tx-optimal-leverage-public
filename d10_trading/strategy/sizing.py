"""口數計算（規格 §4）：整數口數天然構成 deadband，不另設參數。"""
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


def target_contracts(target_leverage: float, equity: float, price: float,
                     *, multiplier: float = config.MULTIPLIER) -> SizingResult:
    """目標口數 = round(L_target × E / (價格 × 乘數))。

    捨入採「四捨五入（half-up）」，非 Python 內建的銀行家捨入——
    0.5 邊界行為必須確定且可測試（規格範例 1.5 口 → 2 口）。
    """
    if equity <= 0:
        raise ValueError(f"權益數異常：{equity}")
    if price <= 0:
        raise ValueError(f"價格異常：{price}")
    raw = target_leverage * equity / (price * multiplier)
    n = math.floor(raw + 0.5)
    return SizingResult(
        target_leverage=target_leverage, equity=equity, price=price,
        raw_contracts=raw, target_contracts=n, rounding_residual=raw - n,
        achieved_leverage=n * price * multiplier / equity,
    )


def check_caps(target: int, order_qty: int) -> None:
    """防呆（不是策略）：目標部位或單次下單口數超過上限 → 拒單。"""
    if target > config.MAX_CONTRACTS:
        raise ContractsCapExceeded(
            f"目標口數 {target} > MAX_CONTRACTS {config.MAX_CONTRACTS}")
    if abs(order_qty) > config.MAX_CONTRACTS:
        raise ContractsCapExceeded(
            f"下單口數 {abs(order_qty)} > MAX_CONTRACTS {config.MAX_CONTRACTS}")


def current_leverage(contracts: int, price: float, equity: float,
                     *, multiplier: float = config.MULTIPLIER) -> float:
    """對帳後的現時槓桿（保險絲檢查用）。"""
    if equity <= 0:
        return math.inf if contracts > 0 else 0.0
    return contracts * price * multiplier / equity


def fuse_triggered(lev_current: float) -> bool:
    """結構保險絲（規格 §7）：L_current ≥ FUSE_L → 立即減碼回 L_target。"""
    return lev_current >= config.FUSE_L
