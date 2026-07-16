"""開場對帳（每次執行的第一步，規格 §7）。

API 查回的實際部位 vs 狀態檔：
- 不一致（手動干預、前日部分成交未記錄…）→ ReconcileMismatch → 告警 + 本日停止。
  **絕不自動「修正」成程式以為的樣子**；人工確認後用 --adopt-state 重置狀態。
- 策略只允許「本商品的多單」；出現空單或其他商品 → 一律視為不一致。
"""
from __future__ import annotations

from dataclasses import dataclass

from live import config
from live.execution.broker import BrokerInterface, PositionInfo
from live.ops.state import PositionState, State


class ReconcileMismatch(RuntimeError):
    pass


@dataclass
class ReconcileResult:
    equity: float
    positions: list[PositionInfo]      # 過濾後（本商品）的實際持倉
    total_qty: int
    held_months: list[str]             # 由合約代碼解析出的 YYYYMM（升冪）


def _fmt(positions) -> str:
    if not positions:
        return "（空）"
    return ", ".join(f"{p.code} x{p.qty if hasattr(p, 'qty') else p.quantity}"
                     for p in positions)


def reconcile(broker: BrokerInterface, state: State,
              month_of_code) -> ReconcileResult:
    """對帳；month_of_code(code)->YYYYMM 由 execution 層提供（查合約檔）。"""
    equity = broker.equity()
    all_pos = broker.positions()

    mine, alien = [], []
    for p in all_pos:
        if p.code.startswith(config.PRODUCT) and p.quantity > 0:
            mine.append(p)
        elif p.quantity > 0:
            alien.append(p)

    if alien:
        raise ReconcileMismatch(
            f"帳戶存在非 {config.PRODUCT} 部位：{_fmt(alien)}——"
            "本系統不代管其他部位，請人工處理後以 --adopt-state 重置")
    for p in mine:
        if p.direction != "Buy":
            raise ReconcileMismatch(
                f"存在空單 {p.code} x{p.quantity}——策略只允許多單，請人工確認")

    actual = sorted((p.code, p.quantity) for p in mine)
    expected = sorted((p.code, p.qty) for p in state.positions)
    if actual != expected:
        raise ReconcileMismatch(
            f"部位不一致：API 實際 {_fmt(mine)} vs 狀態檔 "
            f"{_fmt(state.positions)}——告警停止，等人工確認（--adopt-state 重置）")

    months = sorted({month_of_code(p.code) for p in mine})
    return ReconcileResult(equity=equity, positions=mine,
                           total_qty=sum(p.quantity for p in mine),
                           held_months=months)


def adopt(broker: BrokerInterface, state: State, month_of_code) -> State:
    """人工確認後，把 API 實際部位採納為狀態（手動接管程序用）。"""
    equity = broker.equity()
    mine = [p for p in broker.positions()
            if p.code.startswith(config.PRODUCT) and p.quantity > 0]
    state.positions = [PositionState(code=p.code, month=month_of_code(p.code),
                                     qty=p.quantity, avg_price=p.avg_price)
                       for p in mine]
    state.equity = equity
    state.notes = "adopt-state：以 API 實際部位重置"
    return state
