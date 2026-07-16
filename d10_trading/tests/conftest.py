"""測試基礎設施：FakeBroker（不觸 Shioaji）、合成合約資料、整合 harness。"""
from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # backtest_multimarket / live 皆自專案根 import

from d10_trading import config                     # noqa: E402
from d10_trading.data import calendar              # noqa: E402
from d10_trading.execution.broker import (         # noqa: E402
    BrokerInterface, ContractInfo, OrderHandle, OrderStatusSnapshot,
    PositionInfo, Quote,
)


# ── 合成合約長表 ─────────────────────────────────────────────────────────────

def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    out, cur = [], start
    while cur <= end:
        if calendar.is_trading_day(cur):
            out.append(cur)
        cur += dt.timedelta(days=1)
    return out


def synth_cdf(price_fn, start: dt.date, end: dt.date,
              basis: float = 0.0) -> pd.DataFrame:
    """合成 TX 逐合約長表：每月合約自前兩個月月初掛牌至結算日。

    price_fn(date)->float 為近月價格路徑；次月 = 近月 + basis。
    """
    rows = []
    days = trading_days(start, end)
    months = sorted({(d.year, d.month) for d in days}
                    | {_add_month(d.year, d.month, k) for d in days for k in (1, 2)})
    for (y, m) in months:
        expiry = calendar.settlement_date(y, m)
        y0, m0 = _add_month(y, m, -2)
        listed_from = dt.date(y0, m0, 1)
        for d in days:
            if listed_from <= d <= expiry:
                rows.append({"date": pd.Timestamp(d),
                             "contract": f"{y:04d}{m:02d}",
                             "price": float(price_fn(d)) + basis * _month_rank(d, y, m),
                             "volume": 1000})
    return pd.DataFrame(rows).sort_values(["date", "contract"]).reset_index(drop=True)


def _add_month(y: int, m: int, k: int) -> tuple[int, int]:
    v = (y * 12 + (m - 1)) + k
    return v // 12, v % 12 + 1


def _month_rank(d: dt.date, y: int, m: int) -> int:
    return (y * 12 + m) - (d.year * 12 + d.month)


# ── FakeBroker ───────────────────────────────────────────────────────────────

@dataclass
class _FakeOrder:
    handle: OrderHandle
    fills_left: list          # 依 refresh 次數逐步吐出的 (status, deal_qty, price)
    cancelled: bool = False
    final: OrderStatusSnapshot | None = None
    partial: tuple | None = None    # (qty, px) 部分成交暫存（撤單時定案入帳）


class FakeBroker(BrokerInterface):
    """腳本化券商：fill_mode = fill / never / partial:<n> / fail。"""

    simulation = True

    def __init__(self, *, equity: float, contracts: list[ContractInfo],
                 quotes: dict[str, Quote], positions: list[PositionInfo] = None,
                 closes: dict[tuple, float] = None, fill_mode: str = "fill",
                 combo_fill_mode: str | None = None, combo_raises: bool = False):
        self._equity = equity
        self._contracts = contracts
        self.quotes = quotes
        self._positions = list(positions or [])
        self.closes = closes or {}
        self.fill_mode = fill_mode
        self.combo_fill_mode = combo_fill_mode or fill_mode
        self.combo_raises = combo_raises
        self.placed: list[OrderHandle] = []
        self.login_count = 0

    # ── 連線/查詢 ───────────────────────────────────────────────────────────
    def login(self) -> None:
        self.login_count += 1

    def logout(self) -> None:
        pass

    def monthly_contracts(self, product: str) -> list[ContractInfo]:
        return sorted([c for c in self._contracts if c.product == product],
                      key=lambda c: c.delivery_month)

    def quote(self, code: str) -> Quote:
        return self.quotes[code]

    def day_session_close(self, product, month, date) -> float | None:
        return self.closes.get((product, month, date))

    def equity(self) -> float:
        return self._equity

    def positions(self) -> list[PositionInfo]:
        return list(self._positions)

    # ── 下單（腳本化成交）───────────────────────────────────────────────────
    def _mk_order(self, handle: OrderHandle, mode: str) -> OrderHandle:
        if mode == "fill":
            script = [("Filled", handle.quantity, handle.price)]
        elif mode == "never":
            script = []
        elif mode == "fail":
            script = [("Failed", 0, 0.0)]
        elif mode.startswith("partial:"):
            n = int(mode.split(":")[1])
            script = [("PartFilled", n, handle.price)]
        else:
            raise ValueError(mode)
        handle.raw = _FakeOrder(handle, script)
        self.placed.append(handle)
        return handle

    def place_limit(self, code, action, quantity, price) -> OrderHandle:
        h = OrderHandle(code=code, action=action, quantity=quantity, price=price)
        self._mk_order(h, self.fill_mode)
        return h

    def place_combo_roll(self, sell_code, buy_code, quantity, net_price):
        if self.combo_raises:
            raise RuntimeError("combo not supported")
        h = OrderHandle(code=f"{sell_code}->{buy_code}", action="Roll",
                        quantity=quantity, price=net_price, is_combo=True)
        self._mk_order(h, self.combo_fill_mode)
        return h

    def update_price(self, handle, price) -> None:
        handle.price = price

    def cancel(self, handle) -> None:
        handle.raw.cancelled = True

    def refresh(self, handle) -> OrderStatusSnapshot:
        fo: _FakeOrder = handle.raw
        if fo.final is not None:                   # 終態只入帳一次
            return fo.final
        if fo.fills_left:
            status, qty, px = fo.fills_left.pop(0)
            if status == "Filled":
                self._apply_fill(handle, qty, px)
                fo.final = OrderStatusSnapshot("Filled", qty, px, handle.price)
                return fo.final
            if status == "Failed":
                fo.final = OrderStatusSnapshot("Failed", 0, 0.0, handle.price,
                                               msg="scripted failure")
                return fo.final
            if status == "PartFilled":
                fo.partial = (qty, px)
        qty, px = fo.partial or (0, 0.0)
        if fo.cancelled:                           # 撤單定案（部分成交入帳）
            if qty:
                self._apply_fill(handle, qty, px)
            fo.final = OrderStatusSnapshot("Cancelled", qty, px, handle.price)
            return fo.final
        return OrderStatusSnapshot("PartFilled" if qty else "Submitted",
                                   qty, px, handle.price)

    def _apply_fill(self, handle: OrderHandle, qty: int, px: float) -> None:
        """把成交反映到持倉（多單邏輯，符合本策略）。"""
        if handle.is_combo:
            sell_code, buy_code = handle.code.split("->")
            self._reduce(sell_code, qty)
            self._increase(buy_code, qty, px)
            return
        if handle.action == "Buy":
            self._increase(handle.code, qty, px)
        else:
            self._reduce(handle.code, qty)

    def _increase(self, code, qty, px) -> None:
        for p in self._positions:
            if p.code == code:
                p.avg_price = (p.avg_price * p.quantity + px * qty) / (p.quantity + qty)
                p.quantity += qty
                return
        self._positions.append(PositionInfo(code=code, direction="Buy",
                                            quantity=qty, avg_price=px))

    def _reduce(self, code, qty) -> None:
        for p in self._positions:
            if p.code == code:
                p.quantity -= qty
        self._positions = [p for p in self._positions if p.quantity > 0]


# ── 共用 fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_var(tmp_path, monkeypatch):
    """每個測試用獨立 var 目錄（狀態/日誌/快取互不污染），並關閉輪詢延遲。"""
    monkeypatch.setattr(config, "VAR_DIR", tmp_path / "var")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "var" / "cache")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "var" / "logs")
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "var" / "state.json")
    monkeypatch.setattr(config, "KILL_SWITCH_FILE", tmp_path / "KILL_SWITCH")
    monkeypatch.setattr(config, "ORDER_TIMEOUT_SECS", 0.05)
    monkeypatch.setattr(config, "ORDER_POLL_SECS", 0)
    # 測試絕不觸網：清空通知設定（使用者的 d10_trading/.env 可能已填真實 token）
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", "")
    monkeypatch.setattr(config, "HEALTHCHECK_URL", "")
    monkeypatch.setattr(config, "SJ_PERSON_ID", "")
    yield


def make_contracts(product: str, months: list[str]) -> list[ContractInfo]:
    out = []
    for m in months:
        y, mm = int(m[:4]), int(m[4:])
        out.append(ContractInfo(code=f"{product}{m}", product=product,
                                delivery_month=m,
                                delivery_date=calendar.settlement_date(y, mm)))
    return out


def flat_quote(code: str, px: float) -> Quote:
    return Quote(code=code, price=px, bid=px - 1, ask=px + 1)
