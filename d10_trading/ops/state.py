"""狀態檔（JSON，原子寫入）：程式重啟必須無損恢復（規格 §7）。

欄位：
  regime / exec_leverage      最近一次「已定案」訊號（上一官方訊號日）
  signal_date                 該訊號對應的官方資料日
  preview                     今日暫定訊號（明日以官方收盤定案）{date, leverage}
  executed_date               本日已執行旗標（冪等性 🔴 的第一道防線）
  positions                   持倉明細 [{code, month, qty, avg_price}]（僅多單）
  equity                      最近一次查得的權益數
  last_run / notes            稽核輔助
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field

from d10_trading import config


@dataclass
class PositionState:
    code: str
    month: str            # YYYYMM
    qty: int
    avg_price: float = 0.0


@dataclass
class State:
    regime: str | None = None
    exec_leverage: float | None = None
    signal_date: str | None = None
    preview: dict | None = None          # {"date": ..., "leverage": ...}
    executed_date: str | None = None
    positions: list[PositionState] = field(default_factory=list)
    equity: float | None = None
    last_run: str | None = None
    notes: str = ""
    # 待補記的「執行價 vs 官方收盤價」稽核（規格 §2）；次日官方資料到位後補寫
    pending_exec_audit: list = field(default_factory=list)

    # ── 便利 ────────────────────────────────────────────────────────────────
    def executed_today(self, today: dt.date) -> bool:
        return self.executed_date == today.isoformat()

    def total_qty(self) -> int:
        return sum(p.qty for p in self.positions)


def load() -> State:
    if not config.STATE_FILE.exists():
        return State()
    with open(config.STATE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    positions = [PositionState(**p) for p in raw.pop("positions", [])]
    known = {k: raw.get(k) for k in State.__dataclass_fields__ if k in raw}
    known["positions"] = positions
    known["notes"] = raw.get("notes", "")
    return State(**known)


def save(state: State) -> None:
    """原子寫入：temp 檔 + os.replace，中途斷電不會留下半寫狀態。"""
    config.VAR_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(state)
    fd, tmp = tempfile.mkstemp(dir=config.VAR_DIR, suffix=".state.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, config.STATE_FILE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
