"""轉倉時點判定（規格 §6，第一輪回測實測結論：regime 自適應時點）。

規則：
  結算日 = 近月合約最後交易日（Shioaji delivery_date 為準，第三個週三＋假日順延為互驗）
  月差   = 次月報價 − 近月報價（執行當下的即時報價）
  轉倉日 = 結算日 − ROLL_LEAD_DAYS 交易日  if 月差 > 0  else 結算日

安全補課（規格未明列、fail-safe 必要）：月差在倒數期間由 ≤0 轉正時，
規則上的轉倉日（T−5）已過——此時**當日立即轉倉**並記錄 late_roll，
絕不把近月部位帶進結算日之後。規格 §11 精神：這不是新策略規則，
是「不得持倉進入現金結算」的完備化；已在 README 明列供審查。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from live import config
from live.data import calendar


@dataclass
class RollDecision:
    should_roll: bool
    reason: str                 # "early_roll" | "expiry_roll" | "late_roll" | ""
    settlement: dt.date
    early_roll_date: dt.date    # 結算日 − ROLL_LEAD_DAYS 交易日
    spread: float               # 次月 − 近月


def decide_roll(today: dt.date, settlement: dt.date, spread: float,
                holding_near: bool) -> RollDecision:
    """今日是否執行轉倉（僅在持有近月部位時可能為 True）。"""
    early = calendar.shift_trading_days(settlement, -config.ROLL_LEAD_DAYS)
    dec = RollDecision(False, "", settlement, early, spread)
    if not holding_near or today > settlement:
        return dec
    if spread > 0:
        if today == early:
            dec.should_roll, dec.reason = True, "early_roll"
        elif early < today <= settlement:
            dec.should_roll, dec.reason = True, "late_roll"   # 月差轉正晚於 T−5
    elif today == settlement:
        dec.should_roll, dec.reason = True, "expiry_roll"
    return dec
