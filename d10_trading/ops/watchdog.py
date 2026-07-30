"""到期哨兵：外部到期事項的提前預警（規格 §7 可觀測性）。

檢查兩類對象：
1. **假日檔涵蓋年份**——自 `data_static/taifex_holidays.csv` 的 covered_years
   自動推導，無需人工登錄。這是唯一「到期即 fail-closed、系統完全不跑」的
   項目（calendar.CalendarCoverageError），故最需要提前預警。
   期限視為涵蓋末年的 12/31；跨年後首個交易日就會中斷。
2. **config.EXPIRY_DATES**——人工登錄的外部到期（CA 憑證、API 金鑰、信用卡）。
   這些到期時系統多半是「靜默失效」（下單被拒、主機被停），不會自我察覺。

分級（依剩餘天數）：
    > WARN            靜默
    ≤ WARN (60)       WARNING —— 每週一推播，其餘日子僅入摘要與稽核（防告警疲勞）
    ≤ ALERT (14)      ALERT   —— 每日推播
    < 0（已過期）      CRITICAL —— 每日推播

CA 憑證在真實模式另有 `broker.ca_expiry_days()` 向券商即時查詢（更權威）；
模擬模式查不到（實測 CaError: CA not activated），故 G1 期間本哨兵是唯一
可見性。兩者並存不衝突——一個是即時真值，一個是人工登錄的提前提醒。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from d10_trading import config
from d10_trading.data import calendar

HOLIDAY_ITEM = "TAIFEX 假日檔涵蓋年份"


@dataclass
class ExpiryItem:
    name: str
    deadline: dt.date
    days_left: int
    level: str                # WARNING / ALERT / CRITICAL
    note: str = ""

    def message(self) -> str:
        if self.days_left < 0:
            head = f"{self.name} 已於 {self.deadline} 到期（逾期 {-self.days_left} 天）"
        else:
            head = f"{self.name} 將於 {self.deadline} 到期（剩 {self.days_left} 天）"
        return f"{head}——{self.note}" if self.note else head


def _level(days_left: int) -> str | None:
    if days_left < 0:
        return "CRITICAL"
    if days_left <= config.EXPIRY_ALERT_DAYS:
        return "ALERT"
    if days_left <= config.EXPIRY_WARN_DAYS:
        return "WARNING"
    return None


def holiday_deadline() -> dt.date | None:
    """假日檔的有效期限＝涵蓋末年的 12/31；讀不到檔案回 None。"""
    try:
        years = calendar.covered_years()
    except Exception:  # noqa: BLE001 - 哨兵故障不得影響交易
        return None
    return dt.date(max(years), 12, 31) if years else None


def check_expiries(today: dt.date) -> list[ExpiryItem]:
    """回傳需要提醒的到期項目（未到預警窗者不列入），最急迫者排前。"""
    items: list[ExpiryItem] = []

    deadline = holiday_deadline()
    if deadline is not None:
        days = (deadline - today).days
        lv = _level(days)
        if lv:
            items.append(ExpiryItem(
                HOLIDAY_ITEM, deadline, days, lv,
                note=(f"請依 TAIFEX 公告更新 {config.HOLIDAY_FILE.name} 的 "
                      f"{deadline.year + 1} 年假日與 covered_years，"
                      "否則次年首個交易日起程式將 fail-closed 完全不執行")))

    for name, deadline in config.EXPIRY_DATES.items():
        days = (deadline - today).days
        lv = _level(days)
        if lv:
            items.append(ExpiryItem(name, deadline, days, lv))

    return sorted(items, key=lambda x: x.days_left)


def should_push(item: ExpiryItem, today: dt.date) -> bool:
    """是否推播通知。WARNING 僅每週一推（防疲勞）；ALERT/CRITICAL 每日推。"""
    if item.level in ("ALERT", "CRITICAL"):
        return True
    return today.weekday() == 0        # 週一
