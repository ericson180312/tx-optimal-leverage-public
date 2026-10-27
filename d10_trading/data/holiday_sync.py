"""假日檔自動更新——兩條互補來源，皆為 **additive-only（絕不刪除既有條目）**。

1. **未來排定假日** ← TWSE 開休市日期 API
   交易所約每年秋季公告次年行事曆；到期哨兵進入預警窗後每日嘗試，抓到為止。
   🔴 官方年度表**不含臨時停市（颱風）**——2026-07-10 即是實測案例：TWSE 表裡
   沒有，卻確實休市。故若覆蓋寫入會刪掉該筆，導致 prev_trading_day 誤判、
   序列插入假的一天、訊號相位偏移。additive merge 是正確性要求，不是保守。

2. **臨時停市** ← FinMind 實際行情
   由 calendar.find_adhoc_closures() 自真實行情偵測（前後皆有行情佐證），
   為已發生且被市場資料驗證的日期，補登風險極低。

安全設計：任一健全性檢查不過即放棄寫入，維持現行 fail-closed 行為；
寫入採「暫存檔 + 原子取代」，中途失敗不會留下半寫的假日檔。
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import os
import tempfile
from dataclasses import dataclass, field

import requests

from d10_trading import config
from d10_trading.data import calendar

log = logging.getLogger("d10.holiday_sync")

_HEADERS = {"User-Agent": "Mozilla/5.0"}


class SanityError(RuntimeError):
    """抓回的資料未通過健全性檢查——放棄寫入。"""


@dataclass
class SyncResult:
    status: str                      # added / unavailable / failed / not_needed
    year: int | None = None
    added: list[dt.date] = field(default_factory=list)
    detail: str = ""

    def message(self) -> str:
        if self.status == "added":
            return (f"假日檔已自動加入 {self.year} 年 {len(self.added)} 天休市日"
                    "（來源：TWSE 開休市日期表）——請人工複核一次")
        if self.status == "failed":
            return f"假日檔自動更新失敗（{self.year}）：{self.detail}"
        return self.detail


# ── TWSE 抓取與解析 ─────────────────────────────────────────────────────────

def fetch_twse_year(year: int) -> dict[dt.date, str]:
    """TWSE 該年度「平日休市日」。查無該年度資料回空 dict（尚未公告）。

    解析規則（2025/2026 實測驗證）：
    - 名稱含「交易日」者是**有交易**的特殊日（如春節前最後交易日），排除。
    - 週末本就非交易日，排除（避免與 is_trading_day 的週末判斷重複）。
    - 只保留確實落在目標年度者——API 查無資料時會回退其他年度，必須濾掉。
    """
    r = requests.get(config.TWSE_HOLIDAY_API,
                     params={"response": "json", "date": f"{year}0101"},
                     headers=_HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("stat") != "ok":
        raise RuntimeError(f"TWSE 回應異常：{payload.get('stat')}")
    out: dict[dt.date, str] = {}
    for row in payload.get("data", []):
        day = dt.date.fromisoformat(str(row[0]))
        name = str(row[1]).strip()
        if day.year != year or day.weekday() >= 5 or "交易日" in name:
            continue
        out[day] = name
    return out


def sanity_check(year: int, holidays: dict[dt.date, str]) -> None:
    """健全性檢查——任一不過即 SanityError，維持 fail-closed。"""
    n = len(holidays)
    if not (config.HOLIDAY_MIN_DAYS <= n <= config.HOLIDAY_MAX_DAYS):
        raise SanityError(f"{year} 年平日休市 {n} 天，超出合理區間 "
                          f"{config.HOLIDAY_MIN_DAYS}–{config.HOLIDAY_MAX_DAYS}")
    bad = [d for d in holidays if d.year != year or d.weekday() >= 5]
    if bad:
        raise SanityError(f"含非目標年度或週末日期：{sorted(bad)[:3]}")
    # 春節必有連續 ≥3 個平日休市；缺此特徵幾乎可斷定抓到錯誤資料
    days = sorted(holidays)
    run = best = 1
    for prev, cur in zip(days, days[1:]):
        gap = (cur - prev).days
        run = run + 1 if gap == 1 or (gap == 3 and prev.weekday() == 4) else 1
        best = max(best, run)
    if best < 3:
        raise SanityError(f"{year} 年查無連續 3 日以上休市（春節），資料可疑")


# ── 假日檔讀寫（additive-only）───────────────────────────────────────────────

def _read_file() -> tuple[list[str], dict[dt.date, str], set[int]]:
    """回傳（前置註解行, {日期: 名稱}, covered_years）。"""
    comments: list[str] = []
    rows: dict[dt.date, str] = {}
    covered: set[int] = set()
    with open(config.HOLIDAY_FILE, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.strip().startswith("# covered_years:"):
                covered = {int(y) for y in s.split(":", 1)[1].split(",")}
                continue
            if s.strip().startswith("#"):
                comments.append(s)
                continue
            if not s.strip():
                continue
            row = next(csv.reader([s]))
            if row[0] == "date":
                continue
            rows[dt.date.fromisoformat(row[0])] = row[1] if len(row) > 1 else ""
    return comments, rows, covered


def _write_file(comments: list[str], rows: dict[dt.date, str],
                covered: set[int]) -> None:
    """原子寫入：暫存檔 + os.replace，中途失敗不留半寫檔案。"""
    lines = list(comments)
    lines.append(f"# covered_years: {','.join(str(y) for y in sorted(covered))}")
    lines.append("date,name")
    body = "\n".join(lines) + "\n"
    for d in sorted(rows):
        body += f"{d.isoformat()},{rows[d]}\n"
    d_dir = config.HOLIDAY_FILE.parent
    fd, tmp = tempfile.mkstemp(dir=d_dir, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(body)
        os.replace(tmp, config.HOLIDAY_FILE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    calendar.reload()


def merge(new_rows: dict[dt.date, str], *, cover_year: int | None = None
          ) -> list[dt.date]:
    """把 new_rows 併入假日檔（**只增不刪**），回傳實際新增的日期。

    既有條目一律保留原名稱——人工註記（如「臨時停市」）優先於官方名稱。
    """
    comments, rows, covered = _read_file()
    added = [d for d in sorted(new_rows) if d not in rows]
    for d in added:
        rows[d] = new_rows[d]
    cover_changed = cover_year is not None and cover_year not in covered
    if cover_year is not None:
        covered.add(cover_year)
    if added or cover_changed:
        _write_file(comments, rows, covered)
    return added


# ── 對外流程 ────────────────────────────────────────────────────────────────

def auto_sync_if_needed(today: dt.date) -> SyncResult:
    """涵蓋年份即將用盡（進入哨兵預警窗）時，嘗試自 TWSE 補下一年度。"""
    if not config.HOLIDAY_SYNC_ENABLED:
        return SyncResult("not_needed", detail="自動更新已停用")
    try:
        covered = calendar.covered_years()
    except Exception as e:  # noqa: BLE001
        return SyncResult("failed", detail=f"讀取假日檔失敗：{type(e).__name__}")
    if not covered:
        return SyncResult("failed", detail="假日檔無 covered_years")

    last = max(covered)
    deadline = dt.date(last, 12, 31)
    if (deadline - today).days > config.EXPIRY_WARN_DAYS:
        return SyncResult("not_needed")           # 尚未進入預警窗

    target = last + 1
    try:
        fetched = fetch_twse_year(target)
    except Exception as e:  # noqa: BLE001 - 網路/解析失敗不得影響交易
        return SyncResult("failed", year=target,
                          detail=f"TWSE 抓取失敗：{type(e).__name__}")
    if not fetched:
        return SyncResult("unavailable", year=target,
                          detail=f"TWSE 尚未公告 {target} 年行事曆")
    try:
        sanity_check(target, fetched)
    except SanityError as e:
        return SyncResult("failed", year=target, detail=str(e))

    added = merge(fetched, cover_year=target)
    log.info("假日檔自動更新：%s 年新增 %d 天", target, len(added))
    return SyncResult("added", year=target, added=added)


def record_adhoc_closures(actual_days: list[dt.date],
                          since: dt.date) -> list[dt.date]:
    """把實際行情證實的臨時停市補進假日檔（過去日期，只增不刪）。"""
    if not config.HOLIDAY_SYNC_ENABLED:
        return []
    try:
        found = calendar.find_adhoc_closures(actual_days, since)
        if not found:
            return []
        return merge({d: "臨時停市（實際行情無交易，自動補登）" for d in found})
    except Exception as e:  # noqa: BLE001
        log.warning("臨時停市補登失敗（忽略）：%s", type(e).__name__)
        return []


def record_verified_closure(day: dt.date) -> bool:
    """補登一個已由 Shioaji 分 K 查證「日盤無成交」的臨時停市日（只增不刪）。

    回傳 True 表示該日現已列在假日檔（本次新增或原本就在）；停用或寫入失敗回 False，
    呼叫端應退回原流程。與 record_adhoc_closures 互補：後者要等 FinMind 序列越過
    停市日才看得到缺口，但停市後的第一個交易日，缺口會先被單日遞補填成假的一列。
    """
    if not config.HOLIDAY_SYNC_ENABLED:
        return False
    try:
        merge({day: "臨時停市（Shioaji 分K：日盤無成交，自動補登）"})
        return not calendar.is_trading_day(day)
    except Exception as e:  # noqa: BLE001
        log.warning("臨時停市補登失敗（忽略）：%s", type(e).__name__)
        return False
