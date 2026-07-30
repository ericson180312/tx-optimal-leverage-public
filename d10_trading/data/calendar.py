"""TAIFEX 交易行事曆（規格 §2：不要用「平日」近似）。

資料來源分工：
- **過去**的交易日：以 FinMind 實際行情日為準（資料層天然涵蓋颱風停市等）。
- **未來**的交易日（轉倉倒數需要）：平日 − data_static/taifex_holidays.csv 假日檔。
  假日檔需人工逐年維護並與 TAIFEX 官方行事曆核對（README 有檢核程序）；
  檔頭 covered_years 未涵蓋目標年份 → 拒絕計算並要求停止（fail-closed）。
- 每日批次會用 validate_against_actual() 把假日檔與實際行情日互驗，
  不一致（例如颱風停市未列入）→ 告警。
"""
from __future__ import annotations

import csv
import datetime as dt
from functools import lru_cache

from d10_trading import config


class CalendarCoverageError(RuntimeError):
    """假日檔未涵蓋目標年份——需人工更新後才能繼續。"""


@lru_cache(maxsize=1)
def _load_holiday_file() -> tuple[frozenset[dt.date], frozenset[int]]:
    holidays: set[dt.date] = set()
    covered: set[int] = set()
    with open(config.HOLIDAY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# covered_years:"):
                covered = {int(y) for y in line.split(":", 1)[1].split(",")}
                continue
            if not line or line.startswith("#"):
                continue
            row = next(csv.reader([line]))
            if row[0] == "date":
                continue
            holidays.add(dt.date.fromisoformat(row[0]))
    if not covered:
        raise CalendarCoverageError("假日檔缺少 covered_years 標頭")
    return frozenset(holidays), frozenset(covered)


def covered_years() -> frozenset[int]:
    """假日檔涵蓋的年份集合（到期哨兵用；見 ops/watchdog.py）。"""
    _, covered = _load_holiday_file()
    return covered


def _require_coverage(d: dt.date) -> None:
    _, covered = _load_holiday_file()
    if d.year not in covered:
        raise CalendarCoverageError(
            f"假日檔未涵蓋 {d.year} 年——請依 TAIFEX 官方行事曆更新 "
            f"{config.HOLIDAY_FILE.name} 後再執行")


def is_trading_day(d: dt.date) -> bool:
    _require_coverage(d)
    holidays, _ = _load_holiday_file()
    return d.weekday() < 5 and d not in holidays


def prev_trading_day(d: dt.date) -> dt.date:
    cur = d - dt.timedelta(days=1)
    while not is_trading_day(cur):
        cur -= dt.timedelta(days=1)
    return cur


def next_trading_day(d: dt.date) -> dt.date:
    cur = d + dt.timedelta(days=1)
    while not is_trading_day(cur):
        cur += dt.timedelta(days=1)
    return cur


def shift_trading_days(d: dt.date, n: int) -> dt.date:
    """自 d 起（不含 d）往前/後移 |n| 個交易日。n=0 回傳 d。"""
    cur = d
    step = 1 if n > 0 else -1
    for _ in range(abs(n)):
        cur = next_trading_day(cur) if step > 0 else prev_trading_day(cur)
    return cur


def third_wednesday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (2 - first.weekday()) % 7   # Wed == 2
    return first + dt.timedelta(days=offset + 14)


def settlement_date(year: int, month: int) -> dt.date:
    """該月份合約的最後交易日＝第三個週三，遇假日順延至次一交易日。

    註：實際結算日以 Shioaji 合約的 delivery_date 為準（execution 層），
    此函數作為互驗與測試之用；兩者不一致 → 告警。
    """
    d = third_wednesday(year, month)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


def reload() -> None:
    """清除假日檔快取（holiday_sync 寫入後呼叫，使新資料立即生效）。"""
    _load_holiday_file.cache_clear()


def find_adhoc_closures(actual_days: list[dt.date],
                        since: dt.date) -> list[dt.date]:
    """臨時停市（颱風等）：平日、不在假日檔、卻無實際行情的日期。

    只檢查「實際行情資料範圍之內」的日期——區間為 [max(since, 首個實際行情日),
    最後實際行情日]。兩端都夾在真實資料內，因此：
    - 不會把資料延遲的最新日誤判為停市（其後無行情佐證）；
    - 也不會把資料起點之前的平日全部誤判（since 早於資料起點時）。
    被標記者必然前後都有行情佐證。
    """
    holidays, covered = _load_holiday_file()
    actual = {d for d in actual_days if d >= since}
    if not actual:
        return []
    out, cur, end = [], max(since, min(actual)), max(actual)
    while cur <= end:
        if (cur.year in covered and cur.weekday() < 5
                and cur not in holidays and cur not in actual):
            out.append(cur)
        cur += dt.timedelta(days=1)
    return out


def validate_against_actual(actual_days: list[dt.date],
                            since: dt.date) -> list[str]:
    """假日檔 vs 實際行情日互驗（只驗 since 之後、且在涵蓋年份內的日期）。

    回傳異常訊息清單（空 = 一致）。
    - 平日且不在假日檔、卻沒有行情 → 可能漏列假日（或颱風停市）。
    - 有行情的日子卻列在假日檔 → 假日檔誤列。
    """
    holidays, covered = _load_holiday_file()
    actual = {d for d in actual_days if d >= since}
    if not actual:
        return ["互驗區間內無實際行情日"]
    problems = []
    end = max(actual)
    cur = since
    while cur <= end:
        if cur.year in covered:
            if cur.weekday() < 5 and cur not in holidays and cur not in actual:
                problems.append(f"{cur} 為平日且不在假日檔，但無行情（漏列假日或臨時停市？）")
            if cur in actual and (cur.weekday() >= 5 or cur in holidays):
                problems.append(f"{cur} 有行情但被行事曆判為休市（假日檔誤列？）")
        cur += dt.timedelta(days=1)
    return problems
