"""交易與營運日誌落地（規格 §7）：每日一行摘要 CSV + 成交明細 JSONL，保留全部歷史。"""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict

from d10_trading import config

def _summary_file():
    return config.LOG_DIR / "daily_summary.csv"


def _trades_file():
    return config.LOG_DIR / "trades.jsonl"


_SUMMARY_COLS = [
    "date", "regime", "exec_leverage", "equity", "price", "target_contracts",
    "held_before", "held_after", "achieved_leverage", "action",
    "exec_price", "official_close", "exec_vs_close_pct", "preview_leverage",
    "warnings",
]


def write_summary(row: dict) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _summary_file()
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def write_fills(date: dt.date, fills) -> None:
    if not fills:
        return
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_trades_file(), "a", encoding="utf-8") as f:
        for fill in fills:
            rec = {"date": date.isoformat(), **asdict(fill)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_ROLL_SPREAD_HEADER = [
    "date", "near_month", "next_month", "settlement", "t_minus",
    "near_price", "next_price", "spread", "spread_pct", "rolled"]

_ROLL_SLIPPAGE_HEADER = [
    "date", "near_code", "next_code", "path", "exec_near", "exec_next",
    "exec_spread", "settle_near", "settle_next", "settle_spread",
    "slippage_points", "slippage_pct"]


def _append_csv(name: str, header: list[str], row: dict) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LOG_DIR / name
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def write_roll_spread(row: dict) -> None:
    """近月/次月月差的**每日**紀錄（roll_spread.csv）。

    為何逐日而非只記 T−5 與結算日：使用者要追蹤的是兩者的「關係」，而關係
    是路徑問題——逐日紀錄能回答 T−4/T−3/… 的收斂形狀，兩個點只能給出差值，
    且成本相同（每日一列）。`t_minus` 為距結算的交易日數（結算日 = 0）。
    `spread_pct` = 月差 ÷ 近月價，供跨月份、跨價位水準比較。
    """
    _append_csv("roll_spread.csv", _ROLL_SPREAD_HEADER, row)


def append_roll_slippage(row: dict) -> None:
    """轉倉的**月差滑價**（roll_slippage.csv）：執行月差 − 結算月差。

    🔴 這是轉倉真正的成本口徑。兩腿共同的時間滑價（13:20 執行 vs 13:45 結算）
    在轉倉中完全對消——一腿買、一腿賣——故單看各腿的 exec_vs_close 偏差會
    高估成本。實測（2026-09-09）兩腿偏差皆為 −0.030%，差距 0.000pp，
    而月差滑價為 0 點；同期 8/12 兩腳路徑的月差滑價為 +7.67 點。
    結算月差取自官方結算價，是獨立於自家報價的基準。
    刻意**不設告警門檻**：樣本僅兩次且分屬兩種執行路徑，憑此設門檻會重演
    先前憑推算定風險指標門檻的錯誤。累積數次真實轉倉後再依分布決定。
    """
    _append_csv("roll_slippage.csv", _ROLL_SLIPPAGE_HEADER, row)


def append_exec_vs_close(date: dt.date, code: str, exec_price: float,
                         settlement: float) -> None:
    """執行價 vs 官方結算價偏差記錄（規格 §2：供事後檢視）。結算到位後補寫。

    settlement 為 FinMind TX 大台結算價（權威來源；交易的 TMF 與 TX 每日結算價
    差極小，見 series.SignalSeries.settlement_of）。diff 反映『13:20 執行 vs 收盤
    結算』的時間滑價（含微幅 TMF/TX 基差）。
    """
    path = config.LOG_DIR / "exec_vs_close.csv"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "code", "exec_price", "settlement", "diff_pct"])
        diff = (exec_price / settlement - 1.0) if settlement else ""
        w.writerow([date.isoformat(), code, exec_price, settlement, diff])
