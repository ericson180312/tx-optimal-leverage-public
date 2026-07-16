"""交易與營運日誌落地（規格 §7）：每日一行摘要 CSV + 成交明細 JSONL，保留全部歷史。"""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import asdict

from live import config

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


def append_exec_vs_close(date: dt.date, code: str, exec_price: float,
                         official_close: float) -> None:
    """執行價 vs 官方收盤價偏差記錄（規格 §2：供事後檢視）。次日補寫。"""
    path = config.LOG_DIR / "exec_vs_close.csv"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "code", "exec_price", "official_close", "diff_pct"])
        diff = (exec_price / official_close - 1.0) if official_close else ""
        w.writerow([date.isoformat(), code, exec_price, official_close, diff])
