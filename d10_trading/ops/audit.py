"""操作稽核日誌（audit trail）——G1/G2′/G3 全階段所有操作的統一留痕。

var/logs/audit.jsonl：append-only、每事件一行 JSON、保留全部歷史（規格 §7）。

掛載點（設計原則：掛在「繞不過去的漏斗」上，呼叫端無從遺漏）：
- broker 層（ShioajiBroker）：登入、每筆委託/組合單/改價/撤單——
  日批次、flatten_sim、validate_live_path 等任何呼叫端都必經此層。
- orders 層：每筆委託的終態（成交口數/均價/狀態）。
- main 批次生命週期：batch_start / decision（每日決策快照）/ batch_end /
  batch_stop（告警停止）/ fuse / adopt_state / kill_switch。
- notify：所有等級的通知與告警原文。

事件欄位：ts（台北時間 ISO）、mode（sim/live）、event、其餘依事件而異。
稽核寫入失敗**絕不**影響交易流程（記錯誤日誌後放行）。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from zoneinfo import ZoneInfo

from d10_trading import config

log = logging.getLogger("d10.audit")


def _audit_file():
    return config.LOG_DIR / "audit.jsonl"


def audit(event: str, **fields) -> None:
    """寫入一筆稽核事件。任何失敗只記日誌、不拋出。"""
    rec = {
        "ts": dt.datetime.now(ZoneInfo(config.TIMEZONE)).isoformat(timespec="seconds"),
        "mode": "live" if config.LIVE_TRADING else "sim",
        "event": event,
        **fields,
    }
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_audit_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - 稽核失敗不可影響交易
        log.exception("audit 寫入失敗（事件 %s）", event)
