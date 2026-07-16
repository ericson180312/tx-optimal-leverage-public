"""通知與告警（規格 §7）：Telegram 推播 + 本地日誌，永不輸出金鑰/憑證路徑。

等級：
  INFO      每日摘要、成交回報
  WARNING   雙源核對差異、訊號預覽與定案不一致等（不停止）
  ALERT     資料過期 / 對帳不符 / 下單失敗 / 超過口數上限（本日停止）
  CRITICAL  保險絲觸發（最高等級）
Telegram 未設定時退化為純日誌並提示（G1 前可接受；G2 起 README 要求必須設定）。
"""
from __future__ import annotations

import logging

import requests

from live import config

log = logging.getLogger("live.notify")

_PREFIX = {"INFO": "ℹ️", "WARNING": "⚠️", "ALERT": "🚨", "CRITICAL": "🟥🟥"}


def _telegram(text: str) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text},
            timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 - 通知失敗不可影響交易流程
        log.error("Telegram 推播失敗：%s", type(e).__name__)
        return False


def send(level: str, text: str) -> None:
    tag = _PREFIX.get(level, "")
    body = f"{tag} [D_10] {text}"
    if level in ("ALERT", "CRITICAL"):
        log.error("%s %s", level, text)
    elif level == "WARNING":
        log.warning("%s", text)
    else:
        log.info("%s", text)
    if not _telegram(body):
        log.info("（Telegram 未設定或失敗，僅記錄於日誌）")


def setup_logging() -> None:
    """檔案（月輪替）+ 主控台雙路日誌。"""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:          # 重複呼叫（測試）不疊加 handler
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    import datetime as dt
    fh = logging.FileHandler(
        config.LOG_DIR / f"live_{dt.date.today():%Y%m}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
