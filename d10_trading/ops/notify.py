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

from d10_trading import config
from d10_trading.ops.audit import audit

log = logging.getLogger("d10.notify")

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
    audit("notify", level=level, text=text)
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


def heartbeat(exit_code: int) -> None:
    """healthchecks.io 心跳：批次結尾必 ping（含假日快速退出），
    讓「排程沒跑」與「跑了但異常」都能被外部監控偵測。
    未設定 HEALTHCHECK_URL 即略過；ping 失敗絕不影響批次結果。

    🔴 必須檢查 HTTP 回應碼：URL 打錯或 check 被刪除時 healthchecks 回 404，
    若只看有無例外會一路記成「ping OK」——監控早已失效卻無人察覺
    （「監控系統本身故障」的典型盲點）。非 200 一律升級為 ALERT，
    因為此時外部監控已不可信，缺席偵測形同虛設。
    """
    if not config.HEALTHCHECK_URL:
        return
    url = config.HEALTHCHECK_URL.rstrip("/")
    if exit_code != 0:
        url += "/fail"
    try:
        # 取值一併放進 try：任何非預期回應都在此被吸收，絕不穿透到批次流程
        status = requests.get(url, timeout=10).status_code
    except Exception as e:  # noqa: BLE001 - 心跳失敗不可拖垮批次
        log.warning("heartbeat ping 失敗（忽略）：%s", type(e).__name__)
        audit("heartbeat_failed", reason=type(e).__name__, exit_code=exit_code)
        return
    if status == 200:
        log.info("heartbeat ping OK（exit=%d）", exit_code)
        return
    # 送達了但被拒——URL 失效的機率遠高於服務異常，需人工確認
    log.error("heartbeat 被拒：HTTP %s", status)
    audit("heartbeat_rejected", status=status, exit_code=exit_code)
    send("ALERT", f"healthchecks 心跳被拒（HTTP {status}）——"
                  "URL 可能失效或 check 已刪除，外部缺席偵測目前不可信，"
                  "請至 healthchecks.io 確認後更新 .env 的 HEALTHCHECK_URL")


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
