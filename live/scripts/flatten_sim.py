"""一次性工具：清空【模擬環境】期貨帳戶的所有部位（G1 前置清理用）。

用途：API 開通流程的下單測試會在模擬帳戶留下部位（例如 TXF 一口），
會被開場對帳擋下（本系統不代管非 PRODUCT 部位）。本工具以可成交限價單
逐一平倉。

    python -m live.scripts.flatten_sim --yes

安全鎖：config.SIMULATION 必須為 True（LIVE_TRADING=False），否則直接拒絕——
真實帳戶的部位處理一律人工在券商端操作（README §6 手動接管程序）。
"""
from __future__ import annotations

import argparse
import sys

from live import config
from live.execution import orders
from live.execution.broker import ShioajiBroker
from live.ops import notify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="清空模擬帳戶部位")
    ap.add_argument("--yes", action="store_true", help="確認執行（必填）")
    args = ap.parse_args(argv)

    if not config.SIMULATION:
        print("拒絕：LIVE_TRADING=True（真實模式）。真實部位請人工在券商端處理。")
        return 2
    if not args.yes:
        print("這會平掉模擬帳戶的『全部』期貨部位。確認請加 --yes。")
        return 2

    notify.setup_logging()
    broker = ShioajiBroker()
    broker.login()
    try:
        positions = broker.positions()
        if not positions:
            print("模擬帳戶無部位，無事可做。")
            return 0
        ok = True
        for p in positions:
            action = "Sell" if p.direction == "Buy" else "Buy"
            print(f"平倉 {p.code} {action} x{p.quantity} …")
            rep = orders.execute_limit(broker, p.code, action, p.quantity)
            for f in rep.fills:
                print(f"  成交 {f.quantity}/{f.requested} @ {f.price:.0f}")
            for a in rep.alerts:
                print(f"  ⚠️ {a}")
            ok = ok and rep.ok
        remain = broker.positions()
        print(f"完成。剩餘部位：{[(p.code, p.quantity) for p in remain] or '（空）'}")
        return 0 if ok and not remain else 1
    finally:
        broker.logout()


if __name__ == "__main__":
    sys.exit(main())
