"""G2′ 前置：正式環境「不成交驗證」——委託路徑全鏈驗證，零成交風險。

    python -m d10_trading.scripts.validate_live_path --yes

流程：登入（含 CA 啟用）→ 近月 TMF 報價 → 掛「遠價」限價買單 1 口
（bid × 0.95，且不低於跌停價附近）→ 等交易所回報（Submitted）→ 撤單
→ 確認 Cancelled。整條路徑（CA 簽章/下單/回報/撤單）都走過一遍，
但價格遠離市場、不會成交。

安全設計：
- 只在正式模式執行（LIVE_TRADING=True）；模擬模式直接拒絕——
  模擬路徑已由 G1 覆蓋，本工具的目的就是驗證正式環境。
- 盤中（日盤）執行；跌 5% 以上的異常日不要跑（遠價買單會貼近跌停）。
- 萬一極端行情下竟然成交（理論上僅閃崩），發 CRITICAL 告警並停止，
  由人工處理該 1 口（工具絕不自動反向沖銷）。
"""
from __future__ import annotations

import argparse
import sys
import time

from d10_trading import config
from d10_trading.execution.broker import ShioajiBroker
from d10_trading.ops import notify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="正式環境不成交驗證（G2′ 前置）")
    ap.add_argument("--yes", action="store_true", help="確認執行（必填）")
    args = ap.parse_args(argv)

    if config.SIMULATION:
        print("拒絕：目前為模擬模式（LIVE_TRADING=False）。"
              "本工具驗證的是正式環境，請先依 README §4.1 切換設定。")
        return 2
    if not args.yes:
        print("這會在【正式環境】掛出 1 口遠價限價買單並撤單（不會成交）。"
              "確認請加 --yes。")
        return 2

    notify.setup_logging()
    broker = ShioajiBroker()
    broker.login()
    try:
        import datetime as dt
        today = dt.date.today()
        front, _ = broker.front_and_next(config.PRODUCT, today)
        q = broker.quote(front.code)
        if q.bid <= 0:
            print(f"取不到 {front.code} 買價（bid={q.bid}），盤中再執行。")
            return 1
        # 遠價：bid × 0.95；保底不低於「參考價 × 0.9（跌停）+ 5 點」以免被退單
        floor_px = round(front.reference * 0.9) + 5 if front.reference > 0 else 0
        price = max(round(q.bid * 0.95), floor_px)
        if price >= q.bid:
            print(f"異常：遠價 {price} 未遠離買價 {q.bid}（跌停附近？），中止。")
            return 1

        print(f"掛遠價限價買單：{front.code} 1 口 @ {price}"
              f"（bid={q.bid:.0f}，距離 {q.bid - price:.0f} 點）")
        handle = broker.place_limit(front.code, "Buy", 1, price)

        submitted = False
        for _ in range(10):                      # 最多 ~15 秒等交易所回報
            st = broker.refresh(handle)
            print(f"  委託狀態：{st.status}")
            if st.status in ("Filled", "PartFilled"):
                notify.send("CRITICAL",
                            f"不成交驗證竟然成交 {st.deal_quantity} 口 "
                            f"{front.code} @ {st.avg_fill_price:.0f}——"
                            "請人工處理該部位！")
                return 2
            if st.status == "Failed":
                notify.send("ALERT", f"不成交驗證：委託被拒（{st.msg}）")
                return 1
            if st.status == "Submitted":
                submitted = True
                break
            time.sleep(1.5)
        if not submitted:
            print("未在時限內收到 Submitted 回報，仍嘗試撤單。")

        broker.cancel(handle)
        st = broker.refresh(handle)
        for _ in range(10):
            if st.status in ("Cancelled", "Failed"):
                break
            time.sleep(1.5)
            st = broker.refresh(handle)
        print(f"  撤單後狀態：{st.status}")

        if st.status == "Cancelled" and st.deal_quantity == 0 and submitted:
            msg = (f"不成交驗證通過：{front.code} 掛單 {price} → Submitted → "
                   "Cancelled，全程零成交（CA/下單/回報/撤單路徑正常）")
            print(f"✅ {msg}")
            notify.send("INFO", msg)
            return 0
        notify.send("ALERT",
                    f"不成交驗證未完整通過：submitted={submitted}、"
                    f"最終狀態 {st.status}、成交 {st.deal_quantity} 口——"
                    "請至券商 App 確認委託已消失後再重試")
        return 1
    finally:
        broker.logout()


if __name__ == "__main__":
    sys.exit(main())
