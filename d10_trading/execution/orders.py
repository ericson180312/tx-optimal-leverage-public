"""下單執行流程（規格 §5、§6）。

限價單流程：最佳買/賣價 ± PRICE_TOLERANCE 檔掛單（可成交限價，禁市價）→
輪詢成交回報 → ORDER_TIMEOUT 未成交改價一次（再讓 PRICE_TOLERANCE 檔）→
仍未成交 → 撤單 + 告警 + 今日放棄（明日批次自然重試）。
部分成交：以實際成交口數為準記錄，剩餘依 timeout 流程處理。

轉倉：優先 TAIFEX 跨月價差單（賣近買次，一張單無腳間風險）；
價差單失敗 → 兩腳分開下，先下流動性差的一腳（次月），成交後立即補近月腳，
兩腳間隔與滑價記錄於成交明細。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from d10_trading import config
from d10_trading.execution.broker import BrokerInterface, OrderHandle, Quote
from d10_trading.ops.audit import audit

log = logging.getLogger("d10.orders")

_TICK = config.TICK_POINTS
_TOL = config.PRICE_TOLERANCE_TICKS * _TICK

TERMINAL = {"Filled", "Cancelled", "Failed"}

# 價差單成交淨價與報價月差的容許偏離（點）。8/12 兩腳實測滑價 4.67 點，
# 組合單應更小；20 點足以放行正常滑價，又能抓出正負號慣例錯誤（量級為 2×月差）。
_COMBO_NET_DEV_ALERT = 20.0


@dataclass
class Fill:
    code: str
    action: str
    quantity: int          # 實際成交口數
    requested: int
    price: float           # 成交均價（組合單為淨價）
    order_price: float
    note: str = ""


@dataclass
class ExecutionReport:
    ok: bool               # 全數成交
    fills: list[Fill] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    @property
    def filled_qty(self) -> int:
        return sum(f.quantity for f in self.fills)


def _limit_price(q: Quote, action: str, extra_ticks: int = 0) -> float:
    """可成交限價：買單掛 ask+讓價、賣單掛 bid−讓價（讓價=滑價上限）。"""
    tol = _TOL + extra_ticks * _TICK
    if action == "Buy":
        base = q.ask if q.ask > 0 else q.price
        return base + tol
    base = q.bid if q.bid > 0 else q.price
    return base - tol


def _await_fill(broker: BrokerInterface, handle: OrderHandle,
                timeout: float, sleep_fn=time.sleep):
    """輪詢至成交/終態/逾時，回傳最後狀態快照。"""
    deadline = time.monotonic() + timeout
    while True:
        st = broker.refresh(handle)
        if st.status in TERMINAL:
            return st
        if time.monotonic() >= deadline:
            return st
        sleep_fn(config.ORDER_POLL_SECS)


def execute_limit(broker: BrokerInterface, code: str, action: str,
                  quantity: int, *, sleep_fn=time.sleep) -> ExecutionReport:
    """單腳限價單：掛單 → 逾時改價一次 → 仍未成交撤單告警。"""
    rep = ExecutionReport(ok=False)
    if quantity <= 0:
        rep.ok = True
        return rep

    q = broker.quote(code)
    price = _limit_price(q, action)
    log.info("下單 %s %s x%d 限價 %.0f（bid=%.0f ask=%.0f）",
             action, code, quantity, price, q.bid, q.ask)
    handle = broker.place_limit(code, action, quantity, price)
    st = _await_fill(broker, handle, config.ORDER_TIMEOUT_SECS, sleep_fn)

    if st.status not in TERMINAL:                       # 逾時 → 改價一次
        q2 = broker.quote(code)
        price2 = _limit_price(q2, action, extra_ticks=config.PRICE_TOLERANCE_TICKS)
        # 只朝更積極的方向改價（買調升、賣調降），避免改成更差的排隊價
        if (action == "Buy" and price2 > handle.price) or \
           (action == "Sell" and price2 < handle.price):
            log.info("逾時未成交，改價 %.0f → %.0f", handle.price, price2)
            try:
                broker.update_price(handle, price2)
            except Exception as e:  # noqa: BLE001 - 改價失敗照 timeout 流程收尾
                rep.alerts.append(f"改價失敗（{e}），續等原單")
        st = _await_fill(broker, handle, config.ORDER_TIMEOUT_SECS, sleep_fn)

    if st.status not in TERMINAL:                       # 仍未成交 → 撤單
        try:
            broker.cancel(handle)
        except Exception as e:  # noqa: BLE001
            rep.alerts.append(f"撤單失敗（{e}）——需人工確認委託狀態！")
        st = broker.refresh(handle)

    filled = st.deal_quantity
    audit("order_final", code=code, action=action, status=st.status,
          filled=filled, requested=quantity, avg_price=st.avg_fill_price,
          order_price=handle.price)
    if filled > 0:
        rep.fills.append(Fill(code=code, action=action, quantity=filled,
                              requested=quantity, price=st.avg_fill_price,
                              order_price=handle.price))
    if st.status == "Failed":
        rep.alerts.append(f"委託失敗 {action} {code} x{quantity}：{st.msg}")
    elif filled < quantity:
        rep.alerts.append(
            f"未足額成交 {action} {code}：{filled}/{quantity} 口，"
            "餘量已撤單，今日放棄（明日批次自然重試）")
    rep.ok = filled == quantity
    return rep


def execute_roll(broker: BrokerInterface, near_code: str, next_code: str,
                 near_qty: int, target_qty: int, *,
                 sleep_fn=time.sleep) -> ExecutionReport:
    """轉倉（+與再平衡重疊時一次到位）：平近月 near_qty 口、建次月 target_qty 口。

    共同口數 min(near_qty, target_qty) 走跨月價差單（或兩腳）；
    差額走次月單腳，避免多餘來回（規格 §6）。
    """
    rep = ExecutionReport(ok=True)
    common = min(near_qty, target_qty)

    if common > 0:
        sub = _roll_common(broker, near_code, next_code, common, sleep_fn)
        rep.fills += sub.fills
        rep.alerts += sub.alerts
        if not sub.ok:
            rep.ok = False
            return rep          # 轉倉腿未完成 → 不做後續調整，告警等明日/人工

    if target_qty > near_qty:          # 需加碼：差額買在次月
        sub = execute_limit(broker, next_code, "Buy", target_qty - near_qty,
                            sleep_fn=sleep_fn)
    elif near_qty > target_qty:        # 需減碼：多出的近月直接平掉（不轉）
        sub = execute_limit(broker, near_code, "Sell", near_qty - target_qty,
                            sleep_fn=sleep_fn)
    else:
        return rep
    rep.fills += sub.fills
    rep.alerts += sub.alerts
    rep.ok = rep.ok and sub.ok
    return rep


def _roll_common(broker: BrokerInterface, near_code: str, next_code: str,
                 qty: int, sleep_fn) -> ExecutionReport:
    if config.ROLL_USE_COMBO:
        try:
            return _roll_via_combo(broker, near_code, next_code, qty, sleep_fn)
        except Exception as e:  # noqa: BLE001 - 價差單不可用 → 退回兩腳
            log.warning("跨月價差單失敗（%s），退回兩腳分開下", e)
    return _roll_two_legs(broker, near_code, next_code, qty, sleep_fn)


def _roll_via_combo(broker: BrokerInterface, near_code: str, next_code: str,
                    qty: int, sleep_fn) -> ExecutionReport:
    """跨月價差單：Sell 近月 + Buy 次月，淨價 = 次月 − 近月（買價差）。"""
    rep = ExecutionReport(ok=False)
    qn, qx = broker.quote(near_code), broker.quote(next_code)
    # 合成賣//買價差報價：ask_spread = 次月ask − 近月bid（立即可成交上限）
    net = (qx.ask if qx.ask > 0 else qx.price) - (qn.bid if qn.bid > 0 else qn.price)
    net_limit = net + _TOL
    log.info("跨月價差單 Sell %s / Buy %s x%d 淨價限 %.0f",
             near_code, next_code, qty, net_limit)
    handle = broker.place_combo_roll(near_code, next_code, qty, net_limit)
    # 🔴 委託已送出，市場曝險可能已經發生：此後**絕不可**讓例外往上拋。
    # _roll_common 會把例外解讀成「價差單不可用」而改走兩腳全額下單；若此刻
    # 已部分成交，就會在既有部位之上重複建倉、曝險加倍（2026-09-02 首次真實
    # 價差單前的審查發現）。送出前的例外才屬於 fallback 的職責範圍。
    try:
        st = _await_fill(broker, handle, config.ORDER_TIMEOUT_SECS, sleep_fn)
        if st.status not in TERMINAL:           # 組合單不能改價 → 撤單重下一次
            broker.cancel(handle)
            st = broker.refresh(handle)
            if st.deal_quantity == 0:
                qn, qx = broker.quote(near_code), broker.quote(next_code)
                net2 = ((qx.ask if qx.ask > 0 else qx.price)
                        - (qn.bid if qn.bid > 0 else qn.price)) + 2 * _TOL
                log.info("價差單逾時，重下淨價限 %.0f", net2)
                handle = broker.place_combo_roll(near_code, next_code, qty, net2)
                st = _await_fill(broker, handle, config.ORDER_TIMEOUT_SECS,
                                 sleep_fn)
                if st.status not in TERMINAL:
                    broker.cancel(handle)
                    st = broker.refresh(handle)
    except Exception as e:  # noqa: BLE001 - 見上：送出後一律就地處理，不得外拋
        log.exception("價差單送出後的查詢/撤單失敗")
        audit("combo_post_submit_error", code=f"{near_code}->{next_code}",
              error=f"{type(e).__name__}: {e}")
        rep.alerts.append(
            f"🔴 價差單已送出但後續查詢/撤單失敗（{type(e).__name__}）——"
            f"{near_code}→{next_code} 成交狀態不明，本日不再下任何單。"
            "請至券商 App 確認實際部位，必要時以 --adopt-state 校正")
        return rep                              # ok=False → 上層不做後續調整
    filled = st.deal_quantity
    audit("order_final", code=f"{near_code}->{next_code}", action="Roll",
          status=st.status, filled=filled, requested=qty,
          avg_price=st.avg_fill_price, order_price=handle.price)
    if filled > 0:
        rep.fills.append(Fill(code=f"{near_code}->{next_code}", action="Roll",
                              quantity=filled, requested=qty,
                              price=st.avg_fill_price, order_price=handle.price,
                              note="combo"))
    if filled < qty:
        rep.alerts.append(f"轉倉價差單未足額：{filled}/{qty} 口")
    # 🔴 成交淨價健全性：本模組假設 Shioaji 的組合單淨價 = legs 順序的
    # 「Buy 次月 − Sell 近月」（見 broker.place_combo_roll）。此慣例從未經
    # 真實環境驗證；若正負號或基準與假設不符，限價保護會失效而無聲成交。
    # 故比對成交淨價與下單當時的報價月差，偏離過大即告警（不阻擋，已成交）。
    if filled > 0 and abs(st.avg_fill_price - net) > _COMBO_NET_DEV_ALERT:
        rep.alerts.append(
            f"⚠️ 價差單成交淨價 {st.avg_fill_price:.0f} 與報價月差 {net:.0f} "
            f"相差 {st.avg_fill_price - net:+.0f} 點（限價 {net_limit:.0f}）——"
            "請確認券商的淨價正負號慣例與滑價是否合理")
    rep.ok = filled == qty
    return rep


def _roll_two_legs(broker: BrokerInterface, near_code: str, next_code: str,
                   qty: int, sleep_fn) -> ExecutionReport:
    """兩腳分開：先下流動性差的次月（買），成交後立即補近月（賣）。"""
    rep = ExecutionReport(ok=False)
    t0 = time.monotonic()
    leg1 = execute_limit(broker, next_code, "Buy", qty, sleep_fn=sleep_fn)
    rep.fills += leg1.fills
    rep.alerts += leg1.alerts
    filled1 = leg1.filled_qty
    if filled1 == 0:
        rep.alerts.append("轉倉第一腳（買次月）未成交，近月未動，今日放棄")
        return rep
    # 第二腳只平掉第一腳實際建立的口數，保持兩腳對稱
    leg2 = execute_limit(broker, near_code, "Sell", filled1, sleep_fn=sleep_fn)
    rep.fills += leg2.fills
    rep.alerts += leg2.alerts
    gap = time.monotonic() - t0
    for f in rep.fills:
        f.note = f.note or f"two_legs gap={gap:.1f}s"
    if leg2.filled_qty < filled1:
        rep.alerts.append(
            f"⚠️ 轉倉兩腳不對稱：買次月 {filled1} 口、平近月 {leg2.filled_qty} 口，"
            "淨曝險暫時偏高——明日對帳將顯示不一致，需人工確認")
    rep.ok = leg1.ok and leg2.ok and leg2.filled_qty == filled1 == qty
    return rep
