"""Shioaji 封裝層——全系統唯一 import shioaji 的模組（規格 §10）。

- 查詢類 API：指數退避 retry（上限 RETRY_MAX 次）＋逾時。
- 下單/改單/撤單：**絕不盲目重試**（重複下單風險 > 單次失敗），
  失敗直接拋出，由上層告警停止。
- 介面以 BrokerInterface 抽象，策略核心與測試（FakeBroker）不觸碰 shioaji。
- 日誌永不輸出 API 金鑰或憑證路徑（規格 §11）。
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field

from live import config

log = logging.getLogger("live.broker")


# ── 與策略/執行流程共用的純資料型別 ─────────────────────────────────────────

@dataclass
class ContractInfo:
    code: str                 # Shioaji 合約代碼（如 TMFA6）
    product: str              # TMF / MXF / TXF
    delivery_month: str       # YYYYMM
    delivery_date: dt.date    # 最後交易日（結算日）
    reference: float = 0.0


@dataclass
class Quote:
    code: str
    price: float              # 最新成交價
    bid: float
    ask: float
    ts: dt.datetime | None = None


@dataclass
class PositionInfo:
    code: str
    direction: str            # "Buy"（多）/ "Sell"（空）
    quantity: int
    avg_price: float
    last_price: float = 0.0

    @property
    def signed_qty(self) -> int:
        return self.quantity if self.direction == "Buy" else -self.quantity


@dataclass
class OrderStatusSnapshot:
    status: str               # PendingSubmit/Submitted/Filled/PartFilled/Cancelled/Failed
    deal_quantity: int = 0
    avg_fill_price: float = 0.0
    order_price: float = 0.0
    msg: str = ""


@dataclass
class OrderHandle:
    """封裝一張在途委託（實作內部持有 shioaji Trade 物件）。"""
    code: str
    action: str               # "Buy"/"Sell"
    quantity: int
    price: float
    raw: object = field(default=None, repr=False)
    is_combo: bool = False


class BrokerError(RuntimeError):
    pass


class BrokerInterface:
    """策略/流程層看到的唯一介面；FakeBroker（測試）與 ShioajiBroker 皆實作之。"""

    simulation: bool = True

    def login(self) -> None: raise NotImplementedError
    def logout(self) -> None: raise NotImplementedError
    def monthly_contracts(self, product: str) -> list[ContractInfo]: raise NotImplementedError
    def quote(self, code: str) -> Quote: raise NotImplementedError
    def day_session_close(self, product: str, delivery_month: str,
                          date: dt.date) -> float | None: raise NotImplementedError
    def equity(self) -> float: raise NotImplementedError
    def positions(self) -> list[PositionInfo]: raise NotImplementedError
    def place_limit(self, code: str, action: str, quantity: int,
                    price: float) -> OrderHandle: raise NotImplementedError
    def place_combo_roll(self, sell_code: str, buy_code: str, quantity: int,
                         net_price: float) -> OrderHandle: raise NotImplementedError
    def update_price(self, handle: OrderHandle, price: float) -> None: raise NotImplementedError
    def cancel(self, handle: OrderHandle) -> None: raise NotImplementedError
    def refresh(self, handle: OrderHandle) -> OrderStatusSnapshot: raise NotImplementedError

    # 便利方法（介面層共用邏輯）
    def front_and_next(self, product: str,
                       today: dt.date) -> tuple[ContractInfo, ContractInfo]:
        cs = [c for c in self.monthly_contracts(product) if c.delivery_date >= today]
        if len(cs) < 2:
            raise BrokerError(f"{product} 可用月合約不足（找到 {len(cs)} 檔）")
        cs.sort(key=lambda c: c.delivery_month)
        return cs[0], cs[1]

    def contract_by_month(self, product: str, month: str) -> ContractInfo:
        for c in self.monthly_contracts(product):
            if c.delivery_month == month:
                return c
        raise BrokerError(f"{product} 找不到 {month} 月合約")


def _retry(fn, what: str):
    """查詢類 API 的指數退避重試。"""
    last: Exception | None = None
    for attempt in range(config.RETRY_MAX):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - 網路/API 層一律退避重試
            last = e
            wait = config.RETRY_BASE_SECS * (2 ** attempt)
            log.warning("%s 失敗（第 %d 次）：%s；%.0fs 後重試",
                        what, attempt + 1, type(e).__name__, wait)
            time.sleep(wait)
    raise BrokerError(f"{what} 連續 {config.RETRY_MAX} 次失敗") from last


class ShioajiBroker(BrokerInterface):
    """真實 Shioaji 封裝。simulation 由 config.LIVE_TRADING 決定（G0–G1 恆為模擬）。"""

    def __init__(self):
        self.simulation = config.SIMULATION
        self._api = None

    # ── 連線 ────────────────────────────────────────────────────────────────
    def login(self) -> None:
        import shioaji as sj
        if not config.SJ_API_KEY or not config.SJ_SEC_KEY:
            raise BrokerError("SJ_API_KEY / SJ_SEC_KEY 未設定（live/.env）")
        self._sj = sj
        self._api = sj.Shioaji(simulation=self.simulation)
        _retry(lambda: self._api.login(
            api_key=config.SJ_API_KEY,
            secret_key=config.SJ_SEC_KEY,
            contracts_timeout=10000,
        ), "登入")
        if not self.simulation:
            if not config.SJ_CA_PATH or not config.SJ_CA_PASSWD:
                raise BrokerError("真實模式需要 SJ_CA_PATH / SJ_CA_PASSWD（憑證）")
            self._api.activate_ca(ca_path=config.SJ_CA_PATH,
                                  ca_passwd=config.SJ_CA_PASSWD)
        if self._api.futopt_account is None:
            raise BrokerError("登入成功但無期貨帳戶")
        log.info("Shioaji 登入完成（simulation=%s）", self.simulation)

    def logout(self) -> None:
        if self._api is not None:
            try:
                self._api.logout()
            except Exception:  # noqa: BLE001 - 登出失敗不影響批次結果
                log.warning("登出失敗（忽略）")

    # ── 合約 ────────────────────────────────────────────────────────────────
    # 解析策略（依序）：R1/R2 連續月別名（近月/次月，微台 TMF 群組鍵不可用時
    # 的主要來源）→ 商品群組屬性/鍵 → 全商品檔掃描（代碼前綴）。三者聯集。
    def _to_info(self, c, product: str) -> ContractInfo | None:
        if c is None:
            return None
        dm = str(getattr(c, "delivery_month", "") or "")
        code = str(getattr(c, "code", "") or "")
        if len(dm) != 6 or not dm.isdigit() or not code.startswith(product):
            return None
        if "/" in code or "/" in str(getattr(c, "symbol", "")):
            return None               # 價差/組合商品
        # 連續月別名（TMFR1/R2…）：實測會出現在群組列舉中且帶目標合約的
        # delivery_month，必須排除，否則會被當成「次月」（月差恆為 0）。
        # 別名的辨識：target_code 非空，或代碼為 {product}R+數字
        # （TAIFEX 月份碼僅 A–L，R 不可能是真實月合約）。
        tail = code[len(product):]
        if str(getattr(c, "target_code", "") or "") or \
                (tail[:1] == "R" and tail[1:].isdigit()):
            return None
        dd = self._parse_date(getattr(c, "delivery_date", ""))
        if dd is None:
            return None
        return ContractInfo(code=code, product=product, delivery_month=dm,
                            delivery_date=dd,
                            reference=float(getattr(c, "reference", 0.0) or 0.0))

    def _lookup(self, key: str):
        try:
            return self._api.Contracts.Futures[key]
        except Exception:  # noqa: BLE001 - 查無此鍵（KeyError 等）
            return None

    def _alias_info(self, product: str, n: int) -> ContractInfo | None:
        """{product}R{n} 連續月別名 → target_code 解析回真實月合約。"""
        alias = self._lookup(f"{product}R{n}")
        if alias is None:
            return None
        target = str(getattr(alias, "target_code", "") or "")
        real = self._lookup(target) if target else None
        return self._to_info(real if real is not None else alias, product)

    def _iter_group(self, product: str):
        group = getattr(self._api.Contracts.Futures, product, None)
        if group is None:
            group = self._lookup(product)
        if group is not None and hasattr(group, "__iter__"):
            yield from group
            return
        try:                          # 全檔掃描（元素可能是分類或合約）
            top = list(self._api.Contracts.Futures)
        except Exception:  # noqa: BLE001 - 容器不可迭代（shioaji 以 KeyError 拒絕）
            return
        for item in top:
            try:
                sub = list(item)
            except Exception:  # noqa: BLE001
                sub = [item]
            yield from sub

    def monthly_contracts(self, product: str) -> list[ContractInfo]:
        def _do():
            out: dict[str, ContractInfo] = {}
            for n in (1, 2):          # 近月/次月別名（主要來源）
                info = self._alias_info(product, n)
                if info:
                    out[info.code] = info
            for c in self._iter_group(product):
                info = self._to_info(c, product)
                if info:
                    out.setdefault(info.code, info)
            if not out:
                raise BrokerError(
                    f"{product} 無月合約（R1/R2 別名與群組列舉皆無結果）")
            return sorted(out.values(), key=lambda x: x.delivery_month)
        return _retry(_do, f"查詢 {product} 合約")

    @staticmethod
    def _parse_date(s) -> dt.date | None:
        s = str(s or "").strip()
        for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None

    def _contract_obj(self, code: str):
        c = self._lookup(code)
        if c is None:
            raise BrokerError(f"找不到合約 {code}")
        return c

    # ── 行情 ────────────────────────────────────────────────────────────────
    def quote(self, code: str) -> Quote:
        def _do():
            snap = self._api.snapshots([self._contract_obj(code)])[0]
            return Quote(
                code=code,
                price=float(snap.close),
                bid=float(getattr(snap, "buy_price", 0.0) or 0.0),
                ask=float(getattr(snap, "sell_price", 0.0) or 0.0),
            )
        return _retry(_do, f"snapshot {code}")

    def day_session_close(self, product: str, delivery_month: str,
                          date: dt.date) -> float | None:
        """指定月份合約在 date 的日盤收盤價（08:45–13:45 最後一根 K 棒收盤）。"""
        def _do():
            import pandas as pd
            code = self.contract_by_month(product, delivery_month).code
            kb = self._api.kbars(self._contract_obj(code),
                                 start=date.isoformat(), end=date.isoformat())
            ts = pd.to_datetime(list(kb.ts))
            closes = list(kb.Close)
            best_i, best_ts = -1, None
            for i, t in enumerate(ts):
                if t.date() != date:
                    continue
                hm = t.hour * 60 + t.minute
                if 8 * 60 + 45 <= hm <= 13 * 60 + 45:   # 僅日盤
                    if best_ts is None or t > best_ts:
                        best_i, best_ts = i, t
            return float(closes[best_i]) if best_i >= 0 else None
        try:
            return _retry(_do, f"kbars {product}{delivery_month} {date}")
        except BrokerError:
            return None

    # ── 帳務 ────────────────────────────────────────────────────────────────
    def equity(self) -> float:
        """權益數（規格 §4）。模擬環境 margin() 回 0 → 退回 SIM_EQUITY。"""
        m = _retry(lambda: self._api.margin(account=self._api.futopt_account),
                   "margin 查詢")
        eq = float(getattr(m, "equity", 0.0) or 0.0)
        if eq <= 0 and self.simulation:
            log.info("模擬環境權益數為 0，採用 SIM_EQUITY=%.0f", config.SIM_EQUITY)
            return config.SIM_EQUITY
        return eq

    def positions(self) -> list[PositionInfo]:
        def _do():
            out = []
            for p in self._api.list_positions(account=self._api.futopt_account):
                out.append(PositionInfo(
                    code=str(p.code),
                    direction=str(getattr(p.direction, "value", p.direction)),
                    quantity=int(p.quantity),
                    avg_price=float(getattr(p, "price", 0.0) or 0.0),
                    last_price=float(getattr(p, "last_price", 0.0) or 0.0)))
            return out
        return _retry(_do, "持倉查詢")

    # ── 下單（不重試）────────────────────────────────────────────────────────
    def place_limit(self, code: str, action: str, quantity: int,
                    price: float) -> OrderHandle:
        sj = self._sj
        order = sj.FuturesOrder(
            price=price, quantity=quantity,
            action=sj.Action.Buy if action == "Buy" else sj.Action.Sell,
            price_type=sj.FuturesPriceType.LMT,     # 禁用市價單（規格 §5）
            order_type=sj.OrderType.ROD,
            octype=sj.FuturesOCType.Auto,
            account=self._api.futopt_account)
        trade = self._api.place_order(self._contract_obj(code), order)
        return OrderHandle(code=code, action=action, quantity=quantity,
                           price=price, raw=trade)

    def place_combo_roll(self, sell_code: str, buy_code: str, quantity: int,
                         net_price: float) -> OrderHandle:
        sj = self._sj
        combo = sj.ComboContract(legs=[
            sj.ComboBase.from_contract(self._contract_obj(buy_code),
                                       action=sj.Action.Buy),
            sj.ComboBase.from_contract(self._contract_obj(sell_code),
                                       action=sj.Action.Sell)])
        order = sj.ComboOrder(
            price=net_price, quantity=quantity,
            price_type=sj.FuturesPriceType.LMT,
            order_type=sj.OrderType.ROD,            # 期貨跨月價差可用 ROD
            octype=sj.FuturesOCType.Auto,
            account=self._api.futopt_account)
        trade = self._api.place_comboorder(combo, order)
        return OrderHandle(code=f"{sell_code}->{buy_code}", action="Roll",
                           quantity=quantity, price=net_price, raw=trade,
                           is_combo=True)

    def update_price(self, handle: OrderHandle, price: float) -> None:
        if handle.is_combo:
            raise BrokerError("組合單不支援改價，請撤單重下")
        self._api.update_order(trade=handle.raw, price=price)
        handle.price = price

    def cancel(self, handle: OrderHandle) -> None:
        if handle.is_combo:
            self._api.cancel_comboorder(handle.raw)
        else:
            self._api.cancel_order(handle.raw)

    def refresh(self, handle: OrderHandle) -> OrderStatusSnapshot:
        def _do():
            if handle.is_combo:
                self._api.update_combostatus(self._api.futopt_account)
                for ct in self._api.list_combotrades():
                    if ct is handle.raw or getattr(ct.status, "id", None) == \
                            getattr(handle.raw.status, "id", object()):
                        handle.raw = ct
                        break
            else:
                self._api.update_status(trade=handle.raw)
            st = handle.raw.status
            deals = getattr(st, "deals", None) or []
            fill_qty = int(getattr(st, "deal_quantity", 0) or 0)
            prices, qtys = [], []
            for d in deals:
                px = float(getattr(d, "price", 0.0) or 0.0)
                q = int(getattr(d, "quantity", 0) or 0)
                if px > 0 and q > 0:
                    prices.append(px * q)
                    qtys.append(q)
            avg = (sum(prices) / sum(qtys)) if qtys else 0.0
            return OrderStatusSnapshot(
                status=str(getattr(st.status, "value", st.status)),
                deal_quantity=fill_qty, avg_fill_price=avg,
                order_price=handle.price,
                msg=str(getattr(st, "msg", "") or ""))
        return _retry(_do, "委託狀態查詢")
