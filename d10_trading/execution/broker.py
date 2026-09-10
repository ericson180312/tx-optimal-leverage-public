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

from d10_trading import config
from d10_trading.ops.audit import audit

log = logging.getLogger("d10.broker")


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
    pnl: float = 0.0          # 未實現損益（券商計算，已含正確乘數）

    @property
    def signed_qty(self) -> int:
        return self.quantity if self.direction == "Buy" else -self.quantity


@dataclass
class OrderStatusSnapshot:
    status: str               # PendingSubmit/Submitted/Filled/PartFilled/Cancelled/Failed
    deal_quantity: int = 0
    avg_fill_price: float = 0.0    # 單腳=成交均價；組合單=淨價（買腳 − 賣腳）
    order_price: float = 0.0
    msg: str = ""
    # 組合單的分腳成交均價 {合約代碼: 均價}；單腳為空。轉倉的稽核要能回答
    # 「兩腳各自成交在哪」——淨價本身無法還原這件事。
    leg_prices: dict[str, float] = field(default_factory=dict)


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
    def equity(self) -> float: raise NotImplementedError
    def positions(self) -> list[PositionInfo]: raise NotImplementedError
    def ca_expiry_days(self) -> int | None:
        """CA 憑證剩餘天數；不可用（模擬/未設 person_id/查詢失敗）回 None。"""
        return None

    def risk_indicator(self) -> float | None:
        """風險指標（百分比數值）；不可用（模擬/查詢失敗）回 None。"""
        return None

    def place_limit(self, code: str, action: str, quantity: int,
                    price: float) -> OrderHandle: raise NotImplementedError
    def place_market_range(self, code: str, action: str,
                           quantity: int) -> OrderHandle: raise NotImplementedError
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

    def settlement_reference(self, product: str, month: str) -> float | None:
        """指定月份合約的參考價（reference）＝**前一交易日結算價**。

        TAIFEX 參考價定義為前一交易日結算價，與 FinMind 的結算價同口徑
        （已實測：Shioaji reference 與 FinMind settlement 對得上）。
        用於雙源核對與 FinMind 缺漏遞補——刻意不用日盤 K 棒收盤，因為
        13:45 那根是「收盤集合競價價」，與結算價本有結構性差異（可達 0.3%），
        兩者比對會誤報。查不到（未載入/新合約無結算）回 None。
        """
        try:
            ref = self.contract_by_month(product, month).reference
        except BrokerError:
            return None
        return float(ref) if ref and ref > 0 else None


def _avg_deal_price(deals) -> tuple[float, int]:
    """成交明細的數量加權均價，回傳 (均價, 總口數)；無有效成交回 (0.0, 0)。"""
    tot_px = tot_q = 0.0
    for d in deals or []:
        px = float(getattr(d, "price", 0.0) or 0.0)
        q = int(getattr(d, "quantity", 0) or 0)
        if px > 0 and q > 0:
            tot_px += px * q
            tot_q += q
    return (tot_px / tot_q, int(tot_q)) if tot_q else (0.0, 0)


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
            raise BrokerError("SJ_API_KEY / SJ_SEC_KEY 未設定（d10_trading/.env）")
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
        audit("broker_login", simulation=self.simulation)

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

    # ── 帳務 ────────────────────────────────────────────────────────────────
    def equity(self) -> float:
        """權益數（規格 §4）。

        真實模式：券商權益數（margin().equity）。
        模擬模式：Shioaji 模擬伺服器的 margin() 恆回 0，改以
            **SIM_EQUITY（起始權益）＋累計已實現損益＋未實現損益** 逐日市值計價。
        為何必須市值計價：若權益凍結在 SIM_EQUITY，下跌時目標口數 L×E/(價×乘數)
        反而上升（愈跌愈買），與真實帳戶（權益隨虧損下降 → 目標口數下降）行為相反，
        G1 在下跌段觀察到的行為將無法代表實盤（2026-07-30 檢討）。
        已實現部分不可省略：每月轉倉會把未實現損益實現掉，只計未實現會讓權益
        在轉倉日跳回起始值。
        """
        m = _retry(lambda: self._api.margin(account=self._api.futopt_account),
                   "margin 查詢")
        eq = float(getattr(m, "equity", 0.0) or 0.0)
        if eq > 0 or not self.simulation:
            return eq

        unreal = sum(p.pnl for p in self.positions())
        real = self._realized_pnl()
        total = config.SIM_EQUITY + real + unreal
        log.info("模擬權益（市值計價）：起始 %.0f ＋已實現 %.0f ＋未實現 %.0f ＝ %.0f",
                 config.SIM_EQUITY, real, unreal, total)
        audit("sim_equity", base=config.SIM_EQUITY, realized=real,
              unrealized=unreal, equity=total)
        if total <= 0:
            raise BrokerError(f"模擬權益已歸零或為負（{total:.0f}）——停止")
        return total

    def risk_indicator(self) -> float | None:
        """風險指標＝權益數 ÷ 未沖銷部位維持保證金 ×100%（券商計算）。

        模擬模式 margin() 恆回 0（含本欄）→ 回 None 略過檢查，與 CA 到期
        查詢同樣屬「G2′ 起才有可見性」的項目。查詢失敗亦回 None（監控功能
        失效不得影響交易）。
        """
        if self.simulation:
            return None
        try:
            m = _retry(lambda: self._api.margin(account=self._api.futopt_account),
                       "margin 查詢（風險指標）")
            v = float(getattr(m, "risk_indicator", 0.0) or 0.0)
        except Exception as e:  # noqa: BLE001 - 監控查詢失敗不影響交易
            log.warning("風險指標查詢失敗（忽略）：%s", type(e).__name__)
            return None
        return v if v > 0 else None

    def _realized_pnl(self) -> float:
        """模擬帳戶的累計已實現損益（paper P&L 端點）。僅模擬模式使用。

        只計 **SIM_PNL_SINCE 起、且屬 PRODUCT** 的紀錄——模擬帳戶可能殘留
        本系統之外的活動（實測：2026-07-16 API 開通測試單 TXFH6 留下
        −235,000），計入會讓 G1 權益失真。查詢失敗 → 回 0 並告警。
        """
        def _do():
            rows = self._api.list_profit_loss(
                account=self._api.futopt_account,
                begin_date=config.SIM_PNL_SINCE,
                end_date=dt.date.today().isoformat())
            total = 0.0
            for r in rows:
                code = str(getattr(r, "code", "") or "")
                if not code.startswith(config.PRODUCT):
                    log.info("模擬已實現損益：略過非 %s 紀錄 %s（%s）",
                             config.PRODUCT, code, getattr(r, "pnl", ""))
                    continue
                total += float(getattr(r, "pnl", 0.0) or 0.0)
            return total
        try:
            return _retry(_do, "已實現損益查詢")
        except Exception:  # noqa: BLE001 - 模擬權益輔助查詢，不可中斷批次
            log.warning("已實現損益查詢失敗，模擬權益暫未計入已實現部分")
            return 0.0

    def positions(self) -> list[PositionInfo]:
        def _do():
            out = []
            for p in self._api.list_positions(account=self._api.futopt_account):
                out.append(PositionInfo(
                    code=str(p.code),
                    direction=str(getattr(p.direction, "value", p.direction)),
                    quantity=int(p.quantity),
                    avg_price=float(getattr(p, "price", 0.0) or 0.0),
                    last_price=float(getattr(p, "last_price", 0.0) or 0.0),
                    pnl=float(getattr(p, "pnl", 0.0) or 0.0)))
            return out
        return _retry(_do, "持倉查詢")

    def ca_expiry_days(self) -> int | None:
        """CA 憑證剩餘天數（get_ca_expiretime；到期預警用，失敗回 None）。"""
        if self.simulation or not config.SJ_PERSON_ID:
            return None
        try:
            exp = self._api.get_ca_expiretime(config.SJ_PERSON_ID)
            d = self._to_datetime(exp)
            if d is None:
                log.warning("CA 到期時間格式無法解析：%r", exp)
                return None
            return (d.date() - dt.date.today()).days
        except Exception as e:  # noqa: BLE001 - 預警功能失敗不影響批次
            log.warning("CA 到期查詢失敗：%s", type(e).__name__)
            return None

    @staticmethod
    def _to_datetime(v) -> dt.datetime | None:
        if isinstance(v, dt.datetime):
            return v
        s = str(v or "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                    "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d%H%M%S", "%Y%m%d"):
            try:
                return dt.datetime.strptime(s, fmt)
            except ValueError:
                continue
        try:
            return dt.datetime.fromisoformat(s)
        except ValueError:
            return None

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
        audit("order_submit", code=code, action=action, quantity=quantity,
              price=price)
        trade = self._api.place_order(self._contract_obj(code), order)
        return OrderHandle(code=code, action=action, quantity=quantity,
                           price=price, raw=trade)

    def place_market_range(self, code: str, action: str,
                           quantity: int) -> OrderHandle:
        """一定範圍市價（MKP）單腳委託——**僅供結算日截止前的強制轉倉**。

        🔴 規格 §5 禁用市價單，此處為明列的例外（見 README §8）：適用範圍限於
        「近月最後交易日、限價路徑已耗盡、且尚未到 EXPIRY_HARD_STOP」。
        理由是替代方案更糟——把近月帶進現金結算，成交價完全不受控且無委託軌跡。

        MKP 有價格保護：交易所轉換為「同買賣別最佳一檔 ± 一定範圍點數」的限價，
        故**不保證成交**（價差大於該點數時不成交），呼叫端必須驗證成交結果。
        price 送 0（不指定價格）。
        """
        sj = self._sj
        order = sj.FuturesOrder(
            price=0, quantity=quantity,
            action=sj.Action.Buy if action == "Buy" else sj.Action.Sell,
            price_type=sj.FuturesPriceType.MKP,
            order_type=sj.OrderType.ROD,
            octype=sj.FuturesOCType.Auto,
            account=self._api.futopt_account)
        audit("order_submit_mkp", code=code, action=action, quantity=quantity)
        trade = self._api.place_order(self._contract_obj(code), order)
        return OrderHandle(code=code, action=action, quantity=quantity,
                           price=0.0, raw=trade)

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
        audit("order_submit_combo", sell=sell_code, buy=buy_code,
              quantity=quantity, net_price=net_price)
        trade = self._api.place_comboorder(combo, order)
        return OrderHandle(code=f"{sell_code}->{buy_code}", action="Roll",
                           quantity=quantity, price=net_price, raw=trade,
                           is_combo=True)

    def update_price(self, handle: OrderHandle, price: float) -> None:
        if handle.is_combo:
            raise BrokerError("組合單不支援改價，請撤單重下")
        audit("order_reprice", code=handle.code, old_price=handle.price,
              new_price=price)
        self._api.update_order(trade=handle.raw, price=price)
        handle.price = price

    def cancel(self, handle: OrderHandle) -> None:
        audit("order_cancel", code=handle.code, quantity=handle.quantity)
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
            # 🔴 兩種 deals 口徑（2026-09-09 首次真實價差單實測）：
            #   單腳   → [Deal, ...]
            #   組合單 → {合約代碼: [Deal, ...]}（每腳一組）
            # 原本一律當 list 迭代，組合單迭代到的是 dict 的「key 字串」，
            # getattr(str, "price") 取不到值 → 成交均價回報 0，成交價在
            # trades.jsonl 與稽核中全部遺失，淨價健全性檢查也必然誤報。
            legs: dict[str, float] = {}
            if isinstance(deals, dict):
                for code, ds in deals.items():
                    px, _ = _avg_deal_price(ds)
                    if px > 0:
                        legs[str(code)] = px
                # 淨價 = 買腳 − 賣腳，與 place_combo_roll 的下單口徑一致
                sell_code, _, buy_code = str(handle.code).partition("->")
                buy_px, sell_px = legs.get(buy_code, 0.0), legs.get(sell_code, 0.0)
                avg = (buy_px - sell_px) if (buy_px and sell_px) else 0.0
            else:
                avg, _ = _avg_deal_price(deals)
            return OrderStatusSnapshot(
                status=str(getattr(st.status, "value", st.status)),
                deal_quantity=fill_qty, avg_fill_price=avg,
                order_price=handle.price,
                msg=str(getattr(st, "msg", "") or ""),
                leg_prices=legs)
        return _retry(_do, "委託狀態查詢")
