"""ShioajiBroker 合約解析單元測試（以樁物件模擬商品檔，不觸網路）。

實測發現（2026-07-16 dry-run）：`Contracts.Futures["TMF"]` 會拋
KeyError('Contract not found: TMF')——微台無群組鍵，必須走 TMFR1/TMFR2
連續月別名。本測試鎖定該情境。
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from d10_trading.execution.broker import BrokerError, ShioajiBroker


def _contract(code, month, date, target=""):
    return SimpleNamespace(code=code, symbol=code, delivery_month=month,
                           delivery_date=date, reference=100.0,
                           target_code=target)


class _Futures:
    """模擬 shioaji Contracts.Futures：bracket 查碼、屬性查群組。"""

    def __init__(self, by_code: dict, groups: dict):
        self._by_code = by_code
        for name, items in groups.items():
            setattr(self, name, items)

    def __getitem__(self, key):
        if key not in self._by_code:
            raise KeyError(f"Contract not found: {key}")
        return self._by_code[key]


def _broker(by_code, groups):
    b = ShioajiBroker.__new__(ShioajiBroker)      # 不觸發 __init__ 的 config 邏輯
    b.simulation = True
    b._api = SimpleNamespace(Contracts=SimpleNamespace(
        Futures=_Futures(by_code, groups)))
    return b


def test_tmf_resolved_via_r1_r2_aliases(monkeypatch):
    """無 TMF 群組鍵（真實情境）→ R1/R2 別名 + target_code 解析近月/次月。"""
    real1 = _contract("TMFH6", "202608", "2026/08/19")
    real2 = _contract("TMFI6", "202609", "2026/09/16")
    by_code = {
        "TMFR1": _contract("TMFR1", "202608", "2026/08/19", target="TMFH6"),
        "TMFR2": _contract("TMFR2", "202609", "2026/09/16", target="TMFI6"),
        "TMFH6": real1, "TMFI6": real2,
    }
    b = _broker(by_code, groups={})               # 無 TMF 屬性、bracket 也查不到
    out = b.monthly_contracts("TMF")
    assert [(c.code, c.delivery_month) for c in out] == \
        [("TMFH6", "202608"), ("TMFI6", "202609")]
    assert out[0].delivery_date == dt.date(2026, 8, 19)

    front, nxt = b.front_and_next("TMF", dt.date(2026, 7, 16))
    assert (front.code, nxt.code) == ("TMFH6", "TMFI6")


def test_group_enumeration_excludes_aliases_inside_group():
    """🔴 實測情境（2026-07-16）：群組列舉「包含」R1/R2 別名本身，且別名帶
    目標合約的 delivery_month——若不排除，別名會被排進月份序列、
    front_and_next 回傳（近月, R1 別名）→ 月差恆 0、轉倉買賣同一檔。"""
    months = [_contract("TMFH6", "202608", "2026/08/19"),
              _contract("TMFI6", "202609", "2026/09/16"),
              _contract("TMFJ6", "202610", "2026/10/21"),
              _contract("TMFL6", "202612", "2026/12/16"),
              # 別名在群組內、月份=目標月（真實商品檔行為）
              _contract("TMFR1", "202608", "2026/08/19", target="TMFH6"),
              _contract("TMFR2", "202609", "2026/09/16", target="TMFI6")]
    by_code = {c.code: c for c in months}
    b = _broker(by_code, groups={"TMF": months})
    out = b.monthly_contracts("TMF")
    assert [c.code for c in out] == ["TMFH6", "TMFI6", "TMFJ6", "TMFL6"]
    front, nxt = b.front_and_next("TMF", dt.date(2026, 7, 16))
    assert (front.code, nxt.code) == ("TMFH6", "TMFI6")   # 次月是真合約，非 TMFR1


def test_group_attribute_enumeration_and_alias_union():
    """群組屬性可列舉（TXF 情境）：與別名聯集、過濾價差商品與非月合約。"""
    months = [_contract("TXFG6", "202607", "2026/07/15"),
              _contract("TXFH6", "202608", "2026/08/19"),
              _contract("TXFI6", "202609", "2026/09/16"),
              _contract("TXFG6/H6", "202607", "2026/07/15"),   # 價差 → 濾除
              _contract("TXFR1", "", "", target="TXFH6")]      # 別名 → 濾除
    by_code = {c.code: c for c in months}
    b = _broker(by_code, groups={"TXF": months})
    out = b.monthly_contracts("TXF")
    assert [c.code for c in out] == ["TXFG6", "TXFH6", "TXFI6"]
    assert b.contract_by_month("TXF", "202608").code == "TXFH6"


def test_no_contracts_raises(monkeypatch):
    from d10_trading import config
    monkeypatch.setattr(config, "RETRY_MAX", 1)
    monkeypatch.setattr(config, "RETRY_BASE_SECS", 0.0)
    b = _broker({}, groups={})
    with pytest.raises(BrokerError):
        b.monthly_contracts("TMF")

# ── 成交明細解析：組合單 vs 單腳的兩種 deals 口徑 ───────────────────────────

def _deal(px, q):
    from types import SimpleNamespace
    return SimpleNamespace(price=px, quantity=q)


def _stub_broker_with(trade):
    """只裝出 refresh() 需要的最小 _api。"""
    from types import SimpleNamespace
    from d10_trading.execution.broker import ShioajiBroker
    b = ShioajiBroker.__new__(ShioajiBroker)
    b.simulation = False
    b._api = SimpleNamespace(
        futopt_account=object(),
        update_combostatus=lambda acct: None,
        list_combotrades=lambda: [trade],
        update_status=lambda trade=None: None)
    return b


def test_refresh_parses_combo_deals_dict():
    """🔴 組合單的 deals 是 {合約代碼: [Deal]} 的 dict——2026-09-09 首次真實
    跨月價差單實測。原本一律當 list 迭代，迭代到的是 dict 的 key 字串，
    取不到 price → 成交均價回報 0：成交價在 trades.jsonl 與稽核中全部遺失，
    淨價健全性檢查也必然誤報（該次即誤報「相差 −195 點」）。
    """
    from types import SimpleNamespace as NS
    from d10_trading.execution.broker import OrderHandle
    status = NS(status=NS(value="Filled"), deal_quantity=3, msg="",
                deals={"TMFI6": [_deal(47164, 3)], "TMFJ6": [_deal(47356, 3)]})
    trade = NS(status=status)
    h = OrderHandle(code="TMFI6->TMFJ6", action="Roll", quantity=3,
                    price=197.0, raw=trade, is_combo=True)
    snap = _stub_broker_with(trade).refresh(h)
    assert snap.deal_quantity == 3
    # 淨價 = 買腳 − 賣腳，與 place_combo_roll 的下單口徑一致（實測 192）
    assert snap.avg_fill_price == 192.0
    assert snap.leg_prices == {"TMFI6": 47164.0, "TMFJ6": 47356.0}


def test_refresh_parses_single_leg_deals_list():
    """單腳仍是 [Deal] 的 list，且為數量加權均價。"""
    from types import SimpleNamespace as NS
    from d10_trading.execution.broker import OrderHandle
    status = NS(status=NS(value="Filled"), deal_quantity=3, msg="",
                deals=[_deal(46000, 1), _deal(46030, 2)])
    trade = NS(status=status)
    h = OrderHandle(code="TMFI6", action="Buy", quantity=3, price=46030.0,
                    raw=trade)
    snap = _stub_broker_with(trade).refresh(h)
    assert snap.avg_fill_price == (46000 * 1 + 46030 * 2) / 3
    assert snap.leg_prices == {}


def test_refresh_combo_missing_one_leg_yields_zero_net():
    """只有單腳回報時淨價無意義 → 回 0（由上層的健全性檢查告警）。"""
    from types import SimpleNamespace as NS
    from d10_trading.execution.broker import OrderHandle
    status = NS(status=NS(value="PartFilled"), deal_quantity=3, msg="",
                deals={"TMFI6": [_deal(47164, 3)]})
    trade = NS(status=status)
    h = OrderHandle(code="TMFI6->TMFJ6", action="Roll", quantity=3,
                   price=197.0, raw=trade, is_combo=True)
    snap = _stub_broker_with(trade).refresh(h)
    assert snap.avg_fill_price == 0.0
    assert snap.leg_prices == {"TMFI6": 47164.0}


def test_avg_deal_price_weighted_and_empty():
    from d10_trading.execution.broker import _avg_deal_price
    assert _avg_deal_price([_deal(100, 2), _deal(110, 3)]) == (106.0, 5)
    assert _avg_deal_price([]) == (0.0, 0)
    assert _avg_deal_price([_deal(0, 5), _deal(100, 0)]) == (0.0, 0)   # 無效值略過
