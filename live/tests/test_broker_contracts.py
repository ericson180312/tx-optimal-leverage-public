"""ShioajiBroker 合約解析單元測試（以樁物件模擬商品檔，不觸網路）。

實測發現（2026-07-16 dry-run）：`Contracts.Futures["TMF"]` 會拋
KeyError('Contract not found: TMF')——微台無群組鍵，必須走 TMFR1/TMFR2
連續月別名。本測試鎖定該情境。
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from live.execution.broker import BrokerError, ShioajiBroker


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
    from live import config
    monkeypatch.setattr(config, "RETRY_MAX", 1)
    monkeypatch.setattr(config, "RETRY_BASE_SECS", 0.0)
    b = _broker({}, groups={})
    with pytest.raises(BrokerError):
        b.monthly_contracts("TMF")