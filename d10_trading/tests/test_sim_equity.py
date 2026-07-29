"""模擬權益逐日市值計價（ShioajiBroker.equity）單元測試。

背景（2026-07-30 檢討）：模擬 margin() 恆回 0，原本退回固定 SIM_EQUITY，
導致下跌時目標口數反而上升（愈跌愈買），與真實帳戶行為相反。
改為 SIM_EQUITY ＋已實現 ＋未實現。本測試以樁物件驗證，不觸網路。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d10_trading import config
from d10_trading.execution.broker import BrokerError, ShioajiBroker
from d10_trading.strategy import sizing


def _position(code="TMFH6", qty=3, price=42771.0, last=40963.0, pnl=-54240.0):
    return SimpleNamespace(code=code, direction=SimpleNamespace(value="Buy"),
                           quantity=qty, price=price, last_price=last, pnl=pnl)


def _broker(*, simulation=True, margin_equity=0.0, positions=(), pl_rows=(),
            pl_raises=False):
    b = ShioajiBroker.__new__(ShioajiBroker)
    b.simulation = simulation

    def list_profit_loss(**kw):
        if pl_raises:
            raise RuntimeError("paper P&L unavailable")
        return list(pl_rows)

    b._api = SimpleNamespace(
        futopt_account=SimpleNamespace(),
        margin=lambda **kw: SimpleNamespace(equity=margin_equity),
        list_positions=lambda **kw: list(positions),
        list_profit_loss=list_profit_loss,
    )
    return b


def test_sim_equity_marks_positions_to_market():
    """未實現虧損反映到權益（實測值：3 口 TMF、42771→40963、pnl −54,240）。"""
    b = _broker(positions=[_position()])
    assert b.equity() == pytest.approx(config.SIM_EQUITY - 54240)


def test_sim_equity_includes_realized():
    """已實現損益必須計入——否則轉倉把未實現實現後，權益會跳回起始值。"""
    b = _broker(positions=[_position(pnl=-1000.0)],
                pl_rows=[SimpleNamespace(code="TMFH6", pnl=-53240.0)])
    assert b.equity() == pytest.approx(config.SIM_EQUITY - 53240 - 1000)


def test_realized_excludes_other_products():
    """🔴 回歸：模擬帳戶殘留的非本商品紀錄不得計入。

    實測 2026-07-30：API 開通測試單 TXFH6 留下 −235,000 已實現損益，
    誤計會讓 G1 權益少 33%，並使目標口數由 3 錯降為 2。
    """
    b = _broker(positions=[_position(pnl=-49710.0)],
                pl_rows=[SimpleNamespace(code="TXFH6", pnl=-235000.0),
                         SimpleNamespace(code="TMFH6", pnl=-1000.0)])
    assert b.equity() == pytest.approx(config.SIM_EQUITY - 1000 - 49710)


def test_sim_equity_flat_account_is_base():
    b = _broker(positions=[], pl_rows=[])
    assert b.equity() == pytest.approx(config.SIM_EQUITY)


def test_realized_query_failure_degrades_gracefully():
    """已實現查詢失敗 → 只計未實現，不中斷批次。"""
    b = _broker(positions=[_position()], pl_raises=True)
    assert b.equity() == pytest.approx(config.SIM_EQUITY - 54240)


def test_live_mode_uses_broker_equity_untouched():
    """真實模式一律用 margin().equity，絕不套用 SIM_EQUITY 邏輯。"""
    b = _broker(simulation=False, margin_equity=626710.0,
                positions=[_position()])
    assert b.equity() == pytest.approx(626710.0)


def test_sim_equity_zero_raises():
    b = _broker(positions=[_position(pnl=-config.SIM_EQUITY)])
    with pytest.raises(BrokerError):
        b.equity()


def test_drawdown_no_longer_increases_target():
    """🔴 回歸：這正是本次修正的目的——下跌時目標口數不得反而上升。

    情境：3 口 @42771 起始 70 萬；價格跌到 39000。
    凍結權益會算出 4 口（愈跌愈買）；市值計價算出 3 口（與真實帳戶一致）。
    """
    px = 39000.0
    pnl = 3 * 10 * (px - 42771.0)                       # −113,130
    b = _broker(positions=[_position(last=px, pnl=pnl)])
    eq = b.equity()
    assert eq == pytest.approx(config.SIM_EQUITY + pnl)

    frozen = sizing.target_contracts(2.0, config.SIM_EQUITY, px).target_contracts
    marked = sizing.target_contracts(2.0, eq, px).target_contracts
    assert frozen == 4 and marked == 3                  # 凍結會加碼、市值計價不會
