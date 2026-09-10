"""操作稽核日誌（audit.jsonl）測試：事件格式、漏斗覆蓋、批次全流程留痕。"""
from __future__ import annotations

import json

from d10_trading import config
from d10_trading.ops import notify
from d10_trading.ops.audit import audit
from d10_trading.tests.test_integration import DROP_JULY, run, setup_scenario


def _events():
    path = config.LOG_DIR / "audit.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_audit_writes_jsonl_with_required_fields():
    audit("unit_test_event", foo=1, bar="中文")
    evs = _events()
    assert len(evs) == 1
    e = evs[0]
    assert e["event"] == "unit_test_event" and e["foo"] == 1 and e["bar"] == "中文"
    # 鎖定「對應關係」而非固定值——寫死 "sim" 會在 G2′ 切換閘門時假失敗
    assert e["mode"] == ("live" if config.LIVE_TRADING else "sim")
    assert "T" in e["ts"]                      # ISO 時間戳


def test_audit_mode_reflects_live_trading(monkeypatch):
    """🔴 mode 欄位須忠實反映 LIVE_TRADING——稽核軌跡是事後分辨真單/模擬單的
    唯一依據，兩種模式都必須正確標記（不依賴當前閘門設定）。"""
    monkeypatch.setattr(config, "LIVE_TRADING", False)
    audit("mode_probe_sim")
    monkeypatch.setattr(config, "LIVE_TRADING", True)
    audit("mode_probe_live")
    modes = {e["event"]: e["mode"] for e in _events()}
    assert modes["mode_probe_sim"] == "sim"
    assert modes["mode_probe_live"] == "live"


def test_notify_is_audited():
    notify.send("WARNING", "稽核測試告警")
    evs = [e for e in _events() if e["event"] == "notify"]
    assert evs and evs[-1]["level"] == "WARNING" and "稽核測試" in evs[-1]["text"]


def test_full_batch_leaves_complete_trail(monkeypatch):
    """regime 切換日（會下單）的批次：生命週期＋決策＋委託終態全部留痕。"""
    broker = setup_scenario(monkeypatch, price_fn=DROP_JULY)
    assert run(broker) == 0
    by_event = {}
    for e in _events():
        by_event.setdefault(e["event"], []).append(e)

    assert "batch_start" in by_event
    dec = by_event["decision"][-1]
    assert dec["regime"] == "BEAR" and dec["target"] == 2 and dec["held"] == 3
    fin = by_event["order_final"][-1]
    assert (fin["action"], fin["filled"], fin["status"]) == ("Sell", 1, "Filled")
    end = by_event["batch_end"][-1]
    assert end["orders_ok"] is True and end["exit_code"] == 0
    assert any(e["level"] == "INFO" for e in by_event["notify"])


def test_batch_stop_is_audited(monkeypatch):
    """對帳不符停止 → batch_stop 事件（含退出碼）。"""
    broker = setup_scenario(monkeypatch, held_qty=3, state_qty=2)
    assert run(broker) == 1
    stops = [e for e in _events() if e["event"] == "batch_stop"]
    assert stops and stops[-1]["exit_code"] == 1
    assert "部位不一致" in stops[-1]["reason"]


def test_kill_switch_is_audited(monkeypatch):
    broker = setup_scenario(monkeypatch)
    config.KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.KILL_SWITCH_FILE.write_text("stop")
    assert run(broker) == 0
    assert any(e["event"] == "kill_switch_exit" for e in _events())


def test_audit_failure_never_raises(monkeypatch):
    """稽核寫入失敗絕不影響流程（規格優先序：正確性>安全性>可觀測性）。"""
    monkeypatch.setattr(config, "LOG_DIR", config.LOG_DIR / "im_a_file")
    (config.LOG_DIR.parent).mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR).write_text("occupy")      # 讓 mkdir 失敗
    audit("should_not_raise")                  # 不拋出即通過


# ── 待補稽核紀錄與摘要 exec_price 的口徑（單元）────────────────────────────────

def test_summary_exec_price_picks_outright_leg():
    """組合單取買腳實價、單腳取成交價、無成交回空字串。"""
    import datetime as dt
    from d10_trading import main as main_mod
    from d10_trading.execution.orders import Fill
    combo = Fill(code="TMFI6->TMFJ6", action="Roll", quantity=3, requested=3,
                 price=192.0, order_price=197.0,
                 legs={"TMFI6": 47164.0, "TMFJ6": 47356.0})
    assert main_mod._summary_exec_price([combo]) == 47356.0     # 買腳（新建腿）
    single = Fill(code="TMFI6", action="Buy", quantity=1, requested=1,
                  price=47000.0, order_price=47002.0)
    assert main_mod._summary_exec_price([single]) == 47000.0
    assert main_mod._summary_exec_price([]) == ""
    del dt


def test_pending_exec_records_split_and_month():
    import datetime as dt
    from d10_trading import main as main_mod
    from d10_trading.execution.orders import Fill
    months = {"TMFI6": "202609", "TMFJ6": "202610"}
    today = dt.date(2026, 9, 9)
    combo = Fill(code="TMFI6->TMFJ6", action="Roll", quantity=3, requested=3,
                 price=192.0, order_price=197.0, note="combo",
                 legs={"TMFI6": 47164.0, "TMFJ6": 47356.0})
    recs = main_mod._pending_exec_records(today, combo, months.get, "early_roll")
    assert recs == [
        {"date": "2026-09-09", "code": "TMFI6", "month": "202609",
         "price": 47164.0, "roll": "early_roll", "path": "combo"},
        {"date": "2026-09-09", "code": "TMFJ6", "month": "202610",
         "price": 47356.0, "roll": "early_roll", "path": "combo"}]
    # 非轉倉的單腳：roll 為空，故不會被 _flush_roll_slippage 配對
    single = Fill(code="TMFJ6", action="Buy", quantity=1, requested=1,
                  price=47356.0, order_price=47358.0)
    assert main_mod._pending_exec_records(today, single, months.get) == [
        {"date": "2026-09-09", "code": "TMFJ6", "month": "202610",
         "price": 47356.0, "roll": "", "path": ""}]


def test_flush_roll_slippage_computes_spread_cost(tmp_path, monkeypatch):
    """🔴 轉倉的成本口徑 = 執行月差 − 結算月差。

    兩腿共同的時間滑價（13:20 執行 vs 13:45 結算）在轉倉中完全對消——一腿買、
    一腿賣——故單看各腿的 exec_vs_close 偏差會高估成本。以 2026-09-09 實測值
    驗證：兩腿偏差皆 −0.030%，但月差滑價為 0（價差單把月差鎖住了）。
    """
    import csv
    from d10_trading import main as main_mod
    resolved = [
        ({"date": "2026-09-09", "code": "TMFI6", "month": "202609",
          "price": 47164.0, "roll": "early_roll", "path": "combo"}, 47178.0),
        ({"date": "2026-09-09", "code": "TMFJ6", "month": "202610",
          "price": 47356.0, "roll": "early_roll", "path": "combo"}, 47370.0),
    ]
    main_mod._flush_roll_slippage(resolved)
    rows = list(csv.DictReader(
        open(config.LOG_DIR / "roll_slippage.csv", encoding="utf-8-sig")))
    assert len(rows) == 1
    r = rows[0]
    assert float(r["exec_spread"]) == 192.0        # 47356 − 47164
    assert float(r["settle_spread"]) == 192.0      # 47370 − 47178
    assert float(r["slippage_points"]) == 0.0      # 價差單把月差鎖住
    assert r["path"] == "combo" and r["near_code"] == "TMFI6"


def test_flush_roll_slippage_matches_two_leg_cost():
    """8/12 兩腳的實際數字：執行月差 164.67 − 結算月差 157 = +7.67 點。"""
    import csv
    from d10_trading import main as main_mod
    resolved = [
        ({"date": "2026-08-12", "code": "TMFH6", "month": "202608",
          "price": 45380.333333, "roll": "early_roll", "path": "two_legs"}, 45516.0),
        ({"date": "2026-08-12", "code": "TMFI6", "month": "202609",
          "price": 45545.0, "roll": "early_roll", "path": "two_legs"}, 45673.0),
    ]
    main_mod._flush_roll_slippage(resolved)
    r = list(csv.DictReader(
        open(config.LOG_DIR / "roll_slippage.csv", encoding="utf-8-sig")))[-1]
    assert float(r["slippage_points"]) == 7.67
    assert r["path"] == "two_legs"


def test_flush_roll_slippage_ignores_incomplete_or_non_roll():
    """單腿、非轉倉、同月份 → 不產生任何列（寧缺勿誤）。"""
    from d10_trading import main as main_mod
    base = {"date": "2026-09-09", "month": "202609", "price": 47164.0,
            "path": "combo"}
    for resolved in (
        [({**base, "code": "A", "roll": "early_roll"}, 47178.0)],          # 只一腿
        [({**base, "code": "A", "roll": ""}, 47178.0),
         ({**base, "code": "B", "roll": "", "month": "202610"}, 47370.0)],  # 非轉倉
        [({**base, "code": "A", "roll": "early_roll"}, 47178.0),
         ({**base, "code": "B", "roll": "early_roll"}, 47370.0)],           # 同月份
    ):
        main_mod._flush_roll_slippage(resolved)
    assert not (config.LOG_DIR / "roll_slippage.csv").exists()
