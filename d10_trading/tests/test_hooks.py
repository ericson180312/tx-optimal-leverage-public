"""部署監控 hook 測試：healthchecks 心跳、CA 到期預警。"""
from __future__ import annotations

from types import SimpleNamespace

from d10_trading import config, main as main_mod
from d10_trading.ops import notify
from d10_trading.tests.test_integration import run, setup_scenario


class _Resp:
    """最小 requests 回應樁（心跳只讀 status_code）。"""

    def __init__(self, status_code=200):
        self.status_code = status_code


def _capture_pings(monkeypatch, status=200):
    calls: list[str] = []

    def fake_get(url, timeout=0):
        calls.append(url)
        return _Resp(status)

    monkeypatch.setattr(config, "HEALTHCHECK_URL", "https://hc.example/ping")
    monkeypatch.setattr(notify.requests, "get", fake_get)
    return calls


def test_heartbeat_pings_on_success(monkeypatch):
    broker = setup_scenario(monkeypatch)          # 無動作日 → exit 0
    calls = _capture_pings(monkeypatch)
    assert run(broker) == 0
    assert calls == ["https://hc.example/ping"]


def test_heartbeat_pings_fail_on_alert(monkeypatch):
    broker = setup_scenario(monkeypatch, held_qty=3, state_qty=2)  # 對帳不符 → 1
    calls = _capture_pings(monkeypatch)
    assert run(broker) == 1
    assert calls == ["https://hc.example/ping/fail"]


def test_heartbeat_pings_even_on_holiday_and_kill_switch(monkeypatch):
    import datetime as dt
    broker = setup_scenario(monkeypatch)
    calls = _capture_pings(monkeypatch)
    assert run(broker, today=dt.date(2026, 7, 18)) == 0   # 週六
    config.KILL_SWITCH_FILE.write_text("stop")
    assert run(broker) == 0                               # 斷電開關
    assert len(calls) == 2                                # 兩次都有 ping


def test_heartbeat_failure_never_breaks_batch(monkeypatch):
    broker = setup_scenario(monkeypatch)
    monkeypatch.setattr(config, "HEALTHCHECK_URL", "https://hc.example/ping")

    def boom(url, timeout=0):
        raise OSError("network down")

    monkeypatch.setattr(notify.requests, "get", boom)
    assert run(broker) == 0                       # ping 失敗不影響批次結果


def test_heartbeat_rejected_status_raises_alert(monkeypatch):
    """🔴 URL 失效時 healthchecks 回 404——必須告警，不可假裝正常。

    否則外部缺席偵測早已失效卻無人察覺（監控系統本身故障的盲點）。
    """
    _capture_pings(monkeypatch, status=404)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(notify, "_telegram", lambda text: sent.append(text) or True)
    notify.heartbeat(0)
    assert sent and "HTTP 404" in sent[0] and "不可信" in sent[0]


def test_heartbeat_non_response_object_never_propagates(monkeypatch):
    """回應物件無 status_code（樁/代理異常）也不得穿透到批次流程。"""
    monkeypatch.setattr(config, "HEALTHCHECK_URL", "https://hc.example/ping")
    monkeypatch.setattr(notify.requests, "get", lambda url, timeout=0: None)
    notify.heartbeat(0)                       # 不拋出即通過


def test_heartbeat_ok_status_sends_no_alert(monkeypatch):
    _capture_pings(monkeypatch, status=200)
    sent: list[str] = []
    monkeypatch.setattr(notify, "_telegram", lambda text: sent.append(text) or True)
    notify.heartbeat(0)
    assert sent == []                         # 200 → 靜默


def test_heartbeat_skipped_when_unset(monkeypatch):
    broker = setup_scenario(monkeypatch)
    called = []
    monkeypatch.setattr(notify.requests, "get",
                        lambda *a, **k: called.append(1))
    assert run(broker) == 0                       # HEALTHCHECK_URL=""（conftest）
    assert called == []


# ── CA 到期預警 ──────────────────────────────────────────────────────────────

def _stub_broker(days):
    return SimpleNamespace(ca_expiry_days=lambda: days, simulation=False)


def _capture_notify(monkeypatch):
    sent = []
    monkeypatch.setattr(main_mod.notify, "send",
                        lambda level, text: sent.append((level, text)))
    return sent


def test_ca_check_warns_under_30_days(monkeypatch):
    sent = _capture_notify(monkeypatch)
    main_mod._check_ca(_stub_broker(20))
    assert sent and sent[0][0] == "WARNING" and "20 天" in sent[0][1]


def test_ca_check_alerts_under_7_days(monkeypatch):
    sent = _capture_notify(monkeypatch)
    main_mod._check_ca(_stub_broker(5))
    assert sent and sent[0][0] == "ALERT"


def test_ca_check_silent_when_ok_or_unavailable(monkeypatch):
    sent = _capture_notify(monkeypatch)
    main_mod._check_ca(_stub_broker(200))         # 距到期尚遠
    main_mod._check_ca(_stub_broker(None))        # 模擬模式/未設 person_id
    assert sent == []


# ── 風險指標監控 ─────────────────────────────────────────────────────────────

def _stub_risk(value):
    return SimpleNamespace(risk_indicator=lambda: value)


def _capture_sends(monkeypatch):
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(main_mod.notify, "send",
                        lambda level, text: sent.append((level, text)))
    return sent


def test_risk_indicator_levels(monkeypatch):
    """正常值靜默；低於各門檻依序 WARNING / ALERT / CRITICAL。"""
    cases = [
        (2400.0, None),          # 常態（2 倍槓桿全額入金）
        (500.0, None),           # 恰在門檻上 → 不告警
        (499.0, "WARNING"),
        (200.0, "WARNING"),      # 恰在 ALERT 門檻上 → 仍為 WARNING
        (199.0, "ALERT"),
        (100.0, "ALERT"),        # 恰在 CRITICAL 門檻上
        (99.0, "CRITICAL"),
    ]
    for value, expect in cases:
        sent = _capture_sends(monkeypatch)
        warnings: list[str] = []
        main_mod._check_risk_indicator(_stub_risk(value), warnings)
        if expect is None:
            assert sent == [], f"{value}% 不應告警"
            assert warnings == []
        else:
            assert sent and sent[0][0] == expect, f"{value}% 應為 {expect}"
            assert f"{value:.0f}%" in sent[0][1]
            assert warnings, "應併入每日摘要"


def test_risk_indicator_silent_when_unavailable(monkeypatch):
    """模擬模式／查詢失敗回 None → 完全靜默（G1 期間即為此情形）。"""
    sent = _capture_sends(monkeypatch)
    warnings: list[str] = []
    main_mod._check_risk_indicator(_stub_risk(None), warnings)
    assert sent == [] and warnings == []


def test_broker_risk_indicator_none_in_simulation():
    """模擬模式不查（margin() 恆回 0）。"""
    from d10_trading.execution.broker import ShioajiBroker
    b = ShioajiBroker.__new__(ShioajiBroker)
    b.simulation = True
    assert b.risk_indicator() is None
