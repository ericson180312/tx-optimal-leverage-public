"""整合測試（規格 §8-3）：FakeBroker 下的完整日批次 dry-run。

情境涵蓋：無動作日、首次建倉、regime 切換日、轉倉日、轉倉+切換重疊日、
資料過期、對帳不符、下單 timeout、冪等性、斷電開關、保險絲、口數上限、dry-run。
"""
from __future__ import annotations

import datetime as dt

from d10_trading import config
from d10_trading import main as main_mod
from d10_trading.data import calendar
from d10_trading.data import series as series_mod
from d10_trading.data.series import build_signal_series
from d10_trading.execution.broker import PositionInfo
from d10_trading.main import run_daily
from d10_trading.ops import state as state_mod
from d10_trading.ops.state import PositionState, State
from d10_trading.tests.conftest import FakeBroker, flat_quote, make_contracts, synth_cdf

START = dt.date(2025, 1, 2)
TODAY = dt.date(2026, 7, 16)               # 週四、非結算週轉倉窗
ROLL_TODAY = dt.date(2026, 8, 12)          # 2026-08 結算日 8/19 的 T−5

CONST = lambda d: 46000.0                                        # noqa: E731
DROP_JULY = lambda d: 40000.0 if d >= dt.date(2026, 7, 1) else 46000.0   # noqa: E731
DROP_AUG = lambda d: 40000.0 if d >= dt.date(2026, 8, 1) else 46000.0    # noqa: E731


def _months_around(today: dt.date) -> list[str]:
    base = today.year * 12 + today.month - 1
    return [f"{v // 12:04d}{v % 12 + 1:02d}" for v in range(base - 1, base + 3)]


def _front_month(today: dt.date) -> str:
    for m in _months_around(today):
        if calendar.settlement_date(int(m[:4]), int(m[4:])) >= today:
            return m
    raise AssertionError


def setup_scenario(monkeypatch, *, today=TODAY, price_fn=CONST,
                   equity=700_000.0, held_qty=3, state_qty=None,
                   state_lev=2.0, spread=20.0, fill_mode="fill",
                   cdf_end=None, no_state=False):
    """組出（broker, 前置 state）；價格/持倉/行情皆可控。"""
    cdf = synth_cdf(price_fn, START, cdf_end or calendar.prev_trading_day(today))
    monkeypatch.setattr(series_mod, "load_tx_contracts", lambda **kw: cdf)
    ss = build_signal_series(cdf)

    months = sorted(set(_months_around(today)) | {ss.held_contract})
    # 雙源核對用 Shioaji 參考價（=前日結算價）；令 TXF 持倉月參考價 = FinMind
    # 最後官方價 → 差 0，乾淨情境不觸發告警（與實際運行一致）。
    txf_refs = {ss.held_contract: ss.last_official_price}
    contracts = (make_contracts("TXF", months, references=txf_refs)
                 + make_contracts("TMF", months))
    px = float(price_fn(today))
    front = _front_month(today)
    quotes = {}
    for c in contracts:
        p = px + (spread if c.delivery_month > front else 0.0)
        quotes[c.code] = flat_quote(c.code, p)

    positions = []
    if held_qty > 0:
        positions = [PositionInfo(code=f"TMF{front}", direction="Buy",
                                  quantity=held_qty, avg_price=px)]
    broker = FakeBroker(equity=equity, contracts=contracts, quotes=quotes,
                        positions=positions, fill_mode=fill_mode)

    if not no_state:
        sq = held_qty if state_qty is None else state_qty
        st = State(regime="BULL", exec_leverage=state_lev,
                   signal_date=calendar.prev_trading_day(today).isoformat(),
                   positions=([PositionState(code=f"TMF{front}", month=front,
                                             qty=sq, avg_price=px)] if sq else []),
                   equity=equity)
        state_mod.save(st)
    return broker


def run(broker, today=TODAY, **kw):
    return run_daily(broker, today=today, ignore_window=True, **kw)


# ── 基本流 ───────────────────────────────────────────────────────────────────

def test_noop_day(monkeypatch):
    broker = setup_scenario(monkeypatch)          # L=2、46000 → 目標 3 = 現有 3
    assert run(broker) == 0
    assert broker.placed == []
    st = state_mod.load()
    assert st.executed_date == TODAY.isoformat()
    assert st.regime == "BULL" and st.exec_leverage == 2.0


def test_initial_entry_buys_target(monkeypatch):
    broker = setup_scenario(monkeypatch, held_qty=0)
    assert run(broker) == 0
    assert len(broker.placed) == 1
    h = broker.placed[0]
    assert (h.action, h.quantity) == ("Buy", 3)
    st = state_mod.load()
    assert st.total_qty() == 3


def test_late_run_after_close_keeps_t_plus_1(monkeypatch):
    """收盤後補跑（FinMind 已含今日資料）：執行訊號仍為昨日，行為與 13:20 跑一致。"""
    broker = setup_scenario(monkeypatch, cdf_end=TODAY)   # 官方序列含今日
    assert run(broker) == 0
    assert broker.placed == []                    # 目標=現有，無動作
    st = state_mod.load()
    assert st.signal_date == calendar.prev_trading_day(TODAY).isoformat()
    assert st.preview and st.preview["date"] == TODAY.isoformat()


def test_regime_switch_sells_down(monkeypatch):
    """7/1 起跌破 200MA → 昨日訊號 L=1 → 今日減碼 3→2。"""
    broker = setup_scenario(monkeypatch, price_fn=DROP_JULY)
    assert run(broker) == 0
    assert len(broker.placed) == 1
    h = broker.placed[0]
    assert (h.action, h.quantity) == ("Sell", 1)
    st = state_mod.load()
    assert st.exec_leverage == config.L_LOW and st.regime == "BEAR"
    assert st.total_qty() == 2


# ── 轉倉 ─────────────────────────────────────────────────────────────────────

def test_roll_day_uses_combo(monkeypatch):
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY)
    assert run(broker, today=ROLL_TODAY) == 0
    combos = [h for h in broker.placed if h.is_combo]
    assert len(combos) == 1 and combos[0].quantity == 3
    st = state_mod.load()
    assert st.positions[0].month == "202609" and st.total_qty() == 3


def test_roll_negative_spread_waits_until_expiry(monkeypatch):
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY, spread=-15.0)
    assert run(broker, today=ROLL_TODAY) == 0
    assert broker.placed == []                    # 逆價差 → 結算日才轉


def test_roll_plus_rebalance_overlap_nets_once(monkeypatch):
    """轉倉日 + regime 已切換：combo 共同口數 + 差額平近月，一次到位。"""
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY, price_fn=DROP_AUG)
    assert run(broker, today=ROLL_TODAY) == 0
    combos = [h for h in broker.placed if h.is_combo]
    singles = [h for h in broker.placed if not h.is_combo]
    assert len(combos) == 1 and combos[0].quantity == 2      # 目標 2 口
    assert len(singles) == 1
    assert (singles[0].action, singles[0].quantity) == ("Sell", 1)
    assert singles[0].code == "TMF202608"                    # 差額平近月
    st = state_mod.load()
    assert st.total_qty() == 2 and st.positions[0].month == "202609"


def test_roll_combo_fallback_two_legs(monkeypatch):
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY)
    broker.combo_raises = True
    assert run(broker, today=ROLL_TODAY) == 0
    legs = [(h.action, h.code, h.quantity) for h in broker.placed]
    assert legs == [("Buy", "TMF202609", 3), ("Sell", "TMF202608", 3)]
    st = state_mod.load()
    assert st.positions[0].month == "202609" and st.total_qty() == 3


# ── 防護與異常 ───────────────────────────────────────────────────────────────

def test_stale_data_stops_without_orders(monkeypatch):
    stale_end = calendar.prev_trading_day(
        calendar.prev_trading_day(calendar.prev_trading_day(TODAY)))
    broker = setup_scenario(monkeypatch, cdf_end=stale_end)
    assert run(broker) == 1
    assert broker.placed == []
    assert state_mod.load().executed_date is None


def test_reconcile_mismatch_stops(monkeypatch):
    broker = setup_scenario(monkeypatch, held_qty=3, state_qty=2)
    assert run(broker) == 1
    assert broker.placed == []
    assert state_mod.load().executed_date is None


def test_order_timeout_cancels_and_alerts(monkeypatch):
    broker = setup_scenario(monkeypatch, price_fn=DROP_JULY, fill_mode="never")
    assert run(broker) == 1                       # 未足額 → 告警退出碼
    st = state_mod.load()
    assert st.executed_date == TODAY.isoformat()  # 已執行（明日自然重試）
    assert st.total_qty() == 3                    # 部位以 API 讀回為準（未變）


def test_idempotent_second_run(monkeypatch):
    broker = setup_scenario(monkeypatch)
    assert run(broker) == 0
    n_login, n_orders = broker.login_count, len(broker.placed)
    assert run(broker) == 0                       # 同日重跑
    assert broker.login_count == n_login          # 未再登入
    assert len(broker.placed) == n_orders         # 未重複下單


def test_kill_switch_exits_immediately(monkeypatch):
    broker = setup_scenario(monkeypatch)
    config.KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.KILL_SWITCH_FILE.write_text("stop")
    assert run(broker) == 0
    assert broker.login_count == 0 and broker.placed == []


def test_non_trading_day_exits(monkeypatch):
    broker = setup_scenario(monkeypatch)
    sat = dt.date(2026, 7, 18)
    assert run(broker, today=sat) == 0
    assert broker.login_count == 0


def test_fuse_triggers_reduction_and_critical_exit(monkeypatch):
    # 權益 40 萬、持倉 3 口×46000×10 → L_cur=3.45 ≥ 3.2
    broker = setup_scenario(monkeypatch, equity=400_000.0)
    assert run(broker) == 2
    assert len(broker.placed) == 1
    h = broker.placed[0]
    assert (h.action, h.quantity) == ("Sell", 1)  # 目標 round(1.74)=2
    assert state_mod.load().total_qty() == 2


def test_caps_reject_without_orders(monkeypatch):
    broker = setup_scenario(monkeypatch, equity=1_200_000.0)  # 目標 5 > 上限 4
    assert run(broker) == 1
    assert broker.placed == []
    assert state_mod.load().executed_date is None


def test_dry_run_places_nothing_and_keeps_state(monkeypatch):
    broker = setup_scenario(monkeypatch, price_fn=DROP_JULY)
    assert run(broker, dry_run=True) == 0
    assert broker.placed == []
    st = state_mod.load()
    assert st.executed_date is None               # dry-run 不落地執行旗標
    assert st.exec_leverage == 2.0                # 狀態維持前值


# ── exec_vs_close 稽核（結算價基準）─────────────────────────────────────────

def test_exec_vs_close_uses_settlement(monkeypatch):
    """_flush_exec_audit 以 FinMind TX 結算價（非 K 棒收盤）補記，並移出 pending。"""
    cdf = synth_cdf(CONST, START, calendar.prev_trading_day(TODAY))
    ss = build_signal_series(cdf)
    d, month = ss.last_official_date, ss.held_contract
    st = State(pending_exec_audit=[
        {"date": d.isoformat(), "code": f"TMF{month}", "month": month, "price": 46055.0},
        {"date": TODAY.isoformat(), "code": "TMFZ9", "month": "209912", "price": 1.0},
    ])
    main_mod._flush_exec_audit(st, ss)

    # 有結算者補記並移出；無結算者保留待下次
    assert [r["month"] for r in st.pending_exec_audit] == ["209912"]
    rows = (config.LOG_DIR / "exec_vs_close.csv").read_text(
        encoding="utf-8-sig").splitlines()
    assert rows[0] == "date,code,exec_price,settlement,diff_pct"
    fields = rows[1].split(",")
    assert fields[3] == "46000.0"                 # settlement = TX 結算（CONST）
    assert fields[2] == "46055.0"                 # 執行價
    assert abs(float(fields[4]) - (46055.0 / 46000.0 - 1.0)) < 1e-9


def test_exec_vs_close_skips_roll_records(monkeypatch):
    """轉倉淨價無單一結算對照 → 跳過（不補記、不留 pending）。"""
    cdf = synth_cdf(CONST, START, calendar.prev_trading_day(TODAY))
    ss = build_signal_series(cdf)
    d = ss.last_official_date
    st = State(pending_exec_audit=[
        {"date": d.isoformat(), "code": "TMFH6->TMFI6", "month": "202609", "price": 250.0},
    ])
    main_mod._flush_exec_audit(st, ss)
    assert st.pending_exec_audit == []
    assert not (config.LOG_DIR / "exec_vs_close.csv").exists()


# ── 盤外執行不得佔用當日冪等名額 ─────────────────────────────────────────────

def _at(monkeypatch, hhmm: str):
    """把 _now() 固定在今天的指定時刻（台北時區）。"""
    import datetime as _dt
    from zoneinfo import ZoneInfo
    h, m = (int(x) for x in hhmm.split(":"))
    fixed = _dt.datetime.combine(TODAY, _dt.time(h, m),
                                 tzinfo=ZoneInfo(config.TIMEZONE))
    monkeypatch.setattr(main_mod, "_now", lambda: fixed)


def test_out_of_window_run_does_not_consume_the_day(monkeypatch):
    """🔴 2026-09-01 實際踩到：凌晨手動觸發（無動作日）曾標記 executed_date，
    導致 13:20 正式排程被冪等跳過——盤外價格的『無動作』結論不能代表 13:20。"""
    broker = setup_scenario(monkeypatch)
    _at(monkeypatch, "02:15")
    assert run_daily(broker, today=TODAY) == 0        # 不傳 ignore_window
    st = state_mod.load()
    assert st.executed_date is None, "盤外執行不得標記本日已執行"

    # 13:20 正式排程仍會完整執行並標記
    _at(monkeypatch, "13:20")
    assert run_daily(broker, today=TODAY) == 0
    assert state_mod.load().executed_date == TODAY.isoformat()


def test_in_window_run_commits_the_day(monkeypatch):
    broker = setup_scenario(monkeypatch)
    _at(monkeypatch, "13:20")
    assert run_daily(broker, today=TODAY) == 0
    assert state_mod.load().executed_date == TODAY.isoformat()


def test_ignore_window_still_commits(monkeypatch):
    """--ignore-window 是明確的人為意圖 → 仍標記（避免重複下單）。"""
    broker = setup_scenario(monkeypatch)
    _at(monkeypatch, "02:15")
    assert run_daily(broker, today=TODAY, ignore_window=True) == 0
    assert state_mod.load().executed_date == TODAY.isoformat()


# ── 價差單送出後的例外不得退回兩腳（重複建倉防線）─────────────────────────────

def test_combo_post_submit_error_does_not_fall_back(monkeypatch):
    """🔴 2026-09-02 審查發現：價差單送出後若 cancel/refresh 拋例外，
    原本會被 _roll_common 當成「價差單不可用」而改走兩腳全額下單——
    此時已部分成交，會在既有部位上重複建倉、曝險加倍（真錢路徑）。

    送出後的例外必須就地轉為告警並停止，只有**送出前**的例外才可退兩腳。
    """
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY)
    broker.combo_fill_mode = "partial:2"           # 價差單部分成交 2/3 口

    def boom(handle):
        raise RuntimeError("API 撤單失敗")

    monkeypatch.setattr(broker, "cancel", boom)
    assert run(broker, today=ROLL_TODAY) == 1          # 告警停止

    combos = [h for h in broker.placed if h.is_combo]
    singles = [h for h in broker.placed if not h.is_combo]
    assert len(combos) == 1, "價差單只應送出一次"
    assert singles == [], "🔴 絕不可退回兩腳——已部分成交，會重複建倉"


def test_combo_pre_submit_error_still_falls_back(monkeypatch):
    """對照：送出前失敗（如模擬環境不支援組合單）仍須退回兩腳。"""
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY)
    broker.combo_raises = True
    assert run(broker, today=ROLL_TODAY) == 0
    legs = [(h.action, h.code, h.quantity) for h in broker.placed]
    assert legs == [("Buy", "TMF202609", 3), ("Sell", "TMF202608", 3)]


def test_combo_net_price_deviation_alerts(monkeypatch):
    """成交淨價與報價月差偏離過大 → 告警（抓淨價正負號慣例錯誤）。

    FakeBroker 以「限價成交」模擬，故成交淨價 = 限價 = 月差 + 容忍，
    偏離僅 2 檔，不應告警；把月差放大到量級相當於正負號翻轉時才須告警。
    """
    from d10_trading.execution import orders
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY, spread=20.0)
    assert run(broker, today=ROLL_TODAY) == 0
    st = state_mod.load()
    assert not any("正負號" in w for w in (st.notes or "")), "正常滑價不應告警"

    # 直接驗告警邏輯：成交淨價與報價月差差 40 點（> 20 點門檻）
    assert orders._COMBO_NET_DEV_ALERT == 20.0
    assert abs(-20.0 - 20.0) > orders._COMBO_NET_DEV_ALERT   # 正負號翻轉的量級


# ── 轉倉的執行品質稽核：逐腿月份 + 組合單拆腳 ─────────────────────────────────

def test_combo_roll_yields_per_leg_exec_audit(monkeypatch):
    """🔴 組合單轉倉須產生「每腿一筆」的待補稽核，且月份各自正確。

    2026-09-09 首次真實價差單發現兩個缺陷：整筆的 code 含 "->" 會被補寫端
    跳過（轉倉日零稽核，兩腳路徑卻有 2 列）；月份原本一律寫目標合約，
    使近月腿被拿去與次月結算價比較。
    """
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY, spread=20.0)
    assert run(broker, today=ROLL_TODAY) == 0
    st = state_mod.load()
    recs = {r["code"]: r for r in st.pending_exec_audit}
    assert set(recs) == {"TMF202608", "TMF202609"}, "應每腿一筆，且不含 '->' 整筆"
    assert recs["TMF202608"]["month"] == "202608"      # 近月腿配近月結算
    assert recs["TMF202609"]["month"] == "202609"      # 次月腿配次月結算
    # 分腳價須為單腳實價（實價量級），不可是淨價（數十點量級）
    near, nxt = recs["TMF202608"]["price"], recs["TMF202609"]["price"]
    assert near > 40_000 and nxt > 40_000, "須為實價量級，不可是淨價"
    assert nxt > near, "正價差情境：次月實價高於近月"


def test_combo_roll_summary_exec_price_is_outright(monkeypatch):
    """摘要 exec_price 與 official_close 同欄比較 → 須為新建腿實價，不可放淨價
    （放淨價會讓 exec_vs_close_pct 算出約 −99% 的假偏差）。"""
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY, spread=20.0)
    assert run(broker, today=ROLL_TODAY) == 0
    import csv
    row = [r for r in csv.DictReader(
        open(config.LOG_DIR / "daily_summary.csv", encoding="utf-8-sig"))
        if r["date"] == ROLL_TODAY.isoformat()][-1]
    assert float(row["exec_price"]) > 40_000, "不可是淨價（數十點量級）"
    # 兩個稽核面須互相一致：摘要的 exec_price = 待補紀錄裡新建腿（次月）的實價
    nxt_rec = [r for r in state_mod.load().pending_exec_audit
               if r["code"] == "TMF202609"][0]
    assert float(row["exec_price"]) == nxt_rec["price"]


def test_two_leg_roll_exec_audit_uses_each_leg_month(monkeypatch):
    """兩腳路徑的既存偏差同樣修掉：近月腿不再配次月結算價。"""
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY)
    broker.combo_raises = True                          # 逼走兩腳
    assert run(broker, today=ROLL_TODAY) == 0
    st = state_mod.load()
    recs = {r["code"]: r["month"] for r in st.pending_exec_audit}
    assert recs == {"TMF202609": "202609", "TMF202608": "202608"}


def test_single_leg_rebalance_exec_audit_month(monkeypatch):
    """非轉倉的單腳再平衡：月份取該合約自己的月份。"""
    broker = setup_scenario(monkeypatch, price_fn=DROP_JULY)   # regime 切換 → 賣 1 口
    assert run(broker) == 0
    st = state_mod.load()
    assert len(st.pending_exec_audit) == 1
    r = st.pending_exec_audit[0]
    assert r["month"] == _front_month(TODAY) and r["price"] > 30_000


# ── 月差追蹤：逐日紀錄與轉倉月差滑價 ─────────────────────────────────────────

def _roll_spread_rows():
    import csv
    path = config.LOG_DIR / "roll_spread.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def test_roll_spread_recorded_daily(monkeypatch):
    """每個交易日都留一列月差，且 t_minus 為距結算的交易日數。"""
    broker = setup_scenario(monkeypatch, spread=25.0)
    assert run(broker) == 0
    rows = _roll_spread_rows()
    assert len(rows) == 1
    r = rows[0]
    assert r["date"] == TODAY.isoformat()
    assert float(r["spread"]) == 25.0
    assert r["rolled"] == "", "非轉倉日 rolled 應為空"
    # t_minus：TODAY(7/16) 距 202607 結算日的交易日數，須為正且與行事曆一致
    settle = dt.date.fromisoformat(r["settlement"])
    assert int(r["t_minus"]) == calendar.trading_days_between(TODAY, settle)
    assert int(r["t_minus"]) > 0
    # spread_pct 供跨價位比較
    assert float(r["spread_pct"]) == round(25.0 / float(r["near_price"]), 6)


def test_roll_spread_marks_roll_day_and_t_minus_5(monkeypatch):
    """T−5 提前轉倉日：t_minus 恰為 ROLL_LEAD_DAYS，rolled 記下 reason。"""
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY, spread=20.0)
    assert run(broker, today=ROLL_TODAY) == 0
    r = _roll_spread_rows()[-1]
    assert r["rolled"] == "early_roll"
    assert int(r["t_minus"]) == config.ROLL_LEAD_DAYS
    assert r["near_month"] == "202608" and r["next_month"] == "202609"


def test_roll_spread_not_written_on_dry_run(monkeypatch):
    broker = setup_scenario(monkeypatch)
    assert run_daily(broker, today=TODAY, ignore_window=True, dry_run=True) == 0
    assert _roll_spread_rows() == []


def test_roll_day_stages_slippage_context(monkeypatch):
    """轉倉日把兩腿連同 roll/path 脈絡暫存；月差滑價於**次日**結算到位後才算
    （批次的補寫在成交之前執行，當日結算價尚不存在——與 9/9→9/10 實況一致）。"""
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY, spread=20.0)
    assert run(broker, today=ROLL_TODAY) == 0
    recs = state_mod.load().pending_exec_audit
    assert len(recs) == 2
    assert all(r["roll"] == "early_roll" and r["path"] == "combo" for r in recs)
    assert not (config.LOG_DIR / "roll_slippage.csv").exists(), "當日不應已算出"


def test_plain_rebalance_stages_no_roll_context(monkeypatch):
    """非轉倉的單腳再平衡：roll 標記為空 → 永不會被配對成月差滑價。"""
    broker = setup_scenario(monkeypatch, price_fn=DROP_JULY)   # 減碼 1 口
    assert run(broker) == 0
    recs = state_mod.load().pending_exec_audit
    assert len(recs) == 1 and recs[0]["roll"] == ""


# ── 結算日（近月最後交易日）的轉倉時序護欄 ───────────────────────────────────

EXPIRY_TODAY = dt.date(2026, 8, 19)        # 202608 結算日＝其最後交易日


def _expiry_scenario(monkeypatch, hhmmss: str, *, spread=20.0,
                     combo_fill="fill", mkp_fill="fill"):
    """結算日情境：持近月 3 口，時鐘固定在指定時刻。"""
    from zoneinfo import ZoneInfo
    broker = setup_scenario(monkeypatch, today=EXPIRY_TODAY, spread=spread)
    broker.combo_fill_mode = combo_fill
    broker.mkp_fill_mode = mkp_fill
    h, m, s = (int(x) for x in hhmmss.split(":"))
    fixed = dt.datetime.combine(EXPIRY_TODAY, dt.time(h, m, s),
                                tzinfo=ZoneInfo(config.TIMEZONE))
    monkeypatch.setattr(main_mod, "_now", lambda: fixed)
    return broker


def test_expiry_guard_only_on_last_trading_day(monkeypatch):
    """🔴 護欄只在近月最後交易日建立——其他日行為必須完全不變。"""
    broker = setup_scenario(monkeypatch, today=ROLL_TODAY)
    front, _ = broker.front_and_next(config.PRODUCT, ROLL_TODAY)
    assert main_mod._expiry_guard(ROLL_TODAY, front) is None
    front2, _ = broker.front_and_next(config.PRODUCT, EXPIRY_TODAY)
    g = main_mod._expiry_guard(EXPIRY_TODAY, front2)
    assert g is not None
    assert g.timeout_secs == config.EXPIRY_ORDER_TIMEOUT_SECS
    assert g.force_at == dt.time.fromisoformat(config.EXPIRY_FORCE_TIME)


def test_expiry_limit_path_succeeds_no_mkp(monkeypatch):
    """限價路徑成功 → 不動用一定範圍市價（MKP 只是後備）。"""
    broker = _expiry_scenario(monkeypatch, "13:21:00")
    assert run(broker, today=EXPIRY_TODAY) == 0
    assert [h for h in broker.placed if h.is_combo], "應走價差單"
    assert not any(h.price == 0.0 for h in broker.placed), "不該出現 MKP"
    st = state_mod.load()
    assert st.positions[0].month == "202609" and st.total_qty() == 3


def test_expiry_past_force_time_goes_straight_to_mkp(monkeypatch):
    """🔴 已過強制轉倉時點：略過耗時的兩腳限價流程，直接兩腳 MKP。"""
    broker = _expiry_scenario(monkeypatch, "13:27:30", combo_fill="never")
    assert run(broker, today=EXPIRY_TODAY) == 0
    mkp = [h for h in broker.placed if h.price == 0.0]
    assert len(mkp) == 2, f"應恰兩腳 MKP，實際 {len(mkp)}"
    # 🔴 先平近月、後建次月（與平日相反：近月才有 13:30 死線）
    assert (mkp[0].action, mkp[0].code) == ("Sell", "TMF202608")
    assert (mkp[1].action, mkp[1].code) == ("Buy", "TMF202609")
    st = state_mod.load()
    assert st.positions[0].month == "202609" and st.total_qty() == 3


def test_expiry_mkp_unfilled_alerts_cash_settlement(monkeypatch):
    """MKP 不保證成交（轉為最佳一檔±範圍點數的限價）→ 未成交須明確告警,
    說明近月將由現金結算平倉、價格不受控制。"""
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(main_mod.notify, "send",
                        lambda level, text: sent.append((level, text)))
    broker = _expiry_scenario(monkeypatch, "13:27:30",
                              combo_fill="never", mkp_fill="never")
    assert run(broker, today=EXPIRY_TODAY) == 1        # 告警停止
    texts = " ".join(t for _, t in sent)
    assert "現金結算" in texts and "不受控制" in texts
    assert any(lv == "ALERT" for lv, _ in sent)


def test_expiry_past_hard_stop_sends_nothing(monkeypatch):
    """🔴 已過硬性停止時點：一張單都不能送（近月早已停止交易）。"""
    broker = _expiry_scenario(monkeypatch, "13:29:00", combo_fill="never")
    assert run(broker, today=EXPIRY_TODAY) == 1
    assert not any(h.price == 0.0 for h in broker.placed), "不得送出任何 MKP"


def test_expiry_force_reuses_actual_positions(monkeypatch):
    """強制轉倉以券商實際部位計算需求,不沿用先前嘗試的成交統計。

    先讓價差單部分成交 2/3（近月剩 1、次月已有 2）,再走強制路徑:
    應只賣 1 口近月、買 1 口次月。
    """
    broker = _expiry_scenario(monkeypatch, "13:27:30", combo_fill="partial:2")
    assert run(broker, today=EXPIRY_TODAY) == 0
    mkp = [h for h in broker.placed if h.price == 0.0]
    assert [(h.action, h.code, h.quantity) for h in mkp] == [
        ("Sell", "TMF202608", 1), ("Buy", "TMF202609", 1)]
    st = state_mod.load()
    assert st.positions[0].month == "202609" and st.total_qty() == 3


# ── 臨時停市偵測（2026-10-23 颱風後新增）─────────────────────────────────────
# 10/23 實況：假日檔沒有這天 → 批次照常執行、Telegram 只有一般摘要。若當天剛好
# 需要交易，委託會送進休市的市場。13:20 先以近月最後成交時間確認今日真的在交易。

def _set_last_trade(broker, hhmm: str, day=TODAY):
    h, m = (int(x) for x in hhmm.split(":"))
    for q in broker.quotes.values():
        q.ts = dt.datetime.combine(day, dt.time(h, m))


def _capture(monkeypatch):
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(main_mod.notify, "send",
                        lambda level, text: sent.append((level, text)))
    return sent


def _audit_events():
    import json
    p = config.LOG_DIR / "audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(x)["event"] for x in p.read_text(encoding="utf-8").splitlines()]


def test_closed_market_places_no_orders(monkeypatch):
    """🔴 最後成交是當天凌晨的夜盤（10/23 即 04:59）→ 日盤沒開：不下單、不標記已執行。"""
    broker = setup_scenario(monkeypatch, held_qty=0)    # 正常情況會建倉 3 口
    _set_last_trade(broker, "04:59")
    _at(monkeypatch, "13:20")
    sent = _capture(monkeypatch)
    assert run_daily(broker, today=TODAY) == 0          # 不是系統故障：心跳照常 OK
    assert broker.placed == []
    assert state_mod.load().executed_date is None       # 若誤判，13:40 前可手動重跑
    assert any(lv == "WARNING" and "疑似臨時停市" in t for lv, t in sent)
    assert "market_closed_suspected" in _audit_events()
    assert not (config.LOG_DIR / "daily_summary.csv").exists()   # 不留休市日的假摘要


def test_stalled_quotes_place_no_orders(monkeypatch):
    """開盤後有成交、但已半小時以上沒有新成交 → 疑似盤中停市或報價中斷。"""
    broker = setup_scenario(monkeypatch, held_qty=0)
    _set_last_trade(broker, "12:40")
    _at(monkeypatch, "13:20")
    sent = _capture(monkeypatch)
    assert run_daily(broker, today=TODAY) == 0
    assert broker.placed == []
    assert any("盤中停市或報價中斷" in t for _, t in sent)


def test_live_market_trades_normally(monkeypatch):
    broker = setup_scenario(monkeypatch, held_qty=0)
    _set_last_trade(broker, "13:19")
    _at(monkeypatch, "13:20")
    assert run_daily(broker, today=TODAY) == 0
    assert len(broker.placed) == 1
    assert state_mod.load().executed_date == TODAY.isoformat()
    assert "market_closed_suspected" not in _audit_events()


def test_closure_check_skipped_outside_day_session(monkeypatch):
    """盤外執行（開機補跑等）本來就沒有新成交，不可誤判為停市。"""
    broker = setup_scenario(monkeypatch)
    _set_last_trade(broker, "04:59")
    _at(monkeypatch, "02:15")
    assert run_daily(broker, today=TODAY) == 0
    assert "market_closed_suspected" not in _audit_events()


def test_dry_run_on_closed_market_only_warns(monkeypatch):
    broker = setup_scenario(monkeypatch, held_qty=0)
    _set_last_trade(broker, "04:59")
    _at(monkeypatch, "13:20")
    sent = _capture(monkeypatch)
    assert run_daily(broker, today=TODAY, dry_run=True) == 0
    assert broker.placed == []
    assert any("疑似臨時停市" in t for _, t in sent)
    assert "market_closed_suspected" in _audit_events()


def test_missing_trade_time_does_not_block(monkeypatch):
    """取不到最後成交時間（舊版或查詢欄位缺漏）→ 不擋，維持原流程。"""
    broker = setup_scenario(monkeypatch, held_qty=0)    # flat_quote 的 ts 為 None
    _at(monkeypatch, "13:20")
    assert run_daily(broker, today=TODAY) == 0
    assert len(broker.placed) == 1
