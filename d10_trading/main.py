"""D_10 日批次主程式（規格 §2）——每交易日 13:20 執行一次。

    python -m d10_trading.main [--dry-run] [--ignore-window] [--adopt-state]

流程：斷電開關 → 冪等檢查 → 對帳 → 保險絲 → 資料更新 → 訊號 → 目標口數
     → 轉倉檢查 → 下單執行 → 記錄與通知。
任何 ALERT 級異常：告警 + 本日停止，不自動修正（規格 §7、§11）。

退出碼：0 = 正常/無事可做；1 = 告警停止；2 = 保險絲/嚴重異常。
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from zoneinfo import ZoneInfo

from d10_trading import config
from d10_trading.data import calendar, holiday_sync
from d10_trading.data.calendar import CalendarCoverageError
from d10_trading.data.series import DataStaleError, prepare_series
from d10_trading.execution import orders
from d10_trading.execution.broker import BrokerInterface, ShioajiBroker
from d10_trading.ops import notify, oplog, reconcile as rc, state as state_mod, watchdog
from d10_trading.ops.audit import audit
from d10_trading.ops.state import PositionState
from d10_trading.strategy import regime, sizing
from d10_trading.strategy.rollover import decide_roll

log = logging.getLogger("d10.main")


def _now() -> dt.datetime:
    return dt.datetime.now(ZoneInfo(config.TIMEZONE))


def _in_exec_window(now: dt.datetime) -> bool:
    lo = dt.time.fromisoformat(config.EXEC_WINDOW[0])
    hi = dt.time.fromisoformat(config.EXEC_WINDOW[1])
    return lo <= now.time() <= hi


class DayStop(RuntimeError):
    """告警 + 本日停止。"""
    def __init__(self, msg: str, level: str = "ALERT", code: int = 1):
        super().__init__(msg)
        self.level, self.code = level, code


def run_daily(broker: BrokerInterface, **kwargs) -> int:
    """日批次入口：執行 + 結尾心跳 ping（無論成敗，供外部排程監控）。"""
    code = 1
    try:
        code = _run_daily(broker, **kwargs)
        return code
    finally:
        notify.heartbeat(code)


def _run_daily(broker: BrokerInterface, *, today: dt.date | None = None,
               dry_run: bool = False, ignore_window: bool = False,
               fetch_kwargs: dict | None = None,
               sleep_fn=time.sleep) -> int:
    now = _now()
    today = today or now.date()

    # ── 0. 斷電開關 ─────────────────────────────────────────────────────────
    if config.KILL_SWITCH_FILE.exists():
        log.warning("KILL_SWITCH 存在，程式退出（不做任何事）")
        audit("kill_switch_exit")
        return 0

    # ── 交易日與冪等 ────────────────────────────────────────────────────────
    try:
        if not calendar.is_trading_day(today):
            log.info("%s 非交易日，退出", today)
            return 0
    except CalendarCoverageError as e:
        notify.send("ALERT", str(e))
        return 1

    st = state_mod.load()
    if st.executed_today(today) and not dry_run:
        log.info("本日已執行（executed_date=%s），冪等退出", st.executed_date)
        return 0
    st.last_run = now.isoformat(timespec="seconds")
    audit("batch_start", today=today.isoformat(), dry_run=dry_run,
          ignore_window=ignore_window)

    warnings: list[str] = []
    try:
        return _run_core(broker, st, today, now, dry_run, ignore_window,
                         fetch_kwargs, sleep_fn, warnings)
    except DayStop as e:
        notify.send(e.level, f"{today} 本日停止：{e}")
        audit("batch_stop", reason=str(e), level=e.level, exit_code=e.code)
        st.notes = f"{today} 停止：{e}"
        state_mod.save(st)
        return e.code
    except Exception as e:  # noqa: BLE001 - 未預期異常一律告警停止
        log.exception("未預期異常")
        notify.send("ALERT", f"{today} 未預期異常：{type(e).__name__}: {e}")
        audit("batch_error", error=f"{type(e).__name__}: {e}", exit_code=1)
        return 1


def _run_core(broker: BrokerInterface, st, today, now, dry_run, ignore_window,
              fetch_kwargs, sleep_fn, warnings: list[str]) -> int:
    # ── 0. 到期哨兵（置於登入前：登入失敗的日子也要能收到提醒）──────────────
    _check_expiries(today, warnings)

    # ── 1. 連線與對帳（reconcile）───────────────────────────────────────────
    from d10_trading.execution.broker import BrokerError
    try:
        broker.login()
    except BrokerError as e:
        raise DayStop(f"登入失敗：{e}") from e
    _check_ca(broker)
    try:
        month_map = {c.code: c.delivery_month
                     for c in broker.monthly_contracts(config.PRODUCT)}
        month_of_code = lambda code: month_map.get(code, "??????")  # noqa: E731

        try:
            recon = rc.reconcile(broker, st, month_of_code)
        except rc.ReconcileMismatch as e:
            raise DayStop(str(e)) from e
        log.info("對帳 OK：權益 %.0f、持倉 %d 口", recon.equity, recon.total_qty)
        if len(recon.held_months) > 1:
            raise DayStop(f"持倉橫跨多個月份 {recon.held_months}——"
                          "前日轉倉可能不完整，請人工處理")
        _check_risk_indicator(broker, warnings)

        # ── 1b. 今日是否真的在交易（臨時停市偵測；置於保險絲與下單之前）─────
        closed = _market_closed_reason(broker, today, now)
        if closed:
            audit("market_closed_suspected", reason=closed, dry_run=dry_run)
            if not dry_run:
                raise DayStop(closed + "。本日不下單、不標記已執行；請核對期交所公告，"
                              "若確認有開盤可於 13:40 前手動重跑",
                              level="WARNING", code=0)
            warnings.append(closed)
            notify.send("WARNING", closed)

        # ── 2. 保險絲前置檢查（§7：對帳後）────────────────────────────────
        held_month = recon.held_months[0] if recon.held_months else None
        held_qty = recon.total_qty
        fuse_mode = False
        if held_qty > 0:
            held_code = recon.positions[0].code
            px_now = broker.quote(held_code).price
            l_cur = sizing.current_leverage(held_qty, px_now, recon.equity)
            if sizing.fuse_triggered(l_cur):
                fuse_mode = True
                audit("fuse_triggered", l_current=round(l_cur, 3),
                      held_qty=held_qty, equity=recon.equity)
                notify.send("CRITICAL",
                            f"保險絲觸發：L_current={l_cur:.2f} ≥ {config.FUSE_L}，"
                            "將立即減碼回 L_target")

        # ── 3. 資料更新與訊號 ──────────────────────────────────────────────
        try:
            # reference 僅在日盤收盤（13:45）前 = 前日結算；之後滾動成當日值
            ref_ok = now.time() < dt.time.fromisoformat(config.DAY_SESSION_CLOSE)
            ss = prepare_series(
                today,
                fetch_kwargs=fetch_kwargs,
                prev_settlement_fn=lambda m: broker.settlement_reference(
                    config.SIGNAL_PRODUCT, m),
                today_price_fn=lambda m: broker.quote(
                    broker.contract_by_month(config.SIGNAL_PRODUCT, m).code).price,
                ref_trustworthy=ref_ok,
                contract_exists_fn=lambda m: _tx_contract_exists(broker, m),
                day_traded_fn=lambda d: broker.day_session_traded(
                    config.PRODUCT, d, today),
            )
        except DataStaleError as e:
            if fuse_mode:
                return _fuse_emergency(broker, st, recon, today, sleep_fn, str(e))
            raise DayStop(f"資料過期：{e}") from e
        warnings.extend(ss.warnings)
        for w in ss.warnings:
            notify.send("WARNING", w)
        if ss.closures_recorded:
            audit("adhoc_closure_recorded", source="kbars",
                  dates=[d.isoformat() for d in ss.closures_recorded])

        # 臨時停市自動補登（過去日期，前後皆有實際行情佐證；只增不刪）。
        # 未及時補登會使 prev_trading_day 誤判、序列插入假的一天並偏移訊號相位。
        adhoc = holiday_sync.record_adhoc_closures(
            ss.dates, since=today - dt.timedelta(days=90))
        if adhoc:
            msg = ("假日檔已自動補登臨時停市："
                   + "、".join(d.isoformat() for d in adhoc)
                   + "（依實際行情無交易判定，請與 TAIFEX 公告核對）")
            warnings.append(msg)
            audit("adhoc_closure_recorded", dates=[d.isoformat() for d in adhoc])
            notify.send("WARNING", msg)

        lev, _ = regime.compute_regime(ss.spot)
        exec_idx = ss.exec_index(today)          # T+1：嚴格早於今日的最後官方列
        sig_date = ss.dates[exec_idx]
        prev_td = calendar.prev_trading_day(today)
        if sig_date != prev_td:
            raise DayStop(f"執行訊號日 {sig_date} ≠ 上一交易日 {prev_td}"
                          "（序列缺日或行事曆不符），拒絕下單")
        l_exec = float(lev[exec_idx])
        cur_regime = regime.BULL if l_exec == config.L_HIGH else regime.BEAR
        t_idx = ss.today_index(today)
        preview = float(lev[t_idx]) if t_idx is not None else None

        # 訊號一致性檢查：昨日預覽 vs 今日官方定案；歷史訊號重算漂移
        if st.preview and st.preview.get("date") == sig_date.isoformat() \
                and st.preview.get("leverage") not in (None, l_exec):
            msg = (f"訊號定案與昨日預覽不一致：預覽 {st.preview['leverage']} → "
                   f"定案 {l_exec}（官方結算價跨過緩衝帶邊界，屬正常現象）")
            warnings.append(msg)
            notify.send("WARNING", msg)
        if st.signal_date and st.exec_leverage is not None:
            try:
                idx = ss.dates.index(dt.date.fromisoformat(st.signal_date))
                if float(lev[idx]) != float(st.exec_leverage):
                    msg = (f"資料重建後 {st.signal_date} 的訊號由 "
                           f"{st.exec_leverage} 變為 {lev[idx]}（資料修正）")
                    warnings.append(msg)
                    notify.send("WARNING", msg)
            except ValueError:
                pass
        if st.exec_leverage is not None and l_exec != st.exec_leverage:
            notify.send("INFO", f"regime 切換：今日執行槓桿 {st.exec_leverage} → "
                                f"{l_exec}（{cur_regime}）")

        # ── 補寫前日「執行價 vs 官方結算價」稽核 ───────────────────────────
        _flush_exec_audit(st, ss)

        # ── 4. 目標口數與轉倉檢查 ──────────────────────────────────────────
        front, nxt = broker.front_and_next(config.PRODUCT, today)
        cal_settle = calendar.settlement_date(int(front.delivery_month[:4]),
                                              int(front.delivery_month[4:]))
        if cal_settle != front.delivery_date:
            msg = (f"結算日互驗不一致：Shioaji {front.delivery_date} vs "
                   f"行事曆推算 {cal_settle}（以 Shioaji 為準）")
            warnings.append(msg)
            notify.send("WARNING", msg)

        q_front, q_next = broker.quote(front.code), broker.quote(nxt.code)
        spread = q_next.price - q_front.price
        holding_near = held_month == front.delivery_month and held_qty > 0
        roll = decide_roll(today, front.delivery_date, spread, holding_near)

        # 目標合約：轉倉日→次月；空手時避開轉倉窗口直接建次月；其餘沿用持倉月
        if roll.should_roll:
            target_c, target_q_obj = nxt, q_next
        elif held_qty == 0:
            past_window = (spread > 0 and today >= roll.early_roll_date) \
                          or today >= roll.settlement
            target_c, target_q_obj = (nxt, q_next) if past_window else (front, q_front)
        elif held_month == front.delivery_month:
            target_c, target_q_obj = front, q_front
        else:
            held_code = recon.positions[0].code
            target_c = broker.contract_by_month(config.PRODUCT, held_month)
            target_q_obj = broker.quote(held_code)

        sz = sizing.target_contracts(l_exec, recon.equity, target_q_obj.price)
        delta = sz.target_contracts - held_qty
        try:
            sizing.check_caps(sz.target_contracts, delta)
        except sizing.ContractsCapExceeded as e:
            raise DayStop(f"口數防呆：{e}") from e
        _check_capacity(recon.equity, target_q_obj.price, today, warnings)

        log.info("訊號 %s（L_exec=%.1f，訊號日 %s）；權益 %.0f、目標 %d 口"
                 "（raw=%.2f、殘餘 %.2f）、現有 %d 口；月差 %.0f%s",
                 cur_regime, l_exec, sig_date, recon.equity,
                 sz.target_contracts, sz.raw_contracts, sz.rounding_residual,
                 held_qty, spread,
                 f"；轉倉（{roll.reason}）" if roll.should_roll else "")
        audit("decision", regime=cur_regime, exec_leverage=l_exec,
              signal_date=sig_date.isoformat(), equity=round(recon.equity),
              price=target_q_obj.price, target=sz.target_contracts,
              raw_contracts=round(sz.raw_contracts, 4), held=held_qty,
              delta=delta, roll=roll.reason if roll.should_roll else "",
              spread=round(spread, 1), preview=preview,
              target_code=target_c.code)

        # ── 5. 下單執行 ────────────────────────────────────────────────────
        action_desc = "無動作"
        rep = orders.ExecutionReport(ok=True)
        need_trade = roll.should_roll or delta != 0
        if need_trade:
            if dry_run:
                action_desc = _describe(roll, front, nxt, target_c, held_qty,
                                        sz.target_contracts, delta) + "（dry-run 未下單）"
                log.info("DRY-RUN：%s", action_desc)
            else:
                if not ignore_window and not _in_exec_window(now):
                    raise DayStop(
                        f"不在執行時間窗 {config.EXEC_WINDOW}（現在 "
                        f"{now.time():%H:%M}），拒絕下單")
                if roll.should_roll:
                    rep = orders.execute_roll(broker, front.code, nxt.code,
                                              held_qty, sz.target_contracts,
                                              sleep_fn=sleep_fn,
                                              guard=_expiry_guard(today, front))
                else:
                    act = "Buy" if delta > 0 else "Sell"
                    rep = orders.execute_limit(broker, target_c.code, act,
                                               abs(delta), sleep_fn=sleep_fn)
                action_desc = _describe(roll, front, nxt, target_c, held_qty,
                                        sz.target_contracts, delta)
                for a in rep.alerts:
                    notify.send("ALERT" if not rep.ok else "WARNING", a)
                oplog.write_fills(today, rep.fills)
                for f in rep.fills:
                    legs_txt = ("（" + "、".join(
                        f"{c}@{px:.0f}" for c, px in sorted(f.legs.items()))
                        + "）") if f.legs else ""
                    notify.send("INFO", f"成交：{f.action} {f.code} "
                                        f"{f.quantity}/{f.requested} 口 @ "
                                        f"{f.price:.0f}{legs_txt}")
                    st.pending_exec_audit.extend(
                        _pending_exec_records(
                            today, f, month_of_code,
                            roll.reason if roll.should_roll else ""))

        # ── 6. 讀回實際部位、落地狀態 ──────────────────────────────────────
        if not dry_run:
            post = [p for p in broker.positions()
                    if p.code.startswith(config.PRODUCT) and p.quantity > 0]
            st.positions = [PositionState(code=p.code,
                                          month=month_of_code(p.code),
                                          qty=p.quantity, avg_price=p.avg_price)
                            for p in post]
            held_after = sum(p.qty for p in st.positions)
            # 🔴 只有在執行時間窗內（或明確 --ignore-window）才標記「本日已執行」。
            # 盤外批次（開機補跑、手動 systemctl start）用的是非收盤時段價格，
            # 其「無動作」結論不能代表 13:20 的結論——標記了會讓正式排程被冪等
            # 跳過，錯過當日該有的加減碼（2026-09-01 實際踩到）。
            if _in_exec_window(now) or ignore_window:
                st.executed_date = today.isoformat()
            else:
                msg = (f"盤外執行（{now:%H:%M} 不在 {config.EXEC_WINDOW}）"
                       "——不標記本日已執行，13:20 排程仍會正常運行")
                log.info("%s", msg)
                warnings.append(msg)
                audit("out_of_window_run", time=f"{now:%H:%M}",
                      committed=False)
        else:
            held_after = held_qty

        st.regime = cur_regime
        st.exec_leverage = l_exec
        st.signal_date = sig_date.isoformat()
        st.preview = ({"date": today.isoformat(), "leverage": preview}
                      if preview is not None else None)
        st.equity = recon.equity
        if not dry_run:
            state_mod.save(st)

        # ── 7. 摘要與通知 ──────────────────────────────────────────────────
        ach = sizing.current_leverage(held_after, target_q_obj.price, recon.equity)
        summary = {
            "date": today.isoformat(), "regime": cur_regime,
            "exec_leverage": l_exec, "equity": round(recon.equity),
            "price": target_q_obj.price, "target_contracts": sz.target_contracts,
            "held_before": held_qty, "held_after": held_after,
            "achieved_leverage": round(ach, 3), "action": action_desc,
            "exec_price": _summary_exec_price(rep.fills),
            "official_close": "", "exec_vs_close_pct": "",
            "preview_leverage": preview if preview is not None else "",
            "warnings": "；".join(warnings),
        }
        if not dry_run:
            oplog.write_summary(summary)
            _write_roll_spread(today, front, nxt, q_front, q_next, roll)
        notify.send("INFO",
                    f"{today} 摘要｜{cur_regime} L={l_exec:g}｜權益 "
                    f"{recon.equity:,.0f}｜{held_qty}→{held_after} 口"
                    f"（目標 {sz.target_contracts}）｜{action_desc}"
                    + (f"｜⚠️ {len(warnings)} 則警示" if warnings else ""))
        exit_code = (1 if (need_trade and not dry_run and not rep.ok)
                     else 2 if fuse_mode else 0)
        audit("batch_end", action=action_desc, held_before=held_qty,
              held_after=held_after, target=sz.target_contracts,
              orders_ok=rep.ok, fuse=fuse_mode, dry_run=dry_run,
              warnings=len(warnings), exit_code=exit_code)
        return exit_code
    finally:
        broker.logout()


def _check_expiries(today: dt.date, warnings: list[str]) -> None:
    """到期哨兵（ops/watchdog.py）。哨兵本身故障絕不影響交易。

    進入假日檔預警窗時，先嘗試自 TWSE 自動補次年度（additive-only）；
    補成功後涵蓋年份延長，該項提醒自然消失。
    """
    try:
        res = holiday_sync.auto_sync_if_needed(today)
        if res.status == "added":
            audit("holiday_sync", year=res.year,
                  added=[d.isoformat() for d in res.added])
            notify.send("INFO", res.message())
        elif res.status == "failed":
            audit("holiday_sync_failed", year=res.year, detail=res.detail)
            notify.send("WARNING", res.message())
        elif res.status == "unavailable":
            log.info("%s", res.detail)            # 尚未公告屬正常，不推播
    except Exception as e:  # noqa: BLE001 - 自動更新故障不得影響交易
        log.warning("假日檔自動更新異常（忽略）：%s", type(e).__name__)

    try:
        items = watchdog.check_expiries(today)
    except Exception as e:  # noqa: BLE001
        log.warning("到期檢查失敗（忽略）：%s", type(e).__name__)
        return
    for item in items:
        msg = item.message()
        warnings.append(msg)                       # 一律入每日摘要
        audit("expiry_check", name=item.name, deadline=item.deadline.isoformat(),
              days_left=item.days_left, level=item.level)
        if watchdog.should_push(item, today):      # WARNING 每週一、ALERT 每日
            notify.send(item.level, msg)
        else:
            log.info("到期提醒（未推播）：%s", msg)


def _tx_contract_exists(broker: BrokerInterface, month: str) -> bool:
    """訊號商品（TX）該月份合約是否仍可交易。

    每月結算後 1–2 個交易日，序列的持倉月已下市但 FinMind 資料尚未推進
    （連續序列的到期判定是事後的：last_seen < data_end），此時查價必然失敗。
    交給 series 層據此以單一明確訊息說明並略過預覽／雙源核對。
    """
    from d10_trading.execution.broker import BrokerError
    try:
        broker.contract_by_month(config.SIGNAL_PRODUCT, month)
        return True
    except BrokerError:
        return False


def _expiry_guard(today: dt.date, front) -> "orders.ExpiryGuard | None":
    """近月**最後交易日**才建立轉倉時序護欄；其他日回 None（行為完全不變）。

    🔴 以日期判定，不用 `roll.reason == "expiry_roll"`——`late_roll`（月差在 T−5
    之後才轉正）也可能在該日觸發，面對同一個 13:30 死線。用日期同時涵蓋兩者。

    護欄反映的是**交易所事實**（到期月份契約該日 13:30 停止交易），故
    `--ignore-window` 也不豁免：盤後手動補跑時近月早已結算，拒絕送單才是對的。
    """
    if today != front.delivery_date:
        return None
    return orders.ExpiryGuard(
        force_at=dt.time.fromisoformat(config.EXPIRY_FORCE_TIME),
        hard_stop=dt.time.fromisoformat(config.EXPIRY_HARD_STOP),
        timeout_secs=config.EXPIRY_ORDER_TIMEOUT_SECS,
        now_fn=_now)


def _write_roll_spread(today, front, nxt, q_front, q_next, roll) -> None:
    """逐日寫出近月/次月月差（roll_spread.csv）。追蹤用，失敗不影響流程。"""
    try:
        px = float(q_front.price)
        oplog.write_roll_spread({
            "date": today.isoformat(),
            "near_month": front.delivery_month,
            "next_month": nxt.delivery_month,
            "settlement": front.delivery_date.isoformat(),
            "t_minus": calendar.trading_days_between(today, front.delivery_date),
            "near_price": px, "next_price": float(q_next.price),
            "spread": round(float(roll.spread), 2),
            "spread_pct": round(float(roll.spread) / px, 6) if px else "",
            "rolled": roll.reason if roll.should_roll else "",
        })
    except Exception:  # noqa: BLE001 - 追蹤紀錄絕不可影響交易
        log.warning("roll_spread 紀錄寫入失敗（忽略）", exc_info=True)


def _pending_exec_records(today: dt.date, fill, month_of_code,
                          roll_reason: str = "") -> list[dict]:
    """成交腿 → exec_vs_close 待補紀錄（每腿一筆，各帶**自己的**合約月份）。

    🔴 兩處修正（2026-09-09 首次真實價差單後發現）：

    1. 月份必須逐腿判定。原本一律寫 `target_c.delivery_month`，使近月腿被拿去
       與**次月**結算價比較——8/12 的 `TMFH6 45,380 vs 結算 45,673 → −0.64%`
       其中大半是月差本身，不是執行品質。這是兩腳路徑也有的既存偏差。
    2. 組合單須依 legs 拆成分腳實價。否則整筆的 code 含 "->"，會被補寫端
       （`_flush_exec_audit`）的防線跳過 → 轉倉日完全沒有執行品質稽核
       （兩腳路徑 2 列、組合單 0 列）。當初「淨價無單一結算對照」的取捨成立，
       是因為只有淨價可用；leg_prices 到手後該前提已不存在。
    """
    # roll/path 供 _flush_exec_audit 事後配對兩腿、算出月差滑價
    ctx = {"roll": roll_reason,
           "path": (fill.note or "").split()[0] if fill.note else ""}
    if fill.legs:
        return [{"date": today.isoformat(), "code": code,
                 "month": month_of_code(code), "price": px, **ctx}
                for code, px in sorted(fill.legs.items())]
    return [{"date": today.isoformat(), "code": fill.code,
             "month": month_of_code(fill.code), "price": fill.price, **ctx}]


def _summary_exec_price(fills):
    """每日摘要的 exec_price：與 official_close 同欄比較，故須為**單腳實價**。

    組合單取「新建腿」（買腳）的實價而非淨價——淨價與結算價不可比，
    放淨價會讓 exec_vs_close_pct 算出約 −99% 的假偏差（2026-09-09 實測：
    淨價 192 vs 結算 47,000 量級）。淨價本身仍完整保存於 trades.jsonl。
    """
    if not fills:
        return ""
    f = fills[0]
    if f.legs:
        _, _, buy_code = str(f.code).partition("->")
        return f.legs.get(buy_code, f.price)
    return f.price


def _check_capacity(equity: float, price: float, today: dt.date,
                    warnings: list[str]) -> None:
    """MAX_CONTRACTS 容量餘裕預警——讓靜態上限的過期成為可預期。

    上限是刻意靜態的防呆（見 config 註解），會隨權益成長而過期；不預警的話
    會以「某天突然拒單停機」的方式讓人發現。餘裕 <20% 每週一提醒、<5% 每日提醒。
    """
    try:
        e_bind, headroom = sizing.capacity_headroom(equity, price)
    except Exception:  # noqa: BLE001 - 預警計算失敗不影響交易
        return
    if headroom >= config.CAPACITY_WARN_HEADROOM:
        return
    level = ("ALERT" if headroom < config.CAPACITY_ALERT_HEADROOM else "WARNING")
    msg = (f"口數上限餘裕僅 {headroom:.0%}：權益 {equity:,.0f}，"
           f"MAX_CONTRACTS={config.MAX_CONTRACTS} 對應權益上限約 {e_bind:,.0f}"
           f"（以 L_HIGH={config.L_HIGH:g}、現價 {price:,.0f} 計）——"
           "超過即目標口數大於上限而拒單停止，請及早調高 MAX_CONTRACTS")
    warnings.append(msg)
    audit("capacity_headroom", equity=round(equity), e_bind=round(e_bind),
          headroom=round(headroom, 4), level=level)
    if level == "ALERT" or today.weekday() == 0:      # WARNING 僅週一推播
        notify.send(level, msg)
    else:
        log.info("容量預警（未推播）：%s", msg)


def _check_risk_indicator(broker: BrokerInterface, warnings: list[str]) -> None:
    """風險指標監控（僅真實模式有值）——保證金充足度的異常金絲雀。

    正常值約 1,600–2,700%（2 倍槓桿、全額入金），故任何接近門檻的讀數都代表
    口數、保證金率或權益出現非預期變化。查詢失敗不影響交易。
    """
    ri = broker.risk_indicator()
    if ri is None:
        return
    audit("risk_indicator", value=round(ri, 1))
    if ri < config.RISK_INDICATOR_CRITICAL:
        level, note = "CRITICAL", "已逼近保證金不足，請立即確認帳戶並補繳"
    elif ri < config.RISK_INDICATOR_ALERT:
        level, note = "ALERT", "低於補款門檻，請確認是否需補入資金"
    elif ri < config.RISK_INDICATOR_WARN:
        level, note = "WARNING", "顯著低於常態（約 1,600%+），請確認口數與保證金率"
    else:
        log.info("風險指標 %.0f%%（正常）", ri)
        return
    msg = f"風險指標 {ri:.0f}%——{note}"
    warnings.append(msg)
    notify.send(level, msg)


def _market_closed_reason(broker: BrokerInterface, today: dt.date,
                          now: dt.datetime) -> str | None:
    """確認今日日盤確實在交易；疑似停市回傳說明，否則 None。

    依近月最後成交時間（快照 ts）判斷。只看「最後成交日期 = 今天」不夠：夜盤跨午夜，
    2026-10-23 颱風停市日的最後成交時間戳正是當天 04:59（前一夜盤）。
    - 開盤後沒有任何成交 → 疑似全日停市；
    - 最後成交距今超過 MARKET_STALE_MINUTES → 疑似盤中停市或報價中斷。
    只在日盤時段內判定：盤外執行（開機補跑、手動盤後 dry-run）本來就沒有新成交。
    取不到時間戳或查詢失敗時不擋，交由下單流程既有的逾時／拒單告警處理。
    """
    open_t = dt.time.fromisoformat(config.DAY_SESSION_OPEN)
    close_t = dt.time.fromisoformat(config.DAY_SESSION_CLOSE)
    local_now = now.astimezone(ZoneInfo(config.TIMEZONE)).replace(tzinfo=None)
    if local_now.date() != today or not (open_t < local_now.time() <= close_t):
        return None
    try:
        front, _ = broker.front_and_next(config.PRODUCT, today)
        last = broker.quote(front.code).ts
    except Exception as e:  # noqa: BLE001 - 偵測本身失敗不得擋下交易
        log.warning("開盤確認失敗（略過）：%s", type(e).__name__)
        return None
    if last is None:
        log.warning("取不到 %s 最後成交時間，略過開盤確認", front.code)
        return None
    if last < dt.datetime.combine(today, open_t):
        return (f"近月 {front.code} 今日日盤沒有任何成交（最後成交 {last:%m/%d %H:%M}）"
                "——疑似臨時停市（颱風等）")
    idle = (local_now - last).total_seconds() / 60
    if idle > config.MARKET_STALE_MINUTES:
        return (f"近月 {front.code} 已 {idle:.0f} 分鐘沒有成交（最後成交 {last:%H:%M}）"
                "——疑似盤中停市或報價中斷")
    return None


def _check_ca(broker: BrokerInterface) -> None:
    """CA 憑證到期預警（僅真實模式有值；預警失敗不影響批次）。"""
    days = broker.ca_expiry_days()
    if days is None:
        return
    audit("ca_check", days_left=days)
    if days < 7:
        notify.send("ALERT", f"CA 憑證將於 {days} 天內到期——"
                             "請立即至 API 管理頁更新（README §5）")
    elif days < config.CA_EXPIRY_WARN_DAYS:
        notify.send("WARNING", f"CA 憑證剩餘 {days} 天到期，請儘早更新")


def _describe(roll, front, nxt, target_c, held, target, delta) -> str:
    if roll.should_roll:
        return (f"轉倉（{roll.reason}）：平 {front.code} {held} 口 → "
                f"建 {nxt.code} {target} 口")
    if delta == 0:
        return "無動作（目標=現有）"
    act = "買進" if delta > 0 else "賣出"
    return f"再平衡：{act} {target_c.code} {abs(delta)} 口（{held}→{target}）"


def _fuse_emergency(broker, st, recon, today, sleep_fn, why: str) -> int:
    """資料不可用但保險絲已觸發：以最後定案訊號為 L_target 立即減碼。"""
    l_target = st.exec_leverage if st.exec_leverage is not None else config.L_LOW
    held_code = recon.positions[0].code
    px = broker.quote(held_code).price
    sz = sizing.target_contracts(l_target, recon.equity, px)
    delta = sz.target_contracts - recon.total_qty
    notify.send("CRITICAL",
                f"資料過期（{why}）且保險絲觸發——以最後訊號 L={l_target} 緊急減碼："
                f"{recon.total_qty} → {sz.target_contracts} 口")
    audit("fuse_emergency", reason=why, l_target=l_target,
          held=recon.total_qty, target=sz.target_contracts)
    if delta < 0:
        rep = orders.execute_limit(broker, held_code, "Sell", -delta,
                                   sleep_fn=sleep_fn)
        oplog.write_fills(today, rep.fills)
        for a in rep.alerts:
            notify.send("CRITICAL", a)
    st.executed_date = today.isoformat()
    st.notes = f"{today} 保險絲緊急減碼（資料過期）"
    state_mod.save(st)
    return 2


def _flush_exec_audit(st, ss) -> None:
    """補寫「執行價 vs 官方結算價」（規格 §2）。失敗不影響流程。

    結算價取自當次重建的 FinMind TX 序列（ss.settlement_of）——權威結算來源、
    與策略口徑一致，取代先前的日盤 K 棒收盤（那是收盤集合競價價、非結算）。
    結算價未到位（同日尚無 FinMind）→ 保留待下次批次補記。
    """
    remain, resolved = [], []
    for rec in st.pending_exec_audit:
        try:
            d = dt.date.fromisoformat(rec["date"])
            month = rec.get("month", "")
            if not month or "->" in rec.get("code", ""):
                continue                       # 無單一結算對照（舊格式整筆淨價）
            settle = ss.settlement_of(month, d)
            if settle:
                oplog.append_exec_vs_close(d, rec["code"], rec["price"], settle)
                resolved.append((rec, float(settle)))
            else:
                remain.append(rec)             # 結算未到位 → 次日再補
        except Exception:  # noqa: BLE001 - 稽核補寫失敗不影響交易
            remain.append(rec)
    st.pending_exec_audit = remain
    _flush_roll_slippage(resolved)


def _flush_roll_slippage(resolved: list[tuple[dict, float]]) -> None:
    """由已取得結算價的兩腿算出**月差滑價**（執行月差 − 結算月差）。

    只在同一天恰有兩筆帶 roll 標記、且月份不同的紀錄時成立——那正是一次完整
    轉倉的兩腿。任一腿的結算價尚未到位時整組留待次日（兩腿的結算來自同一天的
    序列，通常同時到位）。近月 = 月份較小者。失敗不影響流程。
    """
    by_date: dict[str, list[tuple[dict, float]]] = {}
    for rec, settle in resolved:
        if rec.get("roll"):
            by_date.setdefault(rec["date"], []).append((rec, settle))
    for date, pair in by_date.items():
        try:
            if len(pair) != 2:
                continue
            pair.sort(key=lambda x: x[0]["month"])       # 近月在前
            (near, s_near), (nxt, s_next) = pair
            if near["month"] == nxt["month"]:
                continue
            exec_spread = float(nxt["price"]) - float(near["price"])
            settle_spread = s_next - s_near
            slip = exec_spread - settle_spread
            oplog.append_roll_slippage({
                "date": date, "near_code": near["code"], "next_code": nxt["code"],
                "path": near.get("path", ""),
                "exec_near": float(near["price"]), "exec_next": float(nxt["price"]),
                "exec_spread": round(exec_spread, 2),
                "settle_near": s_near, "settle_next": s_next,
                "settle_spread": round(settle_spread, 2),
                "slippage_points": round(slip, 2),
                "slippage_pct": (round(slip / settle_spread, 6)
                                 if settle_spread else ""),
            })
            log.info("轉倉月差滑價（%s，%s）：執行 %.2f − 結算 %.2f = %+.2f 點",
                     date, near.get("path", "?"), exec_spread, settle_spread, slip)
        except Exception:  # noqa: BLE001 - 追蹤紀錄絕不可影響交易
            log.warning("roll_slippage 計算失敗（忽略）", exc_info=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D_10 Shioaji 日批次")
    ap.add_argument("--dry-run", action="store_true",
                    help="完整流程但不下單、不落地狀態")
    ap.add_argument("--ignore-window", action="store_true",
                    help="略過 13:05–13:40 執行時間窗檢查（測試用）")
    ap.add_argument("--adopt-state", action="store_true",
                    help="人工確認後：以 API 實際部位重置狀態檔（手動接管程序）")
    args = ap.parse_args(argv)

    notify.setup_logging()
    log.info("=== D_10 批次啟動 simulation=%s live_trading=%s dry_run=%s ===",
             config.SIMULATION, config.LIVE_TRADING, args.dry_run)

    broker = ShioajiBroker()
    if args.adopt_state:
        broker.login()
        try:
            month_map = {c.code: c.delivery_month
                         for c in broker.monthly_contracts(config.PRODUCT)}
            st = rc.adopt(broker, state_mod.load(),
                          lambda code: month_map.get(code, "??????"))
            state_mod.save(st)
            audit("adopt_state",
                  positions=[(p.code, p.qty) for p in st.positions],
                  equity=st.equity)
            log.info("adopt-state 完成：%s", [(p.code, p.qty) for p in st.positions])
            return 0
        finally:
            broker.logout()

    return run_daily(broker, dry_run=args.dry_run,
                     ignore_window=args.ignore_window)


if __name__ == "__main__":
    sys.exit(main())
