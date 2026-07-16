"""還原連續序列的日常維護：重建、新鮮度檢查、雙源核對、當日暫定價追加。

口徑（🔴 與回測一致）：
- 官方序列 = FinMind 結算價 → build_continuous（逐位元移植）。
- 趨勢訊號 spot = 比值還原連續價，公式與 round3/data_mkts.py 完全相同：
      spot = cumprod(1+ret) × (panel.price[0] / (1+ret[0]))
- T+1 執行：今日執行的槓桿 = 「昨日（上一交易日）官方收盤」算出的訊號。
  因此下單只需要官方序列到昨日；今日 13:20 的即時價僅用於：
  (a) 今日訊號的「預覽」（明日以官方收盤重算定案）、(b) 口數換算、(c) 執行價記錄。

新鮮度規則（規格 §3.3）：
- 官方序列最後日期必須 = 上一交易日。FinMind 未更新時，退回 Shioaji 日盤收盤價
  補「暫定官方」列（告警記錄，次日 FinMind 到位後自動以結算價重建覆蓋）；
  兩者皆無 → DataStaleError，停止且絕不用舊資料下單。
- 雙源核對：上一交易日的 FinMind 結算價 vs Shioaji 日盤收盤價，
  相對差 > DUAL_SOURCE_TOL → 告警（結算價與收盤價本有微小差異，僅示警不停止）。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from live import config
from live.data import calendar
from live.data.continuous import ContinuousSeries, build_continuous
from live.data.finmind import load_tx_contracts


class DataStaleError(RuntimeError):
    """資料過期——停止並告警，絕不用舊資料下單（規格 §3.3）。"""


@dataclass
class SignalSeries:
    """訊號用連續序列（官方段 + 可選的當日暫定列）。"""
    dates: list[dt.date]              # 與 spot 對齊（含暫定列）
    spot: np.ndarray                  # 比值還原連續價
    n_official: int                   # 官方列數（dates[:n_official] 為官方）
    held_contract: str                # 最後官方列收盤後持有的合約（YYYYMM）
    last_official_price: float        # 該合約最後官方價（暫定價換算基準）
    cs: ContinuousSeries              # 完整面板（稽核）
    warnings: list[str] = field(default_factory=list)

    @property
    def last_official_date(self) -> dt.date:
        return self.dates[self.n_official - 1]

    @property
    def has_provisional(self) -> bool:
        return len(self.dates) > self.n_official

    def exec_index(self, today: dt.date) -> int:
        """T+1 執行訊號的索引：最後一個「嚴格早於今日」的官方列。

        收盤後補跑時 FinMind 可能已含今日結算價（官方列 = 今日），
        執行訊號仍必須取上一交易日，否則變成以當日收盤訊號當日執行。
        """
        idx = self.n_official - 1
        while idx >= 0 and self.dates[idx] >= today:
            idx -= 1
        if idx < 0:
            raise DataStaleError("官方序列無早於今日的資料")
        return idx

    def today_index(self, today: dt.date) -> int | None:
        """今日列索引（官方或暫定，供訊號預覽）；無今日列回傳 None。"""
        for i in range(len(self.dates) - 1, -1, -1):
            if self.dates[i] == today:
                return i
            if self.dates[i] < today:
                break
        return None


def _spot_from_panel(panel: pd.DataFrame) -> np.ndarray:
    """趨勢訊號 spot——與 round3/data_mkts.py 逐位元相同的建構式。"""
    ret = panel["ret"].to_numpy(float)
    return np.cumprod(1.0 + ret) * float(panel["price"].iloc[0] / (1.0 + ret[0]))


def build_signal_series(cdf: pd.DataFrame) -> SignalSeries:
    """自 FinMind 長表建出官方訊號序列（不含當日暫定列）。"""
    cs = build_continuous(cdf, config.ROLL_OFFSET_DAYS)
    panel = cs.panel
    spot = _spot_from_panel(panel)
    dates = [d.date() for d in panel.index]
    return SignalSeries(
        dates=dates, spot=spot, n_official=len(dates),
        held_contract=str(panel["contract"].iloc[-1]),
        last_official_price=float(panel["price"].iloc[-1]),
        cs=cs,
    )


def append_official_fallback(ss: SignalSeries, date: dt.date, price: float,
                             source: str) -> SignalSeries:
    """FinMind 缺上一交易日時，以 Shioaji 日盤收盤價補一列「暫定官方」。

    以持倉合約的價格比值延伸 spot（同合約內報酬，與逐合約法口徑一致）。
    次日 FinMind 到位後整條序列以結算價重建，本列自動被覆蓋；
    若覆蓋後訊號與本列不同，main 會比對 state 併發告警。
    """
    r = price / ss.last_official_price - 1.0
    ss.dates.append(date)
    ss.spot = np.append(ss.spot, ss.spot[-1] * (1.0 + r))
    ss.n_official += 1
    ss.last_official_price = price
    ss.warnings.append(
        f"{date} 官方結算價缺漏，以 {source} 收盤價 {price:.0f} 暫代（次日自動校正）")
    return ss


def append_provisional(ss: SignalSeries, date: dt.date, price: float) -> SignalSeries:
    """追加今日暫定列（13:20 即時價 ≈ 收盤），僅供今日訊號預覽。"""
    assert not ss.has_provisional, "暫定列只能追加一次"
    r = price / ss.last_official_price - 1.0
    ss.dates.append(date)
    ss.spot = np.append(ss.spot, ss.spot[-1] * (1.0 + r))
    return ss


def prepare_series(today: dt.date, *,
                   fetch_kwargs: dict | None = None,
                   yesterday_close_fn=None,
                   today_price_fn=None) -> SignalSeries:
    """日常批次的序列準備（重建 → 新鮮度 → 暫定列）。

    yesterday_close_fn(contract_yyyymm, date) -> float | None
        Shioaji 日盤收盤價（雙源核對與 FinMind 缺漏遞補用）。
    today_price_fn(contract_yyyymm) -> float | None
        今日即時價（暫定列）。None → 不加暫定列（預覽略過，不影響下單）。
    """
    cdf = load_tx_contracts(**(fetch_kwargs or {"force_current_year": True,
                                                "today": today}))
    ss = build_signal_series(cdf)
    prev_td = calendar.prev_trading_day(today)

    # ── 行事曆 vs 實際行情日互驗（近 90 天）───────────────────────────────
    problems = calendar.validate_against_actual(
        ss.dates, since=today - dt.timedelta(days=90))
    ss.warnings.extend(problems)

    # ── 新鮮度：官方序列必須到上一交易日 ──────────────────────────────────
    if ss.last_official_date < prev_td:
        if ss.last_official_date != calendar.prev_trading_day(prev_td):
            raise DataStaleError(
                f"官方序列最後日期 {ss.last_official_date} 落後上一交易日 "
                f"{prev_td} 超過一日，無法以單日遞補")
        px = yesterday_close_fn(ss.held_contract, prev_td) if yesterday_close_fn else None
        if px is None or px <= 0:
            raise DataStaleError(
                f"FinMind 與 Shioaji 皆無 {prev_td} 收盤資料——停止，絕不用舊資料下單")
        append_official_fallback(ss, prev_td, float(px), "Shioaji")
    elif ss.last_official_date > today:
        raise DataStaleError(
            f"官方序列出現未來日期 {ss.last_official_date}（時鐘或資料異常）")

    # ── 雙源核對：上一交易日 FinMind 結算價 vs Shioaji 收盤價 ─────────────
    if yesterday_close_fn is not None and ss.last_official_date == prev_td:
        sj_px = yesterday_close_fn(ss.held_contract, prev_td)
        if sj_px and sj_px > 0:
            diff = abs(sj_px / ss.last_official_price - 1.0)
            if diff > config.DUAL_SOURCE_TOL:
                ss.warnings.append(
                    f"雙源核對：{prev_td} FinMind 結算 {ss.last_official_price:.0f} vs "
                    f"Shioaji 收盤 {sj_px:.0f}，差 {diff:.3%} > {config.DUAL_SOURCE_TOL:.1%}")

    # ── 今日暫定列（預覽用；序列已含今日官方列則略過）──────────────────────
    if ss.last_official_date < today and today_price_fn is not None:
        try:
            px = today_price_fn(ss.held_contract)
        except Exception as e:  # noqa: BLE001 - 預覽失敗不影響今日執行（T+1）
            px = None
            ss.warnings.append(f"今日即時價查詢失敗（{type(e).__name__}）")
        if px and px > 0:
            append_provisional(ss, today, float(px))
        else:
            ss.warnings.append("今日即時價無法取得，略過訊號預覽（不影響今日執行）")
    return ss
