"""連續合約構建——自回測專案根目錄 continuous.py 逐位元移植（規格 §3.3：移植，不要重寫）。

此檔為純函數模組，無任何 live 系統相依；演算法、數值運算順序與回測完全相同，
黃金測試（tests/test_golden.py）以此為前提。唯一差異是模組路徑。

═══════════════════════════════════════════════════════════════════════════
 連續合約構建說明（沿用回測文件）
═══════════════════════════════════════════════════════════════════════════
到期日：不用「第三個週三」公式硬算，而是取每個月份合約在資料中**最後出現的
交易日**，自然涵蓋假日順延與颱風休市等實際行事曆。

轉倉時點：到期日往前推 ROLL_OFFSET_DAYS 個「全市場交易日」（不是日曆日）。
於轉倉日**收盤（結算價）**賣近月、買次月；持倉合約自轉倉日收盤起改為次月。

假跳空消除 — 兩種方法並行實作、逐日互驗：
【方法 A：逐合約法】日報酬只在「同一口合約內部」計算。
【方法 B：還原法（比值 back-adjustment）】兩法日報酬數學上恆等，
  max|r_A - r_B| < 1e-8 作為實作正確性的互驗；不過即 raise。
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_TOL = 1e-8


@dataclass
class ContinuousSeries:
    """連續合約日面板與稽核資訊。

    panel（index=交易日，自第一個可算報酬的日子起）：
      ret       方法 A/B 互驗後的日報酬
      price     當日「收盤後」持倉合約的結算價（轉倉日 = 次月價）
      contract  當日收盤後持有的合約
      is_roll   是否為轉倉日
      p_near / p_next  轉倉日兩腳的成交價（非轉倉日為 NaN）
    """
    panel: pd.DataFrame
    rolls: pd.DataFrame              # 轉倉事件表（含月差稽核欄位）
    method_max_diff: float           # 兩法日報酬最大差異（互驗）
    n_price_gaps: int                # 持倉合約當日無價、以前值遞補的天數


def contract_expiries(cdf: pd.DataFrame) -> pd.Series:
    """每個月份合約的實際最後交易日（= 到期結算日）。

    最後出現日等於資料最後一日者視為「仍掛牌」，不包含在回傳中。
    """
    last_seen = cdf.groupby("contract")["date"].max()
    data_end = cdf["date"].max()
    return last_seen[last_seen < data_end].sort_index()


def build_continuous(cdf: pd.DataFrame, offset: int) -> ContinuousSeries:
    """以逐合約法為主、比值還原法互驗，建出連續日報酬面板。"""
    cal = pd.DatetimeIndex(np.sort(cdf["date"].unique()))
    piv = cdf.pivot(index="date", columns="contract", values="price").reindex(cal)
    expiries = contract_expiries(cdf)
    all_contracts = sorted(cdf["contract"].unique())

    # ── 持倉鏈：起始前月 → 逐次轉倉 ────────────────────────────────────────
    first_day = cal[0]
    front = min(c for c in all_contracts
                if not pd.isna(piv.at[first_day, c])
                and expiries.get(c, cal[-1]) >= first_day)

    roll_rows: list[dict] = []
    seg_bounds: list[tuple[str, int, int]] = []   # (合約, 報酬起 cal_pos, 報酬迄 cal_pos)
    pos_start = 1                                  # 第一個可算報酬的 cal 位置
    c = front
    while True:
        expiry = expiries.get(c)
        if expiry is None:                         # 仍掛牌 → 持有到資料尾
            seg_bounds.append((c, pos_start, len(cal) - 1))
            break
        rd_pos = cal.get_loc(expiry) - offset
        if rd_pos >= len(cal) - 1:                 # 轉倉日不在資料範圍內 → 持有到尾
            seg_bounds.append((c, pos_start, len(cal) - 1))
            break
        assert rd_pos >= pos_start, f"轉倉日早於持倉起日: {c}"
        nxt = all_contracts[all_contracts.index(c) + 1]
        seg_bounds.append((c, pos_start, rd_pos))
        roll_rows.append({"near": c, "next": nxt, "expiry_near": expiry,
                          "roll_pos": rd_pos, "roll_date": cal[rd_pos]})
        pos_start = rd_pos + 1
        c = nxt

    # ── 各持倉段價格：當日無價（極少數）以前值遞補並計數 ──────────────────
    n_gaps = 0
    seg_price: dict[str, np.ndarray] = {}
    for contract, a, b in seg_bounds:
        s = piv[contract].iloc[a - 1: b + 1]       # 需要 cal[a-1 .. b] 的價格
        n_gaps += int(s.isna().sum())
        s = s.ffill()
        assert not s.isna().any(), f"{contract} 段首即無價，無法遞補"
        seg_price[contract] = s.to_numpy()

    # 轉倉日兩腳價格：與方法 A 同源（遞補後的 pivot），保證互驗有效
    piv_ffill = piv.ffill()
    for ev in roll_rows:
        ev["p_near"] = float(piv_ffill.at[ev["roll_date"], ev["near"]])
        ev["p_next"] = float(piv_ffill.at[ev["roll_date"], ev["next"]])
        assert ev["p_near"] > 0 and ev["p_next"] > 0

    # ── 方法 A：逐合約法 ───────────────────────────────────────────────────
    T = len(cal) - 1                     # 報酬日數（自 cal[1] 起；ret index = cal_pos - 1）
    ret_a = np.empty(T)
    price_held = np.empty(T)             # 當日收盤後持倉合約價（轉倉日稍後覆寫為次月價）
    contract_held = np.empty(T, dtype=object)
    for contract, a, b in seg_bounds:
        p = seg_price[contract]                       # cal[a-1 .. b]
        ret_a[a - 1: b] = p[1:] / p[:-1] - 1.0
        price_held[a - 1: b] = p[1:]
        contract_held[a - 1: b] = contract

    # ── 方法 B：比值還原法 ─────────────────────────────────────────────────
    # 自最新段往回：M_last = 1；M_near = M_next × P_next(rd) / P_near(rd)
    mult = {seg_bounds[-1][0]: 1.0}
    for ev in reversed(roll_rows):
        mult[ev["near"]] = mult[ev["next"]] * ev["p_next"] / ev["p_near"]
    adj = np.empty(len(cal))
    for contract, a, b in seg_bounds:
        adj[a - 1: b + 1] = seg_price[contract] * mult[contract]
    ret_b = adj[1:] / adj[:-1] - 1.0

    method_max_diff = float(np.max(np.abs(ret_a - ret_b)))
    if method_max_diff >= _TOL:
        raise AssertionError(
            f"兩種連續合約構建法互驗失敗：max|r_A - r_B| = {method_max_diff:.3e} ≥ {_TOL}")

    # ── 組面板 ─────────────────────────────────────────────────────────────
    panel = pd.DataFrame(index=cal[1:], data={
        "ret": ret_a,
        "price": price_held,
        "contract": contract_held,
        "is_roll": False,
        "p_near": np.nan,
        "p_next": np.nan,
    })
    rolls = pd.DataFrame(roll_rows)
    if not rolls.empty:
        rolls["spread_pts"] = rolls["p_next"] - rolls["p_near"]
        rolls["spread_pct"] = rolls["spread_pts"] / rolls["p_near"]
        rd = rolls["roll_date"].values
        panel.loc[rd, "is_roll"] = True
        panel.loc[rd, "p_near"] = rolls["p_near"].values
        panel.loc[rd, "p_next"] = rolls["p_next"].values
        # 轉倉日收盤後已持有次月 → 口數換算與持倉合約以次月為準
        panel.loc[rd, "price"] = rolls["p_next"].values
        panel.loc[rd, "contract"] = rolls["next"].values

    return ContinuousSeries(
        panel=panel,
        rolls=rolls,
        method_max_diff=method_max_diff,
        n_price_gaps=n_gaps,
    )
