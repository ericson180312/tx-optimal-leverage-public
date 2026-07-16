"""門檻式再平衡回測引擎（numpy 向量化：一次日迴圈同時模擬 K 組參數）。

═══════════════════════════════════════════════════════════════════════════
 每日流程（規格 §2，順序不可調換 — 特別註明）
═══════════════════════════════════════════════════════════════════════════
 1. 市場結算    E += N·r_t ；N ×= (1+r_t)
 2. 計息        採「閒置現金計息」口徑：期貨為保證金交易，權益全額以現金存放，
                對**日初權益 E_{t-1}** 計 IDLE_CASH_RATE（年化、日頻複利）。
                不對借款計息（純期貨沒有實際借款），規格 §2-2 預設口徑。
 3. 破產檢查    E ≤ 0 → bankrupt，權益歸零、曝險歸零，其後永遠為 0。
 4. 轉倉        （若當日為轉倉日）賣近月＋買次月共 2 腳，各計手續費＋期交稅＋
                滑價；轉倉不改變曝險。機械性動作，先於門檻檢查執行。
 5. 重算槓桿    L = N / E（以此為觸價判斷，收盤執行、不做盤中觸價）
 6. 再平衡      兩種觸發機制（可並用，同日觸發只調整一次）：
                (a) 門檻式：L ≥ X → 減碼回 target；L ≤ Y → 加碼回 target。
                (b) 週期式：每 rebal_every 個交易日無條件調回 target（正 N 倍
                    ETF 的機制；rebal_every=1 即每日再平衡、0 即停用）。
                    週期自「建倉日」起算：第 rebal_every、2×rebal_every… 個
                    交易日收盤執行；rebal_phase 可平移週期相位。
                以「調整名目 |ΔN|」換算口數（連續小數）計成本，成本即時扣入
                權益（先把 N 調到 target×E，再從 E 扣成本，故成交後 L 會比
                target 高/低一點點 — 與實務相同，不遞迴重算）。
 7. 記錄        E、L（成本後）、觸發方向、調整名目、當日成本。

 基準策略以同一引擎表達：BH（X=+inf, Y=-inf 永不觸發）、每日再平衡（X=Y=target）、
 每 N 日再平衡（X=+inf, Y=-inf, rebal_every=N）。
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtest_threshold.params import (
    DAILY_RF, FEE_PER_SIDE, INITIAL_EQUITY, MULTIPLIER, SLIPPAGE_TICKS,
    TAX_RATE, TICK_VALUE,
)


@dataclass
class SimResult:
    """K 組參數在同一條路徑上的完整模擬結果（T 日 × K 組）。"""
    eq: np.ndarray        # (T,K) 每日收盤後權益
    lev: np.ndarray       # (T,K) 每日收盤後（成本後）槓桿
    lev_pre: np.ndarray   # (T,K) 觸價判斷用槓桿（結算與轉倉後、再平衡前）
    trig: np.ndarray      # (T,K) int8：+1 加碼、-1 減碼、0 無
    dN: np.ndarray        # (T,K) 再平衡調整的名目金額（絕對值）
    cost: np.ndarray      # (T,K) 當日總成本（轉倉＋再平衡）
    n_up: np.ndarray      # (K,) 加碼次數
    n_down: np.ndarray    # (K,) 減碼次數
    cost_cum: np.ndarray  # (K,) 累計成本（含建倉）
    bankrupt: np.ndarray  # (K,) bool
    max_lev: np.ndarray   # (K,) 路徑中出現過的最高槓桿（成本前觸價槓桿）


def simulate(ret: np.ndarray, price: np.ndarray, is_roll: np.ndarray,
             p_near: np.ndarray, p_next: np.ndarray,
             X: np.ndarray, Y: np.ndarray, target: np.ndarray,
             entry_price: float,
             *, rebal_every: np.ndarray | None = None,
             rebal_phase: np.ndarray | None = None,
             slippage_ticks: float = SLIPPAGE_TICKS,
             fee: float = FEE_PER_SIDE, tax: float = TAX_RATE,
             rf_daily: float = DAILY_RF, e0: float = INITIAL_EQUITY) -> SimResult:
    """模擬一條報酬路徑上的 K 組 (X, Y, target, rebal_every)。

    ret/price/is_roll/p_near/p_next: (T,)；X/Y/target/rebal_every/rebal_phase: (K,)。
    rebal_every: 週期再平衡的交易日間隔（0 = 停用，只用門檻）。預設全 0。
    rebal_phase: 週期相位平移（0 ≤ phase < rebal_every）。預設全 0。
    entry_price: 建倉日（第一個報酬日前一日收盤）持倉合約價格，供建倉成本換算。
    """
    T, K = len(ret), len(X)
    rebal_every = np.zeros(K, dtype=np.int64) if rebal_every is None \
        else np.asarray(rebal_every, dtype=np.int64)
    rebal_phase = np.zeros(K, dtype=np.int64) if rebal_phase is None \
        else np.asarray(rebal_phase, dtype=np.int64)
    has_period = rebal_every > 0
    slip = slippage_ticks * TICK_VALUE          # 每口每邊滑價（元）
    per_lot = fee + slip                        # 每口每邊固定成本（元）

    E = np.full(K, float(e0))
    Nn = target * E                             # 建倉：買進 target×E 名目
    bankrupt = np.zeros(K, dtype=bool)

    # 建倉成本（單邊買進）
    q = Nn / (MULTIPLIER * entry_price)
    entry_cost = q * per_lot + Nn * tax
    E -= entry_cost
    cost_cum = entry_cost.copy()

    eq = np.empty((T, K)); lev = np.empty((T, K)); lev_pre = np.empty((T, K))
    trig = np.zeros((T, K), dtype=np.int8)
    dN_rec = np.zeros((T, K)); cost_rec = np.zeros((T, K))
    n_up = np.zeros(K, dtype=np.int64); n_down = np.zeros(K, dtype=np.int64)
    max_lev = np.zeros(K)

    def _wipeout():
        """破產：權益歸零、曝險歸零，其後報酬恆為 0（不可反彈）。"""
        nonlocal bankrupt
        dead = E <= 0
        if dead.any():
            bankrupt |= dead
            E[bankrupt] = 0.0
            Nn[bankrupt] = 0.0

    for t in range(T):
        # 1. 市場結算 ＋ 2. 對日初權益計息
        interest = E * rf_daily
        E += Nn * ret[t] + interest
        Nn *= (1.0 + ret[t])
        _wipeout()                              # 3. 破產檢查（結算後）

        # 4. 轉倉（賣近月＋買次月，兩腳成本；不改變曝險）
        if is_roll[t]:
            q_legs = Nn / (MULTIPLIER * p_near[t]) + Nn / (MULTIPLIER * p_next[t])
            roll_cost = q_legs * per_lot + 2.0 * Nn * tax
            E -= roll_cost
            cost_cum += roll_cost
            cost_rec[t] += roll_cost
            _wipeout()

        # 5. 重算槓桿（觸價判斷用，成本前）
        L = np.divide(Nn, E, out=np.zeros(K), where=E > 0)
        lev_pre[t] = L
        np.maximum(max_lev, L, out=max_lev)

        # 6a. 門檻再平衡（先檢查上限 X，再檢查下限 Y — 規格順序）
        trig_down = (L >= X) & ~bankrupt
        trig_up = (L <= Y) & ~bankrupt & ~trig_down
        # 6b. 週期再平衡（每 rebal_every 個交易日，自建倉日起算）
        periodic = np.zeros(K, dtype=bool)
        if has_period.any():
            k = t + 1 - rebal_phase                    # 建倉後的第幾個交易日
            periodic = has_period & (k > 0) & (k % np.maximum(rebal_every, 1) == 0)
            periodic &= ~bankrupt
        do = trig_down | trig_up | periodic
        if do.any():
            Ntgt = target * E
            dN = np.abs(Ntgt - Nn)
            reb_cost = np.where(do, dN / (MULTIPLIER * price[t]) * per_lot + dN * tax, 0.0)
            sell = do & (Ntgt < Nn)             # 實際方向：賣出=減碼、買進=加碼
            n_down += sell
            n_up += do & ~sell
            trig[t] = np.where(sell, -1, np.where(do, 1, 0)).astype(np.int8)
            dN_rec[t] = np.where(do, dN, 0.0)
            Nn = np.where(do, Ntgt, Nn)
            E -= reb_cost
            cost_cum += reb_cost
            cost_rec[t] += reb_cost
            _wipeout()

        # 7. 記錄（成本後）
        eq[t] = E
        lev[t] = np.divide(Nn, E, out=np.zeros(K), where=E > 0)

    return SimResult(eq=eq, lev=lev, lev_pre=lev_pre, trig=trig, dN=dN_rec, cost=cost_rec,
                     n_up=n_up, n_down=n_down, cost_cum=cost_cum,
                     bankrupt=bankrupt, max_lev=max_lev)
