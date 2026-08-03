"""條件式曝險回測引擎（numpy 向量化：一次日迴圈同時模擬 P 條路徑／視窗）。

═══════════════════════════════════════════════════════════════════════════
 每日流程（規格 §2，順序固定）
═══════════════════════════════════════════════════════════════════════════
 1. 市場結算＋計息／融資（皆以「日初」水位計提）：
    - financing="futures"（TX 期貨）：E += N·r_t + E·rf_dep（保證金交易，權益
      全額現金計息；不對名目借款計息 —— 前兩輪口徑）。
    - financing="index"（指數槓桿模擬，規格 §1）：E += N·r_t
      − max(N−E,0)·rf_bor + max(E−N,0)·rf_dep（超過 1 倍的部分付短率+30bp、
      L<1 的閒置現金賺短率；日頻計提）。
 2. 破產檢查：E ≤ 0 → 權益、曝險歸零，其後永遠為 0（不可反彈）。
 3. 轉倉（僅 costs="lot" 的歷史 TX 路徑）：兩腳各計手續費＋期交稅＋滑價。
 4. 重算槓桿 L = N / E（收盤觸價判斷）。
 5. 曝險調整（依策略種類）：
    - periodic（B1）：每 every 日無條件調回 target。
    - bh（B3）：永不調整。
    - fuse（B2）：L ≥ X → 減碼回 target；L ≤ Y → 加碼回 target（同收盤執行，
      觸發變數是自身組合狀態、無 look-ahead —— 前輪口徑）。
    - signal（C/D/F2）：讀取「前一日收盤訊號」的目標槓桿（呼叫端已 shift），
      |L_des − L| > DEADBAND(0.1) 才執行（規格 §2 減少碎單）。
    - signal_fuse（F1）：同 signal，另掛 [3.5] 減碼保險絲 —— L ≥ 3.5 時無條件
      調整至 min(L_des, target)，並記錄觸發次數（結論 §10-3 用）。
    - cppi（E）：L_des = clip(m·(E−Floor)/E, 0, cap)，Floor=(1−D_max)·HWM
      （高水位棘輪只升不降）。訊號於第 t 日收盤以當日權益計算、次日收盤執行。
    調整成本：costs="bp" → |ΔN|×cost_rate（單邊）；costs="lot" → TX 明細模型
    （|ΔN| 換算口數×(手續費+滑價) + |ΔN|×期交稅）。成本即時扣入權益
    （先調 N 再扣成本，成交後 L 與目標有一點點偏差 —— 與實務相同）。
 6. 記錄 E、L（成本後）。
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_threshold import params as root_params
from backtest_multimarket.params import DEADBAND, DERISK_LEV, INITIAL_EQUITY, TRADING_DAYS

_MULT = root_params.MULTIPLIER
_PER_LOT = root_params.FEE_PER_SIDE + root_params.SLIPPAGE_TICKS * root_params.TICK_VALUE
_TAX = root_params.TAX_RATE


@dataclass
class CondResult:
    eq: np.ndarray         # (T,P) 每日收盤後權益
    lev: np.ndarray        # (T,P) 每日收盤後（成本後）槓桿
    n_adj: np.ndarray      # (P,) 曝險調整次數（不含建倉與轉倉）
    cost_cum: np.ndarray   # (P,) 累計成本（含建倉、轉倉）
    fuse_trig: np.ndarray  # (P,) F1/B2 保險絲減碼觸發次數
    bankrupt: np.ndarray   # (P,) bool
    liquidated: np.ndarray | None = None   # (P,) bool：維持保證金強制平倉（第五輪新增）


def _bcast(x, shape) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.broadcast_to(x, shape)
    if x.ndim == 1:                      # (T,) → (T,P)
        return np.broadcast_to(x[:, None], shape)
    return np.broadcast_to(x, shape)


def simulate(strategy: dict, ret: np.ndarray, *,
             financing: str, costs: str,
             rf_dep: np.ndarray | float, rf_bor: np.ndarray | float = 0.0,
             cost_rate: float = 0.0,
             price: np.ndarray | None = None, is_roll: np.ndarray | None = None,
             p_near: np.ndarray | None = None, p_next: np.ndarray | None = None,
             entry_price: np.ndarray | None = None,
             maint_margin_rate: float = 0.0,
             e0: float = INITIAL_EQUITY) -> CondResult:
    """模擬單一策略在 (T,P) 報酬矩陣上的表現。

    strategy: {"kind": "periodic"|"bh"|"fuse"|"signal"|"signal_fuse"|"cppi", ...}
    financing: "futures" | "index"；costs: "lot" | "bp"。
    rf_dep/rf_bor: 每交易日利率（scalar、(T,) 或 (T,P)）。
    costs="lot" 時需 price/is_roll/p_near/p_next/entry_price（皆 (T,P)／(P,)）。
    """
    ret = np.asarray(ret, dtype=float)
    if ret.ndim == 1:
        ret = ret[:, None]
    T, P = ret.shape
    rf_dep = _bcast(rf_dep, (T, P))
    rf_bor = _bcast(rf_bor, (T, P))
    if costs == "lot":
        price = _bcast(price, (T, P))
        p_near = _bcast(p_near, (T, P))
        p_next = _bcast(p_next, (T, P))
        is_roll = np.asarray(is_roll, dtype=bool)
        if is_roll.ndim == 1:
            is_roll = np.broadcast_to(is_roll[:, None], (T, P))
    kind = strategy["kind"]

    # ── 建倉 ────────────────────────────────────────────────────────────────
    if kind in ("periodic", "fuse", "bh"):
        L0 = np.full(P, float(strategy["target"]))
    elif kind in ("signal", "signal_fuse"):
        L0 = np.asarray(strategy["L_exec"][0], dtype=float) * np.ones(P)
    elif kind == "cppi":
        L0 = np.full(P, min(strategy["m"] * strategy["dmax"], strategy["cap"]))
    else:
        raise ValueError(kind)

    E = np.full(P, float(e0))
    N = L0 * E
    if costs == "lot":
        ep = np.asarray(entry_price, dtype=float) * np.ones(P)
        entry_cost = N / (_MULT * ep) * _PER_LOT + N * _TAX
    else:
        entry_cost = N * cost_rate
    E -= entry_cost
    cost_cum = entry_cost.copy()

    dead = np.zeros(P, dtype=bool)
    liq = np.zeros(P, dtype=bool)
    n_adj = np.zeros(P, dtype=np.int64)
    fuse_trig = np.zeros(P, dtype=np.int64)
    eq = np.empty((T, P))
    lev = np.empty((T, P))

    if kind == "cppi":
        m, dmax, cap = strategy["m"], strategy["dmax"], strategy["cap"]
        hwm = np.full(P, float(e0))
        pending = np.minimum(m * dmax, cap) * np.ones(P)   # 次日執行的目標槓桿

    L_exec = strategy.get("L_exec")

    def _wipe():
        nonlocal dead
        d = E <= 0
        if d.any():
            dead |= d
            E[dead] = 0.0
            N[dead] = 0.0

    def _margin_call():
        """維持保證金強制平倉（第五輪新增；maint_margin_rate=0 時完全停用）。

        觸發條件：帳戶權益 < 維持保證金需求（= maint_margin_rate × 名目部位）。
        觸發時全數平倉：曝險歸零、權益保留當下剩餘值，其後永遠不再進場
        （沿用破產的不可反彈紀律；liquidated 與 bankrupt 分開計數）。
        """
        nonlocal liq
        if maint_margin_rate <= 0.0:
            return
        hit = (~dead) & (~liq) & (E < maint_margin_rate * np.abs(N))
        if hit.any():
            liq |= hit
            N[liq] = 0.0

    for t in range(T):
        # 1. 結算＋計息／融資
        if financing == "futures":
            E += N * ret[t] + E * rf_dep[t]
        else:
            borrow = np.maximum(N - E, 0.0)
            idle = np.maximum(E - N, 0.0)
            E += N * ret[t] - borrow * rf_bor[t] + idle * rf_dep[t]
        N *= (1.0 + ret[t])
        _wipe()                                            # 2. 破產檢查

        # 3. 轉倉（歷史 TX）
        if costs == "lot" and is_roll is not None:
            roll_mask = is_roll[t] & ~dead
            if roll_mask.any():
                q_legs = N / (_MULT * p_near[t]) + N / (_MULT * p_next[t])
                rc = np.where(roll_mask, q_legs * _PER_LOT + 2.0 * N * _TAX, 0.0)
                E -= rc
                cost_cum += rc
                _wipe()

        # 3b. 維持保證金檢查（第五輪新增；預設停用）——結算與轉倉成本後、再平衡前
        _margin_call()

        # 4. 觸價槓桿
        L = np.divide(N, E, out=np.zeros(P), where=E > 0)

        # 5. 曝險調整
        if kind == "bh":
            do = np.zeros(P, dtype=bool)
            tgt = L
        elif kind == "periodic":
            do = np.full(P, (t + 1) % strategy["every"] == 0) & ~dead
            tgt = np.full(P, strategy["target"])
        elif kind == "fuse":
            hit_x = L >= strategy["X"]
            do = (hit_x | (L <= strategy["Y"])) & ~dead
            fuse_trig += hit_x & ~dead
            tgt = np.full(P, strategy["target"])
        elif kind in ("signal", "signal_fuse"):
            des = L_exec[t] * np.ones(P)
            do = (np.abs(des - L) > DEADBAND) & ~dead
            if kind == "signal_fuse":
                fuse = (L >= strategy["X"]) & ~dead
                des = np.where(fuse, np.minimum(des, strategy["target"]), des)
                do |= fuse
                fuse_trig += fuse
            tgt = des
        else:                                              # cppi
            do = (np.abs(pending - L) > DEADBAND) & ~dead
            tgt = pending
        do &= ~liq          # 強制平倉後永不再進場（liq 全 False 時為 no-op）
        if do.any():
            Ntgt = tgt * E
            dN = np.abs(Ntgt - N)
            if costs == "lot":
                ac = dN / (_MULT * price[t]) * _PER_LOT + dN * _TAX
            else:
                ac = dN * cost_rate
            ac = np.where(do, ac, 0.0)
            N = np.where(do, Ntgt, N)
            E -= ac
            cost_cum += ac
            n_adj += do
            _wipe()

        # CPPI：以「本日收盤後」權益計算次日目標槓桿（高水位棘輪）
        if kind == "cppi":
            hwm = np.maximum(hwm, E)
            floor = (1.0 - dmax) * hwm
            cushion = np.divide(E - floor, E, out=np.zeros(P), where=E > 0)
            pending = np.clip(m * cushion, 0.0, cap)

        # 6. 記錄
        eq[t] = E
        lev[t] = np.divide(N, E, out=np.zeros(P), where=E > 0)

    return CondResult(eq=eq, lev=lev, n_adj=n_adj, cost_cum=cost_cum,
                      fuse_trig=fuse_trig, bankrupt=dead, liquidated=liq)


# ── 指標（規格 §7：沿用前輪全套，另加三項）──────────────────────────────────

def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    """(T,P) 布林 → (P,) 各欄最長連續 True 長度。"""
    T, P = mask.shape
    out = np.zeros(P, dtype=np.int64)
    run = np.zeros(P, dtype=np.int64)
    for t in range(T):
        run = np.where(mask[t], run + 1, 0)
        np.maximum(out, run, out=out)
    return out


def path_metrics(res: CondResult, rf_dep, *, e0: float = INITIAL_EQUITY) -> pd.DataFrame:
    """(T,P) 模擬結果 → P rows 指標表。破產路徑的風險調整指標記 NaN。"""
    eq, lev = res.eq, res.lev
    T, P = eq.shape
    years = T / TRADING_DAYS
    rf = _bcast(rf_dep, (T, P))

    prev = np.vstack([np.full((1, P), e0), eq[:-1]])
    r = np.divide(eq - prev, prev, out=np.zeros_like(eq), where=prev > 0)
    ex = r - rf

    cumret = eq[-1] / e0 - 1.0
    cagr = np.where(eq[-1] > 0, (np.maximum(eq[-1], 1e-300) / e0) ** (1.0 / years) - 1.0, -1.0)

    mean_ex = ex.mean(axis=0)
    std_ex = ex.std(axis=0, ddof=1)
    vol_ann = r.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = np.divide(mean_ex, std_ex, out=np.full(P, np.nan), where=std_ex > 0) \
        * np.sqrt(TRADING_DAYS)
    neg = np.minimum(ex, 0.0)
    dd_ann = np.sqrt((neg ** 2).mean(axis=0)) * np.sqrt(TRADING_DAYS)
    sortino = np.divide(mean_ex * TRADING_DAYS, dd_ann,
                        out=np.full(P, np.nan), where=dd_ann > 0)

    peak = np.maximum.accumulate(np.vstack([np.full((1, P), e0), eq]), axis=0)[1:]
    dd = eq / peak - 1.0
    mdd = dd.min(axis=0)
    trough = dd.argmin(axis=0)
    lev_at_trough = lev[trough, np.arange(P)]
    calmar = np.divide(cagr, -mdd, out=np.full(P, np.nan), where=mdd < 0)
    uw_months = _longest_true_run(dd < -1e-12) / 21.0

    out = pd.DataFrame({
        "cumret": cumret, "cagr": cagr, "vol_ann": vol_ann, "sharpe": sharpe,
        "sortino": sortino, "mdd": mdd, "calmar": calmar, "uw_months": uw_months,
        "avg_lev": lev.mean(axis=0),
        "max_lev": lev.max(axis=0), "min_lev": lev.min(axis=0),
        "lev_at_trough": lev_at_trough,
        "time_derisked": (lev < DERISK_LEV).mean(axis=0),
        "n_adj": res.n_adj, "cost_ratio": res.cost_cum / e0,
        "fuse_trig": res.fuse_trig, "bankrupt": res.bankrupt,
        # 第五輪新增：維持保證金強制平倉旗標（未啟用保證金模型時恆 False）
        "liquidated": (res.liquidated if res.liquidated is not None
                       else np.zeros(P, dtype=bool)),
    })
    out.loc[out["bankrupt"], ["sharpe", "sortino", "calmar"]] = np.nan
    return out
