"""round3 引擎 sanity check：手算對照 + 與前輪引擎的整合互驗。

執行：conda run -n ML_env python -m round3.test_engine
"""
from __future__ import annotations

import numpy as np

from round3.engine import path_metrics, simulate

OK = {"n": 0}


def check(label: str, got, want, tol=1e-9):
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert np.allclose(got, want, atol=tol, rtol=0), \
        f"{label}: got {got}, want {want}"
    OK["n"] += 1
    print(f"  ok  {label}")


def test_bh1x_index():
    """BH 1x、無成本：權益 = 指數累積（L=1 無借款無閒置）。"""
    ret = np.array([0.01, -0.02, 0.005])
    res = simulate({"kind": "bh", "target": 1.0}, ret, financing="index",
                   costs="bp", rf_dep=0.0, rf_bor=0.001, cost_rate=0.0, e0=100.0)
    check("BH1x 權益", res.eq[:, 0], 100.0 * np.cumprod(1 + ret))
    check("BH1x 槓桿", res.lev[:, 0], [1.0, 1.0, 1.0])


def test_financing_daily2x():
    """每日再平衡 2x＋融資：E_t = E_{t-1}(1 + 2r − r_bor)。"""
    ret = np.array([0.01, -0.02, 0.005])
    rb = 0.0002
    res = simulate({"kind": "periodic", "target": 2.0, "every": 1}, ret,
                   financing="index", costs="bp", rf_dep=0.0, rf_bor=rb,
                   cost_rate=0.0, e0=100.0)
    want = 100.0 * np.cumprod(1 + 2 * ret - rb)
    check("Daily2x 融資權益", res.eq[:, 0], want)


def test_idle_credit():
    """L=0.5（簽號策略）：閒置現金 (E−N) 賺存款利率。"""
    ret = np.array([0.0, 0.0])
    rd = 0.001
    L = np.full(2, 0.5)
    res = simulate({"kind": "signal", "L_exec": L}, ret, financing="index",
                   costs="bp", rf_dep=rd, rf_bor=0.01, cost_rate=0.0, e0=100.0)
    # 日1：E = 100 + 50×0 + 50×0.001 = 100.05；N=50 → L≈0.49975，帶內不調整
    check("閒置計息 日1", res.eq[0, 0], 100.05)
    check("閒置計息 日2", res.eq[1, 0], 100.05 + (100.05 - 50.0) * rd)
    check("無調整", res.n_adj[0], 0)


def test_deadband_and_shift():
    """0.1x 門檻：帶內不動；訊號降至 1.0 當日執行（L_exec 已含 shift）。"""
    ret = np.array([0.0, 0.0, 0.0, 0.0])
    L = np.array([2.0, 2.0, 1.0, 1.0])
    res = simulate({"kind": "signal", "L_exec": L}, ret, financing="index",
                   costs="bp", rf_dep=0.0, cost_rate=0.0, e0=100.0)
    check("帶內不動（前2日）", res.lev[:2, 0], [2.0, 2.0])
    check("第3日執行降槓桿", res.lev[2, 0], 1.0)
    check("僅調整一次", res.n_adj[0], 1)


def test_cost_on_adjust():
    """調整成本 = |ΔN| × cost_rate，先調 N 再扣成本。"""
    ret = np.array([0.0])
    L = np.array([1.0])
    res = simulate({"kind": "signal", "L_exec": L}, ret, financing="index",
                   costs="bp", rf_dep=0.0, cost_rate=0.001, e0=100.0)
    # 建倉：N=100，成本 0.1 → E=99.9；日1：L=100/99.9=1.001，帶內不調整
    check("建倉成本", res.eq[0, 0], 99.9)
    check("累計成本", res.cost_cum[0], 0.1)


def test_cppi_ratchet():
    """CPPI m=4, D_max=0.5：手算兩日（含次日執行與棘輪 floor）。"""
    ret = np.array([-0.10, 0.0, 0.0])
    res = simulate({"kind": "cppi", "m": 4.0, "dmax": 0.5, "cap": 3.0}, ret,
                   financing="index", costs="bp", rf_dep=0.0, cost_rate=0.0,
                   e0=100.0)
    # 建倉 L=2 → N=200。日1：E=80, N=180, L=2.25；pending(初始)=2 → 調回 N=160
    #   收盤後 cushion=(80−50)/80=0.375 → pending=1.5
    # 日2：L=2.0 vs pending 1.5 → 調整 → L=1.5
    check("CPPI 日1 權益", res.eq[0, 0], 80.0)
    check("CPPI 日1 槓桿", res.lev[0, 0], 2.0)
    check("CPPI 日2 槓桿", res.lev[1, 0], 1.5)
    check("CPPI 日3 槓桿（帶內）", res.lev[2, 0], 1.5)


def test_fuse():
    """B2 保險絲：L≥3.5 減碼回 2.0，觸發計數。"""
    ret = np.array([-0.30, 0.0])
    res = simulate({"kind": "fuse", "X": 3.5, "Y": 1.1, "target": 2.0}, ret,
                   financing="index", costs="bp", rf_dep=0.0, cost_rate=0.0,
                   e0=100.0)
    # 日1：E=40, N=140, L=3.5 → 觸發 → N=80
    check("保險絲觸發後槓桿", res.lev[0, 0], 2.0)
    check("觸發次數", res.fuse_trig[0], 1)


def test_bankruptcy():
    """權益 ≤ 0 歸零不反彈。"""
    ret = np.array([-0.60, 0.50])
    res = simulate({"kind": "bh", "target": 2.0}, ret, financing="index",
                   costs="bp", rf_dep=0.0, cost_rate=0.0, e0=100.0)
    check("破產歸零", res.eq[:, 0], [0.0, 0.0])
    assert res.bankrupt[0]
    OK["n"] += 1
    print("  ok  破產旗標")


def test_lot_costs():
    """期貨口徑：建倉成本 = 口數×(手續費+滑價) + 名目×稅。"""
    ret = np.array([0.0])
    price = np.array([20000.0])
    res = simulate({"kind": "periodic", "target": 2.0, "every": 5}, ret,
                   financing="futures", costs="lot", rf_dep=0.0,
                   price=price, is_roll=np.array([False]),
                   p_near=np.array([np.nan]), p_next=np.array([np.nan]),
                   entry_price=np.array([20000.0]), e0=1_000_000.0)
    # q = 2e6/(200×20000) = 0.5 口 → 0.5×220 = 110；稅 = 2e6×2e-5 = 40
    check("期貨建倉成本", res.cost_cum[0], 150.0)
    check("期貨建倉後權益", res.eq[0, 0], 999_850.0)


def test_vs_round2_engine():
    """整合互驗：TX 歷史 20 年、每5日再平衡 2x —— round3 期貨口徑 vs 前輪引擎。"""
    import params as rp
    from continuous import build_continuous
    from data import load_tx_contracts
    from engine import simulate as sim2

    cs = build_continuous(load_tx_contracts(), rp.ROLL_OFFSET_DAYS)
    panel = cs.panel
    idx = panel.index
    s = int(idx.searchsorted(idx[-1] - __import__("pandas").DateOffset(years=20)))
    sl = slice(s, len(idx))
    ret = panel["ret"].to_numpy()[sl]
    price = panel["price"].to_numpy()[sl]
    is_roll = panel["is_roll"].to_numpy()[sl]
    p_near = panel["p_near"].to_numpy()[sl]
    p_next = panel["p_next"].to_numpy()[sl]
    entry = float(panel["price"].to_numpy()[s - 1])

    r2 = sim2(ret, price, is_roll, p_near, p_next,
              X=np.array([np.inf]), Y=np.array([-np.inf]),
              target=np.array([2.0]), entry_price=entry,
              rebal_every=np.array([5], dtype=np.int64))
    r3 = simulate({"kind": "periodic", "target": 2.0, "every": 5}, ret,
                  financing="futures", costs="lot", rf_dep=rp.DAILY_RF,
                  price=price, is_roll=is_roll, p_near=p_near, p_next=p_next,
                  entry_price=np.array([entry]))
    diff = np.max(np.abs(r2.eq[:, 0] - r3.eq[:, 0]) / r2.eq[:, 0])
    assert diff < 1e-10, f"與前輪引擎權益差異 {diff:.2e}"
    OK["n"] += 1
    print(f"  ok  與前輪引擎互驗（20年每5日再平衡，max 相對差 {diff:.1e}）")


def test_metrics():
    """指標：avg_lev / mdd / lev_at_trough。"""
    ret = np.array([0.10, -0.20, 0.05])
    res = simulate({"kind": "bh", "target": 1.0}, ret, financing="index",
                   costs="bp", rf_dep=0.0, cost_rate=0.0, e0=100.0)
    m = path_metrics(res, 0.0, e0=100.0)
    check("metrics cumret", m["cumret"].iloc[0], 1.10 * 0.80 * 1.05 - 1.0)
    check("metrics mdd", m["mdd"].iloc[0], 0.80 - 1.0)
    check("metrics avg_lev", m["avg_lev"].iloc[0], 1.0)


if __name__ == "__main__":
    for fn in [test_bh1x_index, test_financing_daily2x, test_idle_credit,
               test_deadband_and_shift, test_cost_on_adjust, test_cppi_ratchet,
               test_fuse, test_bankruptcy, test_lot_costs, test_metrics,
               test_vs_round2_engine]:
        print(fn.__name__)
        fn()
    print(f"\n全部 {OK['n']} 項檢核通過")
