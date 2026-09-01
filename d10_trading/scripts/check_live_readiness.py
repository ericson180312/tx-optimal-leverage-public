"""G2′ 前置：正式環境**唯讀**就緒檢查（不下單、不改任何設定或狀態）。

    python -m d10_trading.scripts.check_live_readiness

刻意獨立於 `config.LIVE_TRADING`：在檢查期間把 broker 的 simulation 旗標
強制設為 False，讓你在切換 LIVE_TRADING **之前**就能確認正式環境可用。
重用 ShioajiBroker 而非另寫連線邏輯——這樣連 G2′ 實際會走的程式路徑
（合約解析、權益、部位、風險指標）也一併驗證。

全程只呼叫查詢 API。**不 import 也不呼叫任何下單/撤單函式**，
唯一具副作用的動作是 activate_ca（載入憑證，不產生委託）。

回報：正式登入、CA 啟用、帳戶簽署狀態、真實權益、風險指標實際值與單位、
部位明細（標出非 PRODUCT 者）、依真實權益算出的目標口數 vs 現有口數。
"""
from __future__ import annotations

import datetime as dt
import sys

from d10_trading import config
from d10_trading.execution.broker import ShioajiBroker
from d10_trading.strategy import sizing


def _mask(s: str, keep: int = 3) -> str:
    s = str(s or "")
    return s[:keep] + "*" * max(0, len(s) - keep)


def main(argv=None) -> int:
    print("═" * 62)
    print(" 正式環境就緒檢查（唯讀；不下單、不改設定）")
    print("═" * 62)
    if not config.SJ_CA_PATH or not config.SJ_CA_PASSWD:
        print("❌ .env 缺 SJ_CA_PATH / SJ_CA_PASSWD——正式登入需要憑證")
        return 2

    broker = ShioajiBroker()
    broker.simulation = False          # 唯讀檢查：強制正式連線，不動 config
    ok = True
    try:
        broker.login()                 # 內含 activate_ca（正式模式）
        print("✅ 正式登入 + CA 啟用成功")
    except Exception as e:  # noqa: BLE001
        print(f"❌ 正式登入/CA 啟用失敗：{type(e).__name__}: {e}")
        return 2

    try:
        acct = broker._api.futopt_account
        print(f"   期貨帳戶 {_mask(getattr(acct,'account_id',''))}"
              f"（broker {getattr(acct,'broker_id','')}）"
              f" signed={getattr(acct,'signed',None)}")
        if not getattr(acct, "signed", False):
            print("   ❌ 帳戶未完成簽署——下單會被拒")
            ok = False

        print()
        eq = broker.equity()
        print(f"── 權益 ──\n   margin().equity = {eq:,.0f}")
        if eq <= 0:
            print("   ❌ 權益為 0——正式環境不應如此")
            ok = False

        ri = broker.risk_indicator()
        print(f"\n── 風險指標 ──\n   API 回傳 = {ri}")

        print("\n── 部位 ──")
        pos = broker.positions()
        if not pos:
            print("   （空手）")
        for p in pos:
            tag = "" if p.code.startswith(config.PRODUCT) else "  ⚠️ 非 PRODUCT！"
            print(f"   {p.code} {p.direction} {p.quantity} 口 @{p.avg_price:,.0f}"
                  f"  未實現 {p.pnl:,.0f}{tag}")
        alien = [p for p in pos if not p.code.startswith(config.PRODUCT)]
        if alien:
            print(f"   ❌ 有 {len(alien)} 筆非 {config.PRODUCT} 部位——對帳會告警停止")
            ok = False

        held = sum(p.quantity for p in pos if p.code.startswith(config.PRODUCT))
        front, _ = broker.front_and_next(config.PRODUCT, dt.date.today())
        px = broker.quote(front.code).price
        r = sizing.target_contracts(config.L_HIGH, eq, px)
        print(f"\n── 接管日試算（L={config.L_HIGH}，近月 {front.code} @{px:,.0f}）──")
        print(f"   raw {r.raw_contracts:.2f} → 目標 {r.target_contracts} 口"
              f"｜現有 {held} 口 → "
              + ("✅ 無動作" if r.target_contracts == held
                 else f"⚠️ 接管日會交易 {r.target_contracts-held:+d} 口"))
        if ri:
            calc = eq / (held * config.TMF_MARGIN_REF) * 100 if held else 0
            print(f"\n── 風險指標單位交叉核對 ──")
            print(f"   API {ri:,.1f}　vs　以原始保證金推算 {calc:,.0f}"
                  f"（{config.TMF_MARGIN_REF:,.0f}×{held} 口）")
            print("   → 兩者同量級即為百分比；若 API 明顯較大，券商可能以維持保證金為分母")
            print(f"   目前門檻 WARN<{config.RISK_INDICATOR_WARN:.0f}"
                  f" / ALERT<{config.RISK_INDICATOR_ALERT:.0f}")
        if r.target_contracts > config.MAX_CONTRACTS:
            print(f"\n   ❌ 目標 {r.target_contracts} > MAX_CONTRACTS "
                  f"{config.MAX_CONTRACTS} → 會拒單停止")
            ok = False
    finally:
        broker.logout()

    print("\n" + "═" * 62)
    print(" ✅ 就緒" if ok else " ❌ 有項目未通過，請先處理")
    print("═" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
