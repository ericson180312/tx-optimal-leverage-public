"""🔴 黃金測試（規格 §3.2，G0 閘門條件）。

把第三輪回測的還原連續序列餵入 live 模組：
1. live 資料管線（移植版）在同一份 FinMind 快取上重建的序列，
   餵入 live regime 模組後，輸出必須與 round3.signals.trend_leverage 逐位元相同。
2. regime 切換序列必須與第三輪回測輸出 CSV（r3_equity_TX_FUT.csv 的
   lev_D_10 欄）逐日一致：回測執行槓桿（T+1）= 訊號槓桿 shift 一日。

需要根目錄 data/ 的 FinMind parquet 快取與 round3/output CSV；
缺任一者則 skip（在開發機必須實跑，CI 無資料時不誤報）。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from live import config
from live.data.finmind import load_tx_contracts
from live.data.series import build_signal_series
from live.strategy import regime

ROOT = Path(__file__).resolve().parents[2]
R3_CSV = ROOT / "round3" / "output" / "r3_equity_TX_FUT.csv"

pytestmark = pytest.mark.skipif(
    not (R3_CSV.exists() and (ROOT / "data" / "fut_TX_1998.parquet").exists()),
    reason="需要回測專案的資料快取與 round3 輸出")


@pytest.fixture(scope="module")
def signal_series():
    # offline：直接沿用回測所用的 parquet 快取（複製到 live/var/cache）
    cdf = load_tx_contracts(offline=True, today=dt.date.today())
    return build_signal_series(cdf)


def test_golden_bitwise_vs_round3_code(signal_series):
    """live 訊號 = round3 訊號，逐位元。"""
    import round3.signals as r3sg
    ours = regime.trend_leverage(signal_series.spot)
    ref = r3sg.trend_leverage(signal_series.spot, config.L_LOW)
    assert np.array_equal(ours, ref), "live 與 round3 訊號不一致"


def test_golden_switch_dates_vs_backtest_csv(signal_series):
    """regime 切換日期序列 vs 回測 CSV（逐日比對，T+1 對齊）。"""
    csv = pd.read_csv(R3_CSV, parse_dates=["date"])
    csv_dates = [d.date() for d in csv["date"]]
    # 回測執行 regime：實現槓桿以 1.5 為界映回 {1,2}（deadband 漂移 < ±0.3）
    csv_exec = np.where(csv["lev_D_10"].to_numpy() >= 1.5,
                        config.L_HIGH, config.L_LOW)

    lev = regime.trend_leverage(signal_series.spot)
    # T+1：執行槓桿 = 訊號 shift 一日（首日沿用首值，與回測 to_exec 相同）
    exec_lev = np.concatenate([lev[:1], lev[:-1]])
    pos = {d: i for i, d in enumerate(signal_series.dates)}

    missing = [d for d in csv_dates if d not in pos]
    assert not missing, f"回測 CSV 有 {len(missing)} 個日期不在 live 序列中"

    ours = np.array([exec_lev[pos[d]] for d in csv_dates])
    diff = np.nonzero(ours != csv_exec)[0]
    assert diff.size == 0, (
        f"執行 regime 與回測 CSV 不一致共 {diff.size} 日，"
        f"首個：{csv_dates[diff[0]]}（live={ours[diff[0]]}, csv={csv_exec[diff[0]]}）")

    # 切換日期序列（訊號日）完整列印比對用
    sw = regime.regime_switches(signal_series.dates, lev)
    assert len(sw) > 10, "切換次數異常過少（序列疑似不完整）"
