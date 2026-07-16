"""資料層：TX 逐合約每日行情（FinMind 公開 API + 本地 parquet 快取）。

快取策略：
1. 本地 data/ 已有年度檔 → 直接用（當年度檔需最新日期距今 ≤ 3 天）。
2. 本地沒有但姊妹專案（python_台指跨月價差回測）有 → 複製過來。
3. 都沒有 → 向 FinMind 抓（含長退避重試，免費層會節流）。
"""
from __future__ import annotations

import datetime as dt
import shutil
import time

import pandas as pd
import requests

from backtest_threshold.params import (
    DATA_DIR, FINMIND_API, FUTURES_FIRST_YEAR, FUTURES_ID, SIBLING_CACHE_DIR,
)

_SLEEP_BETWEEN_CALLS = 1.5
_FUT_COLUMNS = [
    "date", "futures_id", "contract_date", "open", "max", "min", "close",
    "spread", "spread_per", "volume", "settlement_price", "open_interest",
    "trading_session",
]
_MONTHLY_RE = r"\d{6}"   # 月合約 YYYYMM（排除週合約與價差組合）


def _api_get(params: dict) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            r = requests.get(FINMIND_API, params=params, timeout=(20, 120))
            r.raise_for_status()
            payload = r.json()
            if payload.get("msg") != "success":
                raise RuntimeError(f"FinMind 回應異常: {payload.get('msg')} params={params}")
            return pd.DataFrame(payload.get("data", []))
        except Exception as e:  # noqa: BLE001 - 網路層重試（FinMind 節流時退避拉長）
            last_err = e
            time.sleep(min(90, 3 * 2 ** attempt))
    raise RuntimeError(f"FinMind 連續失敗: {params}") from last_err


def _cache_is_fresh(path, today: dt.date) -> bool:
    if not path.exists():
        return False
    cached = pd.read_parquet(path)
    if cached.empty:
        return False
    latest = pd.to_datetime(cached["date"]).max().date()
    return (today - latest).days <= 3


def fetch_futures_daily() -> pd.DataFrame:
    """TX 全歷史逐合約日行情（逐年快取；優先沿用姊妹專案快取）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    frames: list[pd.DataFrame] = []
    for year in range(FUTURES_FIRST_YEAR, today.year + 1):
        fname = f"fut_{FUTURES_ID}_{year}.parquet"
        path = DATA_DIR / fname
        is_current_year = year == today.year
        if not path.exists():
            sib = SIBLING_CACHE_DIR / fname
            if sib.exists():
                shutil.copy2(sib, path)
        if path.exists() and (not is_current_year or _cache_is_fresh(path, today)):
            frames.append(pd.read_parquet(path))
            continue
        try:
            df = _api_get({
                "dataset": "TaiwanFuturesDaily",
                "data_id": FUTURES_ID,
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
            })
            time.sleep(_SLEEP_BETWEEN_CALLS)
        except RuntimeError:
            if path.exists():   # 抓不到就退回略舊的快取，只影響最後幾天
                print(f"  [warn] FinMind 更新 {year} 失敗，沿用既有快取")
                frames.append(pd.read_parquet(path))
                continue
            raise
        if df.empty:
            df = pd.DataFrame(columns=_FUT_COLUMNS)
        df.to_parquet(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def prepare_contracts(raw: pd.DataFrame) -> pd.DataFrame:
    """清理原始資料 → (date, contract, price, volume) 長表。

    - 只留一般時段；夜盤（after_market）無結算價。
    - 只留月合約（YYYYMM），排除週合約與價差組合。
    - 價格取每日「結算價」，缺漏時退回收盤價；兩者皆 0 視為無行情，剔除。
    """
    df = raw.copy()
    if "trading_session" in df.columns:
        df = df[df["trading_session"].fillna("position") == "position"]
    df["contract"] = df["contract_date"].astype(str).str.strip()
    df = df[df["contract"].str.fullmatch(_MONTHLY_RE)]

    settle = pd.to_numeric(df["settlement_price"], errors="coerce").fillna(0.0)
    close = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df["price"] = settle.where(settle > 0, close)
    df = df[df["price"] > 0]

    df = df[["date", "contract", "price", "volume"]].sort_values(["date", "contract"])
    df = df.drop_duplicates(subset=["date", "contract"], keep="first")
    return df.reset_index(drop=True)


def load_tx_contracts() -> pd.DataFrame:
    """一站式：抓取＋清理後的 TX 逐合約長表。"""
    return prepare_contracts(fetch_futures_daily())
