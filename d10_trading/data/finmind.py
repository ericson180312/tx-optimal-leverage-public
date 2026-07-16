"""FinMind 資料層——移植自回測專案根目錄 data.py（規格 §3.3：移植，不要重寫）。

差異僅限於工程面：
- 快取目錄改為 d10_trading/var/cache；可從既有回測專案的 parquet 快取直接複製。
- 新增 force_current_year（強制更新當年度檔，日常批次用）與
  offline（只用既有快取、不打網路，測試用）兩個模式。
清理邏輯（prepare_contracts）與回測完全相同：只留日盤、月合約、結算價優先。
"""
from __future__ import annotations

import datetime as dt
import shutil
import time

import pandas as pd
import requests

from d10_trading import config

_SLEEP_BETWEEN_CALLS = 1.5
_FUT_COLUMNS = [
    "date", "futures_id", "contract_date", "open", "max", "min", "close",
    "spread", "spread_per", "volume", "settlement_price", "open_interest",
    "trading_session",
]
_MONTHLY_RE = r"\d{6}"   # 月合約 YYYYMM（排除週合約與價差組合）


def _api_get(params: dict) -> pd.DataFrame:
    if config.FINMIND_TOKEN:
        params = {**params, "token": config.FINMIND_TOKEN}
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            r = requests.get(config.FINMIND_API, params=params, timeout=(20, 120))
            r.raise_for_status()
            payload = r.json()
            if payload.get("msg") != "success":
                raise RuntimeError(f"FinMind 回應異常: {payload.get('msg')}")
            return pd.DataFrame(payload.get("data", []))
        except Exception as e:  # noqa: BLE001 - 網路層重試（FinMind 節流時退避拉長）
            last_err = e
            time.sleep(min(90, 3 * 2 ** attempt))
    raise RuntimeError("FinMind 連續失敗") from last_err


def _cache_is_fresh(path, today: dt.date, max_age_days: int) -> bool:
    if not path.exists():
        return False
    cached = pd.read_parquet(path)
    if cached.empty:
        return False
    latest = pd.to_datetime(cached["date"]).max().date()
    return (today - latest).days <= max_age_days


def fetch_futures_daily(*, force_current_year: bool = False,
                        offline: bool = False,
                        today: dt.date | None = None) -> pd.DataFrame:
    """TX 全歷史逐合約日行情（逐年快取；優先沿用既有回測專案快取）。

    force_current_year=True：當年度檔一律重抓（日常批次需要昨日結算價）。
    offline=True：只讀快取，不打網路（單元/黃金測試用）。
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if today is None:
        today = dt.date.today()
    frames: list[pd.DataFrame] = []
    for year in range(config.FUTURES_FIRST_YEAR, today.year + 1):
        fname = f"fut_{config.FUTURES_ID}_{year}.parquet"
        path = config.CACHE_DIR / fname
        is_current_year = year == today.year
        if not path.exists():
            for sib_dir in config.SIBLING_CACHE_DIRS:
                sib = sib_dir / fname
                if sib.exists():
                    shutil.copy2(sib, path)
                    break
        if offline:
            if path.exists():
                frames.append(pd.read_parquet(path))
            continue
        max_age = 0 if (is_current_year and force_current_year) else 3
        if path.exists() and (not is_current_year
                              or _cache_is_fresh(path, today, max_age)):
            frames.append(pd.read_parquet(path))
            continue
        try:
            df = _api_get({
                "dataset": "TaiwanFuturesDaily",
                "data_id": config.FUTURES_ID,
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
            })
            time.sleep(_SLEEP_BETWEEN_CALLS)
        except RuntimeError:
            if path.exists():   # 抓不到就退回略舊的快取；新鮮度由 series 層把關
                frames.append(pd.read_parquet(path))
                continue
            raise
        if df.empty:
            df = pd.DataFrame(columns=_FUT_COLUMNS)
        df.to_parquet(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("FinMind 快取為空且無法抓取（offline 模式需先備妥快取）")
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


def load_tx_contracts(**kwargs) -> pd.DataFrame:
    """一站式：抓取＋清理後的 TX 逐合約長表。kwargs 轉交 fetch_futures_daily。"""
    return prepare_contracts(fetch_futures_daily(**kwargs))
