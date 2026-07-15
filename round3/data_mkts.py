"""第三輪資料層：四市場日頻序列（規格 §1）。

資料來源（各市場實際可得起點於報告揭露）：
- 台股 TX_FUT：沿用根目錄逐合約期貨管線（FinMind parquet 快取；carry 內生）。
- 台股 TX_IDX：TAIEX 發行量加權**報酬指數**（FinMind，2003-01 起；真實 TR）＋融資模型。
- S&P 500：Yahoo ^GSPC 價格指數（1927-12-30 起）；1988-01 起拼接 ^SP500TR
  **真實總報酬指數**，之前以「價格＋年代平均股息率」近似。
- 日經 225：Yahoo ^N225（1965-01-05 起，涵蓋 1989-12 高點）＋股息率近似。
- 恆生：Yahoo ^HSI（1986-12-31 起）＋股息率近似。
- 短率：美國 Ken French 日頻 RF（1926-07 起，真實序列）；日本／香港以分段常數
  近似（params.SHORT_RATE_STEPS）；台灣沿用前輪 1.7% 常數。

快取：round3/data/*.parquet。Yahoo 歷史指數不可變，快取超過 7 天才嘗試更新，
更新失敗一律退回既有快取。
"""
from __future__ import annotations

import datetime as dt
import io
import shutil
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from round3.params import (
    DATA_DIR, DIV_YIELD_STEPS, KENFRENCH_FACTORS_URL, MARKET_LABEL,
    SHORT_RATE_STEPS, TRADING_DAYS, TW_SHORT_RATE, YAHOO_SYMBOLS,
    YAHOO_TR_SYMBOLS,
)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
_SIBLING = r"C:\Users\User\Desktop\python_台指跨月價差回測\analysis\roll_carry\data"


# ── 小工具 ──────────────────────────────────────────────────────────────────

def ann_to_daily(rate) -> np.ndarray | float:
    """年化利率 → 每交易日複利利率（與前輪 DAILY_RF 同口徑）。"""
    return (1.0 + np.asarray(rate, dtype=float)) ** (1.0 / TRADING_DAYS) - 1.0


def step_series(steps: list[tuple[int, float]], dates: pd.DatetimeIndex) -> np.ndarray:
    """分段常數（(自該年起, 年化值)）展開成與 dates 對齊的年化序列。"""
    out = np.full(len(dates), steps[0][1])
    years = dates.year.to_numpy()
    for y0, v in steps:
        out[years >= y0] = v
    return out


def _cache_ok(path, max_age_days: int) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age <= max_age_days * 86400


# ── Yahoo / Ken French 下載 ────────────────────────────────────────────────

def fetch_yahoo(symbol: str) -> pd.Series:
    """Yahoo chart API 全歷史日收盤（快取 7 天）。回傳 close Series（index=日期）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"yahoo_{symbol.replace('^', '')}.parquet"
    if _cache_ok(path, 7):
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        return s
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = requests.get(url, params={"period1": "-2208988800",
                                      "period2": str(int(time.time())),
                                      "interval": "1d"},
                         headers=_HEADERS, timeout=120)
        r.raise_for_status()
        j = r.json()["chart"]["result"][0]
        ts = pd.to_datetime(j["timestamp"], unit="s").normalize()
        close = pd.Series(j["indicators"]["quote"][0]["close"], index=ts, name="close")
        close = close.dropna()
        close = close[close > 0]
        close = close[~close.index.duplicated(keep="last")].sort_index()
        close.to_frame().to_parquet(path)
        return close
    except Exception as e:  # noqa: BLE001 - 網路失敗退回既有快取
        if path.exists():
            print(f"  [warn] Yahoo {symbol} 更新失敗（{type(e).__name__}），沿用快取")
            s = pd.read_parquet(path)["close"]
            s.index = pd.to_datetime(s.index)
            return s
        raise


def fetch_kenfrench_rf() -> pd.Series:
    """Ken French 日頻 RF（1 個月 T-bill 日報酬，%）→ 小數日報酬 Series。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "kenfrench_rf.parquet"
    if _cache_ok(path, 30):
        s = pd.read_parquet(path)["rf"]
        s.index = pd.to_datetime(s.index)
        return s
    try:
        r = requests.get(KENFRENCH_FACTORS_URL, headers=_HEADERS, timeout=180)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        raw = zf.read(zf.namelist()[0]).decode("latin-1").splitlines()
        rows = []
        for ln in raw:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 5 and len(parts[0]) == 8 and parts[0].isdigit():
                rows.append((parts[0], float(parts[4])))
        df = pd.DataFrame(rows, columns=["date", "rf_pct"])
        s = pd.Series(df["rf_pct"].to_numpy() / 100.0,
                      index=pd.to_datetime(df["date"], format="%Y%m%d"), name="rf")
        s.to_frame().to_parquet(path)
        return s
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"  [warn] Ken French RF 更新失敗（{type(e).__name__}），沿用快取")
            s = pd.read_parquet(path)["rf"]
            s.index = pd.to_datetime(s.index)
            return s
        raise


def load_taiex() -> tuple[pd.Series, pd.Series]:
    """TAIEX 報酬指數（2003-01 起）與價格指數（1999-01 起）。優先沿用姊妹專案快取。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for fname, col in [("taiex_tr.parquet", "price"), ("taiex_price.parquet", "close")]:
        path = DATA_DIR / fname
        if not path.exists():
            sib = f"{_SIBLING}\\{fname}"
            shutil.copy2(sib, path)
        df = pd.read_parquet(path)
        c = col if col in df.columns else "price"
        s = pd.Series(pd.to_numeric(df[c], errors="coerce").to_numpy(),
                      index=pd.to_datetime(df["date"]), name=fname).dropna().sort_index()
        s = s[s > 0]
        out.append(s)
    return out[0], out[1]


# ── 市場物件 ────────────────────────────────────────────────────────────────

@dataclass
class Market:
    key: str
    label: str
    dates: pd.DatetimeIndex
    ret: np.ndarray                  # (T,) 日總報酬（策略損益用）
    spot: np.ndarray                 # (T,) 趨勢訊號用價格水準
    rf_ann: np.ndarray               # (T,) 年化短率（Sharpe 無風險利率＝融資基準）
    mode: str                        # "index"（融資模型）| "futures"（期貨口徑）
    price: np.ndarray | None = None  # futures：持倉合約結算價（口數換算）
    is_roll: np.ndarray | None = None
    p_near: np.ndarray | None = None
    p_next: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def rf_daily(self) -> np.ndarray:
        return ann_to_daily(self.rf_ann)

    @property
    def T(self) -> int:
        return len(self.ret)


def _index_market(key: str, close: pd.Series, tr_close: pd.Series | None,
                  rf_ann: np.ndarray | None = None) -> Market:
    """價格指數（＋可得的真實 TR 拼接、股息率近似）組裝成 index-mode 市場。"""
    dates = close.index[1:]
    ret_price = close.to_numpy()[1:] / close.to_numpy()[:-1] - 1.0

    div_d = ann_to_daily(step_series(DIV_YIELD_STEPS[key], dates))
    ret = ret_price + div_d
    tr_used_from = None
    if tr_close is not None:
        tr_ret = (tr_close / tr_close.shift(1) - 1.0).dropna()
        tr_ret = tr_ret.reindex(dates)
        use = tr_ret.notna().to_numpy()
        ret = np.where(use, tr_ret.to_numpy(), ret)
        if use.any():
            tr_used_from = str(dates[use.argmax()].date())

    if rf_ann is None:
        rf_ann = step_series(SHORT_RATE_STEPS[key], dates)

    return Market(
        key=key, label=MARKET_LABEL[key], dates=dates, ret=ret,
        spot=close.to_numpy()[1:], rf_ann=np.asarray(rf_ann, float), mode="index",
        meta={"start": str(dates[0].date()), "end": str(dates[-1].date()),
              "n_days": len(dates), "tr_source": ("真實TR自 " + tr_used_from) if tr_used_from
              else "價格指數＋股息率近似",
              "max_day_ret": float(np.max(ret)), "min_day_ret": float(np.min(ret))})


def load_markets(tx_cs=None) -> dict[str, Market]:
    """載入全部市場。tx_cs 可傳入已建好的 TX ContinuousSeries（避免重複建構）。"""
    mkts: dict[str, Market] = {}

    # ── 海外三市場 ─────────────────────────────────────────────────────────
    kf_rf = fetch_kenfrench_rf()
    for key in ["SPX", "NKX", "HSI"]:
        close = fetch_yahoo(YAHOO_SYMBOLS[key])
        tr = fetch_yahoo(YAHOO_TR_SYMBOLS[key]) if key in YAHOO_TR_SYMBOLS else None
        rf = None
        if key == "SPX":
            rf_d = kf_rf.reindex(close.index[1:]).ffill().bfill()
            rf = (1.0 + rf_d.to_numpy()) ** TRADING_DAYS - 1.0   # 日→年化（揭露用）
        mkts[key] = _index_market(key, close, tr, rf)
        if key == "SPX":
            mkts[key].meta["rf_source"] = (f"Ken French 日頻RF {kf_rf.index[0].date()}"
                                           f"~{kf_rf.index[-1].date()}（其後沿用末值）")
        else:
            mkts[key].meta["rf_source"] = "分段常數近似（params.SHORT_RATE_STEPS）"

    # ── 台股：報酬指數＋融資 ───────────────────────────────────────────────
    taiex_tr, taiex_px = load_taiex()
    common = taiex_tr.index.intersection(taiex_px.index)
    tr_al, px_al = taiex_tr.loc[common], taiex_px.loc[common]
    dates = common[1:]
    ret = tr_al.to_numpy()[1:] / tr_al.to_numpy()[:-1] - 1.0
    mkts["TX_IDX"] = Market(
        key="TX_IDX", label=MARKET_LABEL["TX_IDX"], dates=dates, ret=ret,
        spot=px_al.to_numpy()[1:], rf_ann=np.full(len(dates), TW_SHORT_RATE),
        mode="index",
        meta={"start": str(dates[0].date()), "end": str(dates[-1].date()),
              "n_days": len(dates), "tr_source": "TAIEX 發行量加權報酬指數（真實TR）",
              "rf_source": f"常數 {TW_SHORT_RATE:.1%}（沿用前輪口徑）",
              "max_day_ret": float(np.max(ret)), "min_day_ret": float(np.min(ret))})

    # ── 台股：期貨管線 ─────────────────────────────────────────────────────
    if tx_cs is None:
        import params as root_params
        from data import load_tx_contracts
        from continuous import build_continuous
        tx_cs = build_continuous(load_tx_contracts(), root_params.ROLL_OFFSET_DAYS)
    panel = tx_cs.panel
    dates = panel.index
    ret = panel["ret"].to_numpy(float)
    # 趨勢訊號：比值還原連續價（= cumprod 報酬），與逐合約法恆等、無假跳空。
    spot = np.cumprod(1.0 + ret) * float(panel["price"].iloc[0] / (1.0 + ret[0]))
    mkts["TX_FUT"] = Market(
        key="TX_FUT", label=MARKET_LABEL["TX_FUT"], dates=dates, ret=ret,
        spot=spot, rf_ann=np.full(len(dates), TW_SHORT_RATE), mode="futures",
        price=panel["price"].to_numpy(float),
        is_roll=panel["is_roll"].to_numpy(bool),
        p_near=panel["p_near"].to_numpy(float),
        p_next=panel["p_next"].to_numpy(float),
        meta={"start": str(dates[0].date()), "end": str(dates[-1].date()),
              "n_days": len(dates), "tr_source": "逐合約期貨管線（carry 內生）",
              "rf_source": f"常數 {TW_SHORT_RATE:.1%}（閒置現金計息，前輪口徑）",
              "n_rolls": int(panel["is_roll"].sum()),
              "trend_spot": "比值還原連續期貨價（含 carry 漂移，報告揭露）",
              "max_day_ret": float(np.max(ret)), "min_day_ret": float(np.min(ret))})
    return mkts
