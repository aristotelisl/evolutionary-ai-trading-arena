"""
Candle loading: live data via ccxt (public endpoints, no API key), plus a
synthetic generator for offline tests.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def fetch_ohlcv(symbol: str, timeframe: str = "1h", days: int = 60,
                exchange_id: str = "binance") -> pd.DataFrame:
    """Fetch closed candles for the last `days` days. Paginates politely."""
    import ccxt  # imported lazily so tests don't need it

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})  # ccxt throttles for us
    since = ex.milliseconds() - days * 24 * 60 * 60 * 1000
    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < 2 or batch[-1][0] >= ex.milliseconds() - TF_MINUTES[timeframe] * 60_000:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").set_index(pd.to_datetime(df["ts"].drop_duplicates(), unit="ms", utc=True))
    df = df.drop(columns="ts").astype(float)
    # Drop the still-forming candle: only act on CLOSED bars.
    bar = pd.Timedelta(minutes=TF_MINUTES[timeframe])
    return df[df.index + bar <= pd.Timestamp.now(tz="UTC")]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def synthetic_ohlcv(n: int = 2000, timeframe: str = "1h", seed: int = 0,
                    start_price: float = 100.0) -> pd.DataFrame:
    """Regime-switching random walk with realistic-ish OHLCV. For tests only."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.0015, n // 200 + 1), 200)[:n]       # trending regimes
    vol = np.repeat(rng.uniform(0.004, 0.015, n // 150 + 1), 150)[:n]     # vol regimes
    rets = drift + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.0005, n))
    spread = np.abs(rng.normal(0, 1, n)) * vol * close * 0.6
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(10, 0.5, n) * (1 + 20 * np.abs(rets))
    idx = pd.date_range("2026-01-01", periods=n, freq=f"{TF_MINUTES[timeframe]}min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
