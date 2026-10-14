"""
Indicator library: the fixed "vocabulary" agents build strategies from.

Every indicator declares its parameters (with hard bounds), how many bars it
needs before its values are trustworthy (warmup), and a vectorised function.
Adding an indicator here automatically adds it to the JSON schema and to the
DSL description shown to the LLM.

All functions take a candle DataFrame with columns
open, high, low, close, volume and return a pd.Series aligned to it.
Values at bar t only use data from bars <= t (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- primitives

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rsi(df: pd.DataFrame, period: int) -> pd.Series:
    delta = df["close"].diff()
    gain = _wilder(delta.clip(lower=0), period)
    loss = _wilder(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses in the window -> RSI 100 (but keep warmup NaNs as NaN)
    return out.where(~(loss == 0) | gain.isna(), 100.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _macd_parts(df: pd.DataFrame, fast: int, slow: int, signal: int):
    macd = _ema(df["close"], fast) - _ema(df["close"], slow)
    sig = _ema(macd, signal)
    return macd, sig


# ---------------------------------------------------------------- registry

@dataclass(frozen=True)
class Param:
    lo: float
    hi: float
    kind: type = int  # int or float


@dataclass(frozen=True)
class Indicator:
    name: str
    description: str
    fn: Callable[..., pd.Series]
    params: dict[str, Param] = field(default_factory=dict)
    warmup: Callable[..., int] = lambda **p: 1


INDICATORS: dict[str, Indicator] = {}


def _register(ind: Indicator) -> None:
    INDICATORS[ind.name] = ind


_register(Indicator("close", "Closing price of the bar.",
                    lambda df: df["close"]))
_register(Indicator("open", "Opening price of the bar.",
                    lambda df: df["open"]))
_register(Indicator("sma", "Simple moving average of close.",
                    lambda df, period: _sma(df["close"], period),
                    {"period": Param(2, 200)}, lambda period: period))
_register(Indicator("ema", "Exponential moving average of close.",
                    lambda df, period: _ema(df["close"], period),
                    {"period": Param(2, 200)}, lambda period: 2 * period))
_register(Indicator("rsi", "Relative Strength Index (0-100), Wilder smoothing.",
                    _rsi, {"period": Param(2, 50)}, lambda period: 2 * period + 1))
_register(Indicator("bb_upper", "Upper Bollinger band: SMA + std * stdev.",
                    lambda df, period, std: _sma(df["close"], period)
                    + std * df["close"].rolling(period, min_periods=period).std(),
                    {"period": Param(5, 100), "std": Param(1.0, 3.5, float)},
                    lambda period, std: period))
_register(Indicator("bb_lower", "Lower Bollinger band: SMA - std * stdev.",
                    lambda df, period, std: _sma(df["close"], period)
                    - std * df["close"].rolling(period, min_periods=period).std(),
                    {"period": Param(5, 100), "std": Param(1.0, 3.5, float)},
                    lambda period, std: period))
_register(Indicator("atr_pct", "Average True Range as % of close (volatility).",
                    lambda df, period: 100 * _wilder(_true_range(df), period) / df["close"],
                    {"period": Param(2, 50)}, lambda period: 2 * period + 1))
_register(Indicator("roc", "Rate of change: % change in close over `period` bars.",
                    lambda df, period: 100 * df["close"].pct_change(period),
                    {"period": Param(1, 100)}, lambda period: period + 1))
_register(Indicator("highest", "Highest high of the PREVIOUS `period` bars (excludes current bar, "
                               "so `close > highest` is a true breakout).",
                    lambda df, period: df["high"].shift(1).rolling(period, min_periods=period).max(),
                    {"period": Param(2, 200)}, lambda period: period + 1))
_register(Indicator("lowest", "Lowest low of the PREVIOUS `period` bars (excludes current bar).",
                    lambda df, period: df["low"].shift(1).rolling(period, min_periods=period).min(),
                    {"period": Param(2, 200)}, lambda period: period + 1))
_register(Indicator("volume_ratio", "Current volume divided by its `period`-bar average "
                                    "(>1 means above-average volume).",
                    lambda df, period: df["volume"] / _sma(df["volume"], period).replace(0, np.nan),
                    {"period": Param(2, 100)}, lambda period: period))
_register(Indicator("macd", "MACD line: EMA(fast) - EMA(slow).",
                    lambda df, fast, slow, signal: _macd_parts(df, fast, slow, signal)[0],
                    {"fast": Param(2, 50), "slow": Param(5, 150), "signal": Param(2, 50)},
                    lambda fast, slow, signal: 2 * slow + signal))
_register(Indicator("macd_signal", "MACD signal line: EMA(signal) of the MACD line.",
                    lambda df, fast, slow, signal: _macd_parts(df, fast, slow, signal)[1],
                    {"fast": Param(2, 50), "slow": Param(5, 150), "signal": Param(2, 50)},
                    lambda fast, slow, signal: 2 * slow + signal))


def indicator_key(name: str, params: dict) -> str:
    """Canonical cache key, e.g. 'bb_lower(period=20,std=2.0)'."""
    inner = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{name}({inner})"


def compute(df: pd.DataFrame, name: str, params: dict) -> pd.Series:
    return INDICATORS[name].fn(df, **params)


def warmup_bars(name: str, params: dict) -> int:
    return int(INDICATORS[name].warmup(**params))
