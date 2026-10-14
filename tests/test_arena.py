import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena import Engine, backtest, build_schema, compile_strategy, describe_for_llm, validate  # noqa: E402
from arena.data import synthetic_ohlcv  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def base_strategy(**overrides):
    s = {
        "name": "t", "thesis": "t", "universe": ["BTC/USDT"], "timeframe": "1h",
        "entry": {"left": {"ind": "rsi", "period": 14}, "op": "<", "right": 30},
        "exit": {"stop_loss_pct": 2},
        "sizing": {"position_pct": 50, "max_positions": 1},
    }
    s.update(overrides)
    return s


def flat_candles(prices_open, prices_close=None, highs=None, lows=None):
    o = np.asarray(prices_open, float)
    c = o if prices_close is None else np.asarray(prices_close, float)
    h = np.maximum(o, c) if highs is None else np.asarray(highs, float)
    lo = np.minimum(o, c) if lows is None else np.asarray(lows, float)
    idx = pd.date_range("2026-01-01", periods=len(o), freq="h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, index=idx)


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("f", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.stem)
def test_examples_are_valid(f):
    assert validate(json.loads(f.read_text())) == []


def test_schema_is_json_serialisable():
    json.dumps(build_schema())


@pytest.mark.parametrize("mutate, fragment", [
    (lambda s: s["entry"].update(left={"ind": "rsi", "period": 500}), "entry"),
    (lambda s: s["entry"].update(left={"ind": "telepathy"}), "entry"),
    (lambda s: s.update(universe=["DOGE/USDT"]), "universe"),
    (lambda s: s.update(exit={}), "exit"),
    (lambda s: s.update(sizing={"position_pct": 60, "max_positions": 2}), "exceeds"),
    (lambda s: s.update(entry={"left": 1, "op": "<", "right": 2}), "two constants"),
    (lambda s: s.update(entry={"left": {"ind": "macd", "fast": 30, "slow": 10, "signal": 9},
                               "op": ">", "right": 0}), "fast period"),
    (lambda s: s.update(entry={"left": {"ind": "ema", "period": 200}, "op": ">",
                               "right": {"ind": "macd", "fast": 50, "slow": 150, "signal": 50}}), "bars of history"),
    (lambda s: s.update(hack="import os"), "(root)"),
])
def test_validation_rejects(mutate, fragment):
    s = base_strategy()
    mutate(s)
    errors = validate(s)
    assert errors, "expected validation errors"
    assert any(fragment in e for e in errors), errors


def test_complexity_limits():
    leaf = {"left": {"ind": "close"}, "op": ">", "right": 1}
    s = base_strategy(entry={"all": [{"any": [leaf] * 6}] * 4})   # 1 + 4 + 24 nodes
    assert any("too complex" in e for e in validate(s))
    deep = leaf
    for _ in range(5):
        deep = {"not": deep}
    assert any("nested" in e for e in validate(base_strategy(entry=deep)))


def test_llm_description_mentions_every_indicator():
    from arena.indicators import INDICATORS
    text = describe_for_llm()
    assert all(name in text for name in INDICATORS)


# ------------------------------------------------------------------ signals

def test_not_is_false_during_warmup():
    s = compile_strategy(base_strategy(entry={"not": {"left": {"ind": "sma", "period": 50},
                                                      "op": ">", "right": 1e9}}))
    sig = s.signals(synthetic_ohlcv(200))["entry"]
    assert not sig[: s.warmup].any()
    assert sig[s.warmup:].all()


def test_crosses_above_fires_once():
    df = flat_candles([1, 1, 1, 3, 3, 3, 1, 3])
    s = compile_strategy(base_strategy(entry={"left": {"ind": "close"}, "op": "crosses_above", "right": 2}))
    s.warmup = 0
    assert s.signals(df)["entry"].tolist() == [False, False, False, True, False, False, False, True]


# ------------------------------------------------------------------ execution

def always_enter(**exit_rules):
    s = compile_strategy(base_strategy(
        entry={"left": {"ind": "close"}, "op": ">", "right": 0},
        exit=exit_rules or {"max_hold_bars": 500},
        sizing={"position_pct": 100, "max_positions": 1}))
    s.warmup = 0
    return s


def test_no_lookahead_fills_at_next_open():
    df = flat_candles([10, 20, 30, 40])
    eng = Engine(always_enter(), {"BTC/USDT": df}, cash=1000, fee_rate=0, slippage=0)
    eng.on_bar(0)
    assert not eng.positions           # signal at bar 0 close...
    eng.on_bar(1)
    assert eng.positions["BTC/USDT"].entry_price == 20   # ...fills at bar 1 open


def test_gap_through_stop_fills_at_open():
    # enter at 100 (bar 1 open), bar 2 gaps down to 80 with a 5% stop
    df = flat_candles([100, 100, 80, 80], highs=[100, 100, 80, 80], lows=[100, 100, 79, 79])
    eng = Engine(always_enter(stop_loss_pct=5), {"BTC/USDT": df}, cash=1000, fee_rate=0, slippage=0)
    for t in range(3):
        eng.on_bar(t)
    tr = eng.trades[0]
    assert tr.reason == "stop_loss" and tr.exit_price == 80   # not the 95 stop level


def test_stop_beats_take_profit_in_same_bar():
    df = flat_candles([100, 100, 100], highs=[100, 100, 120], lows=[100, 100, 90])
    eng = Engine(always_enter(stop_loss_pct=5, take_profit_pct=10), {"BTC/USDT": df},
                 cash=1000, fee_rate=0, slippage=0)
    for t in range(3):
        eng.on_bar(t)
    assert eng.trades[0].reason == "stop_loss"


def test_trailing_stop_follows_peak():
    df = flat_candles([100, 100, 110, 120, 120], highs=[100, 100, 110, 120, 120],
                      lows=[100, 100, 110, 120, 113])
    eng = Engine(always_enter(trailing_stop_pct=5), {"BTC/USDT": df}, cash=1000, fee_rate=0, slippage=0)
    for t in range(5):
        eng.on_bar(t)
    tr = eng.trades[0]
    assert tr.reason == "trailing_stop" and tr.exit_price == pytest.approx(114)  # 120 * 0.95


def test_fees_and_slippage_cost_money_on_flat_market():
    df = flat_candles([100] * 50)
    s = always_enter(max_hold_bars=1)
    r = backtest(s, {"BTC/USDT": df}, cash=1000, fee_rate=0.001, slippage=0.0005)
    assert r["n_trades"] > 10
    assert r["final_equity"] < 1000
    assert r["fees_paid"] > 0


def test_cash_conservation_without_costs():
    df = synthetic_ohlcv(600, seed=3)
    r = backtest(always_enter(stop_loss_pct=2, take_profit_pct=2), {"BTC/USDT": df},
                 cash=1000, fee_rate=0, slippage=0)
    realised = sum(t["pnl"] for t in r["trades"])
    if r["open_positions"] == 0:
        assert r["final_equity"] == pytest.approx(1000 + realised)


def test_live_refresh_matches_backtest():
    """Feeding candles one at a time (live mode) must equal a single backtest."""
    df = synthetic_ohlcv(500, seed=7)
    s = compile_strategy(json.loads((EXAMPLES / "dip_buyer.json").read_text()) | {"universe": ["BTC/USDT"]})
    full = backtest(s, {"BTC/USDT": df}, cash=1000)

    eng = Engine(s, {"BTC/USDT": df.iloc[:1]}, cash=1000)
    eng.on_bar(0)
    for t in range(1, len(df)):
        eng.refresh({"BTC/USDT": df.iloc[: t + 1]})
        eng.on_bar(t)
    assert eng.equity_curve[-1][1] == pytest.approx(full["final_equity"])
    assert len(eng.trades) == full["n_trades"]
