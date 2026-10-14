import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.brain import birth, cost_usd, parse_json  # noqa: E402
from arena.data import synthetic_ohlcv  # noqa: E402
from arena.dsl import DEFAULT_SYMBOLS  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
GOOD = (EXAMPLES / "breakout_hunter.json").read_text()   # 1h, trades on synthetic data
NEVER_TRADES = json.dumps(json.loads(GOOD) | {
    "entry": {"left": {"ind": "rsi", "period": 14}, "op": "<", "right": 0}})


@pytest.fixture(scope="module")
def data():
    return {"1h": {s: synthetic_ohlcv(1500, "1h", seed=i) for i, s in enumerate(DEFAULT_SYMBOLS)}}


class FakeClient:
    """Stands in for anthropic.Anthropic(): returns canned replies in order."""

    def __init__(self, *replies, stop_reason="end_turn"):
        self.replies = list(replies)
        self.stop_reason = stop_reason
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"])))  # snapshot
        text = self.replies.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=2000, output_tokens=500),
            stop_reason=self.stop_reason, model=kwargs["model"])


def test_valid_on_first_try(data):
    r = birth(data, client=FakeClient(GOOD))
    assert r.ok and r.spec["name"] == "Breakout Hunter"
    assert r.backtest["n_trades"] >= 3 and "trades" not in r.backtest
    assert len(r.calls) == 1 and r.calls[0].ok


def test_bad_json_then_fixed(data):
    client = FakeClient("here you go: {not json", GOOD)
    r = birth(data, client=client)
    assert r.ok and len(r.calls) == 2
    assert not r.calls[0].ok and r.calls[1].ok
    assert "not valid JSON" in client.requests[1][-1]["content"]   # error fed back


def test_invalid_spec_errors_fed_back(data):
    bad = json.dumps(json.loads(GOOD) | {"sizing": {"position_pct": 60, "max_positions": 3}})
    client = FakeClient(bad, GOOD)
    r = birth(data, client=client)
    assert r.ok
    assert "exceeds 100% of equity" in client.requests[1][-1]["content"]


def test_failed_birth_check_then_fixed(data):
    client = FakeClient(NEVER_TRADES, GOOD)
    r = birth(data, client=client)
    assert r.ok and len(r.calls) == 2
    assert "trades in the backtest window" in client.requests[1][-1]["content"]


def test_missing_timeframe_is_rejected(data):
    four_h = json.dumps(json.loads(GOOD) | {"timeframe": "4h"})
    client = FakeClient(four_h, GOOD)
    r = birth(data, client=client)
    assert r.ok and "has no market data" in client.requests[1][-1]["content"]


def test_gives_up_after_max_attempts(data):
    r = birth(data, client=FakeClient(NEVER_TRADES, NEVER_TRADES, NEVER_TRADES, GOOD), max_attempts=3)
    assert not r.ok and r.spec is None and len(r.calls) == 3
    assert any("trades" in p for p in r.problems)


def test_refusal_stops_immediately(data):
    r = birth(data, client=FakeClient("", stop_reason="refusal"))
    assert not r.ok and len(r.calls) == 1


def test_calls_record_cost_and_prompt(data):
    r = birth(data, lessons="Stops tighter than 1% got hit by noise.", client=FakeClient(GOOD))
    c = r.calls[0]
    assert c.purpose == "birth" and c.input_tokens == 2000 and c.output_tokens == 500
    assert c.cost_usd == pytest.approx(cost_usd("claude-sonnet-5-5", 2000, 500)) == pytest.approx(0.009)
    assert "Stops tighter than 1%" in c.prompt and r.cost_usd == pytest.approx(c.cost_usd)


def test_parse_json_strips_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no object here")
