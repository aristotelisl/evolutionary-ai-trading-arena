import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.brain import LLMCall  # noqa: E402
from arena.db import connect  # noqa: E402
from arena.tax import charge_llm_call, charge_llm_calls  # noqa: E402

T0 = "2026-10-14T08:00:00Z"


def make_call(cost=0.01, ok=True):
    return LLMCall("birth", "claude-sonnet-5-5", 2000, 500, cost, '{"system": "s"}', "{}", ok)


@pytest.fixture
def conn():
    c = connect(":memory:")
    c.execute("INSERT INTO agents (name, born_at, starting_cash, cash, equity) VALUES ('a', ?, 1000, 1000, 1000)",
              (T0,))
    return c


def wallet(conn):
    return tuple(conn.execute("SELECT cash, equity FROM agents WHERE id = 1").fetchone())


def test_call_logged_and_wallet_charged(conn):
    tax = charge_llm_call(conn, 1, make_call(0.01), T0)
    assert tax == pytest.approx(10.0)                       # $0.01 x 1000
    assert wallet(conn) == pytest.approx((990.0, 990.0))
    row = conn.execute("SELECT * FROM llm_calls").fetchone()
    assert row["agent_id"] == 1 and row["cost_usd"] == pytest.approx(0.01)
    assert row["tax_charged"] == pytest.approx(10.0) and row["ok"] == 1


def test_all_calls_charged_including_failed(conn):
    tax = charge_llm_calls(conn, 1, [make_call(0.01, ok=False), make_call(0.02)], T0)
    assert tax == pytest.approx(30.0)
    assert wallet(conn) == pytest.approx((970.0, 970.0))
    assert [r["ok"] for r in conn.execute("SELECT ok FROM llm_calls ORDER BY id")] == [0, 1]


def test_custom_multiplier(conn):
    charge_llm_call(conn, 1, make_call(0.01), T0, multiplier=100)
    assert wallet(conn) == pytest.approx((999.0, 999.0))


def test_unknown_agent_charges_nothing(conn):
    with pytest.raises(ValueError):
        charge_llm_call(conn, 999, make_call(), T0)
    assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 0


def test_cash_can_go_negative(conn):
    conn.execute("UPDATE agents SET cash = 5, equity = 1000 WHERE id = 1")
    charge_llm_call(conn, 1, make_call(0.01), T0)
    assert wallet(conn) == pytest.approx((-5.0, 990.0))
