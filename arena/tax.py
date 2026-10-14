"""
Compute tax: every LLM call is logged and its real cost x multiplier is taken
from the agent's wallet, so thinking competes with trading for survival.

The charge comes out of cash (which may go negative, like a debt repaid when a
position closes) and equity. Logging and charging happen in one transaction.
Failed calls are charged too.
"""
from __future__ import annotations

import sqlite3

from .brain import LLMCall

COMPUTE_TAX_MULTIPLIER = 1000


def charge_llm_calls(conn: sqlite3.Connection, agent_id: int, calls: list[LLMCall], now: str,
                     multiplier: float = COMPUTE_TAX_MULTIPLIER) -> float:
    """Log each call to llm_calls and charge the agent. Returns the total tax charged."""
    total = 0.0
    with conn:
        if conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone() is None:
            raise ValueError(f"no agent with id {agent_id}")
        for c in calls:
            tax = c.cost_usd * multiplier
            conn.execute(
                "INSERT INTO llm_calls (agent_id, time, purpose, model, input_tokens, output_tokens,"
                " cost_usd, tax_charged, prompt, response, ok) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (agent_id, now, c.purpose, c.model, c.input_tokens, c.output_tokens,
                 c.cost_usd, tax, c.prompt, c.response, int(c.ok)))
            total += tax
        conn.execute("UPDATE agents SET cash = cash - ?, equity = equity - ? WHERE id = ?",
                     (total, total, agent_id))
    return total


def charge_llm_call(conn: sqlite3.Connection, agent_id: int, call: LLMCall, now: str,
                    multiplier: float = COMPUTE_TAX_MULTIPLIER) -> float:
    return charge_llm_calls(conn, agent_id, [call], now, multiplier)
