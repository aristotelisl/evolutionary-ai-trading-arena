"""
SQLite storage for the arena.

connect(path) opens the database and creates any missing tables. Use ":memory:"
for tests. Times are UTC ISO-8601 text; money is simulated dollars (REAL).

Founders have parent_id NULL and generation 0. lineage_id is the founder's id
(a founder points to itself), set right after the founder's row is inserted.
agents.strategy_id and strategies.agent_id point at each other, so insert the
agent first, then its strategy, then set agents.strategy_id.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    lineage_id      INTEGER REFERENCES agents(id),
    parent_id       INTEGER REFERENCES agents(id),
    generation      INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'alive' CHECK (status IN ('alive', 'dead')),
    born_at         TEXT NOT NULL,
    died_at         TEXT,
    cause_of_death  TEXT,
    starting_cash   REAL NOT NULL,
    cash            REAL NOT NULL,
    equity          REAL NOT NULL,
    strategy_id     INTEGER REFERENCES strategies(id),
    lessons         TEXT NOT NULL DEFAULT '',
    engine_state    TEXT NOT NULL DEFAULT '{}'   -- JSON: positions, pending orders, last bar
);

CREATE TABLE IF NOT EXISTS strategies (
    id              INTEGER PRIMARY KEY,
    agent_id        INTEGER NOT NULL REFERENCES agents(id),
    version         INTEGER NOT NULL,
    previous_id     INTEGER REFERENCES strategies(id),
    spec            TEXT NOT NULL,               -- JSON genome
    change_reason   TEXT NOT NULL,
    backtest        TEXT,                        -- JSON summary from birth check
    created_at      TEXT NOT NULL,
    UNIQUE (agent_id, version)
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY,
    agent_id        INTEGER NOT NULL REFERENCES agents(id),
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    symbol          TEXT NOT NULL,
    entry_time      TEXT NOT NULL,
    exit_time       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    qty             REAL NOT NULL,
    pnl             REAL NOT NULL,
    pnl_pct         REAL NOT NULL,
    bars_held       INTEGER NOT NULL,
    reason          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY,
    agent_id        INTEGER REFERENCES agents(id),   -- NULL for arena-wide events
    time            TEXT NOT NULL,
    type            TEXT NOT NULL,
    summary         TEXT NOT NULL,
    details         TEXT NOT NULL DEFAULT '{}'       -- JSON, stored verbatim
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY,
    agent_id        INTEGER REFERENCES agents(id),
    time            TEXT NOT NULL,
    purpose         TEXT NOT NULL,
    model           TEXT NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    tax_charged     REAL NOT NULL,
    prompt          TEXT NOT NULL,
    response        TEXT NOT NULL,
    ok              INTEGER NOT NULL CHECK (ok IN (0, 1))
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    agent_id        INTEGER NOT NULL REFERENCES agents(id),
    time            TEXT NOT NULL,
    cash            REAL NOT NULL,
    equity          REAL NOT NULL,
    PRIMARY KEY (agent_id, time)
);

CREATE INDEX IF NOT EXISTS idx_strategies_agent ON strategies(agent_id);
CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(time);
CREATE INDEX IF NOT EXISTS idx_llm_calls_agent ON llm_calls(agent_id);
"""

TABLES = ["agents", "strategies", "trades", "events", "llm_calls", "equity_snapshots"]


def connect(path: str | Path = "data/arena.db") -> sqlite3.Connection:
    """Open (and if needed create) the arena database."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
