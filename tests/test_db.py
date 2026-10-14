import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.db import TABLES, connect  # noqa: E402

T0 = "2026-10-14T08:00:00Z"


def add_agent(conn, name, parent_id=None, lineage_id=None, generation=0):
    cur = conn.execute(
        "INSERT INTO agents (name, parent_id, lineage_id, generation, born_at, starting_cash, cash, equity)"
        " VALUES (?, ?, ?, ?, ?, 1000, 1000, 1000)",
        (name, parent_id, lineage_id, generation, T0))
    return cur.lastrowid


def test_all_tables_created():
    conn = connect(":memory:")
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLES) <= names


def test_connect_twice_keeps_data(tmp_path):
    path = tmp_path / "sub" / "arena.db"
    conn = connect(path)
    add_agent(conn, "a")
    conn.commit()
    conn.close()
    assert connect(path).execute("SELECT COUNT(*) FROM agents").fetchone()[0] == 1


def test_founder_child_and_strategy_flow():
    """Founder points to itself; child shares the lineage; strategy links both ways."""
    conn = connect(":memory:")
    founder = add_agent(conn, "Founder")
    conn.execute("UPDATE agents SET lineage_id = id WHERE id = ?", (founder,))
    child = add_agent(conn, "Child", parent_id=founder, lineage_id=founder, generation=1)

    sid = conn.execute(
        "INSERT INTO strategies (agent_id, version, spec, change_reason, created_at)"
        " VALUES (?, 1, ?, 'birth', ?)", (child, json.dumps({"name": "x"}), T0)).lastrowid
    conn.execute("UPDATE agents SET strategy_id = ? WHERE id = ?", (sid, child))

    row = conn.execute("SELECT * FROM agents WHERE id = ?", (child,)).fetchone()
    assert row["lineage_id"] == founder and row["generation"] == 1 and row["strategy_id"] == sid
    lineage = conn.execute("SELECT lineage_id FROM agents WHERE id = ?", (founder,)).fetchone()[0]
    assert lineage == founder


def test_foreign_keys_enforced():
    conn = connect(":memory:")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO events (agent_id, time, type, summary) VALUES (999, ?, 'birth', 'x')", (T0,))


def test_status_must_be_alive_or_dead():
    conn = connect(":memory:")
    aid = add_agent(conn, "a")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE agents SET status = 'zombie' WHERE id = ?", (aid,))


def test_strategy_versions_unique_per_agent():
    conn = connect(":memory:")
    aid = add_agent(conn, "a")
    sql = "INSERT INTO strategies (agent_id, version, spec, change_reason, created_at) VALUES (?, 1, '{}', 'birth', ?)"
    conn.execute(sql, (aid, T0))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, (aid, T0))
