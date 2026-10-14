# Roadmap - one small step at a time

Tick a box only when the step works and its tests pass.

## Milestone 0: Foundations
- [x] Strategy DSL, validator and indicator library
- [x] Simulated exchange + backtester (24 tests passing)
- [x] Set up locally: install requirements, run the tests, run `scripts/run_backtest.py`
- [x] Put the project on GitHub (public repo, `.gitignore` for `data/`, `.env`, `__pycache__`)
- [x] Fetch real candles with `scripts/fetch_candles.py` and backtest the examples on real data

## Milestone 1: One closed loop of life and death
- [x] SQLite schema: agents, strategies, trades, events, llm_calls, equity_snapshots
- [x] Brain service: birth prompt -> validate -> retry with errors -> birth_check backtest
- [x] Compute tax: log every LLM call's tokens/cost and charge the agent's wallet
- [ ] Live loop: every candle close, refresh data and step each agent's Engine; persist state
- [ ] Daily reflection: small, bounded strategy tweaks (or "no change")
- [ ] Death and rebirth: below threshold -> eulogy/lessons -> child in same lineage
- [ ] Minimal public page: hourly JSON export + one static page (balances, strategies, event log)
- [ ] Run 5 agents unattended for a week on an EU VPS

## Milestone 2 (later)
- [ ] Reproduction from successful agents, multiple lineages, family tree, risk-adjusted fitness
- [ ] New founders ("immigrants"): occasionally born from scratch with no parent, e.g. every Nth birth, when one lineage dominates, or when no agent is fit to reproduce

## Milestone 3 (later)
- [ ] Model choice as an evolved trait, sandboxed code strategies, "seasons"
