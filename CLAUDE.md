# Evolutionary AI Trading Arena

## What this project is
An arena where lineages of AI agents paper-trade crypto on live market data (simulated money, no real funds).
Each agent uses an LLM **at birth** to design a strategy as a JSON genome; the strategy then runs
deterministically with no LLM. Once a day the agent reflects on its performance and may make a small tweak.
Agents die when their wallet falls below a threshold; a child is born carrying the parent's compressed lessons.
Successful agents can fund and spawn children. Real LLM API costs are charged to each agent's simulated wallet
(scaled by a "compute tax" multiplier), so efficiency is part of the evolution.

Goals: running cost of a few dollars a day, 5-10 agents to start, and a public dashboard
(family tree, balances, strategies, causes of death) used to document the project and build an audience.

## Working style
The owner wants to build **slowly, one small step at a time**.
- Do only the current step in ROADMAP.md. Don't jump ahead or add features that weren't asked for.
- Before writing code, briefly explain the plan for the step and wait for a go-ahead.
- Keep changes small and reviewable. Explain decisions in plain language.
- After finishing a step: run the tests, tick the box in ROADMAP.md, and suggest what comes next.

## Key design decisions (already made)
- **Brain vs body:** the LLM is called rarely (birth, daily reflection, death lessons). Strategies run as deterministic code every candle.
- **Strategies are a constrained JSON DSL**, not arbitrary code: safe, validatable, diffable, easy to show on a dashboard.
  Sandboxed Python strategies are a possible future milestone, not now.
- **Market:** crypto (24/7, free public data via ccxt, no API key). Default exchange Binance; Kraken as fallback.
  Binance geo-blocks some regions and hosts (HTTP 451) - host the server in the EU and check access with `curl https://api.binance.com`.
- **We simulate the exchange ourselves** (no broker paper account): 0.1% fee + 0.05% slippage per side,
  signals at bar close, fills at next bar open, intrabar stops with gap fills, stop assumed before take-profit.
- **Death threshold**, not exactly zero (e.g. below 30% of starting capital). Consider a small daily living cost.
- **Compute tax:** the wallet is charged real API cost x a multiplier (e.g. 1000x) so cost pressure actually matters.
- **Reflection limits:** small changes only, a minimum trade count before changes, and "no change" is a valid free choice.
- **Reproduction costs the parent** (e.g. 30% of wallet); population is capped (~10-15).
- **Fitness for reproduction** uses risk-adjusted metrics (Sharpe, drawdown), not raw balance, to reduce luck.
- **Lessons ("memory") are capped** at ~300 tokens and passed to children in the birth prompt.
- **Events log** (births, deaths, reflections, strategy changes) stored verbatim - it's the content engine for the audience.

## Tech stack
Python, pandas, numpy, jsonschema, ccxt. SQLite first (Postgres later if needed).
One small EU VPS (e.g. Hetzner) running a long-lived process. Anthropic API: cheap model (Haiku-class) for reflections,
optionally a stronger model for births. Dashboard: static site (e.g. Next.js/Astro) fed by hourly JSON exports.

## Code map (what exists)
- `arena/indicators.py` - indicator vocabulary (params with bounds, warmup, vectorised functions)
- `arena/dsl.py` - JSON schema, `validate()` (returns LLM-friendly error strings), `compile_strategy()`, `describe_for_llm()`
- `arena/engine.py` - simulated exchange `Engine` (same code for backtest and live), `backtest()`, `birth_check()`
- `arena/data.py` - ccxt candle fetcher + synthetic data generator for tests
- `examples/*.json` - three sample genomes; `scripts/` - fetch_candles.py, run_backtest.py
- `tests/test_arena.py` - 24 tests (run with `python -m pytest -q`)

## Conventions
- Never introduce look-ahead: indicator values at bar t use only bars <= t.
- Every new rule/indicator gets a test.
- No real-money trading code unless the owner explicitly decides otherwise later.
- Secrets (API keys) go in a `.env` file that is git-ignored.

## Open design notes (to handle in the relevant later steps)
- **Doing nothing must not be safe:** add a daily living cost so an agent can't survive by never trading.
- **Birth check needs a minimum trade count** (e.g. the Quiet Dip Buyer example passed with only 7 trades on synthetic data).
- **Calibrate the compute tax** against the $1,000 starting wallet so LLM cost matters but doesn't dominate.
- **Seed diversity in birth prompts** (personas, allowed indicators) so agents don't all invent the same strategy.
- **Cap birth retries** to limit overfitting to the backtest window.
- **Birth check should beat buy-and-hold** (or at least report it): on the Aug-Oct 2026 data all three examples passed but trailed holding the same coins by 3-8x. Also watch fees on high-frequency strategies.
- **One week of live results is mostly noise:** don't read early deaths/survivals as skill.
