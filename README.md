# Arena core: strategy DSL + simulated exchange

The "body" of each agent: a validated JSON strategy (the genome) and an engine that
runs it identically in backtests and live paper trading.

```
arena/indicators.py   indicator vocabulary (params, bounds, warmup)
arena/dsl.py          JSON schema, validation, compilation, LLM reference text
arena/engine.py       simulated exchange, backtest(), birth_check()
arena/data.py         ccxt candle fetcher + synthetic data for tests
examples/*.json       three sample genomes
scripts/              fetch_candles.py, run_backtest.py
strategy.schema.json  exported JSON Schema (for structured output / docs)
dsl_reference.txt     the text you paste into birth/reflection prompts
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                       # 24 tests, offline
python scripts/run_backtest.py            # synthetic data if no CSVs
python scripts/fetch_candles.py --timeframe 1h --days 60
python scripts/fetch_candles.py --timeframe 4h --days 120
python scripts/run_backtest.py            # now on real candles
```

## How it plugs into the agent loop

Birth:
1. Prompt the LLM with `describe_for_llm()` + parent's lessons; ask for JSON only.
2. `errors = validate(spec)`; if errors, send them back verbatim and retry (charge the agent).
3. `result = backtest(compile_strategy(spec), last_60_days)`; reject if `birth_check(result)` fails.

Live (every candle close):
```python
engine.refresh(latest_candles)        # dict symbol -> DataFrame of closed bars
engine.on_bar(len(engine.index) - 1)
```
Keep at least ~3x the strategy's warmup bars of history in the window so EMA/RSI
values match the backtest. Persist `cash`, `positions` and `trades` to the DB after each bar.

Reflection: send the strategy, its recent trades and `exit_reasons` (great signal:
"17 of 25 exits were stop-losses") and ask for a small diff, then validate again.

## Execution model
Signals at bar close, fills at next open; intrabar stops on high/low with gap fills
at the open; stop assumed before take-profit when both touch; 0.1% fee + 0.05%
slippage per side by default. Long-only spot.
