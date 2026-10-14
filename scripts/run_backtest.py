"""Validate and backtest every strategy in examples/.

Uses data/<timeframe>/*.csv if present (run fetch_candles.py first),
otherwise falls back to synthetic data so you can try it offline.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena import backtest, birth_check, compile_strategy  # noqa: E402
from arena.data import load_csv, synthetic_ohlcv  # noqa: E402

root = Path(__file__).resolve().parents[1]
for f in sorted((root / "examples").glob("*.json")):
    spec = json.loads(f.read_text())
    strat = compile_strategy(spec)
    data, source = {}, "real"
    for i, sym in enumerate(strat.universe):
        path = root / "data" / spec["timeframe"] / f"{sym.replace('/', '_')}.csv"
        if path.exists():
            data[sym] = load_csv(path)
        else:
            data[sym], source = synthetic_ohlcv(1500, spec["timeframe"], seed=i), "synthetic"
    r = backtest(strat, data, cash=1000)
    gate = birth_check(r)
    print(f"\n== {spec['name']} ({spec['timeframe']}, {source} data, warmup {strat.warmup} bars)")
    print(f"   return {r['total_return_pct']:+.2f}%  maxDD {r['max_drawdown_pct']:.2f}%  "
          f"sharpe {r['sharpe']:.2f}  trades {r['n_trades']}  win {r['win_rate_pct']:.0f}%  "
          f"fees ${r['fees_paid']:.2f}")
    print(f"   exits: {r['exit_reasons']}")
    print(f"   birth check: {'PASS' if not gate else 'FAIL - ' + '; '.join(gate)}")
