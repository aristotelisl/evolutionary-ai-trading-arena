"""Run one real birth with the Anthropic API and print the result (costs ~$0.01-0.06).

Needs ANTHROPIC_API_KEY in .env and candles in data/ (run fetch_candles.py first).
Usage: python scripts/try_birth.py ["optional parent lessons"]
"""
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from arena.brain import birth  # noqa: E402
from arena.data import load_csv  # noqa: E402

for line in (root / ".env").read_text().splitlines() if (root / ".env").exists() else []:
    key, sep, value = line.partition("=")
    if sep and not line.lstrip().startswith("#"):
        os.environ.setdefault(key.strip(), value.strip())

data = {}
for tf_dir in sorted((root / "data").glob("*")):
    if tf_dir.is_dir() and any(tf_dir.glob("*.csv")):
        data[tf_dir.name] = {f.stem.replace("_", "/"): load_csv(f) for f in tf_dir.glob("*.csv")}
if not data:
    sys.exit("no candles in data/ - run scripts/fetch_candles.py first")

r = birth(data, lessons=sys.argv[1] if len(sys.argv) > 1 else "")
for i, c in enumerate(r.calls, 1):
    print(f"call {i}: {c.model}  {c.input_tokens} in / {c.output_tokens} out  ${c.cost_usd:.4f}  "
          f"{'ok' if c.ok else 'rejected'}")
print(f"total cost: ${r.cost_usd:.4f}")
if r.ok:
    print(json.dumps(r.spec, indent=2))
    b = r.backtest
    print(f"backtest: return {b['total_return_pct']:+.2f}%  maxDD {b['max_drawdown_pct']:.2f}%  "
          f"sharpe {b['sharpe']:.2f}  trades {b['n_trades']}  exits {b['exit_reasons']}")
else:
    print("birth FAILED:", *r.problems, sep="\n- ")
