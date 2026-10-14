"""Download candles to data/<timeframe>/<SYMBOL>.csv.

Usage: python scripts/fetch_candles.py --timeframe 1h --days 60 --exchange binance
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.data import fetch_ohlcv, save_csv  # noqa: E402
from arena.dsl import DEFAULT_SYMBOLS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--timeframe", default="1h")
p.add_argument("--days", type=int, default=60)
p.add_argument("--exchange", default="binance", help="any ccxt id: binance, kraken, coinbase, bybit...")
p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
args = p.parse_args()

for sym in args.symbols:
    df = fetch_ohlcv(sym, args.timeframe, args.days, args.exchange)
    out = Path("data") / args.timeframe / f"{sym.replace('/', '_')}.csv"
    save_csv(df, out)
    print(f"{sym}: {len(df)} candles -> {out}")
