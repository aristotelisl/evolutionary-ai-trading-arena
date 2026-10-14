"""
Simulated exchange + strategy runner.

The same Engine is used for birth-time backtests and for live paper trading,
so a strategy behaves identically in both. Live mode: append the newest closed
candle, call engine.refresh(data), then engine.on_bar(len(index) - 1).

Execution model (deliberately pessimistic):
  * Signals are computed at bar t close, orders fill at bar t+1 OPEN.
  * Stops / take-profits are checked intrabar against high/low.
    If the bar opens through the stop (gap), fill at the open, not the stop.
    If stop and take-profit are both touched in one bar, assume the stop hit first.
  * Every fill pays slippage and a percentage fee.
  * Long-only, spot, one position per symbol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .dsl import CompiledStrategy

BARS_PER_YEAR = {"15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    cost: float          # cash spent including fee
    peak: float          # highest high since entry (for trailing stops)
    bars_held: int = 0


@dataclass
class Trade:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    bars_held: int
    reason: str          # stop_loss | trailing_stop | take_profit | signal | max_hold


class Engine:
    def __init__(self, strategy: CompiledStrategy, data: dict[str, pd.DataFrame],
                 cash: float, fee_rate: float = 0.001, slippage: float = 0.0005):
        self.s = strategy
        self.cash = float(cash)
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.positions: dict[str, Position] = {}
        self.pending_entries: list[str] = []
        self.pending_exits: dict[str, str] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.fees_paid = 0.0
        self.last_equity = float(cash)
        self.refresh(data)

    # ------------------------------------------------------------ data

    def refresh(self, data: dict[str, pd.DataFrame]) -> None:
        """Load/replace candle data and recompute signals. Keeps positions and cash."""
        missing = [s for s in self.s.universe if s not in data]
        if missing:
            raise KeyError(f"no candle data for {missing}")
        idx = None
        for sym in self.s.universe:
            idx = data[sym].index if idx is None else idx.intersection(data[sym].index)
        self.index = idx.sort_values()
        self.o, self.h, self.l, self.c, self.sig = {}, {}, {}, {}, {}
        for sym in self.s.universe:
            df = data[sym].loc[self.index]
            self.o[sym] = df["open"].to_numpy(float)
            self.h[sym] = df["high"].to_numpy(float)
            self.l[sym] = df["low"].to_numpy(float)
            self.c[sym] = df["close"].to_numpy(float)
            self.sig[sym] = self.s.signals(df)

    # ------------------------------------------------------------ fills

    def _buy(self, sym: str, ref_price: float, t: int) -> None:
        target = self.last_equity * self.s.sizing["position_pct"] / 100
        spend = min(target, self.cash)
        if spend <= 1e-9:
            return
        fill = ref_price * (1 + self.slippage)
        fee = spend * self.fee_rate
        qty = (spend - fee) / fill
        self.cash -= spend
        self.fees_paid += fee
        self.positions[sym] = Position(sym, qty, fill, self.index[t], spend, fill)

    def _sell(self, sym: str, ref_price: float, t: int, reason: str) -> None:
        pos = self.positions.pop(sym)
        fill = ref_price * (1 - self.slippage)
        gross = pos.qty * fill
        fee = gross * self.fee_rate
        net = gross - fee
        self.cash += net
        self.fees_paid += fee
        pnl = net - pos.cost
        self.trades.append(Trade(
            symbol=sym, entry_time=str(pos.entry_time), exit_time=str(self.index[t]),
            entry_price=pos.entry_price, exit_price=fill, qty=pos.qty,
            pnl=pnl, pnl_pct=100 * pnl / pos.cost, bars_held=pos.bars_held, reason=reason,
        ))

    # ------------------------------------------------------------ main step

    def on_bar(self, t: int) -> None:
        ex = self.s.exit

        # 1. Fill orders decided at the previous close, at this bar's open.
        for sym, reason in list(self.pending_exits.items()):
            if sym in self.positions:
                self._sell(sym, self.o[sym][t], t, reason)
        self.pending_exits.clear()
        for sym in self.pending_entries:
            if sym not in self.positions and len(self.positions) < self.s.sizing["max_positions"]:
                self._buy(sym, self.o[sym][t], t)
        self.pending_entries.clear()

        # 2. Intrabar protective exits.
        for sym, pos in list(self.positions.items()):
            o, h, l = self.o[sym][t], self.h[sym][t], self.l[sym][t]
            stop, stop_reason = None, None
            if "stop_loss_pct" in ex:
                stop, stop_reason = pos.entry_price * (1 - ex["stop_loss_pct"] / 100), "stop_loss"
            if "trailing_stop_pct" in ex:
                trail = pos.peak * (1 - ex["trailing_stop_pct"] / 100)
                if stop is None or trail > stop:
                    stop, stop_reason = trail, "trailing_stop"
            tp = pos.entry_price * (1 + ex["take_profit_pct"] / 100) if "take_profit_pct" in ex else None

            if stop is not None and l <= stop:
                self._sell(sym, min(o, stop), t, stop_reason)       # gap-down fills at open
            elif tp is not None and h >= tp:
                self._sell(sym, max(o, tp), t, "take_profit")       # gap-up fills at open
            else:
                pos.peak = max(pos.peak, h)

        # 3. Decisions at this bar's close (executed next bar).
        for sym, pos in self.positions.items():
            pos.bars_held += 1
            if "max_hold_bars" in ex and pos.bars_held >= ex["max_hold_bars"]:
                self.pending_exits[sym] = "max_hold"
            elif self.sig[sym]["exit"][t]:
                self.pending_exits[sym] = "signal"
        if t >= self.s.warmup:
            for sym in self.s.universe:
                if sym not in self.positions and self.sig[sym]["entry"][t]:
                    self.pending_entries.append(sym)

        # 4. Mark to market.
        equity = self.cash + sum(p.qty * self.c[s][t] for s, p in self.positions.items())
        self.last_equity = equity
        self.equity_curve.append((self.index[t], equity))


# ---------------------------------------------------------------- backtest

def backtest(strategy: CompiledStrategy, data: dict[str, pd.DataFrame], cash: float = 1000.0,
             fee_rate: float = 0.001, slippage: float = 0.0005) -> dict:
    eng = Engine(strategy, data, cash, fee_rate, slippage)
    for t in range(len(eng.index)):
        eng.on_bar(t)
    return summarize(eng, cash)


def summarize(eng: Engine, start_cash: float) -> dict:
    eq = pd.Series([e for _, e in eng.equity_curve], index=[ts for ts, _ in eng.equity_curve])
    rets = eq.pct_change().dropna()
    bpy = BARS_PER_YEAR.get(eng.s.spec["timeframe"], 8760)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bpy)) if len(rets) > 1 and rets.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    pnls = [tr.pnl for tr in eng.trades]
    reasons: dict[str, int] = {}
    for tr in eng.trades:
        reasons[tr.reason] = reasons.get(tr.reason, 0) + 1
    return {
        "final_equity": float(eq.iloc[-1]) if len(eq) else start_cash,
        "total_return_pct": float(100 * (eq.iloc[-1] / start_cash - 1)) if len(eq) else 0.0,
        "max_drawdown_pct": float(100 * dd),
        "sharpe": sharpe,
        "n_trades": len(eng.trades),
        "win_rate_pct": float(100 * np.mean([p > 0 for p in pnls])) if pnls else 0.0,
        "fees_paid": eng.fees_paid,
        "exit_reasons": reasons,
        "open_positions": len(eng.positions),
        "trades": [asdict(tr) for tr in eng.trades],
        "equity_curve": eq,
    }


def birth_check(result: dict, min_trades: int = 3, max_drawdown_pct: float = -60.0) -> list[str]:
    """Cheap sanity gate before a newborn strategy goes live.

    This rejects BROKEN genomes (never trades, instantly blows up), not bad ones;
    weak-but-valid strategies should be allowed to live and die on their own.
    """
    problems = []
    if result["n_trades"] < min_trades:
        problems.append(f"only {result['n_trades']} trades in the backtest window "
                        f"(need >= {min_trades}); entry conditions may be impossible to satisfy")
    if result["max_drawdown_pct"] < max_drawdown_pct:
        problems.append(f"backtest drawdown {result['max_drawdown_pct']:.1f}% is catastrophic")
    return problems
