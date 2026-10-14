"""
Strategy DSL: JSON schema, semantic validation, compilation and LLM docs.

A strategy ("genome") looks like:

{
  "name": "Quiet Dip Buyer",
  "thesis": "Buy oversold dips in an uptrend, exit fast.",
  "universe": ["BTC/USDT", "ETH/USDT"],
  "timeframe": "1h",
  "entry": {"all": [
      {"left": {"ind": "rsi", "period": 14}, "op": "<", "right": 30},
      {"left": {"ind": "close"}, "op": ">", "right": {"ind": "sma", "period": 200}}
  ]},
  "exit": {"take_profit_pct": 4, "stop_loss_pct": 2, "max_hold_bars": 48},
  "sizing": {"position_pct": 25, "max_positions": 2}
}

Conditions are trees of all / any / not nodes with comparison leaves.
An operand is either a number or an indicator reference, optionally scaled
with "mult" (e.g. {"ind": "sma", "period": 50, "mult": 1.02} = 2% above SMA50).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import jsonschema
import numpy as np
import pandas as pd

from .indicators import INDICATORS, compute, indicator_key, warmup_bars

# ---------------------------------------------------------------- config

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
OPS = ["<", ">", "<=", ">=", "crosses_above", "crosses_below"]

MAX_NODES = 20        # total condition nodes in entry + exit signal
MAX_DEPTH = 4         # nesting depth of all/any/not
MAX_WARMUP = 400      # bars of history the strategy may need
MAX_EXPOSURE = 100    # position_pct * max_positions must not exceed this


# ---------------------------------------------------------------- schema

def _operand_schema() -> dict:
    variants: list[dict] = [{"type": "number"}]
    for name, ind in INDICATORS.items():
        props: dict[str, Any] = {
            "ind": {"const": name},
            "mult": {"type": "number", "minimum": 0.5, "maximum": 2.0},
        }
        required = ["ind"]
        for pname, p in ind.params.items():
            props[pname] = {
                "type": "integer" if p.kind is int else "number",
                "minimum": p.lo,
                "maximum": p.hi,
            }
            required.append(pname)
        variants.append({"type": "object", "properties": props,
                         "required": required, "additionalProperties": False})
    return {"oneOf": variants}


def build_schema(symbols: list[str] | None = None) -> dict:
    symbols = symbols or DEFAULT_SYMBOLS
    cond_ref = {"$ref": "#/$defs/condition"}
    group = lambda key: {  # noqa: E731
        "type": "object",
        "properties": {key: {"type": "array", "items": cond_ref, "minItems": 1, "maxItems": 6}},
        "required": [key], "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ArenaStrategy",
        "type": "object",
        "required": ["name", "thesis", "universe", "timeframe", "entry", "exit", "sizing"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 40},
            "thesis": {"type": "string", "minLength": 1, "maxLength": 280},
            "universe": {"type": "array", "items": {"enum": symbols},
                         "minItems": 1, "maxItems": len(symbols), "uniqueItems": True},
            "timeframe": {"enum": TIMEFRAMES},
            "entry": cond_ref,
            "exit": {
                "type": "object", "additionalProperties": False, "minProperties": 1,
                "properties": {
                    "take_profit_pct": {"type": "number", "minimum": 0.3, "maximum": 50},
                    "stop_loss_pct": {"type": "number", "minimum": 0.3, "maximum": 30},
                    "trailing_stop_pct": {"type": "number", "minimum": 0.3, "maximum": 30},
                    "max_hold_bars": {"type": "integer", "minimum": 1, "maximum": 500},
                    "signal": cond_ref,
                },
            },
            "sizing": {
                "type": "object", "additionalProperties": False,
                "required": ["position_pct", "max_positions"],
                "properties": {
                    "position_pct": {"type": "number", "minimum": 1, "maximum": 100},
                    "max_positions": {"type": "integer", "minimum": 1, "maximum": 5},
                },
            },
        },
        "$defs": {
            "operand": _operand_schema(),
            "condition": {"oneOf": [
                group("all"),
                group("any"),
                {"type": "object", "properties": {"not": cond_ref},
                 "required": ["not"], "additionalProperties": False},
                {"type": "object", "additionalProperties": False,
                 "required": ["left", "op", "right"],
                 "properties": {"left": {"$ref": "#/$defs/operand"},
                                "op": {"enum": OPS},
                                "right": {"$ref": "#/$defs/operand"}}},
            ]},
        },
    }


# ---------------------------------------------------------------- validation

def _walk(node: dict, depth: int = 1):
    """Yield (node, depth) for every condition node."""
    yield node, depth
    for key in ("all", "any"):
        if key in node:
            for child in node[key]:
                yield from _walk(child, depth + 1)
    if "not" in node:
        yield from _walk(node["not"], depth + 1)


def _operands(node: dict):
    for n, _ in _walk(node):
        if "op" in n:
            yield n["left"]
            yield n["right"]


def validate(strategy: dict, symbols: list[str] | None = None) -> list[str]:
    """Return a list of human-readable errors (empty list = valid).

    Errors are phrased so they can be fed straight back to the LLM on retry.
    """
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(build_schema(symbols))
    for err in sorted(validator.iter_errors(strategy), key=lambda e: list(e.path)):
        path = "/".join(str(p) for p in err.path) or "(root)"
        # oneOf errors are noisy; summarise them
        msg = "does not match any allowed form (check indicator names, params and bounds)" \
            if err.validator == "oneOf" else err.message
        errors.append(f"{path}: {msg}")
    if errors:
        return errors

    conditions = [strategy["entry"]]
    if "signal" in strategy["exit"]:
        conditions.append(strategy["exit"]["signal"])

    n_nodes = sum(1 for c in conditions for _ in _walk(c))
    if n_nodes > MAX_NODES:
        errors.append(f"too complex: {n_nodes} condition nodes (max {MAX_NODES})")
    depth = max(d for c in conditions for _, d in _walk(c))
    if depth > MAX_DEPTH:
        errors.append(f"too deeply nested: depth {depth} (max {MAX_DEPTH})")

    for c in conditions:
        for n, _ in _walk(c):
            if "op" in n and not isinstance(n["left"], dict) and not isinstance(n["right"], dict):
                errors.append(f"comparison {n['left']} {n['op']} {n['right']} compares two constants")
            if "op" in n:
                for side in ("left", "right"):
                    o = n[side]
                    if isinstance(o, dict) and o["ind"] == "macd" and o.get("fast", 0) >= o.get("slow", 1):
                        errors.append("macd: fast period must be smaller than slow period")

    wu = required_warmup(strategy)
    if wu > MAX_WARMUP:
        errors.append(f"needs {wu} bars of history (max {MAX_WARMUP}); use shorter periods")

    sz = strategy["sizing"]
    if sz["position_pct"] * sz["max_positions"] > MAX_EXPOSURE:
        errors.append(f"position_pct * max_positions = {sz['position_pct'] * sz['max_positions']} "
                      f"exceeds {MAX_EXPOSURE}% of equity")
    return errors


def required_warmup(strategy: dict) -> int:
    conditions = [strategy["entry"]] + ([strategy["exit"]["signal"]] if "signal" in strategy["exit"] else [])
    wu = 1
    for c in conditions:
        for o in _operands(c):
            if isinstance(o, dict):
                params = {k: v for k, v in o.items() if k not in ("ind", "mult")}
                wu = max(wu, warmup_bars(o["ind"], params))
    return wu + 1  # +1 so crosses_* has a valid previous bar


# ---------------------------------------------------------------- compilation

@dataclass
class CompiledStrategy:
    spec: dict
    warmup: int

    @property
    def universe(self) -> list[str]:
        return self.spec["universe"]

    @property
    def exit(self) -> dict:
        return self.spec["exit"]

    @property
    def sizing(self) -> dict:
        return self.spec["sizing"]

    def signals(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Evaluate entry/exit conditions on every bar (vectorised).

        Signal at index t uses only bars <= t. The engine acts on it at t+1.
        """
        cache: dict[str, pd.Series] = {}
        entry = _eval(self.spec["entry"], df, cache).to_numpy(dtype=bool, copy=True)
        if "signal" in self.exit:
            exit_sig = _eval(self.exit["signal"], df, cache).to_numpy(dtype=bool, copy=True)
        else:
            exit_sig = np.zeros(len(df), dtype=bool)
        # Nothing is trustworthy before warmup (also neutralises not(NaN) == True)
        entry[: self.warmup] = False
        exit_sig[: self.warmup] = False
        return {"entry": entry, "exit": exit_sig}


def compile_strategy(strategy: dict, symbols: list[str] | None = None) -> CompiledStrategy:
    errors = validate(strategy, symbols)
    if errors:
        raise ValueError("invalid strategy:\n  " + "\n  ".join(errors))
    return CompiledStrategy(spec=copy.deepcopy(strategy), warmup=required_warmup(strategy))


def _operand_series(o, df: pd.DataFrame, cache: dict) -> pd.Series:
    if not isinstance(o, dict):
        return pd.Series(float(o), index=df.index)
    params = {k: v for k, v in o.items() if k not in ("ind", "mult")}
    key = indicator_key(o["ind"], params)
    if key not in cache:
        cache[key] = compute(df, o["ind"], params)
    s = cache[key]
    return s * o["mult"] if "mult" in o else s


def _eval(node: dict, df: pd.DataFrame, cache: dict) -> pd.Series:
    if "all" in node:
        out = pd.Series(True, index=df.index)
        for child in node["all"]:
            out &= _eval(child, df, cache)
        return out
    if "any" in node:
        out = pd.Series(False, index=df.index)
        for child in node["any"]:
            out |= _eval(child, df, cache)
        return out
    if "not" in node:
        return ~_eval(node["not"], df, cache)

    left = _operand_series(node["left"], df, cache)
    right = _operand_series(node["right"], df, cache)
    op = node["op"]
    if op == "<":
        res = left < right
    elif op == ">":
        res = left > right
    elif op == "<=":
        res = left <= right
    elif op == ">=":
        res = left >= right
    elif op == "crosses_above":
        res = (left > right) & (left.shift(1) <= right.shift(1))
    elif op == "crosses_below":
        res = (left < right) & (left.shift(1) >= right.shift(1))
    else:  # unreachable after validation
        raise ValueError(op)
    return res.fillna(False).astype(bool)


# ---------------------------------------------------------------- LLM docs

def describe_for_llm(symbols: list[str] | None = None) -> str:
    """Compact DSL reference to paste into birth / reflection prompts."""
    symbols = symbols or DEFAULT_SYMBOLS
    lines = [
        "STRATEGY DSL REFERENCE",
        "Return ONLY a JSON object with keys: name, thesis, universe, timeframe, entry, exit, sizing.",
        f"universe: subset of {symbols}",
        f"timeframe: one of {TIMEFRAMES}",
        "entry: a condition. exit.signal (optional): a condition.",
        "condition: {\"all\": [cond,...]} | {\"any\": [cond,...]} | {\"not\": cond} | "
        "{\"left\": operand, \"op\": OP, \"right\": operand}",
        f"OP: {OPS}  (crosses_* = crossed on this bar)",
        "operand: a number, or {\"ind\": NAME, <params>, \"mult\": optional 0.5-2.0 scale factor}",
        "",
        "INDICATORS (name: params [bounds] - meaning)",
    ]
    for name, ind in INDICATORS.items():
        ps = ", ".join(f"{k} [{p.lo}-{p.hi}]" for k, p in ind.params.items()) or "no params"
        lines.append(f"  {name}: {ps} - {ind.description}")
    lines += [
        "",
        "exit (at least one): take_profit_pct [0.3-50], stop_loss_pct [0.3-30], "
        "trailing_stop_pct [0.3-30], max_hold_bars [1-500], signal (condition)",
        "sizing: position_pct [1-100] of equity per position, max_positions [1-5]; "
        f"position_pct * max_positions <= {MAX_EXPOSURE}",
        f"Limits: <= {MAX_NODES} condition nodes, nesting depth <= {MAX_DEPTH}, "
        f"warmup <= {MAX_WARMUP} bars.",
        "Execution: long-only spot. Signals are evaluated at bar close and filled at the next bar's "
        "open. Fees ~0.1% per side plus slippage, so frequent trading is costly.",
    ]
    return "\n".join(lines)
