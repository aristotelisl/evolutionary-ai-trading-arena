"""
The agent's "brain": LLM calls that design and revise strategies.

birth() asks the LLM for a strategy, validates it, backtests it and runs
birth_check(). Any problem is sent back verbatim and the LLM retries, up to
max_attempts calls in total. Every call is returned as an LLMCall so the
caller can log it and charge the agent's wallet (compute tax).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from .dsl import compile_strategy, describe_for_llm, validate
from .engine import backtest, birth_check

BIRTH_MODEL = "claude-sonnet-5-5"
BIRTH_EFFORT = "medium"

# USD per million tokens (input, output). Update when prices change.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass
class LLMCall:
    purpose: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    prompt: str          # JSON: system + messages sent
    response: str        # text the model returned
    ok: bool             # True if the response was usable


@dataclass
class BirthResult:
    ok: bool
    spec: dict | None
    backtest: dict | None              # summary without trades / equity curve
    problems: list[str]                # last rejection reasons (empty if ok)
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICES.get(model, PRICES[BIRTH_MODEL])
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


def parse_json(text: str) -> dict:
    """Pull the JSON object out of a reply, tolerating code fences or stray prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return json.loads(text[start:end + 1])


def birth_prompt(lessons: str, timeframes: list[str]) -> tuple[str, str]:
    system = (
        "You are the brain of a newborn trading agent in a paper-trading arena. "
        "Your strategy runs automatically on every candle; agents that lose too much money die, "
        "and every LLM call you make is charged to your wallet.\n\n"
        f"{describe_for_llm()}\n\n"
        f"Market data is currently available only for these timeframes: {timeframes}. "
        "Reply with the JSON object only: no prose, no code fences."
    )
    if lessons.strip():
        user = f"Lessons inherited from your parent:\n{lessons.strip()}\n\nDesign your strategy now."
    else:
        user = "You are a founder: no parent, no inherited lessons. Design your strategy now."
    return system, user


def check_candidate(spec: dict, data: dict[str, dict[str, pd.DataFrame]]) -> tuple[list[str], dict | None]:
    """Validate and backtest one candidate. Returns (problems, backtest summary)."""
    errors = validate(spec)
    if errors:
        return errors, None
    tf_data = data.get(spec["timeframe"])
    if tf_data is None:
        return [f"timeframe {spec['timeframe']!r} has no market data; choose one of {sorted(data)}"], None
    missing = [s for s in spec["universe"] if s not in tf_data]
    if missing:
        return [f"no market data for {missing} on {spec['timeframe']}"], None
    result = backtest(compile_strategy(spec), {s: tf_data[s] for s in spec["universe"]})
    summary = {k: v for k, v in result.items() if k not in ("trades", "equity_curve")}
    return birth_check(result), summary


def _ask(client, model: str, system: str, messages: list[dict]):
    return client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=messages,
        output_config={"effort": BIRTH_EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def birth(data: dict[str, dict[str, pd.DataFrame]], lessons: str = "", client=None,
          model: str = BIRTH_MODEL, max_attempts: int = 3) -> BirthResult:
    """Design a strategy for a newborn agent.

    data: {timeframe: {symbol: candles}}; the backtest uses all candles given.
    lessons: the parent's compressed lessons ("" for a founder).
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    system, user = birth_prompt(lessons, sorted(data))
    messages = [{"role": "user", "content": user}]
    calls: list[LLMCall] = []
    problems: list[str] = []

    for _ in range(max_attempts):
        prompt_record = json.dumps({"system": system, "messages": messages})
        resp = _ask(client, model, system, messages)
        text = "".join(b.text for b in resp.content if b.type == "text")
        served = getattr(resp, "model", model) or model
        call = LLMCall("birth", served, resp.usage.input_tokens, resp.usage.output_tokens,
                       cost_usd(served, resp.usage.input_tokens, resp.usage.output_tokens),
                       prompt_record, text, ok=False)
        calls.append(call)

        if resp.stop_reason == "refusal":
            return BirthResult(False, None, None, ["the model refused the request"], calls)

        spec, summary = None, None
        try:
            spec = parse_json(text)
            problems, summary = check_candidate(spec, data)
        except (ValueError, json.JSONDecodeError) as e:
            problems = [f"reply was not valid JSON: {e}"]

        if not problems:
            call.ok = True
            return BirthResult(True, spec, summary, [], calls)

        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Your strategy was rejected:\n"
                                        + "\n".join(f"- {p}" for p in problems)
                                        + "\nFix these problems and reply with the corrected JSON object only."},
        ]

    return BirthResult(False, None, None, problems, calls)
