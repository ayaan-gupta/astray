"""Token→USD cost math. Rates are USD per 1M tokens, verified 2026-07-28."""

from dataclasses import dataclass


class UnknownModelError(ValueError):
    """Raised when a model has no price entry — fail loudly, never silently bill $0."""


@dataclass(frozen=True)
class Rate:
    input_miss: float
    input_hit: float
    output: float


RATES: dict[str, Rate] = {
    "deepseek-v4-flash": Rate(input_miss=0.14, input_hit=0.0028, output=0.28),
    "deepseek-v4-pro": Rate(input_miss=0.435, input_hit=0.003625, output=0.87),
    # Gemini has no separate cache-hit tier for our usage.
    "gemini-3.5-flash-lite": Rate(input_miss=0.30, input_hit=0.30, output=2.50),
    "gemini-2.5-flash-lite": Rate(input_miss=0.10, input_hit=0.10, output=0.40),
}


def cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
) -> float:
    try:
        rate = RATES[model]
    except KeyError as exc:
        raise UnknownModelError(f"no price entry for model {model!r}") from exc

    cached = min(max(cached_tokens, 0), max(prompt_tokens, 0))
    miss = max(prompt_tokens - cached, 0)
    total = (
        miss * rate.input_miss
        + cached * rate.input_hit
        + completion_tokens * rate.output
    )
    return total / 1_000_000
