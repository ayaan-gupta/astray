import pytest

from server.llm.accounting import UnknownModelError, cost_usd


def test_pro_cost_with_cache_hits():
    # 600 miss * 0.435 + 400 hit * 0.003625 + 500 out * 0.87, per 1M
    got = cost_usd("deepseek-v4-pro", prompt_tokens=1000, completion_tokens=500, cached_tokens=400)
    assert got == pytest.approx(0.00069745, rel=1e-6)


def test_flash_cost_no_cache():
    got = cost_usd("deepseek-v4-flash", prompt_tokens=1000, completion_tokens=1000)
    assert got == pytest.approx((1000 * 0.14 + 1000 * 0.28) / 1_000_000, rel=1e-9)


def test_gemini_vision_cost():
    got = cost_usd("gemini-3.5-flash-lite", prompt_tokens=1125, completion_tokens=83)
    assert got == pytest.approx((1125 * 0.30 + 83 * 2.50) / 1_000_000, rel=1e-9)


def test_cached_exceeding_prompt_does_not_go_negative():
    got = cost_usd("deepseek-v4-flash", prompt_tokens=100, completion_tokens=0, cached_tokens=500)
    assert got >= 0.0


def test_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        cost_usd("gpt-9", prompt_tokens=1, completion_tokens=1)
