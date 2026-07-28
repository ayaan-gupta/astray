import pytest

from server.llm.accounting import UnknownModelError, cost_usd


def test_pro_cost_with_cache_hits():
    # 600 miss * 0.435 + 400 hit * 0.003625 + 500 out * 0.87, per 1M
    got = cost_usd("deepseek-v4-pro", prompt_tokens=1000, completion_tokens=500, cached_tokens=400)
    assert got == pytest.approx(0.00069745, rel=1e-6)


def test_flash_cost_no_cache():
    # 1000 prompt * 0.14 + 1000 output * 0.28 = 420 / 1M = 0.00042
    got = cost_usd("deepseek-v4-flash", prompt_tokens=1000, completion_tokens=1000)
    assert got == pytest.approx(0.00042, rel=1e-9)


def test_gemini_vision_cost():
    # 1125 prompt * 0.30 + 83 output * 2.50 = 545 / 1M = 0.000545
    got = cost_usd("gemini-3.5-flash-lite", prompt_tokens=1125, completion_tokens=83)
    assert got == pytest.approx(0.000545, rel=1e-9)


def test_cached_exceeding_prompt_does_not_go_negative():
    # cached capped to 100; miss=0; 100 cached * 0.0028 = 0.28 / 1M = 2.8e-7
    got = cost_usd("deepseek-v4-flash", prompt_tokens=100, completion_tokens=0, cached_tokens=500)
    assert got == pytest.approx(2.8e-7)


def test_negative_completion_tokens_clamped():
    # negative completion_tokens clamped to 0, cost should be 0
    got = cost_usd("deepseek-v4-flash", prompt_tokens=0, completion_tokens=-1000, cached_tokens=0)
    assert got == 0.0


def test_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        cost_usd("gpt-9", prompt_tokens=1, completion_tokens=1)
