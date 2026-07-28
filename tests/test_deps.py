from server.config import Settings
from server.deps import build_llm_client, build_vision
from server.llm.vision import GeminiVision, NullVision


def _settings(**overrides) -> Settings:
    base = {"_env_file": None, "deepseek_api_key": "sk-test"}
    base.update(overrides)
    return Settings(**base)


def test_build_vision_returns_null_vision_without_gemini_key():
    assert isinstance(build_vision(_settings()), NullVision)


def test_build_vision_returns_gemini_vision_when_enabled_and_not_fake():
    settings = _settings(gemini_api_key="AQ.test")
    assert isinstance(build_vision(settings), GeminiVision)


def test_build_vision_returns_null_vision_when_fake_llm_even_with_gemini_key():
    """fake_transport only knows how to answer DeepSeek-shaped chat-completions
    requests (server/llm/fake.py) -- it cannot stand in for Gemini's entirely
    different response shape. If fake_llm is on, offline mode must not reach the
    real Gemini API for photo input even when a real GEMINI_API_KEY is set."""
    settings = _settings(gemini_api_key="AQ.test", fake_llm=True)
    assert isinstance(build_vision(settings), NullVision)


def test_build_llm_client_does_not_raise_when_fake_llm_enabled():
    settings = _settings(fake_llm=True)
    client = build_llm_client(settings)
    assert client is not None
