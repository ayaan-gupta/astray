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


def test_build_llm_client_wires_max_retries_from_settings():
    """Settings.llm_max_retries used to be dead config: complete_json hard-coded
    max_retries=2 and nothing read the setting, so configuring it in .env silently
    did nothing. build_llm_client must now pass it through to DeepSeekClient's
    constructor, which is what makes it take effect (see test_deepseek.py for the
    end-to-end proof that the constructor value actually changes retry behavior)."""
    settings = _settings(llm_max_retries=5)
    client = build_llm_client(settings)
    assert client._max_retries == 5


def test_build_llm_client_default_max_retries_matches_settings_default():
    settings = _settings()
    client = build_llm_client(settings)
    assert client._max_retries == settings.llm_max_retries == 2
