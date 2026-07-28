import pytest
from pydantic import ValidationError

from server.config import Settings, get_settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.deepseek_api_key == "sk-test"
    assert s.deepseek_model_reasoning == "deepseek-v4-pro"
    assert s.deepseek_model_fast == "deepseek-v4-flash"
    assert s.gemini_model_vision == "gemini-3.5-flash-lite"
    assert s.render_max_repairs == 2


def test_missing_deepseek_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_vision_disabled_without_gemini_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert Settings(_env_file=None).vision_enabled is False


def test_vision_enabled_with_gemini_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    assert Settings(_env_file=None).vision_enabled is True


def test_get_settings_reflects_first_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-first")
    assert get_settings().deepseek_api_key == "sk-first"


def test_get_settings_not_stale_across_tests(monkeypatch):
    """Proves the conftest autouse fixture clears get_settings()'s lru_cache between tests.

    Without the fixture clearing the cache after test_get_settings_reflects_first_env, this
    would incorrectly see "sk-first" (the cached singleton from the previous test) instead of
    the value patched in this test.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-second")
    settings = get_settings()
    assert settings.deepseek_api_key == "sk-second"
    assert settings.deepseek_api_key != "sk-first"
