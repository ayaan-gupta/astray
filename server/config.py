from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets come from server/.env only."""

    deepseek_api_key: str
    gemini_api_key: str | None = None

    deepseek_base_url: str = "https://api.deepseek.com"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    deepseek_model_reasoning: str = "deepseek-v4-pro"
    deepseek_model_fast: str = "deepseek-v4-flash"
    gemini_model_vision: str = "gemini-3.5-flash-lite"

    # Rendering needs a working Docker daemon and the manim image. Disabled, the
    # chain still plans the animation (s2-s8) and persists the beats, but never
    # starts a container -- which is what tests do (no test may touch docker, for
    # the same reason none may touch the network) and what a host without Docker
    # should do rather than failing every session.
    render_enabled: bool = True
    render_timeout_s: int = 300
    render_max_repairs: int = 2
    llm_timeout_s: int = 240
    llm_max_retries: int = 2

    fake_llm: bool = False
    db_path: Path = Path("data/tutor.db")
    media_root: Path = Path("media")

    model_config = SettingsConfigDict(
        env_file="server/.env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def vision_enabled(self) -> bool:
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
