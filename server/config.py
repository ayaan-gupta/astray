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

    # Narration. Written after the render, never before: the script is budgeted
    # against each beat's *measured* duration, so it cannot be produced until the
    # renderer has reported its clock. s2.1-pro is Fish Audio's current top
    # model; latency is irrelevant here because the render already took minutes,
    # so every setting below trades speed for naturalness.
    fish_api_key: str | None = None
    fish_base_url: str = "https://api.fish.audio"
    # `s2.1-pro-free` is the same model as `s2.1-pro` on the free developer tier,
    # under Fish's fair-use policy. The paid string returns HTTP 402 unless the
    # account has *API* credit, which Fish bills separately from platform credit,
    # so the free string is the working default and the paid one is opt-in.
    fish_model: str = "s2.1-pro-free"
    # A voice id from fish.audio. Unset uses the model's own default voice, which
    # is a reasonable narrator; set it to keep one voice across every render.
    fish_voice_id: str | None = None
    narration_enabled: bool = True
    narration_timeout_s: int = 120
    # Measured, not assumed, and set near the slow end rather than the mean.
    # Six real s2.1-pro lines at speed 0.96 averaged 2.17 words/second but ranged
    # from 1.69 to 2.66: the more of a line is spoken maths, the slower it goes,
    # and "a squared plus two a b plus b squared" is nearly all maths. Budgeting
    # at the 2.5 first guessed here made a 16-word line 8.4s long inside a 7.0s
    # beat; at 2.0 a 13-word closing line still came out at 1.82 w/s and overran.
    # A short line costs nothing, so the bias belongs on this side.
    narration_words_per_second: float = 1.75
    # Slightly under real time. Maths needs a fraction more room than prose.
    narration_speed: float = 0.96

    fake_llm: bool = False
    db_path: Path = Path("data/tutor.db")
    media_root: Path = Path("media")

    model_config = SettingsConfigDict(
        env_file="server/.env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def vision_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def narration_available(self) -> bool:
        return self.narration_enabled and bool(self.fish_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
