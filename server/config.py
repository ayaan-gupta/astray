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
    # Pinning a voice is not optional, and leaving it unset was a real bug: every
    # beat is a separate request, so with no reference_id each one comes back in a
    # different voice and a six-beat video has six narrators.
    #
    # "Adam - Calm, Smart", chosen on measured evidence. Across five candidates on
    # the same five lines it was the fastest of the calm voices (2.84 words/second
    # mean against 2.21 for the steadiest), which matters because the video is a
    # fixed 34.8s: a faster voice fits ~89 words of explanation where a slower one
    # fits ~69, and too few words is what makes narration sound like disconnected
    # captions. Re-measure with scripts/measure_voice.py when changing this.
    fish_voice_id: str | None = "ba1cd26ca87b42b2bf7d60c1f65f9242"
    narration_enabled: bool = True
    narration_timeout_s: int = 120
    # A property of the chosen voice, not a constant, and measured rather than
    # guessed: scripts/measure_voice.py reports it, on the text actually sent,
    # phoneme tags included. Forcing the letter names costs about 5% in duration,
    # so a rate measured on untagged text is optimistic by that much: for this
    # voice the mean fell from 2.84 to 2.76 and the slowest line from 2.57 to 2.39
    # once tagging was included, and 2.49 (set from the untagged figure) was above
    # the slowest line, which is exactly the beat that then overran.
    #
    # This sits just under the slowest measured line, because a line that overruns
    # pushes the next one late while a short line costs nothing. Every voice
    # measured slows markedly on dense spoken maths, which is why the slow end is
    # the one that matters. Re-measure when changing fish_voice_id.
    narration_words_per_second: float = 2.32
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
