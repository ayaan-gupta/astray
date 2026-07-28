"""Provider construction. Keeps secret handling in one place."""

from server.config import Settings
from server.llm.deepseek import DeepSeekClient
from server.llm.fake import FIXTURES, fake_transport
from server.llm.vision import GeminiVision, NullVision, VisionProvider


def build_llm_client(settings: Settings) -> DeepSeekClient:
    transport = fake_transport(FIXTURES) if settings.fake_llm else None
    return DeepSeekClient(
        settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        transport=transport,
        timeout_s=settings.llm_timeout_s,
    )


def build_vision(settings: Settings) -> VisionProvider:
    """Return the vision provider for photo transcription.

    ``fake_transport`` only knows how to answer DeepSeek-shaped chat-completions
    requests (see server/llm/fake.py) -- it has no idea how to answer Gemini's
    ``generateContent`` shape, so it cannot stand in for GeminiVision. If
    ``fake_llm`` is on, offline mode must not attempt a real call to the live
    Gemini API for photo input: return NullVision so a photo upload fails with
    the same clear "GEMINI_API_KEY" 503 a genuinely-missing key would produce,
    rather than hanging or erroring against the real network.
    """
    if settings.fake_llm or not settings.vision_enabled:
        return NullVision()
    return GeminiVision(
        settings.gemini_api_key or "",
        model=settings.gemini_model_vision,
        base_url=settings.gemini_base_url,
    )
