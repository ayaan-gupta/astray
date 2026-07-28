import json

import httpx
import pytest

from server.llm.vision import GeminiVision, NullVision, VisionUnavailable


def _gemini_reply(payload: dict, prompt_tokens=1125, out_tokens=83):
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}], "role": "model"}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": out_tokens,
        },
    }


async def test_transcribe_returns_structured_work():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "AQ.test"
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        # the image must be inlined as base64
        parts = body["contents"][0]["parts"]
        assert any("inline_data" in p or "inlineData" in p for p in parts)
        return httpx.Response(
            200,
            json=_gemini_reply(
                {
                    "problem": "Solve: (x + 3)^2 = 25",
                    "steps": ["x^2 + 9 = 25", "x^2 = 16", "x = 4"],
                    "confidence": 0.99,
                    "unreadable": [],
                }
            ),
        )

    provider = GeminiVision("AQ.test", transport=httpx.MockTransport(handler))
    transcription, meta = await provider.transcribe(b"\x89PNG fake", "image/png")
    assert transcription.steps[0] == "x^2 + 9 = 25"
    assert transcription.confidence == pytest.approx(0.99)
    assert meta.cost_usd > 0


async def test_prompt_forbids_correcting_the_student():
    from server.llm.vision import TRANSCRIBE_PROMPT

    lowered = TRANSCRIBE_PROMPT.lower()
    assert "do not correct" in lowered
    assert "exactly" in lowered


async def test_null_vision_raises():
    with pytest.raises(VisionUnavailable):
        await NullVision().transcribe(b"x", "image/png")


async def test_http_error_raises_vision_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "bad key"}})

    provider = GeminiVision("AQ.bad", transport=httpx.MockTransport(handler))
    with pytest.raises(VisionUnavailable):
        await provider.transcribe(b"x", "image/png")
