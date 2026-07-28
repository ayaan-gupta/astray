"""Vision transcription. DeepSeek has no image support, so photo input uses Gemini.

The product-critical property: transcription must preserve the student's errors verbatim.
A vision model that "helpfully" corrects the arithmetic destroys the diagnosis stage, since
the whole point is to catch the mistake. ``TRANSCRIBE_PROMPT`` is written — and was verified
against the live API — to keep the model from correcting anything it transcribes.
"""

import base64
import json
import time
from typing import Protocol

import httpx
from pydantic import ValidationError

from server.charter.contracts import LlmCallMeta, Transcription
from server.llm.accounting import cost_usd

TRANSCRIBE_PROMPT = (
    "You are transcribing a photo of a student's handwritten math work.\n\n"
    "Transcribe EXACTLY what is written, character for character. Do NOT correct errors, "
    "do NOT simplify, do NOT complete unfinished work, and do NOT fix arithmetic even if it "
    "is wrong. Preserving the student's mistakes exactly is the entire purpose of this "
    "transcription — a corrected transcription is worse than useless.\n\n"
    "Use LaTeX-style notation for math. Put the problem statement in `problem` and each "
    "line of the student's work as a separate entry in `steps`, in the order written.\n"
    "Set `confidence` to your overall confidence (0-1) that the transcription is accurate. "
    "List any region you could not read in `unreadable`.\n\n"
    "Return only JSON with exactly these fields: "
    '{"problem": str, "steps": [str], "confidence": number, "unreadable": [str]}'
)


class VisionUnavailable(Exception):
    """No vision provider is configured, or the provider call failed."""


class VisionProvider(Protocol):
    async def transcribe(
        self, image_bytes: bytes, mime_type: str
    ) -> tuple[Transcription, LlmCallMeta]: ...


class NullVision:
    """Used when GEMINI_API_KEY is absent. Photo input is disabled, typed input still works."""

    async def transcribe(
        self, image_bytes: bytes, mime_type: str
    ) -> tuple[Transcription, LlmCallMeta]:
        raise VisionUnavailable("photo input requires GEMINI_API_KEY; typed input is unaffected")


def _text_part(body: object) -> str:
    """Validate and return ``candidates[0].content.parts[0].text`` from a decoded body.

    Raises ``VisionUnavailable`` (never a bare ``KeyError``/``IndexError``/``TypeError``)
    for any malformed shape, mirroring the DeepSeek client's response-shape discipline —
    Gemini's response nests deeply, so unguarded indexing is a live risk.
    """
    if not isinstance(body, dict):
        raise VisionUnavailable(f"gemini response body is not a JSON object: {type(body).__name__}")
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise VisionUnavailable(f"gemini response missing non-empty 'candidates': {body!r}"[:500])
    first = candidates[0]
    content = first.get("content") if isinstance(first, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise VisionUnavailable(f"gemini candidate missing 'content.parts': {first!r}"[:500])
    text = parts[0].get("text") if isinstance(parts[0], dict) else None
    if not isinstance(text, str):
        raise VisionUnavailable(f"gemini part missing 'text': {parts[0]!r}"[:500])
    return text


class GeminiVision:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-3.5-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: int = 120,
    ) -> None:
        self._model = model
        # Auth is via the x-goog-api-key HEADER, never a query param or bearer token — a
        # query param would leak the key into logs and proxy access records.
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=timeout_s,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self, image_bytes: bytes, mime_type: str
    ) -> tuple[Transcription, LlmCallMeta]:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": TRANSCRIBE_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode(),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
        started = time.monotonic()
        try:
            response = await self._client.post(
                f"/models/{self._model}:generateContent", json=payload
            )
        except httpx.HTTPError as exc:
            # str(exc) for httpx errors never includes headers/auth, only URL + message.
            raise VisionUnavailable(f"gemini transport failure: {exc}") from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        if response.status_code >= 400:
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError):
                data = None
            error = data.get("error") if isinstance(data, dict) else None
            detail = error.get("message", "") if isinstance(error, dict) else ""
            if not detail:
                detail = response.text[:200]
            raise VisionUnavailable(f"gemini HTTP {response.status_code}: {detail}")

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise VisionUnavailable(f"gemini response is not valid JSON: {exc}") from exc

        text = _text_part(body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VisionUnavailable(f"gemini transcription is not valid JSON: {exc}") from exc
        try:
            transcription = Transcription.model_validate(parsed)
        except ValidationError as exc:
            raise VisionUnavailable(
                f"gemini transcription failed schema validation: {exc}"
            ) from exc

        usage = body.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        meta = LlmCallMeta(
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd(self._model, prompt_tokens, completion_tokens),
            ms=elapsed_ms,
        )
        return transcription, meta
