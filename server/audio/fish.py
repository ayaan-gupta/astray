"""Fish Audio text-to-speech client.

Same shape as `server/llm/deepseek.py`: an injectable transport so tests never
touch the network, upstream text never forwarded to a client, and a typed error
for every transport, HTTP and protocol failure.

Every knob here is set for naturalness over speed, which costs nothing: the
render this narrates already took minutes, so `latency="normal"` (Fish's
best-quality path, not `"low"`) and the `quality-guard` feature are free wins.
Model `s2.1-pro` is their current top model.

Voice consistency comes from `reference_id`. Each beat is a separate request, so
there is no cross-request conditioning to rely on, and without a pinned voice a
single video can drift between narrators between one beat and the next.
"""

import json
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Fish bills per character of input. Recorded so narration shows up in the same
# ledger the model calls do, rather than being invisible spend.
USD_PER_MILLION_BYTES = 15.0


class SpeechError(Exception):
    """Transport, HTTP, or protocol failure from Fish Audio."""


@dataclass(frozen=True)
class SpeechClip:
    """One synthesised utterance. `audio` is encoded bytes, not samples."""

    audio: bytes
    characters: int
    ms: int

    @property
    def cost_usd(self) -> float:
        return self.characters / 1_000_000 * USD_PER_MILLION_BYTES


class FishAudioClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.fish.audio",
        model: str = "s2.1-pro",
        voice_id: str | None = None,
        speed: float = 0.96,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: int = 120,
    ) -> None:
        self._model = model
        self._voice_id = voice_id
        self._speed = speed
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # Fish selects the engine by header, not by a body field.
                "model": model,
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, text: str) -> dict:
        payload: dict = {
            "text": text,
            "format": "mp3",
            "mp3_bitrate": 128,
            # Fish's best-quality path. The render already took minutes, so
            # there is nothing to gain from "low".
            "latency": "normal",
            "features": ["quality-guard"],
            # Text normalisation for numbers and units. `speech.py` has already
            # turned notation into words, so this only has ordinary prose left.
            "normalize": True,
            # One beat's narration is a sentence or two, comfortably inside a
            # single chunk, which keeps prosody coherent across the whole line
            # instead of resetting mid-sentence.
            "chunk_length": 300,
            "prosody": {"speed": self._speed, "normalize_loudness": True},
        }
        if self._voice_id:
            payload["reference_id"] = self._voice_id
        return payload

    async def synthesize(self, text: str) -> SpeechClip:
        """Synthesise one utterance. Raises `SpeechError` on any failure."""
        if not text.strip():
            raise SpeechError("refusing to synthesize empty text")

        started = time.monotonic()
        try:
            response = await self._client.post("/v1/tts", json=self._payload(text))
        except httpx.HTTPError as exc:
            # str() on an httpx error carries the URL and message, never headers.
            raise SpeechError(f"fish transport failure: {exc}") from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        if response.status_code >= 400:
            raise SpeechError(f"fish HTTP {response.status_code}: {_detail(response)}")

        audio = response.content
        if not audio:
            raise SpeechError("fish returned an empty audio body")
        # A JSON body with a 200 means an error shaped like a success. Catching it
        # here stops a JSON blob being written out with an .mp3 extension and
        # failing much later inside ffmpeg, where the cause is unrecoverable.
        if audio[:1] in (b"{", b"["):
            raise SpeechError(f"fish returned JSON, not audio: {_detail(response)}")

        return SpeechClip(audio=audio, characters=len(text), ms=elapsed_ms)


def _detail(response: httpx.Response) -> str:
    """A short, safe description of an error body, whatever shape it arrives in."""
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("detail") or body)[:200]
    return str(body)[:200]
