"""Narration end to end: script, speech, timeline, mux.

Runs after a successful render, and is deliberately non-fatal. A session with a
silent video is a working session; a session whose render is discarded because a
voice API was down is not. Every failure here logs and leaves the original video
in place.

Narration publishes to the render's own path so the URL a session already serves
keeps working, and copies the untouched render to `silent.mp4` beside it first.
That copy is what makes the step re-runnable: a second pass reads it rather than
feeding ffmpeg its own previous output, which ffmpeg refuses outright. The
published file is swapped in atomically, so a crash mid-mux cannot leave a
half-written video being served.
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from server.audio import mux, narrate, speech
from server.audio.fish import FishAudioClient, SpeechError
from server.charter.contracts import StageName
from server.config import Settings
from server.llm.deepseek import DeepSeekClient, LlmError
from server.store import repo

logger = logging.getLogger(__name__)

# The untouched render, kept beside the published video.
SILENT_NAME = "silent.mp4"


def silent_source(video: Path) -> Path:
    """The file to narrate: the preserved silent original if there is one."""
    silent = video.parent / SILENT_NAME
    return silent if silent.exists() else video


@dataclass(frozen=True)
class NarrationResult:
    ok: bool
    video_path: str | None = None
    lines: int = 0
    characters: int = 0
    cost_usd: float = 0.0
    worst_drift_s: float = 0.0
    skipped_reason: str | None = None


async def add_narration(
    conn,
    *,
    session_id: str,
    video_path: str,
    settings: Settings,
    llm: DeepSeekClient,
) -> NarrationResult:
    """Write, synthesise and mux narration for one finished render. Never raises."""
    if not settings.narration_available:
        return NarrationResult(ok=False, skipped_reason="narration disabled or no fish key")
    if not mux.ffmpeg_available():
        return NarrationResult(ok=False, skipped_reason="ffmpeg and ffprobe not on PATH")

    video = Path(video_path)
    if not video.exists():
        return NarrationResult(ok=False, skipped_reason=f"render missing at {video}")

    try:
        script, meta = await narrate.write_script(
            conn,
            llm,
            session_id=session_id,
            model=settings.deepseek_model_fast,
            words_per_second=settings.narration_words_per_second,
        )
    except (LlmError, ValueError) as exc:
        logger.warning("narration script failed for %s: %s", session_id, exc)
        return NarrationResult(ok=False, skipped_reason=f"script failed: {exc}")

    if not script:
        return NarrationResult(ok=False, skipped_reason="script came back empty")

    repo.record_artifact(
        conn,
        session_id=session_id,
        stage=StageName.NARRATE,
        payload={"lines": [{"beat_id": b, "line": t} for b, t in script]},
        meta=meta,
    )

    clip_dir = video.parent / "narration"
    clip_dir.mkdir(parents=True, exist_ok=True)

    client = FishAudioClient(
        settings.fish_api_key or "",
        base_url=settings.fish_base_url,
        model=settings.fish_model,
        voice_id=settings.fish_voice_id,
        speed=settings.narration_speed,
        timeout_s=settings.narration_timeout_s,
    )
    try:
        clips, characters, cost = await _synthesize(client, script, clip_dir)
    except SpeechError as exc:
        logger.warning("narration synthesis failed for %s: %s", session_id, exc)
        return NarrationResult(ok=False, skipped_reason=f"synthesis failed: {exc}")
    finally:
        await client.aclose()

    if not clips:
        return NarrationResult(ok=False, skipped_reason="no clips were synthesised")

    starts = {
        row["beat_id"]: float(row["start_s"])
        for row in repo.list_beats(conn, session_id)
        if row["start_s"] is not None
    }
    # Narrating publishes to the render's own path, so the URL the session already
    # serves keeps working and nothing downstream has to be repointed. The silent
    # original is preserved beside it under SILENT_NAME first, which is also what
    # makes this re-runnable: a second pass reads the silent copy rather than
    # feeding ffmpeg its own output, which it refuses outright.
    source = silent_source(video)
    try:
        placements = mux.plan_timeline(
            clips, starts, video_duration_s=await mux.probe_duration_s(source)
        )
        staged = video.parent / "narrated.staging.mp4"
        await mux.mux_in(source, placements, staged, timeout_s=settings.render_timeout_s)
        if source == video:
            # Copy rather than move: a move would leave the served path missing
            # for the moment between the two operations.
            shutil.copy2(video, video.parent / SILENT_NAME)
        # Atomic, so a crash here cannot leave a half-written video being served.
        staged.replace(video)
    except (RuntimeError, KeyError, ValueError, TimeoutError, OSError) as exc:
        logger.warning("narration mux failed for %s: %s", session_id, exc)
        return NarrationResult(ok=False, skipped_reason=f"mux failed: {exc}")

    out = video
    # Worst *absolute* drift. Taking a signed max reported 0.00s for a run whose
    # final clip had been pulled 1.27s earlier, hiding the one placement that had
    # actually moved.
    worst = max((abs(p.drift_s) for p in placements), default=0.0)
    logger.info(
        "narrated %s: %d lines, %d chars, $%.4f, worst drift %.2fs",
        session_id,
        len(placements),
        characters,
        cost,
        worst,
    )
    return NarrationResult(
        ok=True,
        video_path=str(out),
        lines=len(placements),
        characters=characters,
        cost_usd=cost,
        worst_drift_s=worst,
    )


async def _synthesize(
    client: FishAudioClient, script: list[tuple[str, str]], clip_dir: Path
) -> tuple[list[tuple[str, Path, float]], int, float]:
    """Synthesise every line concurrently, then measure what actually came back.

    Concurrent because the lines are independent and a six-beat render would
    otherwise serialise six round trips for no reason. Bounded by the number of
    beats, which is single digits, so there is no semaphore to add.
    """
    # Phoneme tags are added here rather than in the script, so what gets stored
    # and shown stays readable text and only the API sees the markup.
    results = await asyncio.gather(
        *(client.synthesize(speech.with_letter_phonemes(text)) for _, text in script),
        return_exceptions=True,
    )

    clips: list[tuple[str, Path, float]] = []
    characters = 0
    cost = 0.0
    for (beat_id, _), result in zip(script, results, strict=True):
        if isinstance(result, BaseException):
            # One failed line is a quieter video, not a failed render. Skipping it
            # keeps every other line in place at its correct offset.
            logger.warning("narration for beat %s failed: %s", beat_id, result)
            continue
        path = clip_dir / f"{beat_id}.mp3"
        path.write_bytes(result.audio)
        characters += result.characters
        cost += result.cost_usd
        clips.append((beat_id, path, await mux.probe_duration_s(path)))
    return clips, characters, cost
