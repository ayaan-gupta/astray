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
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from server.audio import mux, narrate, pad, speech
from server.audio.fish import FishAudioClient, SpeechError
from server.charter.contracts import BeatTiming, StageName
from server.config import Settings
from server.llm.deepseek import DeepSeekClient, LlmError
from server.store import repo

logger = logging.getLogger(__name__)

# The untouched render, kept beside the published video.
SILENT_NAME = "silent.mp4"

# The render's own beat timings, kept beside the silent original and paired with
# it. Written before the first pad and never rewritten, because it is the only
# record of where the beats were *in the render* once the beats table has been
# updated to describe the padded video that gets published.
SPANS_NAME = "silent.spans.json"


def silent_source(video: Path) -> Path:
    """The file to narrate: the preserved silent original if there is one."""
    silent = video.parent / SILENT_NAME
    return silent if silent.exists() else video


def render_spans(video: Path, from_db: dict[str, tuple[float, float]]) -> dict:
    """The render's beat spans, from the sidecar if written, else from the DB.

    First narration: the beats table still holds the render's own measured
    timings, so those are the truth and get written down. Every narration after
    that: the table describes the published video, holds included, so the sidecar
    is the truth and the table is not consulted.

    Without this, re-narrating compounds. Pass two would read padded spans, decide
    each beat was already long enough, and mux against the unpadded silent
    original -- putting every line late and cutting the last one off.
    """
    sidecar = video.parent / SPANS_NAME
    if sidecar.exists():
        try:
            raw = json.loads(sidecar.read_text())
            return {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            logger.warning("unreadable %s; falling back to the beats table", sidecar.name)
    try:
        sidecar.write_text(json.dumps({k: list(v) for k, v in from_db.items()}))
    except OSError:
        # A sidecar we cannot write costs re-runnability, not this run.
        logger.warning("could not write %s; re-narration will not be idempotent", sidecar.name)
    return dict(from_db)


@dataclass(frozen=True)
class NarrationResult:
    ok: bool
    video_path: str | None = None
    lines: int = 0
    characters: int = 0
    cost_usd: float = 0.0
    worst_drift_s: float = 0.0
    # Seconds of frozen frame added so the lines fit. 0.0 means every line came in
    # under its beat and the render was published unchanged.
    held_s: float = 0.0
    skipped_reason: str | None = None


async def add_narration(
    conn,
    *,
    session_id: str,
    video_path: str,
    settings: Settings,
    llm: DeepSeekClient,
    explaining: bool = False,
) -> NarrationResult:
    """Write, synthesise and mux narration for one finished render. Never raises."""
    if not settings.narration_available:
        return NarrationResult(ok=False, skipped_reason="narration disabled or no fish key")
    if not mux.ffmpeg_available():
        return NarrationResult(ok=False, skipped_reason="ffmpeg and ffprobe not on PATH")

    video = Path(video_path)
    if not video.exists():
        return NarrationResult(ok=False, skipped_reason=f"render missing at {video}")

    db_spans = {
        row["beat_id"]: (float(row["start_s"]), float(row["end_s"]))
        for row in repo.list_beats(conn, session_id)
        if row["start_s"] is not None
    }
    spans = render_spans(video, db_spans)

    try:
        script, meta = await narrate.write_script(
            conn,
            llm,
            session_id=session_id,
            model=settings.deepseek_model_fast,
            words_per_second=settings.narration_words_per_second,
            spans=spans,
            explaining=explaining,
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

    # Narrating publishes to the render's own path, so the URL the session already
    # serves keeps working and nothing downstream has to be repointed. The silent
    # original is preserved beside it under SILENT_NAME first, which is also what
    # makes this re-runnable: a second pass reads the silent copy rather than
    # feeding ffmpeg its own output, which it refuses outright.
    source = silent_source(video)
    held = 0.0
    try:
        source, spans, held = await _make_room(
            source, spans, clips, session_id=session_id, timeout_s=settings.render_timeout_s
        )
        starts = {beat_id: start for beat_id, (start, _) in spans.items()}
        placements = mux.plan_timeline(
            clips, starts, video_duration_s=await mux.probe_duration_s(source)
        )
        staged = video.parent / "narrated.staging.mp4"
        await mux.mux_in(source, placements, staged, timeout_s=settings.render_timeout_s)
        if silent_source(video) == video:
            # Copy rather than move: a move would leave the served path missing
            # for the moment between the two operations.
            shutil.copy2(video, video.parent / SILENT_NAME)
        # Atomic, so a crash here cannot leave a half-written video being served.
        staged.replace(video)
    except (RuntimeError, KeyError, ValueError, TimeoutError, OSError) as exc:
        logger.warning("narration mux failed for %s: %s", session_id, exc)
        return NarrationResult(ok=False, skipped_reason=f"mux failed: {exc}")
    finally:
        # The padded intermediate is only an input to the mux above. `silent.mp4`
        # plus `silent.spans.json` are what a re-run needs, and both survive, so
        # keeping this as well would be a second copy of every narrated video.
        if source != silent_source(video):
            source.unlink(missing_ok=True)

    if held:
        # The published video is no longer the render, so the beats table has to
        # describe the published one: every chip on the rail and every `[beat:bN]`
        # citation seeks against these numbers.
        repo.save_beat_timings(
            conn,
            session_id,
            [
                BeatTiming(id=beat_id, start=start, end=end)
                for beat_id, (start, end) in spans.items()
            ],
        )

    out = video
    # Worst *absolute* drift. Taking a signed max reported 0.00s for a run whose
    # final clip had been pulled 1.27s earlier, hiding the one placement that had
    # actually moved.
    worst = max((abs(p.drift_s) for p in placements), default=0.0)
    logger.info(
        "narrated %s: %d lines, %d chars, $%.4f, %.1fs held, worst drift %.2fs",
        session_id,
        len(placements),
        characters,
        cost,
        held,
        worst,
    )
    return NarrationResult(
        ok=True,
        video_path=str(out),
        lines=len(placements),
        characters=characters,
        cost_usd=cost,
        worst_drift_s=worst,
        held_s=held,
    )


async def _make_room(
    source: Path,
    spans: dict[str, tuple[float, float]],
    clips: list[tuple[str, Path, float]],
    *,
    session_id: str,
    timeout_s: int,
) -> tuple[Path, dict[str, tuple[float, float]], float]:
    """Hold each beat open long enough for its line, if any of them need it.

    Returns `(video_to_mux, spans_of_that_video, seconds_added)`. When nothing
    needs room -- every line came in under its beat -- the render is returned
    untouched with its own spans, so the common case still copies the video stream
    rather than re-encoding it.

    Non-fatal like the rest of this module. A padding failure returns the
    unpadded render, which narrates exactly as it did before this step existed:
    lines pushed later by `mux.plan_timeline` and a shorter video. Worse pacing,
    still a working session.
    """
    beats = sorted(
        (pad.Beat(beat_id, start, end) for beat_id, (start, end) in spans.items()),
        key=lambda beat: beat.start,
    )
    holds = pad.plan_holds(beats, {beat_id: duration for beat_id, _, duration in clips})
    if not holds:
        return source, spans, 0.0

    added = sum(hold.seconds for hold in holds)
    shifted = pad.shift(beats, holds)
    padded = source.parent / "padded.staging.mp4"
    try:
        await pad.apply(
            source,
            beats,
            holds,
            padded,
            duration_s=await mux.probe_duration_s(source),
            fps=await pad.probe_fps(source),
            timeout_s=timeout_s,
        )
    except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
        logger.warning("could not pad %s for narration: %s", session_id, exc)
        return source, spans, 0.0

    logger.info(
        "held %d beats open for narration on %s, adding %.1fs", len(holds), session_id, added
    )
    return padded, {beat.id: (beat.start, beat.end) for beat in shifted}, added


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
