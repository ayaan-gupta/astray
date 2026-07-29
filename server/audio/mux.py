"""Placing narration clips on the video's timeline and muxing them in.

Two jobs, deliberately separated. `plan_timeline` is pure arithmetic over
measured beat starts and real clip durations, so the part that decides *when*
each line is heard is testable without ffmpeg, audio, or a container. `mux_in`
is the thin shell that hands that plan to ffmpeg.

The rule the timeline enforces is that narration never overlaps narration. A
clip that runs past its beat pushes the next one later rather than talking over
it: two voices at once is unintelligible, whereas a line arriving a fraction late
is barely noticeable and self-corrects at the next beat with spare room. Speeding
audio up to fit was the other option and is rejected on purpose, because
time-stretched speech is exactly the artificial sound this feature exists to
avoid.
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Breathing room between consecutive lines, so a pushed-back clip does not begin
# on the same frame the previous one ends.
GAP_S = 0.18

# Drift past this is worth a log line: it means the script consistently outran
# its budget and the narration is no longer tracking the picture.
DRIFT_WARN_S = 1.5


@dataclass(frozen=True)
class Placement:
    beat_id: str
    path: Path
    at_s: float
    duration_s: float
    drift_s: float


def plan_timeline(
    clips: list[tuple[str, Path, float]],
    starts: dict[str, float],
    *,
    gap_s: float = GAP_S,
    video_duration_s: float | None = None,
) -> list[Placement]:
    """Lay clips out at their beats' measured starts, never overlapping.

    `clips` is `[(beat_id, path, duration_s), ...]` in beat order and `starts`
    maps beat id to its measured start. A clip whose beat start falls before the
    previous clip has finished is pushed to just after it, and the gap it was
    pushed by is reported as `drift_s` rather than hidden.

    Passing `video_duration_s` also handles the one case the push-later rule
    cannot: the final clip. Every other overrun is absorbed by the beat after it,
    but a last line running past the end of the video gets its final words cut
    off by `-shortest`, and losing the end of a sentence is the most audible
    failure this whole module has. So the last clip is pulled *earlier* instead,
    far enough to fit and no further than the previous clip allows. The final beat
    is a summary, so starting it a moment early is imperceptible next to having it
    truncated mid-word.
    """
    placements: list[Placement] = []
    cursor = 0.0
    for beat_id, path, duration in clips:
        start = starts[beat_id]
        at = max(start, cursor)
        placements.append(
            Placement(
                beat_id=beat_id,
                path=path,
                at_s=round(at, 3),
                duration_s=round(duration, 3),
                drift_s=round(at - start, 3),
            )
        )
        cursor = at + duration + gap_s

    if video_duration_s is not None and placements:
        placements[-1] = _fit_last(placements, video_duration_s, gap_s)

    worst = max((abs(p.drift_s) for p in placements), default=0.0)
    if worst > DRIFT_WARN_S:
        logger.warning(
            "narration is %.1fs out of step with its beat; the script overran its budget", worst
        )
    return placements


def _fit_last(placements: list[Placement], video_duration_s: float, gap_s: float) -> Placement:
    """Pull the final clip earlier if it would otherwise be cut off by the video end."""
    last = placements[-1]
    overrun = (last.at_s + last.duration_s) - video_duration_s
    if overrun <= 0:
        return last

    floor = 0.0
    if len(placements) > 1:
        previous = placements[-2]
        floor = previous.at_s + previous.duration_s + gap_s
    at = max(floor, last.at_s - overrun)
    if at >= last.at_s:
        # Nowhere to move it to. Better to report this than to pretend it fits:
        # the tail will be trimmed and the line will end mid-word.
        logger.warning(
            "narration for %s overruns the video by %.2fs and cannot be moved earlier",
            last.beat_id,
            overrun,
        )
        return last

    logger.info(
        "pulled narration for %s %.2fs earlier so its ending is not trimmed",
        last.beat_id,
        last.at_s - at,
    )
    return Placement(
        beat_id=last.beat_id,
        path=last.path,
        at_s=round(at, 3),
        duration_s=last.duration_s,
        drift_s=round(at - (last.at_s - last.drift_s), 3),
    )


def build_filter(placements: list[Placement]) -> str:
    """The ffmpeg filter_complex that delays each clip and mixes them into one track.

    `amix` normalises by default, which divides every input's volume by the number
    of inputs and would leave a six-beat narration almost inaudible. Since the
    timeline guarantees no two clips overlap, `normalize=0` is both safe and
    required.
    """
    parts = []
    for index, placement in enumerate(placements, start=1):
        delay_ms = int(round(placement.at_s * 1000))
        # `all=1` applies the delay to every channel; without it only the first
        # channel is delayed and a stereo clip arrives smeared.
        parts.append(f"[{index}:a]adelay={delay_ms}:all=1[d{index}]")
    labels = "".join(f"[d{i}]" for i in range(1, len(placements) + 1))
    # `apad` matters more than it looks. The mix ends when the last clip does,
    # which is normally before the video ends, and `-shortest` then trims the
    # *video* down to the audio and silently cuts the final beat off the
    # animation. Padding the audio with silence makes the video the shorter
    # stream again, so -shortest trims the padding instead of the picture.
    parts.append(f"{labels}amix=inputs={len(placements)}:normalize=0,apad[mixed]")
    return ";".join(parts)


def build_command(video: Path, placements: list[Placement], out: Path) -> list[str]:
    """The full ffmpeg argv. Separate from running it so tests can assert on it."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video)]
    for placement in placements:
        cmd += ["-i", str(placement.path)]
    cmd += [
        "-filter_complex",
        build_filter(placements),
        "-map",
        "0:v",
        "-map",
        "[mixed]",
        # The render is already the picture we want. Re-encoding it would cost
        # minutes and lose quality for no reason.
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        # Without this the output runs as long as the audio mix, which can exceed
        # the video when a final line overruns, leaving a frozen last frame.
        "-shortest",
        str(out),
    ]
    return cmd


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


async def _run(cmd: list[str], timeout_s: int) -> tuple[int, str, str]:
    """Run a command, returning (returncode, stdout, stderr).

    Both streams, because the two callers here need different ones: ffprobe
    reports its answer on stdout while ffmpeg reports its failures on stderr.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    return (
        process.returncode or 0,
        stdout.decode("utf-8", "replace"),
        stderr.decode("utf-8", "replace")[:2000],
    )


async def probe_duration_s(path: Path, *, timeout_s: int = 30) -> float:
    """Measured duration of an encoded clip.

    Measured rather than estimated from its character count, because that
    estimate is exactly what the word budget already was: reading the real length
    is the only way to find out whether the budget held.
    """
    code, stdout, stderr = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        timeout_s,
    )
    if code != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {stderr}")
    return _parse_duration(stdout)


def _parse_duration(payload: str) -> float:
    try:
        return float(json.loads(payload)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"could not read a duration from ffprobe output: {exc}") from exc


async def mux_in(
    video: Path, placements: list[Placement], out: Path, *, timeout_s: int = 180
) -> None:
    """Write `video` plus the placed narration to `out`. Raises on failure."""
    if not placements:
        raise ValueError("no narration to mux")
    out.parent.mkdir(parents=True, exist_ok=True)
    code, _, stderr = await _run(build_command(video, placements, out), timeout_s)
    if code != 0:
        raise RuntimeError(f"ffmpeg mux failed: {stderr}")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("ffmpeg reported success but produced no output file")
