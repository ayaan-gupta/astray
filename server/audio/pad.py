"""Making the video long enough for the explanation, instead of the reverse.

Until now the render was fixed and the script had to fit inside it. Every beat's
word budget came from its measured duration, and a five-beat 34-second animation
budgets about seventy words -- for stating the student's rule, contradicting it,
giving the reason, walking the correct expansion, and checking it with a number.
Seventy words cannot do that, so the script did the only thing it could and wrote
a label per beat. The animation was not under-narrated; it was too short to be
explained over.

Lengthening the render is the wrong lever: an animation stretched to fill a
narration is the same picture with longer pauses inside each beat, and s6's own
runtime estimate is already clamped precisely because more seconds bought dead
air rather than more explanation. What is actually wanted is the picture holding
still while the voice finishes the sentence about it, which is what a person
does when they explain a diagram.

So this pads each beat's *tail* with its own frozen last frame, by exactly the
shortfall between how long its line takes to say and how long the beat is on
screen. The picture is untouched; the video grows only where the voice needed
room, and only at a boundary where nothing is moving.

Two consequences worth naming. Beat timings shift, so they are recomputed here
and rewritten -- a citation that seeks to `[beat:b3]` must land on b3 in the
published video, not in the render it came from. And this is the one step that
re-encodes, because the pad frames do not exist in the source; everything else in
the narration path copies the video stream untouched.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Silence left after a line before the next beat starts. Slightly more than the
# gap between clips, because this one is heard as the end of a thought.
TAIL_S = 0.45

# Not worth a segment boundary and a re-encode. Below this the line already
# effectively fits.
MIN_HOLD_S = 0.3

# A single beat holding a still frame longer than this is a symptom, not a
# feature: it means one line came back far over budget, and freezing the picture
# for a quarter of a minute is worse than the line being cut.
MAX_HOLD_S = 10.0

# Ceiling on the whole video's growth. Past this the holds are scaled down
# together rather than any one beat being singled out, so the pacing stays even.
MAX_TOTAL_HOLD_S = 40.0


@dataclass(frozen=True)
class Beat:
    """One beat's measured span in the original render."""

    id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Hold:
    """A frozen-frame pad appended to one beat's tail."""

    beat_id: str
    at_s: float  # the cut point, on the ORIGINAL timeline
    seconds: float


def plan_holds(
    beats: list[Beat],
    needs: dict[str, float],
    *,
    tail_s: float = TAIL_S,
    max_hold_s: float = MAX_HOLD_S,
    max_total_s: float = MAX_TOTAL_HOLD_S,
) -> list[Hold]:
    """How long to freeze each beat so its line finishes before the next one.

    `needs` maps beat id to the measured duration of its narration clip. A beat
    with no clip needs nothing: a line that failed to synthesise leaves silence,
    and silence does not need room made for it.

    Pure arithmetic, so the decision about how long the video becomes is testable
    without ffmpeg, audio, or a container -- the same split `mux.plan_timeline`
    uses for the same reason.
    """
    holds: list[Hold] = []
    for beat in beats:
        need = needs.get(beat.id)
        if not need:
            continue
        shortfall = (need + tail_s) - beat.duration
        if shortfall < MIN_HOLD_S:
            continue
        if shortfall > max_hold_s:
            logger.info(
                "beat %s wants %.1fs of hold; capping at %.1fs, so its line may run into the next",
                beat.id,
                shortfall,
                max_hold_s,
            )
        holds.append(Hold(beat.id, round(beat.end, 3), round(min(shortfall, max_hold_s), 3)))

    total = sum(hold.seconds for hold in holds)
    if total <= max_total_s or not holds:
        return holds

    # Every line overran. Scaling together keeps the pacing even, where capping
    # each in turn would leave the last beats unpadded and the earlier ones long.
    scale = max_total_s / total
    logger.warning(
        "narration wants %.1fs of hold across %d beats; scaling to %.1fs",
        total,
        len(holds),
        max_total_s,
    )
    return [
        Hold(hold.beat_id, hold.at_s, round(hold.seconds * scale, 3))
        for hold in holds
        if hold.seconds * scale >= MIN_HOLD_S
    ]


def shift(beats: list[Beat], holds: list[Hold]) -> list[Beat]:
    """Where each beat lands once the holds are inserted.

    A beat's own hold sits at its end, so it extends that beat rather than
    delaying it: `start` moves by the holds of every *earlier* beat, `end` by
    those plus its own. This is what keeps a citation pointing at the same
    picture it pointed at before the pad.
    """
    by_id = {hold.beat_id: hold.seconds for hold in holds}
    out: list[Beat] = []
    offset = 0.0
    for beat in beats:
        own = by_id.get(beat.id, 0.0)
        out.append(Beat(beat.id, round(beat.start + offset, 3), round(beat.end + offset + own, 3)))
        offset += own
    return out


def build_filter(beats: list[Beat], holds: list[Hold], duration_s: float, fps: float) -> str:
    """The filter_complex that cuts, freezes and re-joins the video.

    One chain per segment: trim it out, rebase its timestamps, and where a hold
    belongs, clone the last frame for that long. `fps` is pinned on every chain
    because `concat` requires its inputs to agree, and a segment boundary that
    lands between frames can otherwise leave one chain a frame short.

    Segments are cut at beat *ends*, and the first runs from zero rather than from
    the first beat's start, so anything the scene drew outside a beat is carried
    through rather than silently dropped.
    """
    by_id = {hold.beat_id: hold.seconds for hold in holds}
    parts: list[str] = []
    labels: list[str] = []
    previous = 0.0

    for beat in beats:
        chain = f"[0:v]trim=start={previous:.3f}:end={beat.end:.3f},setpts=PTS-STARTPTS,fps={fps}"
        hold = by_id.get(beat.id)
        if hold:
            chain += f",tpad=stop_mode=clone:stop_duration={hold:.3f}"
        label = f"v{len(labels)}"
        parts.append(f"{chain}[{label}]")
        labels.append(label)
        previous = beat.end

    # Whatever the scene did after its last beat. A closing FadeOut lives here.
    if duration_s - previous > 1.0 / fps:
        label = f"v{len(labels)}"
        parts.append(f"[0:v]trim=start={previous:.3f},setpts=PTS-STARTPTS,fps={fps}[{label}]")
        labels.append(label)

    joined = "".join(f"[{label}]" for label in labels)
    parts.append(f"{joined}concat=n={len(labels)}:v=1:a=0[padded]")
    return ";".join(parts)


def build_command(
    video: Path, beats: list[Beat], holds: list[Hold], out: Path, duration_s: float, fps: float
) -> list[str]:
    """The full ffmpeg argv. Separate from running it so tests can assert on it."""
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-filter_complex",
        build_filter(beats, holds, duration_s, fps),
        "-map",
        "[padded]",
        # The one re-encode in the narration path: the frozen frames are new
        # pictures, so there is no stream to copy. veryfast at crf 20 is visually
        # indistinguishable on 480p line art and costs seconds, not minutes.
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        # A keyframe about every second. Not a quality setting -- a seeking one.
        # x264 defaults to a 250-frame GOP, which at this render's 15fps is one
        # keyframe every 16.7 seconds: the whole video came back with three. Every
        # chip on the beat rail seeks to an arbitrary timestamp, so a sparse GOP
        # makes each of those seeks decode from up to 16 seconds earlier. The
        # untouched manim render carries ~25 keyframes over the same footage, and
        # this keeps that property rather than quietly spending it on file size.
        "-g",
        str(max(1, round(fps))),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out),
    ]


async def probe_fps(path: Path, *, timeout_s: int = 30) -> float:
    """The video's frame rate, as a float. Falls back to 30 if unreadable."""
    code, stdout, _ = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        timeout_s,
    )
    if code != 0:
        return 30.0
    try:
        raw = json.loads(stdout)["streams"][0]["r_frame_rate"]
        numerator, _, denominator = raw.partition("/")
        rate = float(numerator) / float(denominator or 1)
        return rate if rate > 0 else 30.0
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        # A frame rate we cannot read is not worth failing a render over; 30 is
        # only used to size sub-frame boundaries.
        return 30.0


async def _run(cmd: list[str], timeout_s: int) -> tuple[int, str, str]:
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


async def apply(
    video: Path,
    beats: list[Beat],
    holds: list[Hold],
    out: Path,
    *,
    duration_s: float,
    fps: float,
    timeout_s: int = 300,
) -> None:
    """Write `video` with its beats held open to `out`. Raises on failure."""
    if not holds:
        raise ValueError("no holds to apply")
    out.parent.mkdir(parents=True, exist_ok=True)
    code, _, stderr = await _run(
        build_command(video, beats, holds, out, duration_s, fps), timeout_s
    )
    if code != 0:
        raise RuntimeError(f"ffmpeg padding failed: {stderr}")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("ffmpeg reported success but produced no padded video")
