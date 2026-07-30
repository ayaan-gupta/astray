"""The `beat()` context manager -- the grounding contract, container side.

This module runs INSIDE the render container, imported by LLM-authored scene
code as `from primitives.beats import beat`. Nothing here may import anything
outside the standard library and manim: the container runs with `--network=none`
and a read-only filesystem, and `server/render/validator.py` rejects any import
outside its allow-list before this code is ever reached.

Why a measured manifest rather than the storyboard's estimates: chat cites
`[beat:b3]` and the client turns that into a chip that seeks the player to a
timestamp. An estimated timestamp that drifts by two seconds points the student
at the wrong moment, which is worse than not citing at all -- it makes the tutor
look like it is describing a different video. `beat()` therefore brackets each
section with the renderer's own clock and writes what actually happened.

The manifest is rewritten on every beat exit rather than once at the end. A
render that dies partway (a Manim error inside beat 5 of 6) still leaves the
first four beats' real timings on disk, so the session degrades to a partially
grounded video instead of an ungrounded one.
"""

import json
import os
from contextlib import contextmanager

# /out is the only writable mount in the container (tmpfs); everything else is
# read-only. Overridable for host-side unit tests, which have no /out.
MANIFEST_DIR = os.environ.get("ASTRAY_OUT_DIR", "/out")
MANIFEST_PATH = os.path.join(MANIFEST_DIR, "manifest.json")

_TIMINGS: list[dict] = []

# The shortest a beat is allowed to be, held open on exit if its animations
# finished sooner. This is a guarantee rather than a request, for the same reason
# `clear_frame` lives in the primitives: the s7 prompt asks for pacing, and a live
# render came back with no `self.wait()` anywhere in the file, six beats, and a
# total runtime of 12.5s against a storyboard estimate of 90. One beat lasted
# 0.8 seconds.
#
# Two things break at that length, and neither is cosmetic. A citation seeking to
# a beat 0.8s wide points at a moment gone before the student can look at it, so
# the grounding contract stops meaning anything. And the narration budget is
# computed from measured duration, so a 0.8s beat gets a three-word line -- which
# is exactly the "random four- or five-word phrases" failure the narration work
# was meant to end. At 5.0s a beat holds about ten spoken words and is long enough
# to seek to.
MIN_BEAT_S = 5.0


def _scene_time(scene) -> float:
    """Elapsed animation time, from the renderer clock.

    `Scene.time` is the documented accessor in Manim CE 0.20; the
    `renderer.time` fallback covers renderer backends that do not surface it,
    and 0.0 covers a scene that has not animated yet. Never raises -- a timing
    helper that can abort a render would trade a grounded video for no video.
    """
    for source in (scene, getattr(scene, "renderer", None)):
        value = getattr(source, "time", None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _flush() -> None:
    try:
        os.makedirs(MANIFEST_DIR, exist_ok=True)
        with open(MANIFEST_PATH, "w") as handle:
            json.dump({"beats": _TIMINGS}, handle)
    except OSError:
        # A manifest we cannot write is a degraded render, not a failed one:
        # the video is still correct and still worth showing. The host treats a
        # missing/partial manifest as "beats untimed" (see runner.py).
        pass


@contextmanager
def beat(scene, beat_id: str):
    """Bracket one storyboard beat, recording its measured start and end.

    Usage inside a generated scene::

        with beat(self, "b3"):
            self.play(Write(formula))

    `s8_validate` enforces that every planned beat id appears exactly once in a
    `with beat(...)` block, so this is a contract the pipeline checks statically
    before the container ever runs, not a convention the model is asked to honour.

    On a clean exit the beat is held open to `MIN_BEAT_S`, so no beat is too short
    to seek to or to narrate over.
    """
    start = _scene_time(scene)
    try:
        yield
    except BaseException:
        # A beat whose animations raise still records its span: the traceback goes
        # to the repair loop, and beats that completed before it stay citable. No
        # hold here -- padding a beat that just died would only delay the failure.
        _record(scene, beat_id, start)
        raise
    _hold(scene, start)
    _record(scene, beat_id, start)


def _hold(scene, start: float) -> None:
    """Wait out the remainder of `MIN_BEAT_S`, if the beat ran short."""
    remaining = MIN_BEAT_S - (_scene_time(scene) - start)
    if remaining <= 0.05:
        return
    try:
        scene.wait(remaining)
    except Exception:
        # Pacing is a nicety; a video is not. Never let the hold be what fails a
        # render that had already drawn its beat successfully.
        pass


def _record(scene, beat_id: str, start: float) -> None:
    _TIMINGS.append({"id": beat_id, "start": start, "end": _scene_time(scene)})
    _flush()


def reset() -> None:
    """Clear accumulated timings. For tests; a container renders one scene once."""
    _TIMINGS.clear()
