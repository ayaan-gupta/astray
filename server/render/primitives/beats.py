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
    """
    start = _scene_time(scene)
    try:
        yield
    finally:
        # `finally`, so a beat whose animations raise still records its span --
        # the traceback goes to the repair loop, and any beats that completed
        # before it stay citable.
        _TIMINGS.append({"id": beat_id, "start": start, "end": _scene_time(scene)})
        _flush()


def reset() -> None:
    """Clear accumulated timings. For tests; a container renders one scene once."""
    _TIMINGS.clear()
