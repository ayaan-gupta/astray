"""One `docker run` per render, with the sandbox flags the spec mandates.

Layer one of two (the static validator is the other). The code being executed
was authored by a model whose prompt chain starts with untrusted student text,
so this container is treated as hostile-by-default:

    --network=none      no egress; a payload cannot phone home or fetch a stage 2
    --read-only         root filesystem immutable; only tmpfs and /out are writable
    --user <non-root>   no root inside the namespace
    --memory=2g --cpus=2 --pids-limit=256   resource ceilings; fork bombs die
    timeout 300s        hard wall-clock kill, enforced twice (see below)

The wall clock is enforced in two places on purpose: `timeout` inside the
container ends a hung manim process, and an outer `asyncio.wait_for` plus
`docker kill` ends a container that ignores it (a wedged runtime, a process that
traps signals). Relying on the inner one alone would let a stuck container hold
a slot in the run forever.

**Host path constraint, found by measurement, not documentation.** Under Docker
Desktop/colima the daemon runs in a VM that only shares certain host paths --
`$HOME` among them, `/private/tmp` not. A bind mount of an unshared path does
not fail: it silently presents an EMPTY directory inside the container, and
manim reports `FileNotFoundError: /work/scene.py not found`, which reads like a
codegen bug rather than a mount problem. The render workspace therefore lives
under `settings.media_root` (inside the project, under $HOME) and never in the
system temp directory. Single-file bind mounts are avoided for the same family
of reasons -- under colima one resolved to a directory.
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from server.charter.contracts import BeatTiming, RenderResult

logger = logging.getLogger(__name__)

IMAGE = "manimcommunity/manim:stable"
SCENE_FILENAME = "scene.py"

# Quality flag: -ql is 480p15. A misconception animation is a diagram, not a
# film, and 480p renders several times faster -- latency is the scarce resource
# here (see the chain's progress events), not resolution.
QUALITY_FLAG = "-ql"
_RESOLUTION_DIR = "480p15"

# Bounded so a runaway manim traceback cannot become an unbounded DB row or an
# unbounded repair prompt. The tail is what carries the actual error.
MAX_LOG_CHARS = 8_000


@dataclass(frozen=True)
class RenderPaths:
    """Filesystem layout for one render attempt."""

    root: Path  # media/<session_id>/attempt-<n>
    work: Path  # mounted read-only at /work
    out: Path  # mounted read-write at /out

    @property
    def video(self) -> Path:
        return self.out / "media" / "videos" / "scene" / _RESOLUTION_DIR / "video.mp4"

    @property
    def manifest(self) -> Path:
        return self.out / "manifest.json"

    @property
    def log(self) -> Path:
        return self.root / "render.log"


def prepare(media_root: Path, session_id: str, code: str, attempt: int = 1) -> RenderPaths:
    """Lay out the workspace for one attempt and copy the primitives in.

    The primitives are COPIED rather than bind-mounted from the source tree so
    that a render cannot be affected by an edit to the source tree mid-run, and
    so the whole workspace is one self-contained directory to mount and to
    inspect afterwards when a render is being debugged.
    """
    root = Path(media_root) / session_id / f"attempt-{attempt}"
    work = root / "work"
    out = root / "out"
    if root.exists():
        shutil.rmtree(root)
    work.mkdir(parents=True)
    out.mkdir(parents=True)

    primitives_src = Path(__file__).parent / "primitives"
    shutil.copytree(primitives_src, work / "primitives")
    # __pycache__ from a host-side test run would be stale bytecode for a
    # different Python than the container's.
    for cached in (work / "primitives").rglob("__pycache__"):
        shutil.rmtree(cached, ignore_errors=True)

    (work / SCENE_FILENAME).write_text(code)
    return RenderPaths(root=root, work=work, out=out)


def docker_argv(paths: RenderPaths, scene_class_name: str, timeout_s: int) -> list[str]:
    """The exact `docker run` invocation. Separated out so tests can assert the
    sandbox flags are present without running a container."""
    return [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--user",
        "1000:1000",
        "--memory=2g",
        "--cpus=2",
        "--pids-limit=256",
        "--security-opt",
        "no-new-privileges",
        # manim writes caches and TeX intermediates outside /out; with
        # --read-only these must be tmpfs or the render dies on a read-only FS.
        "--tmpfs",
        "/tmp:rw,exec,size=512m",
        "--tmpfs",
        "/manim:rw,exec,size=512m",
        "--tmpfs",
        "/home/manimuser:rw,exec,size=128m",
        "-v",
        f"{paths.work.resolve()}:/work:ro",
        "-v",
        f"{paths.out.resolve()}:/out",
        "-w",
        "/work",
        "-e",
        "PYTHONPATH=/work",
        "-e",
        "ASTRAY_OUT_DIR=/out",
        "-e",
        "HOME=/home/manimuser",
        # Fontconfig writes its cache to $XDG_CACHE_HOME (or ~/.cache) and, with
        # --read-only, emitted "No writable cache directories" once per font
        # lookup: twenty identical lines at the head of every render log, ahead of
        # the tail that actually carries a traceback. Harmless, but it is the log
        # a failed render gets diagnosed from, and noise there costs real time.
        "-e",
        "XDG_CACHE_HOME=/tmp/cache",
        IMAGE,
        "timeout",
        str(timeout_s),
        "manim",
        QUALITY_FLAG,
        "--media_dir",
        "/out/media",
        "-o",
        "video.mp4",
        f"/work/{SCENE_FILENAME}",
        scene_class_name,
    ]


def read_manifest(paths: RenderPaths) -> list[BeatTiming]:
    """Parse the container-written manifest into measured beat timings.

    A missing or malformed manifest yields an empty list rather than raising: the
    video may still be fine, and an ungrounded video is a degraded result, not a
    failed one. The chain records which beats got timings, so the beat rail can
    show the rest as untimed.
    """
    try:
        raw = json.loads(paths.manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    timings = []
    for entry in raw.get("beats", []) if isinstance(raw, dict) else []:
        try:
            timings.append(BeatTiming.model_validate(entry))
        except Exception:
            # One malformed entry must not discard the beats that are well-formed.
            continue
    return timings


async def run(
    paths: RenderPaths,
    scene_class_name: str,
    *,
    timeout_s: int = 300,
    mode: str = "generated",
) -> RenderResult:
    """Execute one render. Never raises -- a failure is a RenderResult."""
    argv = docker_argv(paths, scene_class_name, timeout_s)
    loop = asyncio.get_running_loop()
    started = loop.time()

    try:
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except (OSError, FileNotFoundError) as exc:
        # Docker missing or not running: a deployment problem, not a bad scene.
        # Distinguished in the text so it is not mistaken for a codegen failure
        # and fed pointlessly into the repair loop.
        return RenderResult(
            ok=False, mode=mode, error_text=f"docker unavailable: {exc}", duration_s=0.0
        )

    try:
        # Outer bound: `timeout` inside the container handles a hung manim, this
        # handles a container that ignores it. +15s so the inner one wins when
        # both would fire, giving a cleaner error.
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_s + 15)
        output = stdout.decode(errors="replace")
    except TimeoutError:
        process.kill()
        await process.wait()
        output = f"render exceeded {timeout_s}s wall clock and was killed"
        logger.warning("render timeout for %s", paths.root)

    duration = loop.time() - started
    tail = output[-MAX_LOG_CHARS:]
    try:
        paths.log.write_text(output)
    except OSError:
        pass

    if not paths.video.exists():
        return RenderResult(ok=False, mode=mode, error_text=tail, duration_s=duration)

    return RenderResult(
        ok=True,
        mode=mode,
        video_path=str(paths.video),
        timings=read_manifest(paths),
        duration_s=duration,
    )
