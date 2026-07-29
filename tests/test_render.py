"""Validator, runner invocation, fallback renderer, and beat/render persistence.

No container is started here -- `docker_argv` is separated from `run` precisely
so the sandbox flags can be asserted without a 2GB image and a real render. The
container's actual behaviour (network blocked, read-only root, non-root uid,
measured beat manifest) was verified against the live image; these tests pin the
invocation that produces it.
"""

import ast

import pytest

from server.charter.contracts import Beat, BeatTiming, MathContent, Storyboard
from server.render import runner
from server.render import storyboard as fallback
from server.render.validator import validate, validate_scene
from server.store import repo
from server.store.db import connect


def _storyboard(ids=("b1", "b2"), targets=("b1",)) -> Storyboard:
    return Storyboard(
        beats=[
            Beat(
                id=i,
                title=f"title {i}",
                teaching_purpose="p",
                on_screen="o",
                targets_misconception=i in targets,
                primitive="algebra_steps",
            )
            for i in ids
        ],
        total_estimated_seconds=45,
    )


def _scene(body: str) -> str:
    return (
        "from manim import *\n"
        "from primitives.beats import beat\n"
        "class AstrayScene(Scene):\n"
        "    def construct(self):\n" + body
    )


CLEAN_BODY = (
    '        with beat(self, "b1"): self.wait(1)\n        with beat(self, "b2"): self.wait(1)\n'
)


def test_clean_scene_passes():
    assert validate(_scene(CLEAN_BODY), _storyboard(), "AstrayScene").ok


@pytest.mark.parametrize(
    "code,kind",
    [
        ("import os\n" + _scene(CLEAN_BODY), "import"),
        ("from subprocess import run\n" + _scene(CLEAN_BODY), "import"),
        (_scene('        eval("1")\n' + CLEAN_BODY), "name"),
        (_scene("        x = ().__class__.__bases__\n" + CLEAN_BODY), "name"),
        (_scene("        f = open('/etc/passwd')\n" + CLEAN_BODY), "name"),
        ("class AstrayScene(Scene:\n    pass", "syntax"),
    ],
    ids=["import-os", "from-subprocess", "eval", "dunder-escape", "open", "syntax-error"],
)
def test_hostile_code_is_rejected(code, kind):
    report = validate(code, _storyboard(), "AstrayScene")
    assert not report.ok
    assert any(issue.kind == kind for issue in report.issues), report.failure_text()


def test_missing_beat_fails_the_gate():
    """Grounding cannot degrade silently: an unwrapped beat is a citation that
    would seek nowhere."""
    body = '        with beat(self, "b1"): self.wait(1)\n'
    report = validate(_scene(body), _storyboard(), "AstrayScene")
    assert not report.ok
    assert "b2" in report.failure_text()


def test_duplicate_beat_fails_the_gate():
    """A repeated id makes every citation to it ambiguous between two moments."""
    body = '        with beat(self, "b1"): self.wait(1)\n' * 2 + (
        '        with beat(self, "b2"): self.wait(1)\n'
    )
    report = validate(_scene(body), _storyboard(), "AstrayScene")
    assert not report.ok
    assert "wrapped 2 times" in report.failure_text()


def test_computed_beat_id_cannot_satisfy_coverage():
    """A beat id built at runtime cannot be checked statically, which is the
    whole point of checking before the container runs."""
    body = '        n = "b1"\n        with beat(self, n): self.wait(1)\n'
    report = validate(_scene(body), _storyboard(), "AstrayScene")
    assert not report.ok


def test_docker_argv_carries_every_sandbox_flag(tmp_path):
    paths = runner.prepare(tmp_path, "s1", _scene(CLEAN_BODY))
    argv = runner.docker_argv(paths, "AstrayScene", 300)
    joined = " ".join(argv)
    for flag in (
        "--network=none",
        "--read-only",
        "--pids-limit=256",
        "--memory=2g",
        "--cpus=2",
        "no-new-privileges",
    ):
        assert flag in joined, flag
    assert "--user" in argv and "1000:1000" in argv
    assert f"{paths.work.resolve()}:/work:ro" in joined, "work mount must be read-only"
    assert "timeout" in argv and "300" in argv


def test_prepare_copies_primitives_and_writes_scene(tmp_path):
    paths = runner.prepare(tmp_path, "s1", "print('x')")
    assert (paths.work / "primitives" / "beats.py").exists()
    assert (paths.work / "scene.py").read_text() == "print('x')"


def test_missing_manifest_yields_no_timings(tmp_path):
    """An ungrounded video is degraded, not failed -- never an exception."""
    paths = runner.prepare(tmp_path, "s1", "x = 1")
    assert runner.read_manifest(paths) == []


def test_malformed_manifest_entry_does_not_discard_good_ones(tmp_path):
    paths = runner.prepare(tmp_path, "s1", "x = 1")
    paths.manifest.write_text('{"beats": [{"id": "b1", "start": 0, "end": 2}, {"broken": true}]}')
    timings = runner.read_manifest(paths)
    assert [t.id for t in timings] == ["b1"]


def test_fallback_scene_passes_its_own_validator():
    """The fallback must clear the same gate as generated code -- it is not a
    trusted path, and it must stay grounded."""
    board = _storyboard(("b1", "b2", "b3"), ("b2",))
    math = MathContent(
        worked_example=["a", "b"], counter_example=["c"], key_identity="k", concrete_numbers=[]
    )
    scene = fallback.build_scene(board, math)
    assert validate_scene(scene, board).ok
    assert scene.beats_covered == ["b1", "b2", "b3"]


def test_fallback_neutralizes_injection_through_math_content():
    """Math content reaches us through a model; the fallback builds Python source
    from it by string assembly, so a naive quote would hand code to the container."""
    board = _storyboard(("b1", "b2"), ("b2",))
    math = MathContent(
        worked_example=["x'); import os; os.system('touch /tmp/pwn"],
        counter_example=["c"],
        key_identity="k",
        concrete_numbers=[],
    )
    scene = fallback.build_scene(board, math)
    tree = ast.parse(scene.code)
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "os" not in imported, "payload became a real import"
    assert validate_scene(scene, board).ok


def test_beats_persist_planned_then_timed(tmp_path):
    """s6 writes the plan before any render exists; the render fills timings in.
    Untimed beats are a real UI state (greyed on the rail), not a placeholder."""
    conn = connect(tmp_path / "t.db")
    from server.charter.contracts import StudentSubmission

    sid = repo.create_session(
        conn, handle="h", submission=StudentSubmission(problem="p", source="typed")
    )
    board = _storyboard(("b1", "b2"), ("b1",))
    repo.save_beats(conn, sid, board)

    rows = repo.list_beats(conn, sid)
    assert [r["beat_id"] for r in rows] == ["b1", "b2"]
    assert all(r["start_s"] is None for r in rows)
    assert rows[0]["targets_misconception"] == 1

    updated = repo.save_beat_timings(
        conn,
        sid,
        [BeatTiming(id="b1", start=0.0, end=2.5), BeatTiming(id="b2", start=2.5, end=6.0)],
    )
    assert updated == 2
    rows = repo.list_beats(conn, sid)
    assert rows[0]["start_s"] == 0.0 and rows[1]["end_s"] == 6.0


def test_timings_for_unplanned_beats_are_discarded(tmp_path):
    """An unplanned beat has no title or purpose; inserting one would put a blank,
    unciteable segment on the rail."""
    conn = connect(tmp_path / "t.db")
    from server.charter.contracts import StudentSubmission

    sid = repo.create_session(
        conn, handle="h", submission=StudentSubmission(problem="p", source="typed")
    )
    repo.save_beats(conn, sid, _storyboard(("b1",), ("b1",)))
    updated = repo.save_beat_timings(conn, sid, [BeatTiming(id="b9", start=0.0, end=1.0)])
    assert updated == 0
    assert len(repo.list_beats(conn, sid)) == 1


def test_replanning_beats_preserves_measured_timings(tmp_path):
    """A re-plan must not wipe timings already measured for beats it keeps."""
    conn = connect(tmp_path / "t.db")
    from server.charter.contracts import StudentSubmission

    sid = repo.create_session(
        conn, handle="h", submission=StudentSubmission(problem="p", source="typed")
    )
    repo.save_beats(conn, sid, _storyboard(("b1", "b2"), ("b1",)))
    repo.save_beat_timings(conn, sid, [BeatTiming(id="b1", start=1.0, end=2.0)])
    repo.save_beats(conn, sid, _storyboard(("b1", "b2"), ("b1",)))
    assert repo.list_beats(conn, sid)[0]["start_s"] == 1.0


def test_render_ledger_records_every_attempt(tmp_path):
    """Failed attempts are kept: the fallback rate is only measurable if the
    failures that led to it are in the data."""
    conn = connect(tmp_path / "t.db")
    from server.charter.contracts import StudentSubmission

    sid = repo.create_session(
        conn, handle="h", submission=StudentSubmission(problem="p", source="typed")
    )
    repo.record_render(conn, session_id=sid, attempt=1, status="failed", error_text="boom")
    repo.record_render(
        conn,
        session_id=sid,
        attempt=2,
        status="ok",
        video_path="/v.mp4",
        mode="storyboard_fallback",
    )
    rows = conn.execute("SELECT * FROM renders WHERE session_id = ? ORDER BY id", (sid,)).fetchall()
    assert [r["status"] for r in rows] == ["failed", "ok"]
    latest = repo.latest_render(conn, sid)
    assert latest["mode"] == "storyboard_fallback" and latest["video_path"] == "/v.mp4"
