"""The container-side primitives: their numeric rules, and their import discipline.

Two things are testable here without Docker or manim.

`server/render/primitives/sampling.py` holds the plotting decisions and imports
`math` alone, so it runs on the host. Those rules are where a plausible-looking
choice is quietly wrong on real input, which is why they were pulled out of
`graph.py` rather than left beside the drawing code.

The rest of the package cannot be imported here at all -- `manim` lives only in
the render image -- but it can still be parsed, and one property is worth
asserting statically: the primitives are mounted into the sandbox and imported by
generated code, and `validator.py` checks the *generated scene* rather than the
package it imports. A primitive reaching for `os` would therefore pass every
existing gate. `test_no_primitive_imports_outside_the_allow_list` is that gate.

Drawing itself is verified by rendering: `scripts/check_primitives.py` runs every
helper through the real container and writes one frame per beat to look at.
"""

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest

from server.audio.speech import budget_words
from server.render.runner import prepare
from server.render.validator import ALLOWED_IMPORT_ROOTS

PRIMITIVES_DIR = Path("server/render/primitives")


def load_primitive(name: str):
    """Load one primitive module by path, deliberately bypassing the package.

    `primitives/__init__.py` re-exports with container-style absolute imports
    (`from primitives.layout import ...`), which resolve only inside the render
    container where `/work` is on the path. Importing
    `server.render.primitives.sampling` would execute that `__init__` and fail on
    `manim`. Loading the two modules that depend on nothing but the standard
    library keeps this test honest about what the package is: container code that
    happens to live in the source tree.

    Loaded fresh on every call, because `beats.py` reads `ASTRAY_OUT_DIR` at import
    time and accumulates timings in a module global.
    """
    spec = importlib.util.spec_from_file_location(f"astray_{name}", PRIMITIVES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampling = load_primitive("sampling")
_prose_module = load_primitive("prose")
prose = _prose_module.prose
mathify = _prose_module.mathify
MIN_GAP_SAMPLES = sampling.MIN_GAP_SAMPLES
SAMPLES = sampling.SAMPLES
defined_range = sampling.defined_range
guard = sampling.guard
decimal_places = sampling.decimal_places
tick_step = sampling.tick_step
y_window = sampling.y_window


def _modules() -> list[Path]:
    return sorted(PRIMITIVES_DIR.glob("*.py"))


def _import_roots(module: Path) -> set[str]:
    """Top-level module names `module` imports, by either import form.

    Both forms matter: `contextlib` reaches `beats.py` as `from contextlib import
    contextmanager`, so a check that walked only `ast.Import` would report it as
    unused and pass an incomplete picture of what the sandbox loads.
    """
    roots = set()
    for node in ast.walk(ast.parse(module.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.level, f"{module.name}: relative import"
            roots.add((node.module or "").split(".")[0])
    return roots


def test_primitives_directory_is_not_empty():
    """Guards the two tests below: a bad glob would make both vacuously pass."""
    assert len(_modules()) >= 8


# Imports a primitive may hold beyond what generated code is allowed, each one
# reviewed. The primitives are the trusted layer, so this is not the same list the
# validator enforces on model-authored scenes -- but it is enumerated rather than
# waived, so adding an import to this package is a deliberate act with a reason
# attached, not a silent widening of what runs in the sandbox.
VETTED_EXTRA_IMPORTS = {
    # Writes the beat manifest to /out, the container's only writable mount. This
    # is the module that makes chat citations point at real timestamps, and it
    # cannot do that without touching the filesystem. `contextlib` is what makes
    # `beat()` a `with` block, which is the shape the validator enforces.
    "beats.py": {"contextlib", "json", "os"},
    # Classifies a label as prose or as mathematics. Pure string inspection.
    "layout.py": {"re"},
    # Translates the LaTeX a model writes into a caption. Pure string rewriting.
    "prose.py": {"re"},
    # Reads a number out of a side label, so a square labelled `a = 1`, `b = 3` is
    # drawn to those proportions instead of contradicting them.
    "area.py": {"re"},
}


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_no_primitive_imports_outside_the_allow_list(module: Path):
    """A primitive imports only what generated code may, plus a reviewed exception.

    The static validator inspects the generated scene, never the package it
    imports, so an import added here is seen by no existing gate. The container is
    still the second layer, but neither layer is trusted alone -- and an import
    that appears in this package without a line in `VETTED_EXTRA_IMPORTS` should
    fail a test rather than reach the sandbox unremarked.
    """
    allowed = ALLOWED_IMPORT_ROOTS | VETTED_EXTRA_IMPORTS.get(module.name, set())
    roots = _import_roots(module)
    assert roots <= allowed, f"{module.name} imports {roots - allowed}"


def test_vetted_exceptions_are_all_still_needed():
    """An exception that stops being used must stop being granted.

    Without this, `VETTED_EXTRA_IMPORTS` only ever grows: an import removed from a
    primitive leaves its waiver behind, silently pre-approving the next use.
    """
    for name, extras in VETTED_EXTRA_IMPORTS.items():
        assert extras <= _import_roots(PRIMITIVES_DIR / name), (
            f"{name} no longer imports all of {extras}"
        )


def test_prepare_copies_every_primitive_module(tmp_path):
    """A new primitive must reach the container, not just the source tree."""
    paths = prepare(tmp_path, "session", "code", attempt=1)
    copied = {p.name for p in (paths.work / "primitives").glob("*.py")}
    assert copied == {p.name for p in _modules()}


class TestTickStep:
    @pytest.mark.parametrize(
        ("span", "expected"),
        [(6.0, 1.0), (12.0, 2.0), (60.0, 10.0), (0.6, 0.1), (1.0, 0.2)],
    )
    def test_snaps_to_a_readable_multiple(self, span, expected):
        assert tick_step(span) == pytest.approx(expected)

    @pytest.mark.parametrize("span", [4.0, 12.0, 100.0, 0.5])
    def test_gives_between_four_and_eight_ticks(self, span):
        assert 4 <= span / tick_step(span) <= 8

    def test_a_degenerate_span_does_not_divide_by_zero(self):
        assert tick_step(0.0) == 1.0


class TestYWindow:
    def test_covers_an_ordinary_function(self):
        lo, hi = y_window([lambda x: x], (-2.0, 2.0))
        assert lo < -1.9 and hi > 1.9

    def test_a_pole_does_not_set_the_scale(self):
        """The trimmed window keeps the interesting part visible.

        Without trimming, one sample beside x = 0 is on the order of 10^15 and the
        whole plot flattens onto the x-axis.
        """
        lo, hi = y_window([lambda x: 1.0 / x], (-2.0, 2.0))
        assert hi < 100.0 and lo > -100.0

    def test_a_constant_function_still_gets_height(self):
        lo, hi = y_window([lambda x: 3.0], (-1.0, 1.0))
        assert hi - lo > 0.5

    def test_a_function_defined_nowhere_falls_back(self):
        lo, hi = y_window([lambda x: math.sqrt(-1 - x * x)], (-1.0, 1.0))
        assert (lo, hi) == (-1.0, 1.0)


class TestDefinedRange:
    def test_a_total_function_keeps_the_whole_window(self):
        span = defined_range(lambda x: x**2, (-3.0, 3.0))
        assert span == pytest.approx((-3.0, 3.0))

    def test_a_pole_is_bridged_rather_than_splitting_the_curve(self):
        """`1/x` must be drawn right across the window, not on one side of zero.

        Splitting here was the first implementation, and it silently discarded half
        of every curve with a pole in view.
        """
        span = defined_range(lambda x: 1.0 / x, (-2.0, 2.0))
        assert span == pytest.approx((-2.0, 2.0))

    def test_an_undefined_half_is_dropped(self):
        """`log x` on [-2, 2] exists only for x > 0 and must be drawn only there.

        Clamping instead drew a flat line along the top of the frame across the
        half where the function has no value at all.
        """
        lo, hi = defined_range(lambda x: math.log(x), (-2.0, 2.0))
        assert lo >= 0.0
        assert hi == pytest.approx(2.0)

    def test_a_leading_domain_edge_is_not_bridged(self):
        """A window opening just below a domain edge is an edge, not a pole.

        `sqrt` is undefined for only one sample of [-0.01, 4], which is shorter
        than MIN_GAP_SAMPLES; bridging by length alone would draw the curve where
        the function does not exist, so only interior runs are bridged.
        """
        assert MIN_GAP_SAMPLES > 1
        lo, _ = defined_range(lambda x: math.sqrt(x), (-0.01, 4.0))
        assert lo >= 0.0

    def test_a_function_defined_nowhere_returns_none(self):
        assert defined_range(lambda x: math.log(x), (-2.0, -1.0)) is None

    def test_the_returned_span_lies_inside_the_window(self):
        lo, hi = defined_range(lambda x: math.log(x), (-5.0, 5.0))
        assert -5.0 <= lo <= hi <= 5.0


class TestProse:
    """Captions are `Text`, which has no LaTeX, and the model writes LaTeX anyway.

    Both inputs below are verbatim from live renders, and both reached the screen.
    """

    def test_dollar_delimiters_are_removed(self):
        assert prose("The two $3y$ rectangles are missing!") == (
            "The two 3y rectangles are missing!"
        )

    def test_a_caption_full_of_maths_becomes_readable(self):
        assert prose(r"For y=1, (1+3)^2=16, but 1^2+3^2=10, so 16\neq10.") == (
            "For y=1, (1+3)²=16, but 1²+3²=10, so 16≠10."
        )

    def test_a_command_is_translated_rather_than_deleted(self):
        """Deleting it would turn a claim into its opposite-looking remains.

        "so 16 \\neq 10" with the command dropped reads "so 16 10", which is worse
        than the raw backslash: the sentence still looks finished and no longer says
        anything.
        """
        assert "≠" in prose(r"16 \neq 10")
        assert "16 10" not in prose(r"16 \neq 10")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (r"a \times b", "a × b"),
            (r"x \le 3", "x ≤ 3"),
            (r"x = \pm 4", "x = ± 4"),
            (r"\sqrt{16}", "√16"),
            (r"x^{-1}", "x⁻¹"),
            (r"x^n", "xⁿ"),
            (r"a \rightarrow b", "a → b"),
        ],
    )
    def test_common_constructs(self, raw, expected):
        assert prose(raw) == expected

    def test_a_longer_command_wins_over_its_own_prefix(self):
        r"""`\neq` must not be matched as `\ne` followed by a stray `q`."""
        assert prose(r"16 \neq 10") == "16 ≠ 10"

    def test_an_unmapped_command_is_dropped_without_its_backslash(self):
        assert "\\" not in prose(r"the \varnothing case")

    def test_ordinary_prose_is_left_alone(self):
        text = "these two are the middle term"
        assert prose(text) == text

    def test_whitespace_is_collapsed(self):
        assert prose("a   b\n c") == "a b c"


class TestMathify:
    r"""Math mode has no spaces, so prose handed to `MathTex` loses them all.

    Both broken inputs below are verbatim from a live render, which showed
    `Lety = 1` and `Thestudent'srulefails` on screen.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Let y = 1", r"\text{Let }y = 1"),
            ("Correct : (1+3)^2 = 16", r"\text{Correct }: (1+3)^2 = 16"),
        ],
    )
    def test_prose_words_are_wrapped(self, raw, expected):
        assert mathify(raw) == expected

    def test_a_whole_sentence_keeps_every_space(self):
        wrapped = mathify("The student's rule fails")
        assert wrapped.count(r"\text{") == 4
        assert "student's " in wrapped

    @pytest.mark.parametrize("expression", ["y^2 + 6y + 9", "ab", "2ab", "x = 4", "a + b"])
    def test_mathematics_is_left_alone(self, expression):
        """Short letter runs are the variables this is protecting.

        `ab` must stay a product and a lone `y` must stay italic, so only runs of
        three or more letters are wrapped.
        """
        assert mathify(expression) == expression

    @pytest.mark.parametrize("command", [r"\sin(x)", r"\log x", r"\frac{1}{2}", r"\cdot"])
    def test_latex_commands_are_not_rewrapped(self, command):
        assert mathify(command) == command

    def test_a_bare_function_name_stays_upright(self):
        """`sin` written without its backslash is still mathematics, not a word."""
        assert mathify("sin(x) + cos(x)") == "sin(x) + cos(x)"

    def test_the_trailing_space_goes_inside_the_braces(self):
        r"""`\text{Let} y` renders "Lety": outside the braces it is math mode again."""
        assert r"\text{Let }" in mathify("Let y = 1")
        assert r"\text{Let} " not in mathify("Let y = 1")

    def test_a_word_at_the_end_needs_no_trailing_space(self):
        assert mathify("total") == r"\text{total}"


class FakeScene:
    """The two things `beat()` touches on a Scene: a clock and `wait`.

    Manim's own Scene advances `time` as animations play; here only `wait` moves
    it, which is exactly what the hold is measured in.
    """

    def __init__(self) -> None:
        self.time = 0.0
        self.waits: list[float] = []

    def wait(self, duration: float) -> None:
        self.waits.append(duration)
        self.time += duration


class TestBeatHold:
    """Every beat is held to a floor, because a 0.8s beat cannot be used.

    A live render produced six beats and 12.5s of video with no `self.wait()`
    anywhere in the file. A citation into a beat that narrow points at a moment
    already gone, and the narration budget derived from it yields a three-word
    line.
    """

    @pytest.fixture
    def beats(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASTRAY_OUT_DIR", str(tmp_path))
        return load_primitive("beats")

    def test_a_short_beat_is_padded_to_the_floor(self, beats):
        scene = FakeScene()
        with beats.beat(scene, "b1"):
            scene.time += 0.8
        assert scene.time == pytest.approx(beats.MIN_BEAT_S)

    def test_a_long_beat_is_left_alone(self, beats):
        scene = FakeScene()
        with beats.beat(scene, "b1"):
            scene.time += beats.MIN_BEAT_S + 3.0
        assert scene.waits == []

    def test_the_recorded_span_includes_the_hold(self, beats):
        """The manifest must describe the video, not the animations inside it.

        A span ending before the padding would send a citation to a timestamp the
        beat is still occupying, and would hand narration a budget for a beat
        shorter than the one on screen.
        """
        scene = FakeScene()
        with beats.beat(scene, "b1"):
            scene.time += 1.0
        (span,) = beats._TIMINGS
        assert span == {"id": "b1", "start": 0.0, "end": pytest.approx(beats.MIN_BEAT_S)}

    def test_consecutive_beats_do_not_overlap(self, beats):
        scene = FakeScene()
        for beat_id in ("b1", "b2", "b3"):
            with beats.beat(scene, beat_id):
                scene.time += 0.5
        spans = beats._TIMINGS
        assert [s["id"] for s in spans] == ["b1", "b2", "b3"]
        for earlier, later in zip(spans, spans[1:], strict=False):
            assert earlier["end"] == pytest.approx(later["start"])

    def test_a_failing_beat_still_records_and_is_not_padded(self, beats):
        """The traceback goes to the repair loop; padding a dead beat helps nobody."""
        scene = FakeScene()
        with pytest.raises(ValueError):
            with beats.beat(scene, "b1"):
                scene.time += 0.5
                raise ValueError("manim exploded")
        assert scene.waits == []
        assert beats._TIMINGS == [{"id": "b1", "start": 0.0, "end": 0.5}]

    def test_the_manifest_is_written_for_a_partial_render(self, beats, tmp_path):
        scene = FakeScene()
        with beats.beat(scene, "b1"):
            scene.time += 1.0
        written = json.loads((tmp_path / "manifest.json").read_text())
        assert written == {"beats": [{"id": "b1", "start": 0.0, "end": beats.MIN_BEAT_S}]}

    def test_a_scene_whose_wait_raises_still_yields_a_beat(self, beats):
        """Pacing is a nicety; a video is not."""

        class Brittle(FakeScene):
            def wait(self, duration):
                raise RuntimeError("no")

        scene = Brittle()
        with beats.beat(scene, "b1"):
            scene.time += 0.5
        assert beats._TIMINGS == [{"id": "b1", "start": 0.0, "end": 0.5}]

    def test_the_floor_leaves_room_for_a_spoken_line(self, beats):
        """The floor exists to serve narration, so it is checked against it."""
        assert budget_words(beats.MIN_BEAT_S, 2.32) >= 10


class TestGuard:
    def test_clamps_to_the_window(self):
        guarded = guard(lambda x: 1000.0, -2.0, 2.0)
        assert guarded(0.0) == 2.0

    def test_leaves_values_inside_the_window_alone(self):
        guarded = guard(lambda x: 0.5, -2.0, 2.0)
        assert guarded(0.0) == 0.5

    def test_an_exception_becomes_a_clamp_not_a_crash(self):
        """Manim evaluates the plotted function directly, so raising ends the render."""
        guarded = guard(lambda x: 1.0 / 0.0, -2.0, 2.0)
        assert guarded(0.0) == 2.0

    def test_nan_becomes_a_clamp(self):
        guarded = guard(lambda x: float("nan"), -2.0, 2.0)
        assert guarded(0.0) == 2.0

    def test_samples_enough_points_to_see_a_narrow_feature(self):
        assert SAMPLES >= 100
