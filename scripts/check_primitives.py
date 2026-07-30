"""Render every primitive in the container and extract a frame from each beat.

The primitives import `manim`, which is not a host dependency -- it exists only
inside the render image -- so they cannot be unit-tested in `tests/`. That is also
why no test may touch Docker. This script is the gap: it exercises each helper for
real, through the same sandboxed runner a session uses, and leaves one PNG per
beat to look at.

It deliberately passes the abuse the live pipeline produced as well as the correct
usage: a Mobject where a string belongs, a function with a pole inside the plot
window, a label too wide for its cell. Each of those cost a render or a frame
before it was handled here.

    uv run python scripts/check_primitives.py [--out DIR]
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.charter.contracts import Beat, Storyboard  # noqa: E402
from server.config import get_settings  # noqa: E402
from server.render import runner  # noqa: E402
from server.render.validator import validate  # noqa: E402

SCENE = """
import numpy as np
from manim import FadeIn, FadeOut, MathTex, Scene

from primitives.algebra_steps import compare_rules, step_sequence
from primitives.area import area_totals, binomial_square, compare_areas, missing_area
from primitives.beats import beat
from primitives.graph import compare_functions, mark_divergence
from primitives.layout import caption, title_card
from primitives.numberline import missing_roots


class AstrayScene(Scene):
    def construct(self):
        with beat(self, "b1"):
            # `title_card` and `math_lines` are the two helpers that build without
            # animating, so the caller plays them. Everything below animates itself.
            card = title_card("Primitive check", "every helper, once")
            self.play(FadeIn(card))
            self.wait(0.5)
            self.play(FadeOut(card))

        with beat(self, "b2"):
            # Prose mixed into maths lines. A live render showed "Lety = 1" and
            # "Thestudent'srulefails" from exactly this shape.
            step_sequence(self, [r"Let y = 1", r"(y+3)^2 = y^2 + 6y + 9",
                                 r"Your rule gives y^2 + 9"])
            self.wait(0.4)

        with beat(self, "b3"):
            # Mobjects where strings belong. This is what the live pipeline wrote,
            # and it used to typeset the repr of the Mobject.
            result = compare_rules(self, [MathTex(r"y^2 + 9")], [r"y^2 + 6y + 9"])
            FadeOut(result)  # must not raise: a tuple return did
            self.wait(0.4)

        with beat(self, "b4"):
            corners, middles = binomial_square(
                self, a_label="y", b_label="3", a_term="y^2", b_term="9", middle_term="3y"
            )
            missing_area(self, corners, middles, "these two are the $6y$ you dropped")
            area_totals(self, r"y^2 + 9", r"y^2 + 6y + 9")
            # A third claimant on the bottom edge. A live render put this line
            # straight on top of the totals above it.
            caption(self, r"The buggy expansion misses the 2ab term, so 16\\neq10.")
            self.wait(0.4)

        with beat(self, "b8"):
            # The frame the model kept hand-rolling into an unreadable mess.
            compare_areas(
                self,
                a_label="1",
                b_label="3",
                a_term="1",
                b_term="9",
                middle_term="3",
                correct_total="16",
                buggy_total="10",
            )
            self.wait(0.4)

        with beat(self, "b5"):
            right = lambda x: 2 * x * np.cos(x**2)
            wrong = lambda x: np.cos(x**2)
            axes, _, _ = compare_functions(
                self, right, wrong, x_range=(-3.0, 3.0),
                correct_label=r"2x\\cos(x^2)", wrong_label=r"\\cos(x^2)",
            )
            mark_divergence(self, axes, right, wrong, 1.0)
            self.wait(0.4)

        with beat(self, "b6"):
            # A pole inside the window, and a function undefined over half of it.
            compare_functions(
                self,
                lambda x: 1.0 / x,
                lambda x: np.log(x),
                x_range=(-2.0, 2.0),
                correct_label="1/x",
                wrong_label=r"\\log x",
            )
            self.wait(0.4)

        with beat(self, "b7"):
            missing_roots(self, [4.0], [-4.0])
            caption(self, "squaring loses the sign")
            self.wait(0.4)
"""

SPACE_SCENE = """
import numpy as np
from manim import ThreeDScene

from primitives.beats import beat
from primitives.space import composition_lift, gap_pillars, pace_marks, rule_surfaces


class AstraySpaceScene(ThreeDScene):
    def construct(self):
        with beat(self, "b1"):
            # The freshman's dream as two surfaces. They touch exactly along the
            # two axes, which is why the rule feels right.
            surfaces = rule_surfaces(
                self,
                lambda a, b: (a + b) ** 2,
                lambda a, b: a**2 + b**2,
                u_range=(0.0, 3.0),
                v_range=(0.0, 3.0),
                correct_label=r"(y+3)^2",
                wrong_label=r"y^2+9",
                orbit_seconds=2.0,
            )

        with beat(self, "b2"):
            gap_pillars(
                self,
                surfaces,
                lambda a, b: (a + b) ** 2,
                lambda a, b: a**2 + b**2,
                [(1.0, 3.0)],
            )

        with beat(self, "b3"):
            # A surface undefined over part of its own rectangle, which is the
            # shape a student's rule usually arrives in.
            rule_surfaces(
                self,
                lambda a, b: np.log(a + b),
                lambda a, b: np.log(a) + np.log(b),
                u_range=(-1.0, 3.0),
                v_range=(-1.0, 3.0),
                orbit_seconds=1.0,
            )

        with beat(self, "b4"):
            lift = composition_lift(
                self,
                lambda x: x**2,
                lambda u: np.sin(u),
                x_range=(-2.0, 2.0),
                orbit_seconds=2.0,
            )

        with beat(self, "b5"):
            pace_marks(self, lift, lambda x: x**2, x_range=(-2.0, 2.0))
"""

SPACE_BEATS = [
    ("b1", "two rules as two surfaces"),
    ("b2", "the gap, as a length"),
    ("b3", "a surface with a domain hole"),
    ("b4", "composition lifted into space"),
    ("b5", "equal x steps, unequal u steps"),
]

BEATS = [
    ("b1", "title card"),
    ("b2", "step sequence"),
    ("b3", "compare_rules, abused"),
    ("b4", "area model"),
    ("b5", "graph comparison"),
    ("b6", "graph with a pole"),
    ("b7", "missing root"),
    ("b8", "area comparison"),
]


PASSES = {
    "flat": (SCENE, BEATS, "AstrayScene"),
    "space": (SPACE_SCENE, SPACE_BEATS, "AstraySpaceScene"),
}


async def check(name: str, out: Path, settings) -> int:
    """Validate, render and sample one pass. Returns a shell exit code."""
    scene_code, beats, class_name = PASSES[name]
    board = Storyboard(
        beats=[
            Beat(id=bid, title=title, teaching_purpose=title, on_screen=title, primitive="custom")
            for bid, title in beats
        ],
        total_estimated_seconds=40,
    )

    report = validate(scene_code, board, class_name)
    if not report.ok:
        for issue in report.issues:
            print(f"  validator: {issue.kind}: {issue.detail}")
        return 1
    print(f"[{name}] validator ok")

    paths = runner.prepare(
        Path(settings.media_root), f"_primitive_check_{name}", scene_code, attempt=1
    )
    result = await runner.run(paths, class_name, timeout_s=settings.render_timeout_s)
    if not result.ok:
        print(f"[{name}] render FAILED\n{(result.error_text or '')[-3000:]}")
        return 1

    print(f"[{name}] render ok in {result.duration_s:.1f}s -> {result.video_path}")
    out.mkdir(parents=True, exist_ok=True)

    timings = {t.id: t for t in result.timings}
    for bid, title in beats:
        span = timings.get(bid)
        if span is None:
            print(f"  {bid}: no timing recorded")
            continue
        # Sampled just before the beat ends, when everything it draws is present.
        at = max(span.start, span.end - 0.35)
        target = out / f"{bid}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{at:.2f}",
                "-i",
                str(result.video_path),
                "-frames:v",
                "1",
                str(target),
                "-y",
            ],
            check=True,
        )
        print(f"  {bid}  {span.start:6.2f}-{span.end:6.2f}s  {title:<28} {target}")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="media/_primitive_check")
    parser.add_argument("--pass", dest="passes", action="append", choices=sorted(PASSES))
    args = parser.parse_args()

    settings = get_settings()
    out = Path(args.out)
    status = 0
    for name in args.passes or sorted(PASSES):
        status |= await check(name, out / name, settings)
    return status


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
