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
from manim import FadeOut, MathTex, Scene

from primitives.algebra_steps import compare_rules, step_sequence
from primitives.area import area_totals, binomial_square, missing_area
from primitives.beats import beat
from primitives.graph import compare_functions, mark_divergence
from primitives.layout import caption, title_card
from primitives.numberline import missing_roots


class AstrayScene(Scene):
    def construct(self):
        with beat(self, "b1"):
            card = title_card("Primitive check", "every helper, once")
            self.play(FadeOut(card.copy().scale(0.01)), run_time=0.1)
            self.add(card)
            self.wait(0.5)
            self.play(FadeOut(card))

        with beat(self, "b2"):
            step_sequence(self, [r"(y+3)^2", r"= y^2 + 6y + 9"])
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
            missing_area(self, corners, middles, "these two are the 6y you dropped")
            area_totals(self, r"y^2 + 9", r"y^2 + 6y + 9")
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

BEATS = [
    ("b1", "title card"),
    ("b2", "step sequence"),
    ("b3", "compare_rules, abused"),
    ("b4", "area model"),
    ("b5", "graph comparison"),
    ("b6", "graph with a pole"),
    ("b7", "missing root"),
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="media/_primitive_check")
    args = parser.parse_args()

    settings = get_settings()
    board = Storyboard(
        beats=[
            Beat(id=bid, title=title, teaching_purpose=title, on_screen=title, primitive="custom")
            for bid, title in BEATS
        ],
        total_estimated_seconds=40,
    )

    report = validate(SCENE, board, "AstrayScene")
    if not report.ok:
        for issue in report.issues:
            print(f"  validator: {issue.kind}: {issue.detail}")
        return 1
    print("validator ok")

    paths = runner.prepare(Path(settings.media_root), "_primitive_check", SCENE, attempt=1)
    result = await runner.run(paths, "AstrayScene", timeout_s=settings.render_timeout_s)
    if not result.ok:
        print(f"render FAILED\n{(result.error_text or '')[-3000:]}")
        return 1

    print(f"render ok in {result.duration_s:.1f}s -> {result.video_path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    timings = {t.id: t for t in result.timings}
    for bid, title in BEATS:
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
        print(f"  {bid}  {span.start:6.2f}-{span.end:6.2f}s  {title:<24} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
