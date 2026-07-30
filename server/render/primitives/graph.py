"""Function-graph primitives. Container side.

The storyboard vocabulary has offered a `graph` beat since s6 was written, and
nothing implemented it, so every graph beat was raw improvised Manim: no fitting,
no frame ownership, and an unguarded call into a function the model chose. That is
the pairing this module closes.

`compare_functions` is the graph-shaped `compare_rules`. For a misconception whose
two sides are functions rather than strings -- a dropped chain-rule factor, a
mishandled log, a root never taken -- plotting both argues better than either
derivation, because the curves separate on screen and the gap is visibly not a
rounding detail. The factor dropped from `d/dx sin(x^2)` is `2x`, and
`2x cos(x^2)` against `cos(x^2)` is two unrelated curves that happen to touch.

**Every function reaching this module is guarded.** A model asked for "the
student's expression as a function" will eventually hand over something with a
pole in the window (`1/x`, `tan x`, `log` of a negative), and Manim samples
whatever it is given: one non-finite value, or one raised exception, ends the
render. `sampling.guard` clamps instead, so a pole becomes a curve running off
the top of the frame -- which is what a pole looks like, and is a diagram rather
than a lost video. `sampling` also decides where each function exists, so an
undefined stretch is drawn as a gap and not as a flat line.
"""

import math

from manim import (
    BLACK,
    DOWN,
    GREEN,
    GREY_B,
    RED,
    RIGHT,
    UP,
    WHITE,
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    VGroup,
)
from primitives.layout import clear_frame, fit, legend, safe_math
from primitives.sampling import defined_range, guard, tick_step, y_window

# Sized to leave room for axis numbers, a legend across the top, and a caption
# along the bottom edge without any of the three touching.
PLOT_WIDTH = 9.6
PLOT_HEIGHT = 4.9


def make_axes(x_range: tuple[float, float], y_range: tuple[float, float]) -> Axes:
    """Axes with readable ticks, sized to the safe area."""
    return Axes(
        x_range=[x_range[0], x_range[1], tick_step(x_range[1] - x_range[0])],
        y_range=[y_range[0], y_range[1], tick_step(y_range[1] - y_range[0])],
        x_length=PLOT_WIDTH,
        y_length=PLOT_HEIGHT,
        tips=False,
        axis_config={
            "include_numbers": True,
            "font_size": 20,
            "decimal_number_config": {"num_decimal_places": 0},
            "color": GREY_B,
        },
    )


def compare_functions(
    scene,
    correct,
    wrong,
    x_range: tuple[float, float] = (-3.0, 3.0),
    correct_label: str = "correct",
    wrong_label: str = "your answer",
    y_range: tuple[float, float] | None = None,
    run_time: float = 1.2,
):
    """Plot the correct function and the student's, on one set of axes.

    `correct` and `wrong` are plain callables -- pass lambdas over `numpy` or
    `math`. The correct curve is drawn first and in green, so the frame reads as
    "here is the shape, and here is what yours does instead" rather than as two
    anonymous lines.

    Returns a VGroup that unpacks as
    `axes, right, wrong = compare_functions(...)`. `axes` is what
    `mark_divergence` needs next.
    """
    clear_frame(scene)

    window = y_range or y_window([correct, wrong], x_range)
    axes = make_axes(x_range, window)

    # Each curve is plotted only where its own function exists, so a function
    # undefined over part of the window leaves a gap rather than a false line. A
    # function defined nowhere still contributes an empty VGroup, keeping the
    # return arity fixed at three so `axes, right, wrong = ...` cannot raise.
    def curve_for(fn, color):
        span = defined_range(fn, x_range)
        if span is None:
            return VGroup()
        return axes.plot(guard(fn, *window), x_range=list(span), color=color, stroke_width=5)

    right = curve_for(correct, GREEN)
    left = curve_for(wrong, RED)

    key = legend([(correct_label, GREEN), (wrong_label, RED)])
    frame = VGroup(key, VGroup(axes, right, left)).arrange(DOWN, buff=0.25)
    fit(frame)

    scene.play(Create(axes), run_time=run_time * 0.6)
    scene.play(FadeIn(key), run_time=run_time * 0.4)
    for curve in (right, left):
        if len(curve.get_all_points()):
            scene.play(Create(curve), run_time=run_time)
    return VGroup(axes, right, left)


def mark_divergence(scene, axes, correct, wrong, x: float, run_time: float = 0.8):
    """At `x`, drop a vertical line and dot both curves with their values.

    The most useful follow-up to `compare_functions`: two curves that differ
    everywhere still invite "but they are close, surely". A labelled pair of
    values at one x turns the gap into two numbers. Returns the VGroup, or None if
    either function is undefined or off-frame there -- an unmarked graph is fine,
    a dot pointing at nothing is a lie, and a dead render is worse than both.
    """
    try:
        y_right = float(correct(x))
        y_wrong = float(wrong(x))
    except Exception:
        return None
    if not (math.isfinite(y_right) and math.isfinite(y_wrong)):
        return None

    lo, hi = axes.y_range[0], axes.y_range[1]
    if not (lo <= y_right <= hi and lo <= y_wrong <= hi):
        return None

    rule = DashedLine(axes.c2p(x, lo), axes.c2p(x, hi), color=WHITE, stroke_width=2)
    right_dot = Dot(axes.c2p(x, y_right), color=GREEN, radius=0.08)
    wrong_dot = Dot(axes.c2p(x, y_wrong), color=RED, radius=0.08)

    right_text = safe_math(f"{y_right:.2f}", font_size=24, color=GREEN)
    wrong_text = safe_math(f"{y_wrong:.2f}", font_size=24, color=RED)
    # Pushed diagonally apart, not both placed to the right. The two curves are
    # closest exactly where marking them is most interesting, and side-by-side
    # labels overlapped each other and the dots: at x=1 on the chain-rule case the
    # second value rendered as ".54" with its leading digit behind a dot.
    above, below = (UP + RIGHT), (DOWN + RIGHT)
    right_text.next_to(right_dot, above if y_right >= y_wrong else below, buff=0.1)
    wrong_text.next_to(wrong_dot, below if y_right >= y_wrong else above, buff=0.1)
    # Separating them is still not enough on its own: a value near zero lands on
    # the x-axis and its tick labels whatever direction it is pushed. A backing
    # panel is what makes the number readable wherever the maths happens to put it.
    for text in (right_text, wrong_text):
        text.add_background_rectangle(color=BLACK, opacity=0.8, buff=0.05)

    group = VGroup(rule, right_dot, wrong_dot, right_text, wrong_text)
    scene.play(Create(rule), run_time=run_time * 0.5)
    scene.play(FadeIn(right_dot, wrong_dot, right_text, wrong_text), run_time=run_time * 0.5)
    return group
