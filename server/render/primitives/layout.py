"""Shared layout helpers. Container side.

Every helper here is deliberately conservative about vertical space: generated
scenes routinely try to put a title, a worked example and a counter-example on
one frame, and Manim will happily render them overlapping off-screen. These
helpers scale-to-fit rather than trusting the caller's sizing.
"""

import re

from manim import DOWN, LEFT, RIGHT, UP, FadeIn, FadeOut, Line, MathTex, Mobject, Text, VGroup
from primitives.prose import mathify, prose

# Manim's default frame is 8 units tall, 14.22 wide. Leaving a margin keeps text
# clear of the edge on the 854x480 preview resolution the runner uses.
SAFE_WIDTH = 12.0
SAFE_HEIGHT = 6.5

# How many occupants a caption will step over before giving up and overlapping.
# Enough for `area_totals`' two lines plus the diagram's own bottom edge.
_CAPTION_LIFTS = 4


def fit(mobject, width: float = SAFE_WIDTH, height: float = SAFE_HEIGHT):
    """Scale `mobject` down until it fits the safe area. Never scales up."""
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    if mobject.height > height:
        mobject.scale_to_fit_height(height)
    return mobject


def title_card(text: str, subtitle: str = "") -> VGroup:
    """A title, optionally with a subtitle beneath it, pre-fitted to the frame."""
    parts = [Text(text, font_size=44)]
    if subtitle:
        parts.append(Text(subtitle, font_size=28))
    group = VGroup(*parts).arrange(DOWN, buff=0.35)
    return fit(group)


def safe_math(line, font_size: int = 40, color=None):
    """A `MathTex`, degrading to plain `Text` if LaTeX refuses the string.

    `MathTex`, not `Tex`: these lines are expressions, and `Tex` renders in LaTeX
    *text* mode where a bare `^` is the error "Missing $ inserted" and kills the
    whole render. Transcribed student work arrives as math (`x^{2} + 25 = 36`),
    so math mode is the correct default, not a convenience.

    The `Text` fallback is load-bearing. Every label in every primitive is
    ultimately a string that reached us through a model whose prompt chain began
    with student-supplied text, so any one of them can be invalid LaTeX. One bad
    label must cost its own styling and nothing more -- a raised exception here
    would take the whole video with it.

    **An already-built Mobject is passed straight through**, and that branch was
    written against a real failure rather than in anticipation of one. Generated
    code called `compare_rules(self, [MathTex(...)], [MathTex(...)])`, since a
    list of *rendered maths* is a perfectly reasonable reading of "lines of
    maths". `MathTex` accepted the Mobject, stringified it, and typeset its repr:
    the misconception beat -- the one frame the whole video exists for -- read
    `MathTex('fracdydx = cos(x^2)')` in crossed-out red, and the pipeline recorded
    the render as a success. The `Text` fallback above actively concealed it, by
    turning what would have been a loud `TypeError` into a quiet wrong frame. So
    the fix belongs here, in the primitive, not in a prompt asking the model to
    read the type hint more carefully.
    """
    if isinstance(line, Mobject):
        return line
    try:
        return MathTex(mathify(str(line)), font_size=font_size, color=color)
    except Exception:
        return Text(prose(str(line)), font_size=max(font_size - 8, 18), color=color)


def math_lines(lines: list, font_size: int = 40) -> VGroup:
    """Stack lines of math vertically, fitted to the frame.

    Each item is a LaTeX string or an already-built Mobject; see `safe_math`. A
    Mobject keeps whatever size it was made at, so mixing the two forms in one
    call can give an uneven stack -- pass strings and let `font_size` govern.
    """
    rendered = [safe_math(line, font_size) for line in lines]
    if not rendered:
        return VGroup()
    # Spacing scales with the tallest line, not a fixed gap. A stack of `\frac`
    # expressions is roughly three times the height of `x = 7`, and a constant
    # 0.3 buff let consecutive fractions overlap -- observed on the live
    # "1/2 + 1/3" render, where the numerator of one line sat inside the
    # denominator of the line above it.
    tallest = max(item.height for item in rendered)
    # `aligned_edge` must be PERPENDICULAR to the arrange direction. Passing UP
    # while arranging DOWN silently produces a line pitch smaller than the line
    # height -- measured at 0.661 pitch against 0.723-tall `\frac` lines, a
    # -0.062 overlap that put each numerator inside the denominator above it.
    # Increasing buff does not fix it; the conflicting alignment overrides the
    # spacing. LEFT gives a left-aligned stack, which is also easier to read for
    # a derivation than centring every line.
    group = VGroup(*rendered).arrange(DOWN, buff=max(0.3, tallest * 0.35), aligned_edge=LEFT)
    return fit(group)


# LaTeX markup, or a digit. Either says the string is meant as mathematics; a
# label with neither is prose and belongs in text mode.
_MATHISH = re.compile(r"[\\^_{}]|\d")


def label(value, font_size: int = 26, color=None):
    """Typeset a caller-supplied label as prose or as mathematics, whichever it is.

    Labels are the one place where both arrive through the same argument.
    `compare_functions` is called with `wrong_label=r"\\cos(x^2)"` on one beat and
    `wrong_label="your answer"` on the next, and the two need opposite treatment:
    math mode discards spaces, so a phrase typeset as maths comes out as one run of
    italics. The number line's legend read "youfound" and "alsoasolution" before
    this existed, which is a rendered frame quietly losing its own caption.
    """
    if isinstance(value, Mobject):
        return value
    if _MATHISH.search(str(value)):
        return safe_math(value, font_size=font_size, color=color)
    return Text(str(value), font_size=font_size, color=color)


def legend(entries: list[tuple[str, object]]) -> VGroup:
    """A row of swatch-and-label pairs. `entries` is [(label, color), ...].

    Colour alone does not say which curve is which, and Manim's
    `get_graph_label` attaches text to the curve itself, which collides with the
    other curve exactly where they are most interesting -- near a crossing. A
    legend outside the plot area never collides with anything.
    """
    rows = []
    for text, color in entries:
        swatch = Line(LEFT * 0.22, RIGHT * 0.22, stroke_width=6, color=color)
        rows.append(VGroup(swatch, label(text, 26, color)).arrange(RIGHT, buff=0.18))
    group = VGroup(*rows).arrange(RIGHT, buff=0.7)
    if group.width > SAFE_WIDTH:
        group.scale_to_fit_width(SAFE_WIDTH)
    return group


def _overlaps(a, b, pad: float = 0.05) -> bool:
    """Do two mobjects' bounding boxes intersect?"""
    return (
        a.get_right()[0] + pad > b.get_left()[0]
        and a.get_left()[0] - pad < b.get_right()[0]
        and a.get_top()[1] + pad > b.get_bottom()[1]
        and a.get_bottom()[1] - pad < b.get_top()[1]
    )


def caption(scene, text: str, run_time: float = 0.6):
    """One line of prose along the bottom edge, stacked above whatever is there.

    `Text`, not `MathTex`: a caption is a sentence. Callers wanting an expression
    on the bottom edge should compose `safe_math` themselves.

    The stacking is the part that matters. `caption` and `area.area_totals` both
    want the bottom edge, and a beat legitimately uses both -- a live render put
    *"The buggy expansion misses the 2ab term."* directly on top of the two totals
    it was describing, so the frame carried three overlapping lines of text. Each
    primitive reserving its own band cannot fix this, because neither knows the
    other ran; asking the frame what is already there can.
    """
    line = Text(prose(text), font_size=26)
    if line.width > SAFE_WIDTH:
        line.scale_to_fit_width(SAFE_WIDTH)
    line.to_edge(DOWN, buff=0.28)

    # Lift clear of whatever already occupies the bottom edge, one occupant at a
    # time. A single pass is not enough and that is not a corner case: `area_totals`
    # adds its two lines separately, so clearing the lower one lands the caption
    # squarely on the upper one. Bounded, because the caption must not walk up
    # through an entire diagram looking for space, and monotonic, so it terminates.
    for _ in range(_CAPTION_LIFTS):
        blocking = [item for item in scene.mobjects if _overlaps(line, item)]
        if not blocking:
            break
        highest = max(item.get_top()[1] for item in blocking)
        lifted = highest + 0.18 - line.get_bottom()[1]
        if line.get_top()[1] + lifted > SAFE_HEIGHT / 2:
            break  # nowhere left to go; an overlapping caption beats one off-frame
        line.shift(UP * lifted)

    scene.play(FadeIn(line), run_time=run_time)
    return line


def clear_frame(scene, run_time: float = 0.4) -> None:
    """Fade out everything currently on screen.

    Primitives call this before drawing, so a beat cannot be rendered on top of
    the previous beat's leftovers. The s7 prompt asks generated code to clean up
    after itself, but a prompt instruction is not a guarantee: on the live
    "(x+5)^2" render the model left a title card up and drew the comparison
    straight over it, making the most important beat in the animation unreadable.
    Making the primitives own the frame removes the dependency on the model
    remembering.
    """
    if scene.mobjects:
        scene.play(*[FadeOut(item) for item in scene.mobjects], run_time=run_time)
