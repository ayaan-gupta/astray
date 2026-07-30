"""Shared layout helpers. Container side.

Every helper here is deliberately conservative about vertical space: generated
scenes routinely try to put a title, a worked example and a counter-example on
one frame, and Manim will happily render them overlapping off-screen. These
helpers scale-to-fit rather than trusting the caller's sizing.
"""

import re

from manim import DOWN, LEFT, RIGHT, FadeIn, FadeOut, Line, MathTex, Mobject, Text, VGroup

# Manim's default frame is 8 units tall, 14.22 wide. Leaving a margin keeps text
# clear of the edge on the 854x480 preview resolution the runner uses.
SAFE_WIDTH = 12.0
SAFE_HEIGHT = 6.5


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
        return MathTex(line, font_size=font_size, color=color)
    except Exception:
        return Text(str(line), font_size=max(font_size - 8, 18), color=color)


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


def caption(scene, text: str, run_time: float = 0.6):
    """One line of prose along the bottom edge, leaving the diagram above it alone.

    `Text`, not `MathTex`: a caption is a sentence. Callers wanting an expression
    on the bottom edge should compose `safe_math` themselves.
    """
    line = Text(text, font_size=26)
    if line.width > SAFE_WIDTH:
        line.scale_to_fit_width(SAFE_WIDTH)
    line.to_edge(DOWN, buff=0.28)
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
