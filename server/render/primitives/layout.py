"""Shared layout helpers. Container side.

Every helper here is deliberately conservative about vertical space: generated
scenes routinely try to put a title, a worked example and a counter-example on
one frame, and Manim will happily render them overlapping off-screen. These
helpers scale-to-fit rather than trusting the caller's sizing.
"""

from manim import DOWN, LEFT, FadeOut, MathTex, Text, VGroup

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


def math_lines(lines: list[str], font_size: int = 40) -> VGroup:
    """Stack lines of math vertically, fitted to the frame.

    `MathTex`, not `Tex`: these lines are expressions, and `Tex` renders in LaTeX
    *text* mode where a bare `^` is the error "Missing $ inserted" and kills the
    whole render. Transcribed student work arrives as math (`x^{2} + 25 = 36`),
    so math mode is the correct default, not a convenience.

    Falls back to plain Text for any line LaTeX still refuses. Student work
    reaches this function and a student can write something that is not valid
    LaTeX; one bad line must degrade to unstyled text rather than lose the video.
    """
    rendered = []
    for line in lines:
        try:
            rendered.append(MathTex(line, font_size=font_size))
        except Exception:
            rendered.append(Text(line, font_size=max(font_size - 8, 18)))
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
