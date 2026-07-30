"""Area-model primitives. Container side.

`binomial_square` is the strongest single argument this codebase can draw. The
misconception `(a+b)^2 -> a^2 + b^2` survives a symbolic derivation, because a
student who believes it watches the correct expansion, agrees with each line, and
keeps the belief -- the two sides are just competing strings. It does not survive
a picture of the square: the two `ab` rectangles are visibly, measurably there,
occupying area that `a^2 + b^2` does not account for. The claim stops being
"my teacher says the middle term exists" and becomes "the corner squares do not
fill the square".

That is why the geometry here is built from real side lengths rather than a
schematic sketch. If `a` and `b` are drawn to scale, the `ab` rectangles are as
large as the arithmetic says they are, and the frame is the proof. A decorative
diagram with arbitrary proportions would show the same four labels while quietly
giving up the only thing the picture adds.
"""

from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Create,
    FadeIn,
    Rectangle,
    Text,
    VGroup,
)
from primitives.layout import SAFE_HEIGHT, SAFE_WIDTH, clear_frame, fit, prose, safe_math

# The square is drawn `a`-major on purpose: with a and b too close in length the
# four cells look like a 2x2 grid of equals, and the point is that a^2 dominates
# while the two ab rectangles are still far too big to ignore. 2:1 reads clearly
# at 480p and keeps `b^2` large enough to label.
DEFAULT_A = 2.6
DEFAULT_B = 1.3

CORNER_COLOR = BLUE  # what the student kept: a^2 and b^2
MIDDLE_COLOR = YELLOW  # what they dropped: the two ab rectangles

# Manim's frame height. The bands below are measured from the frame edges rather
# than from `SAFE_HEIGHT`, because `to_edge` positions against the frame: taking
# them out of the safe height instead double-counts its margin and shrinks the
# square to well under half the frame for no reason.
FRAME_HEIGHT = 8.0

# Vertical bands kept clear of the square, for `missing_area`'s caption above and
# `area_totals`' two lines below. Each is the band's own height plus its edge buff
# plus a gap, so the square can use everything that is left.
CAPTION_ROOM = 0.85
TOTALS_ROOM = 1.62


def _cell(width: float, height: float, color, label: str, opacity: float = 0.35):
    """One labelled cell of the square, as a (rectangle, label) VGroup."""
    box = Rectangle(
        width=width,
        height=height,
        stroke_color=WHITE,
        stroke_width=2,
        fill_color=color,
        fill_opacity=opacity,
    )
    text = safe_math(label, font_size=30)
    # A label wider than its cell is the failure mode here: `ab` fits anywhere,
    # but a caller passing `2xy` into the thin b^2 corner does not. Scale to the
    # cell rather than letting it bleed across the border it belongs inside.
    if text.width > width * 0.8:
        text.scale_to_fit_width(width * 0.8)
    if text.height > height * 0.8:
        text.scale_to_fit_height(height * 0.8)
    text.move_to(box.get_center())
    return VGroup(box, text)


def binomial_square(
    scene,
    a_label: str = "a",
    b_label: str = "b",
    a_term: str = "a^2",
    b_term: str = "b^2",
    middle_term: str = "ab",
    a_len: float = DEFAULT_A,
    b_len: float = DEFAULT_B,
    run_time: float = 0.7,
):
    """Decompose a square of side (a+b) into a^2, two ab rectangles, and b^2.

    Reveals in the order the argument runs: the outer square and its sides, then
    the two corner squares the student kept, then the two rectangles they dropped.

    Returns a VGroup that unpacks as `corners, middles = binomial_square(...)`,
    so the caller can keep arguing with the middle pair while the return value is
    still a single Mobject. See `algebra_steps.compare_rules` for why that shape
    rather than a tuple.
    """
    clear_frame(scene)

    total = a_len + b_len
    origin = LEFT * (total / 2) + DOWN * (total / 2)

    def at(x: float, y: float, width: float, height: float):
        """Cell centre, from the square's bottom-left corner."""
        return origin + RIGHT * (x + width / 2) + UP * (y + height / 2)

    a_square = _cell(a_len, a_len, CORNER_COLOR, a_term)
    b_square = _cell(b_len, b_len, CORNER_COLOR, b_term)
    top_right = _cell(b_len, a_len, MIDDLE_COLOR, middle_term)
    bottom_left = _cell(a_len, b_len, MIDDLE_COLOR, middle_term)

    a_square.move_to(at(0, b_len, a_len, a_len))
    top_right.move_to(at(a_len, b_len, b_len, a_len))
    bottom_left.move_to(at(0, 0, a_len, b_len))
    b_square.move_to(at(a_len, 0, b_len, b_len))

    outline = Rectangle(width=total, height=total, stroke_color=WHITE, stroke_width=4)
    outline.move_to(at(0, 0, total, total))
    # `missing_area` captions along the top edge and `area_totals` writes two lines
    # along the bottom, and a beat may use all three. Reserving the room here is
    # what keeps them from colliding: the first version fitted the square to the
    # full safe height, and the totals landed on top of the caption.
    reserved_height = min(SAFE_HEIGHT, FRAME_HEIGHT - CAPTION_ROOM - TOTALS_ROOM)

    # Side labels along the top and left edges, so the reader can check the
    # lengths against the cells rather than taking them on trust.
    top_a = safe_math(a_label, font_size=30, color=GREY_B)
    top_b = safe_math(b_label, font_size=30, color=GREY_B)
    left_a = safe_math(a_label, font_size=30, color=GREY_B)
    left_b = safe_math(b_label, font_size=30, color=GREY_B)
    top_a.next_to(a_square, UP, buff=0.18)
    top_b.next_to(top_right, UP, buff=0.18)
    left_a.next_to(a_square, LEFT, buff=0.18)
    left_b.next_to(bottom_left, LEFT, buff=0.18)

    corners = VGroup(a_square, b_square)
    middles = VGroup(top_right, bottom_left)
    edges = VGroup(top_a, top_b, left_a, left_b)
    whole = VGroup(outline, corners, middles, edges)

    # Fit the assembled group, never the pieces: scaling cells individually
    # would break the very proportions the argument rests on.
    fit(whole, height=reserved_height)
    # Centred within what is left after the two reserved bands, not on the frame.
    # The bottom band is the taller of the two, so the centre of the free space sits
    # ABOVE the frame's centre: the usable interval is
    # [-4 + TOTALS_ROOM, 4 - CAPTION_ROOM], whose midpoint is
    # (TOTALS_ROOM - CAPTION_ROOM) / 2. Getting this sign backwards pushed the
    # square down into the very band it was reserving.
    whole.shift(UP * (TOTALS_ROOM - CAPTION_ROOM) / 2)

    scene.play(Create(outline), FadeIn(edges), run_time=run_time)
    scene.play(FadeIn(corners), run_time=run_time)
    scene.play(FadeIn(middles), run_time=run_time)
    return VGroup(corners, middles)


def missing_area(
    scene,
    corners,
    middles,
    caption: str = "these two are the middle term",
    run_time: float = 0.8,
):
    """Dim the corner squares and hold the dropped rectangles, with a caption.

    Separated from `binomial_square` because it is the argument, not the diagram:
    a beat may want the square drawn plainly, and the beat that targets the
    misconception wants the two rectangles isolated and named. Dimming rather
    than removing keeps the whole square on screen, which is the entire point --
    the missing area has to be missing *from something*.

    Both effects touch the cell rectangles only, never the cell labels, and both
    of those were rendered wrong before being narrowed. `set_opacity` on the whole
    cell took `y^2` and `9` down with the fill and left the corner terms
    unreadable, which loses the two things the student *did* get right. Worse,
    `set_stroke` on the whole cell gave the `3y` labels a 5px yellow outline,
    turning the one number the beat exists to name into a blob.
    """
    text = Text(prose(caption), font_size=26, color=MIDDLE_COLOR)
    if text.width > SAFE_WIDTH:
        text.scale_to_fit_width(SAFE_WIDTH)
    text.to_edge(UP, buff=0.3)

    boxes = VGroup(*[cell[0] for cell in corners])
    edges = VGroup(*[cell[0] for cell in middles])

    scene.play(boxes.animate.set_fill(opacity=0.08), run_time=run_time * 0.6)
    scene.play(edges.animate.set_stroke(color=MIDDLE_COLOR, width=5), run_time=run_time * 0.6)
    scene.play(FadeIn(text), run_time=run_time)
    return text


def area_totals(scene, kept: str, actual: str, run_time: float = 0.8):
    """Two totals beneath the square: what the student's rule accounts for, and
    what the square actually measures.

    Placed at the bottom edge so the diagram above stays visible. The comparison
    only lands while the reader can still see the area both numbers refer to.
    """
    kept_line = safe_math(kept, font_size=32, color=CORNER_COLOR)
    actual_line = safe_math(actual, font_size=32, color=GREEN)
    stack = VGroup(kept_line, actual_line).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    fit(stack, height=TOTALS_ROOM - 0.35)
    stack.to_edge(DOWN, buff=0.3)
    scene.play(FadeIn(kept_line), run_time=run_time)
    scene.play(FadeIn(actual_line), run_time=run_time)
    return stack
