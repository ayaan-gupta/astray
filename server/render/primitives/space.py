"""Three-dimensional primitives. Container side.

Everything else in this package argues on a flat page: a square cut into four
cells, two curves on one set of axes, dots on a line. Those are the right tools
when the misconception is about a *value*. They are the wrong tools when it is
about a *shape*, and two of the most common misconceptions are exactly that.

`(a+b)^2 -> a^2 + b^2` is a claim about a surface. Written as algebra it is one
term against three, and a student who does not already believe the middle term
matters reads it as bookkeeping. Drawn as two surfaces over the same square of
inputs it stops being arguable: they touch along the two axes -- which is why the
rule *feels* right, since it is exactly correct whenever `a` or `b` is zero -- and
everywhere else the true surface climbs away from the student's. The gap is a
solid object with a size you can point at.

`d/dx sin(x^2) -> cos(x^2)` is a claim about a composition, and a composition is a
path through an intermediate space. Flattened to one plane that middle stage is
invisible, which is precisely why the factor coming from it goes missing. Given
its own axis, the whole mechanism is one curve: the inner stage is its shadow on
the floor, the outer stage its shadow on the wall, the answer its shadow on the
back. `pace_marks` then shows the thing the algebra hides -- that equal steps in
`x` are unequal steps in `u`, which is the dropped factor, seen rather than
asserted.

**Every one of these degrades rather than raises.** A helper here can be called
from a scene whose base class is a plain `Scene`, in which case there is no
camera to orient and no fixed frame to pin a legend to. The static validator
rejects that combination before a container starts (see `validator.py`), so it
should not reach here -- but a primitive that raises takes the whole video with
it, and a flat projection of a 3D scene is still a picture.
"""

import math

from manim import (
    BLUE_B,
    DEGREES,
    DOWN,
    GREEN,
    GREY_B,
    ORIGIN,
    RED,
    TEAL,
    UL,
    UP,
    YELLOW,
    Create,
    DashedLine,
    Dot3D,
    FadeIn,
    Line,
    ParametricFunction,
    Surface,
    ThreeDAxes,
    VGroup,
)
from primitives.layout import clear_frame, legend
from primitives.sampling import guard, guard2, tick_step, y_window, z_window

# Samples per side of a surface. Each surface is RESOLUTION^2 quads, drawn by the
# cairo renderer one at a time, and two of them share a beat -- 22 is the point
# where the silhouette is smooth and the render still fits inside its budget.
RESOLUTION = 22

# Where the camera sits before anything is drawn. High enough to read the two
# surfaces as stacked rather than intersecting, turned far enough off-axis that
# the depth is legible on the first frame instead of after the orbit starts.
PHI = 62 * DEGREES
THETA = -55 * DEGREES
# Pulled back from 0.95 after a render ran the plan axes out through the bottom
# edge. An axis whose arrowhead is cropped reads as a line, not an axis.
ZOOM = 0.82

# Radians per second of ambient rotation. A slow orbit is what turns a still
# render into a solid; fast enough to see, slow enough not to be the subject.
ORBIT_RATE = 0.12

# How far up the frame the whole drawing is nudged. See `_centre`.
LIFT = 0.55

# Manim's frame is 14.22 x 8 units. These lengths put a box of axes inside it
# with room for the key, at the camera angle above. Sized up once already: at
# 5.6 x 5.6 x 3.6 the drawing used a little over half the frame width, which on a
# 480p render is a diagram the viewer has to lean towards.
PLAN_LENGTH = 7.0
RISE_LENGTH = 4.4


def _stage(scene, *, phi: float = PHI, theta: float = THETA, zoom: float = ZOOM) -> bool:
    """Point the camera. False if this scene has no camera to point (plain `Scene`)."""
    setter = getattr(scene, "set_camera_orientation", None)
    if setter is None:
        return False
    try:
        setter(phi=phi, theta=theta, zoom=zoom)
        return True
    except Exception:
        return False


def fixed(scene, *mobjects) -> None:
    """Pin mobjects to the screen rather than to the space.

    Legends and captions must not tumble with the camera: a key that reads
    "your rule" is a caption on the frame, not an object in the scene. Falls back
    to a plain `add` where the scene has no fixed frame, which puts the text into
    the space -- readable from the opening angle, which is better than absent.
    """
    adder = getattr(scene, "add_fixed_in_frame_mobjects", None)
    if adder is None:
        scene.add(*mobjects)
        return
    try:
        adder(*mobjects)
    except Exception:
        scene.add(*mobjects)


def _orbit(scene, seconds: float) -> None:
    """Turn the camera slowly for `seconds`, then stop it.

    Stopping is in a `finally` on purpose. Ambient rotation is scene state, not
    beat state: a beat that begins the rotation and dies before stopping it leaves
    every later beat spinning, including the flat ones.
    """
    if seconds <= 0:
        return
    begin = getattr(scene, "begin_ambient_camera_rotation", None)
    stop = getattr(scene, "stop_ambient_camera_rotation", None)
    if begin is None or stop is None:
        scene.wait(seconds)
        return
    try:
        begin(rate=ORBIT_RATE)
        scene.wait(seconds)
    except Exception:
        pass
    finally:
        try:
            stop()
        except Exception:
            pass


def _axes(u_range, v_range, z_range) -> ThreeDAxes:
    return ThreeDAxes(
        x_range=[u_range[0], u_range[1], tick_step(u_range[1] - u_range[0])],
        y_range=[v_range[0], v_range[1], tick_step(v_range[1] - v_range[0])],
        z_range=[z_range[0], z_range[1], tick_step(z_range[1] - z_range[0])],
        x_length=PLAN_LENGTH,
        y_length=PLAN_LENGTH,
        z_length=RISE_LENGTH,
    )


def _key(scene, entries, scale: float, corner=UL) -> VGroup:
    """The frame's only text: a fixed key naming what each colour is.

    There are no labels on the axes themselves, and that is a decision rather
    than an omission. Three attempts are in the render history and each landed
    badly, because in a box that turns there is no good place to write on an
    axis: `axes.get_axis_labels` orients for a camera these primitives do not
    use and arrives squashed to nothing, with the vertical label lying sideways
    across the surfaces; tags placed past each tip fell out through the bottom
    edge; and once the frame was pulled back to fit them, the tip of the axis
    running away from the camera sat in the middle of the picture, so its tag
    printed over the very surfaces it was meant to annotate.

    A key in the corner has none of those problems and answers the question the
    axes were being asked to answer -- which line, or which sheet, is which.
    """
    key = legend(entries)
    key.scale(scale).to_corner(corner, buff=0.35)
    fixed(scene, key)
    return key


def _centre(*mobjects) -> VGroup:
    """Centre everything drawn in the space on the point the camera looks at.

    Manim centres an `Axes` on its own coordinate ranges, so a box whose ranges
    are not symmetric about zero -- `0..3` in the plan, `0..40` in the rise, which
    is exactly what a squared quantity gives -- sits off to one side once the
    camera rotates it. The first render of `rule_surfaces` left the whole left
    third of the frame empty and ran the surfaces off the right edge.
    """
    group = VGroup(*[item for item in mobjects if item is not None])
    group.move_to(ORIGIN)
    # Then lifted, because a centred 3D box does not project to a centred picture.
    # Looking down at 62 degrees puts the plan axes low and the rise axis high, so
    # balancing the box left the two horizontal axes running out through the
    # bottom edge -- and with them the tags that sit past their tips.
    group.shift(LIFT * UP)
    return group


def _restore(scene, group) -> None:
    """Re-add anything from `group` that a beat boundary faded out.

    `gap_pillars` and `pace_marks` annotate a picture the previous helper drew,
    and each is worth its own beat -- a pillar deserves a citation of its own.
    But `beat()` clears the frame on entry, so used that way they would draw onto
    black: the first render of these primitives produced one bar floating in an
    empty frame with no axes and no surfaces to be a bar between. Putting the
    scene back is cheaper than forbidding the split, and Manim's `FadeOut`
    restores opacity when it cleans up, so a re-added mobject is visible.
    """
    missing = [item for item in group if item not in scene.mobjects]
    if missing:
        scene.add(*missing)


def _surface(axes, fn, u_range, v_range, z_range, color, opacity: float = 0.6) -> Surface:
    """One z = f(u, v) sheet, clamped into the drawn box.

    Clamped rather than dropped: a student's rule can be undefined over part of
    the rectangle (`\\log(u+v)` on a square straddling `u + v = 0`), and Manim
    samples the function directly, so one nan ends the render. Clamping puts the
    sheet flat against the ceiling where the expression leaves its domain, which
    reads as "it goes off the top", which is what happened.
    """
    safe = guard2(fn, *z_range)
    surface = Surface(
        lambda u, v: axes.c2p(u, v, safe(u, v)),
        u_range=list(u_range),
        v_range=list(v_range),
        resolution=(RESOLUTION, RESOLUTION),
        fill_color=color,
        fill_opacity=opacity,
        checkerboard_colors=False,
        # The wireframe takes the sheet's own colour. White gridlines were tried
        # first and washed both sheets towards the same pale grey at 480p, which
        # is the one thing that must not happen to a frame whose whole argument is
        # "these are two different things".
        stroke_color=color,
        stroke_width=0.6,
    )
    surface.set_fill(color, opacity=opacity)
    return surface


def rule_surfaces(
    scene,
    correct,
    wrong,
    *,
    u_range: tuple[float, float] = (-2.0, 2.0),
    v_range: tuple[float, float] = (-2.0, 2.0),
    correct_label: str = "the truth",
    wrong_label: str = "your rule",
    orbit_seconds: float = 3.5,
) -> VGroup:
    """Two rules over the same square of inputs, as two surfaces in one space.

    `correct` and `wrong` are CALLABLES OF TWO ARGUMENTS -- write them as lambdas
    over plain arithmetic, e.g. `lambda a, b: (a + b) ** 2` against
    `lambda a, b: a ** 2 + b ** 2`. Both are drawn on one shared z scale, because
    two sheets on separate scales would sit at the same apparent height and deny
    the only thing the frame is for.

    Unpacks as `axes, right, left`, and the whole return value is what
    `gap_pillars` wants -- pass it straight on.
    """
    clear_frame(scene)
    z_range = z_window([correct, wrong], u_range, v_range)
    axes = _axes(u_range, v_range, z_range)
    right = _surface(axes, correct, u_range, v_range, z_range, GREEN)
    left = _surface(axes, wrong, u_range, v_range, z_range, RED)
    _centre(axes, right, left)

    _stage(scene)
    _key(scene, [(correct_label, GREEN), (wrong_label, RED)], 0.8)

    scene.play(Create(axes), run_time=0.9)
    scene.play(FadeIn(left), run_time=0.9)
    scene.play(FadeIn(right), run_time=0.9)
    _orbit(scene, orbit_seconds)
    # The labels and the legend are on screen but deliberately outside the return
    # value: the documented unpacking is `axes, right, left`, and a group whose
    # length does not match its docstring is a ValueError in generated code.
    return VGroup(axes, right, left)


def gap_pillars(
    scene,
    surfaces,
    correct,
    wrong,
    points,
    *,
    correct_label: str = "the truth",
    wrong_label: str = "your rule",
) -> VGroup:
    """At chosen inputs, the bar between the two surfaces: the error, as a length.

    `surfaces` is exactly what `rule_surfaces` returned, and `points` is a list of
    `(a, b)` pairs. The surfaces answer "are these different"; a pillar answers
    "by how much, here", which is the number the student can check against their
    own working. This is `graph.mark_divergence` one dimension up, and it is used
    the same way: same two callables, plus the picture to mark.

    Safe to call in a beat of its own -- the surfaces are put back if the beat
    boundary faded them.

    Pillars whose ends are not both finite are skipped rather than drawn at a
    guessed height: an invented bar is worse than a missing one.
    """
    _restore(scene, surfaces)
    axes = surfaces[0]
    drawn = []
    readings = []
    for a, b in points:
        try:
            high = float(correct(a, b))
            low = float(wrong(a, b))
        except Exception:
            continue
        if not (math.isfinite(high) and math.isfinite(low)):
            continue
        pillar = Line(
            axes.c2p(a, b, low),
            axes.c2p(a, b, high),
            color=YELLOW,
            stroke_width=7,
        )
        drawn.append(
            VGroup(
                pillar,
                Dot3D(axes.c2p(a, b, low), color=RED, radius=0.055),
                Dot3D(axes.c2p(a, b, high), color=GREEN, radius=0.055),
            )
        )
        readings.append((low, high))

    if not drawn:
        return VGroup()

    group = VGroup(*drawn)
    scene.play(*[Create(item) for item in drawn], run_time=1.0)

    low, high = readings[0]
    reading = legend(
        [
            (f"{wrong_label}: {_number(low)}", RED),
            (f"{correct_label}: {_number(high)}", GREEN),
            (f"missing: {_number(high - low)}", YELLOW),
        ]
    )
    reading.scale(0.75).to_edge(DOWN, buff=0.3)
    fixed(scene, reading)
    scene.play(FadeIn(reading), run_time=0.5)
    return VGroup(group, reading)


def composition_lift(
    scene,
    inner,
    outer,
    *,
    x_range: tuple[float, float] = (-2.0, 2.0),
    orbit_seconds: float = 3.5,
) -> VGroup:
    """A composition as one curve in space, with each stage as one of its shadows.

    `inner` and `outer` are CALLABLES OF ONE ARGUMENT: for `sin(x^2)` pass
    `lambda x: x ** 2` and `lambda u: np.sin(u)`. The middle value `u` gets its
    own axis, which is the whole point -- flattened onto one plane it is invisible,
    and a factor that comes from a stage you cannot see is a factor that goes
    missing.

    The three shadows are the three stages: on the floor, `u` against `x`; on the
    wall, `y` against `u`; on the back, the answer, `y` against `x`.

    Unpacks as `axes, lifted, floor, wall, back`, and the whole return value is
    what `pace_marks` wants -- pass it straight on.
    """
    clear_frame(scene)
    u_range = y_window([inner], x_range, pad=0.08)
    inner_safe = guard(inner, *u_range)

    def composed(x):
        return outer(inner_safe(x))

    # Two windows, unioned, because the two functions live over different
    # intervals: the composition is read across `x`, the outer stage across `u`.
    # Sampling both over one of the two would size the box to a stretch of curve
    # that is never drawn.
    answer_lo, answer_hi = y_window([composed], x_range, pad=0.12)
    outer_lo, outer_hi = y_window([outer], u_range, pad=0.12)
    y_range = (min(answer_lo, outer_lo), max(answer_hi, outer_hi))
    outer_safe = guard(outer, *y_range)

    axes = _axes(x_range, u_range, y_range)
    step = (x_range[1] - x_range[0]) / 160.0
    u_step = (u_range[1] - u_range[0]) / 160.0

    lifted = ParametricFunction(
        lambda t: axes.c2p(t, inner_safe(t), outer_safe(inner_safe(t))),
        t_range=[x_range[0], x_range[1], step],
        color=YELLOW,
        stroke_width=6,
    )
    floor = ParametricFunction(
        lambda t: axes.c2p(t, inner_safe(t), y_range[0]),
        t_range=[x_range[0], x_range[1], step],
        color=BLUE_B,
        stroke_width=4,
    )
    wall = ParametricFunction(
        lambda s: axes.c2p(x_range[0], s, outer_safe(s)),
        t_range=[u_range[0], u_range[1], u_step],
        color=TEAL,
        stroke_width=4,
    )
    back = ParametricFunction(
        lambda t: axes.c2p(t, u_range[1], outer_safe(inner_safe(t))),
        t_range=[x_range[0], x_range[1], step],
        color=GREEN,
        stroke_width=4,
    )

    _centre(axes, lifted, floor, wall, back)
    _stage(scene)
    # Prose, not the caller's labels. `layout.label` typesets anything containing
    # a digit or a backslash as mathematics, where spaces are discarded, and a
    # legend built from `u = x^2` plus the word "from" came out reading
    # "u = x^2from x". The three colours need naming, not restating.
    _key(
        scene,
        [
            ("inner stage, on the floor", BLUE_B),
            ("outer stage, up the wall", TEAL),
            ("the answer, on the back", GREEN),
        ],
        0.62,
    )

    scene.play(Create(axes), run_time=0.9)
    scene.play(Create(floor), run_time=0.8)
    scene.play(Create(wall), run_time=0.8)
    scene.play(Create(lifted), run_time=1.1)
    scene.play(Create(back), run_time=0.8)
    _orbit(scene, orbit_seconds)
    return VGroup(axes, lifted, floor, wall, back)


def pace_marks(
    scene,
    lift,
    inner,
    *,
    x_range: tuple[float, float] = (-2.0, 2.0),
    count: int = 9,
) -> VGroup:
    """Equal steps along `x`, and the unequal steps they become along `u`.

    This is the dropped factor, drawn. Ticks are placed at evenly spaced `x`, each
    carried up to the inner curve and across to the `u` axis; the marks that land
    there are bunched where the inner function is flat and spread where it is
    steep. The ratio between the two spacings IS the inner derivative, so a
    student looking at the picture is looking at the `2x` their answer left out.

    Pass what `composition_lift` returned and the same `inner`. Safe in a beat of
    its own: the lift is put back if the beat boundary faded it.
    """
    if count < 2:
        return VGroup()
    _restore(scene, lift)
    axes = lift[0]
    lo, hi = x_range
    span = (hi - lo) / (count - 1)
    floor_z = axes.z_range[0]

    rungs = []
    for index in range(count):
        x = lo + index * span
        try:
            u = float(inner(x))
        except Exception:
            continue
        if not math.isfinite(u):
            continue
        # Clamped to the drawn box. The caller can legitimately mark a wider `x`
        # window than the lift was built over, and a rung leaving the box would
        # be a dashed line into empty space with a dot on the end of it.
        u = max(float(axes.y_range[0]), min(float(axes.y_range[1]), u))
        base = axes.c2p(x, axes.y_range[0], floor_z)
        up = axes.c2p(x, u, floor_z)
        across = axes.c2p(lo, u, floor_z)
        rungs.append(
            VGroup(
                DashedLine(base, up, color=GREY_B, stroke_width=2, dash_length=0.06),
                DashedLine(up, across, color=GREY_B, stroke_width=2, dash_length=0.06),
                Dot3D(base, color=BLUE_B, radius=0.05),
                Dot3D(across, color=YELLOW, radius=0.06),
            )
        )

    if not rungs:
        return VGroup()
    group = VGroup(*rungs)
    scene.play(*[Create(rung) for rung in rungs], run_time=1.4, lag_ratio=0.06)

    key = legend([("equal steps in", BLUE_B), ("unequal steps out", YELLOW)])
    key.scale(0.7).to_edge(DOWN, buff=0.3)
    fixed(scene, key)
    scene.play(FadeIn(key), run_time=0.5)
    return VGroup(group, key)


def _number(value: float) -> str:
    """A reading a student can compare against their own working."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")
