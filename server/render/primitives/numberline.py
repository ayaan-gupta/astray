"""Number-line primitives. Container side.

The lost-root family (`x^2 = 16 -> x = 4`) is the case a symbolic derivation
cannot show. Every line the student wrote is true; the error is an *absence*, and
you cannot cross out something that is not on the page. `compare_rules` has
nothing to strike through and `step_sequence` reveals a derivation the student
already agrees with.

On a line the absence becomes a place. A second dot appearing where the student
had nothing is the entire explanation, and it needs no words. The same shape
covers sign errors and magnitude claims, which are the other two things a line
says better than a string.
"""

from manim import BLUE, DOWN, RED, Create, Dot, FadeIn, NumberLine, VGroup
from primitives.layout import clear_frame, fit, legend
from primitives.sampling import decimal_places, tick_step

# Matches graph.PLOT_WIDTH so a beat mixing the two does not visibly change scale
# between them.
LINE_WIDTH = 9.6

# Green would be wrong for either dot here, and that is a real distinction rather
# than a palette preference. Everywhere else in this codebase red marks the
# student's error and green marks something checked and correct. On this line the
# student's own answer is correct as far as it goes, so it is not green; the value
# they omitted is correct mathematics, so colouring *it* red would invert the
# scheme the rest of the app teaches. Blue is `area.CORNER_COLOR`, already meaning
# "what the student kept", and red marks the omission itself, which genuinely is
# the error.
FOUND = BLUE
MISSED = RED


def make_line(window: tuple[float, float]) -> NumberLine:
    """A labelled number line across `window`, sized to the safe area.

    Decimal places follow the tick step rather than being fixed. Integer roots
    labelled "-4.0" and "4.0" read as measurements rather than as solutions, and
    the lost-root case is almost always integers; a fractional step still needs
    its decimal, so the choice is made from the step and not hardcoded.
    """
    step = tick_step(window[1] - window[0])
    return NumberLine(
        x_range=[window[0], window[1], step],
        length=LINE_WIDTH,
        include_numbers=True,
        font_size=24,
        decimal_number_config={"num_decimal_places": decimal_places(step)},
    )


def _window(values: list[float], given: tuple[float, float] | None) -> tuple[float, float]:
    """A window spanning the marked values with room to breathe.

    Spans the values rather than centring on zero, and the difference is not
    cosmetic. Forcing a zero-centred window put a comparison of 10 against 16 on a
    line running -20 to 20, with both dots crowded into one eighth of it and most
    of the line carrying nothing.

    Symmetry survives anyway, and that is why spanning is safe: a lost root marks
    -4 and 4, whose span is already centred on zero, so the symmetry that explains
    why the second root exists still reads as symmetry.
    """
    if given:
        return given
    if not values:
        return (-1.0, 1.0)
    low, high = min(values), max(values)
    pad = max((high - low) * 0.25, abs(high) * 0.15, 1.0)
    return (low - pad, high + pad)


def missing_roots(
    scene,
    found: list[float],
    missed: list[float],
    x_range: tuple[float, float] | None = None,
    found_label: str = "you found",
    missed_label: str = "you missed this one",
    run_time: float = 0.8,
):
    """Mark the solutions the student found, then the ones they did not.

    The pause between the two reveals is the primitive's whole substance: the
    student's own answer sits alone on the line long enough to feel complete
    before the missing one lands.

    Returns a VGroup that unpacks as
    `line, found_dots, missed_dots = missing_roots(...)`.
    """
    clear_frame(scene)

    line = make_line(_window([*found, *missed], x_range))
    found_dots = VGroup(*[Dot(line.number_to_point(v), color=FOUND, radius=0.1) for v in found])
    missed_dots = VGroup(*[Dot(line.number_to_point(v), color=MISSED, radius=0.1) for v in missed])

    key = legend([(found_label, FOUND), (missed_label, MISSED)])
    frame = VGroup(key, VGroup(line, found_dots, missed_dots)).arrange(DOWN, buff=0.8)
    fit(frame)

    scene.play(Create(line), run_time=run_time)
    scene.play(FadeIn(key), run_time=run_time * 0.5)
    scene.play(FadeIn(found_dots, scale=1.6), run_time=run_time)
    scene.wait(0.6)
    scene.play(FadeIn(missed_dots, scale=1.6), run_time=run_time)
    return VGroup(line, found_dots, missed_dots)
