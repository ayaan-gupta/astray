"""Numeric decisions behind the plotting primitives. No manim, no drawing.

Separated from `graph.py` for one concrete reason: `manim` exists only inside the
render image, so anything importing it cannot be reached by a test on the host,
and every rule in this module is a judgement call that deserves one. Choosing a y
window, deciding where a function exists, and picking a tick step are all places
where a plausible-looking rule is quietly wrong on a real input, and the pole case
below took three attempts to get right.

Everything here depends on `math` alone, which is inside the container's import
allow-list, so the split costs nothing at render time.
"""

import math

# How many points to sample across a window. Enough that a narrow feature is not
# missed, cheap enough to run several times per beat.
SAMPLES = 240

# Trim this fraction from each end of the sampled values before choosing the y
# window. A single sample beside a pole is astronomically large; using the raw min
# and max would scale the plot to that one point and flatten everything else.
TRIM = 0.03

# How many consecutive undefined samples count as the function genuinely not
# existing there, rather than a pole passed through. This is the whole difference
# between two cases that look identical to a naive check: `log x` on [-2, 2] is
# undefined across half the window and must be drawn only on the half where it
# exists, while `1/x` is defined everywhere except one point and must be drawn
# right across. Splitting on any undefined sample cuts `1/x` in half at x = 0;
# never splitting draws `log x` as a flat line along the top of the frame,
# asserting a value it does not have.
MIN_GAP_SAMPLES = 3


def grid(lo: float, hi: float) -> list[float]:
    """The sample abscissae across [lo, hi]."""
    step = (hi - lo) / max(SAMPLES - 1, 1)
    return [lo + index * step for index in range(SAMPLES)]


def defined_mask(fn, lo: float, hi: float) -> list[bool]:
    """Per-sample: does `fn` return a real number there?

    `numpy` returns nan rather than raising for `log` of a negative, so nan counts
    as undefined here even though `guard` has to clamp it to something later.
    """
    mask = []
    for x in grid(lo, hi):
        try:
            value = float(fn(x))
            mask.append(math.isfinite(value))
        except Exception:
            mask.append(False)
    return mask


def finite_values(fn, lo: float, hi: float) -> list[float]:
    """Finite values of `fn` across [lo, hi]. Failures and infinities dropped."""
    values = []
    for x in grid(lo, hi):
        try:
            value = float(fn(x))
        except Exception:
            # A function undefined on part of the window is normal, not an error:
            # sqrt and log both are. The window comes from where it IS defined.
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def y_window(functions, x_range: tuple[float, float], pad: float = 0.18) -> tuple[float, float]:
    """A y window that shows the functions rather than their worst outlier.

    Trimmed rather than min/max, for the reason `TRIM` documents. Always returns a
    non-degenerate interval: a constant function would otherwise give lo == hi and
    an Axes of zero height.
    """
    values: list[float] = []
    for fn in functions:
        values.extend(finite_values(fn, *x_range))
    if not values:
        return (-1.0, 1.0)

    values.sort()
    cut = int(len(values) * TRIM)
    trimmed = values[cut : len(values) - cut] or values
    lo, hi = trimmed[0], trimmed[-1]

    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    margin = (hi - lo) * pad
    return (lo - margin, hi + margin)


def _bridge(mask: list[bool]) -> list[bool]:
    """Fill undefined runs shorter than `MIN_GAP_SAMPLES`, so poles do not split.

    Only interior runs are bridged. A window opening or closing on an undefined
    stretch is a domain edge, not a pole, however short it is: `sqrt(x)` on
    [-0.01, 4] is undefined at its first sample only, and bridging that would draw
    the curve where the function does not exist.
    """
    filled = list(mask)
    index = 0
    while index < len(filled):
        if filled[index]:
            index += 1
            continue
        end = index
        while end < len(filled) and not filled[end]:
            end += 1
        if end - index < MIN_GAP_SAMPLES and index > 0 and end < len(filled):
            for position in range(index, end):
                filled[position] = True
        index = end
    return filled


def defined_range(fn, x_range: tuple[float, float]) -> tuple[float, float] | None:
    """The widest sub-interval of `x_range` where `fn` exists, poles bridged.

    Returns None if the function exists nowhere in the window, which is a
    legitimate outcome for a student's expression rather than an error.
    """
    lo, hi = x_range
    mask = _bridge(defined_mask(fn, lo, hi))

    best_start, best_end = 0, -1
    start = None
    for position, ok in enumerate(mask):
        if ok:
            if start is None:
                start = position
            if position - start > best_end - best_start:
                best_start, best_end = start, position
        else:
            start = None

    if best_end <= best_start:
        return None
    step = (hi - lo) / max(SAMPLES - 1, 1)
    return (lo + best_start * step, lo + best_end * step)


def guard(fn, lo: float, hi: float):
    """`fn` clamped into [lo, hi], with exceptions and non-finites clamped too.

    Manim evaluates the plotted function directly, so this wrapper is the only
    thing between a model-chosen expression and the renderer. It never raises.
    """

    def guarded(x):
        try:
            value = float(fn(x))
        except Exception:
            return hi
        if math.isnan(value):
            return hi
        return max(lo, min(hi, value))

    return guarded


def surface_grid(lo: float, hi: float, count: int) -> list[float]:
    """`count` evenly spaced abscissae across [lo, hi].

    Coarser than `grid` on purpose. A surface is sampled in two dimensions, so the
    240 samples a curve uses would be 57,600 evaluations per surface and two
    surfaces per beat -- seconds of pure Python before Manim draws anything.
    """
    if count < 2:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + index * step for index in range(count)]


def finite_values_2d(
    fn, u_range: tuple[float, float], v_range: tuple[float, float], count: int = 21
):
    """Finite values of a two-argument `fn` over the sample grid.

    Same contract as `finite_values`: an expression undefined on part of its
    rectangle is normal (`\\log(u+v)` over a rectangle straddling `u+v = 0`), so
    failures and infinities are dropped rather than raised.
    """
    values = []
    for u in surface_grid(*u_range, count):
        for v in surface_grid(*v_range, count):
            try:
                value = float(fn(u, v))
            except Exception:
                continue
            if math.isfinite(value):
                values.append(value)
    return values


# The fraction trimmed from each end before a surface's spread is measured.
# Much wider than `TRIM`, because a singularity in two variables is a curve
# rather than a point: `1/(a+b)` is bad along the whole diagonal of its
# rectangle, which is one full row of the grid, so roughly `1/count` of every
# sample. At count = 21 that is 4.8%, and a 3% trim leaves the pole in charge of
# the scale. 8% clears a row with room to spare.
SURFACE_TRIM = 0.08


def z_window(
    functions,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    pad: float = 0.1,
) -> tuple[float, float]:
    """A z window holding every surface, shared across all of them.

    Shared rather than computed per surface, for the reason the whole primitive
    exists: two sheets scaled separately would sit at the same apparent height,
    denying the one thing the frame is for.

    The extreme is kept unless it is an outlier, and that is the difference from
    `y_window`. A curve's peak is usually somewhere in the middle of the window,
    so trimming the top few samples costs nothing; a surface's peak is at a corner
    of the rectangle, and it is the corner the argument is about. Trimming flat
    cut the top off `(a+b)^2` over `0..3` at 28.6 against a true 36, which draws
    the maximum divergence -- the whole point of the beat -- as a plateau clamped
    against the ceiling. So the trimmed bound is used only to measure the spread,
    and the real extreme is kept whenever it lies within one spread of it. A pole
    does not: `1/(a+b)` puts a sample near the domain edge many orders of
    magnitude out, and that one is still discarded.
    """
    values: list[float] = []
    for fn in functions:
        values.extend(finite_values_2d(fn, u_range, v_range))
    if not values:
        return (-1.0, 1.0)

    values.sort()
    cut = int(len(values) * SURFACE_TRIM)
    trimmed = values[cut : len(values) - cut] or values
    lo, hi = trimmed[0], trimmed[-1]
    spread = max(hi - lo, 1e-6)
    if values[-1] - hi <= spread:
        hi = values[-1]
    if lo - values[0] <= spread:
        lo = values[0]

    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    margin = (hi - lo) * pad
    return (lo - margin, hi + margin)


def guard2(fn, lo: float, hi: float):
    """A two-argument `guard`. Manim samples a surface directly; this never raises."""

    def guarded(u, v):
        try:
            value = float(fn(u, v))
        except Exception:
            return hi
        if math.isnan(value):
            return hi
        return max(lo, min(hi, value))

    return guarded


def decimal_places(step: float) -> int:
    """How many decimals a tick label needs, given the step between ticks.

    Paired with `tick_step`, and not optional: a fractional step printed with zero
    decimals produces *duplicate* labels, not merely coarse ones. A plot over
    x = 0 to 3 gets a 0.5 step, which rounded to integers reads
    "0 . 2 . 2 . 2 . 3" -- observed on a live chain-rule graph, where it looks like
    a rendering fault rather than a rounding choice.

    One decimal is enough for every step this module produces, since the steps are
    snapped to 1/2/2.5/5 times a power of ten.
    """
    return 0 if step >= 1 and float(step).is_integer() else 1


def tick_step(span: float) -> float:
    """A tick step giving roughly 4-8 labelled ticks across `span`.

    Axes and number lines both need this, and both look wrong without it: Manim's
    default step of 1 puts fifty labels on a range of fifty and one label on a
    range of two. Snapped to a 1/2/2.5/5 multiple of a power of ten, which is
    where readable axis labels come from.
    """
    if span <= 0:
        return 1.0
    raw = span / 6.0
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= factor * power:
            return factor * power
    return 10.0 * power
