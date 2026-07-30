"""Vetted Manim helpers mounted read-only into the render container.

Scene code authored by the model may import from `manim`, `numpy`, `math` and
this package, and nothing else -- `server/render/validator.py` enforces that
list statically before any execution.

The package exists so generated code has a small, known vocabulary to reach for
instead of inventing its own scaffolding on every run. `beat()` is the only
member that is load-bearing for correctness: it is what makes chat citations
point at real timestamps.

Two conventions hold across every module here, and both were written against
observed render failures rather than in the abstract:

* **A primitive owns the frame.** Each one calls `clear_frame` before drawing, so
  a beat can never be rendered over the previous beat's leftovers. Callers do not
  need to fade anything out.
* **A primitive returns one Mobject that unpacks into its parts.** `VGroup` is
  iterable, so `corners, middles = binomial_square(...)` still works while
  `FadeOut(binomial_square(...))` no longer raises `TypeError: Animation only
  works on Mobjects` -- which is exactly how a live render died when these
  returned bare tuples.
"""

from primitives.algebra_steps import compare_rules, step_sequence
from primitives.area import area_totals, binomial_square, compare_areas, missing_area
from primitives.beats import beat
from primitives.graph import compare_functions, mark_divergence
from primitives.layout import caption, math_lines, title_card
from primitives.numberline import missing_roots

__all__ = [
    "area_totals",
    "beat",
    "binomial_square",
    "caption",
    "compare_areas",
    "compare_functions",
    "compare_rules",
    "mark_divergence",
    "math_lines",
    "missing_area",
    "missing_roots",
    "step_sequence",
    "title_card",
]
