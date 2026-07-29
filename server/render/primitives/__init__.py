"""Vetted Manim helpers mounted read-only into the render container.

Scene code authored by the model may import from `manim`, `numpy`, `math` and
this package, and nothing else -- `server/render/validator.py` enforces that
list statically before any execution.

The package exists so generated code has a small, known vocabulary to reach for
instead of inventing its own scaffolding on every run. `beat()` is the only
member that is load-bearing for correctness: it is what makes chat citations
point at real timestamps.
"""

from primitives.algebra_steps import compare_rules, step_sequence
from primitives.beats import beat
from primitives.layout import title_card

__all__ = ["beat", "step_sequence", "compare_rules", "title_card"]
