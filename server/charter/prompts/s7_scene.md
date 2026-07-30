Write a single Manim Community scene file that renders the given storyboard.

This code runs in a locked-down container and is statically validated before it
executes. Violating any rule below fails the render:

1. The ONLY imports permitted are `manim`, `numpy`, `math`, and `primitives`.
   No `os`, `sys`, `subprocess`, `pathlib`, or any other module. No `eval`,
   `exec`, `compile`, `open`, `getattr`, or `__import__`. No dunder attribute
   access of any kind.
2. Exactly ONE Scene subclass, named exactly as specified.
3. Every beat id in the storyboard must appear EXACTLY ONCE, wrapped as:

       with beat(self, "b1"):
           ...animations for this beat...

   Import it as `from primitives.beats import beat`. A beat id that is missing,
   duplicated, or not in the storyboard fails validation. The id must be a string
   literal, never a variable or f-string.
4. Use `MathTex` for mathematics, never `Tex` -- `Tex` renders in LaTeX text mode
   where a bare `^` raises "Missing $ inserted" and kills the render. Do not wrap
   expressions in `$`; `MathTex` is already math mode.
5. **Pace each beat to about `seconds_per_beat` seconds**, animation included. The
   helpers animate for one to four seconds on their own, so most beats need two to
   six seconds of waiting on top.

   Reach that length by **showing more, not by holding longer**. A single unchanging
   frame is a still image and a viewer reads one in about two seconds, so never
   write one wait longer than 4. If a beat needs more time than that, give it more
   to look at: reveal it in stages with a short wait after each, add a
   `mark_divergence` to a graph, add a `caption`. Two `self.wait(3)` calls either
   side of a second reveal are worth far more than one `self.wait(6)`.

   Both failure modes here are real and equally bad. A scene with no waits at all
   ran 12.5s across six beats, too short to seek to or narrate. A scene told to
   reach a total spent it entirely on waiting -- 30, 35, 40, 35 -- and ran 155s as
   four static frames. Every beat is held open to a five second floor whatever you
   write, so shortness is survivable; a 35 second still is not.

## Reach for a primitive before writing your own

Each helper below clears the previous beat before drawing and scales what it
draws to fit the frame. Hand-rolled equivalents are the main source of
unreadable renders, so prefer a primitive whenever one covers the beat, and use
the primitive the beat's `primitive` field names.

**Three rules apply to all of them.**

- **Do not fade out what a primitive returned, and do not track a "last group".**
  The next primitive clears the frame itself.
- **Each returns a single Mobject that also unpacks into its parts**, so both
  `corners, middles = binomial_square(self)` and `FadeOut(result)` are safe.
- **Pass LaTeX strings, never built Mobjects**, wherever a helper takes maths.
  `compare_rules(self, [MathTex("x^2")], ...)` is wrong; pass `["x^2"]`.

**Two things not to do to a primitive's output**, both of which produced bad frames:

- **Do not restyle or relabel what it returned.** Reaching into the groups from
  `binomial_square` to recolour them and write your own `3y` labels over the cells
  reproduces, worse, what `missing_area` already does. If you want different
  emphasis, call the helper for it.
- **Do not stack your own text under a diagram with chained `.next_to(...)`.**
  Nothing fits that chain to the frame, so the second or third line runs off the
  bottom edge and is clipped. `area_totals` places two totals and `caption` places
  one line, both inside the frame and both aware of each other.

### Symbolic derivations -- `from primitives.algebra_steps import ...`

    step_sequence(scene, lines: list[str]) -> VGroup
        Reveal LaTeX lines one at a time, top to bottom.

    compare_rules(scene, wrong_lines: list[str], right_lines: list[str],
                  wrong_label: str = "What you did",
                  right_label: str = "What the rule actually gives") -> VGroup
        The student's derivation crossed out in red beside the correct one in
        green. The default choice for a beat targeting the misconception.

### Function graphs -- `from primitives.graph import ...`

    compare_functions(scene, correct, wrong,
                      x_range: tuple[float, float] = (-3.0, 3.0),
                      correct_label: str = "correct",
                      wrong_label: str = "your answer",
                      y_range: tuple[float, float] | None = None) -> VGroup
        Plot both functions on one set of axes, correct in green and the
        student's in red. `correct` and `wrong` are CALLABLES of one argument;
        write them as lambdas over numpy, e.g.
        `lambda x: 2 * x * np.cos(x ** 2)`. The y window, tick spacing and
        legend are chosen for you. Unpacks as `axes, right, wrong`.

    mark_divergence(scene, axes, correct, wrong, x: float) -> VGroup | None
        At one x, a dashed vertical line and a labelled dot on each curve, so the
        gap becomes two numbers. Pass the same callables and the `axes` that
        `compare_functions` returned. Returns None where either is undefined.

### Area arguments -- `from primitives.area import ...`

    binomial_square(scene, a_label="a", b_label="b", a_term="a^2",
                    b_term="b^2", middle_term="ab") -> VGroup
        A square of side (a+b) split into a^2, two ab rectangles and b^2, drawn
        to scale. Unpacks as `corners, middles`. The strongest argument against
        `(a+b)^2 -> a^2 + b^2`: the two ab rectangles are visibly there. Pass the
        student's actual letters and terms, e.g. `a_label="y", b_label="3"`.
        Numeric side labels are drawn to their real proportions, so
        `a_label="1", b_label="3"` gives a small corner and a large one.

    compare_areas(scene, a_label="a", b_label="b", a_term="a^2", b_term="b^2",
                  middle_term="ab", correct_total="", buggy_total="",
                  correct_label="Correct", buggy_label="Your rule") -> VGroup
        The full square beside the student's, whose two ab regions are left as
        dashed holes, with a total under each. Use this instead of building two
        area models side by side yourself: the same outline accounting for less
        area is the whole argument, and hand-rolled versions of this frame come
        out unreadable.

    missing_area(scene, corners, middles, caption: str) -> Mobject
        Dim the corner squares, hold the two ab rectangles, caption them. Takes
        the two groups `binomial_square` returned.

    area_totals(scene, kept: str, actual: str) -> VGroup
        Two LaTeX totals along the bottom edge: what the student's rule accounts
        for, and what the square actually measures.

### Lost solutions -- `from primitives.numberline import ...`

    missing_roots(scene, found: list[float], missed: list[float],
                  found_label="you found",
                  missed_label="also a solution") -> VGroup
        A number line marking the solutions the student found, a pause, then the
        ones they missed. When the error is an absence (`x^2 = 16 -> x = 4`) there
        is nothing to cross out, so this replaces `compare_rules`.

### Layout -- `from primitives.layout import ...`

    title_card(text: str, subtitle: str = "") -> VGroup
    math_lines(lines: list[str], font_size: int = 40) -> VGroup
    caption(scene, text: str) -> Mobject

`title_card` and `math_lines` build a Mobject and return it *without* animating,
so play them yourself and fade them out yourself. Every other helper above
animates, and needs no `self.play` around it.

Return only JSON matching the schema: `scene_class_name`, `code`, `beats_covered`.
`code` is the complete file as a single string.
