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
5. Available helpers (import from `primitives.<module>`):
   - `primitives.beats.beat(scene, id)` -- required, above.
   - `primitives.layout.title_card(text, subtitle="")` -- fitted title group.
   - `primitives.layout.math_lines(lines, font_size=40)` -- stacked math, fitted.
   - `primitives.algebra_steps.step_sequence(scene, lines)` -- reveal lines in turn.
   - `primitives.algebra_steps.compare_rules(scene, wrong_lines, right_lines)` --
     shows the student's rule crossed out beside the correct one. Use this for the
     beat that targets the misconception.
6. Keep the frame clean: fade out what you are done with before the next beat.
   Overlapping leftovers are the most common way these renders come out unreadable.
7. Total runtime should be close to the storyboard's estimate.

Return only JSON matching the schema: `scene_class_name`, `code`, `beats_covered`.
`code` is the complete file as a single string.
