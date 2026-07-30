Plan the animation as an ordered sequence of BEATS.

A beat is one addressable moment. The tutor chat cites beats by id to point the
student at a specific point in the video, so a beat must be a coherent thing worth
pointing at -- not an arbitrary time slice.

Rules, all enforced downstream:

- 4 to 8 beats. Ids are `b1`, `b2`, ... in order, no gaps.
- At least one beat MUST have `targets_misconception: true` -- the moment the
  student's own rule is shown to fail. This is the beat the whole animation exists
  for.
- `title`: short, shown on the beat rail under the player. Write it so a student
  scanning the rail knows what they would be jumping to.
- `teaching_purpose`: why this beat exists. If you cannot say what it does that its
  neighbours do not, cut it.
- `on_screen`: what the viewer literally sees. Where the beat needs concrete
  values -- an x window to plot over, the side lengths of a square, the solutions
  to mark -- name them here. The stage that writes the code has only this text to
  work from.
- `total_estimated_seconds`: about 8 to 12 seconds per beat. This is a diagram, not
  a lecture: a beat is one thing being shown, and a viewer reads a static frame in
  roughly two seconds. A generous estimate does not buy more explanation, it buys a
  longer hold on the same picture, so it is clamped to a realistic band before the
  scene is written.

## Choosing `primitive`

Every value below is backed by a real builder, so this field decides what the beat
actually looks like. Pick whichever argues best, not whichever is easiest to
typeset.

**Two of them are not optional when they apply.** If the misconception is a rule
for **combining two quantities** -- anything of the shape `f(a+b) = f(a) + f(b)`,
which covers `(a+b)^2`, `\log(x+y)`, `\sqrt{a^2+b^2}`, `1/(a+b)` -- then the beat
that targets it is `surface`, followed by a second `surface` beat marking the gap
at the student's own numbers. If the misconception is a **dropped or misapplied
stage of a composition**, that pair is `lift` then `lift`. Flat beats around them
are still right, and welcome; the targeting beat is not one of them.

- `algebra_steps` -- a symbolic derivation, or the student's derivation crossed out
  beside the correct one. Right when the error is a visible wrong step.
- `graph` -- two functions plotted on one set of axes, correct against the
  student's. Right whenever both sides of the misconception are functions of a
  variable: a dropped chain-rule factor, `\log(x+y)` split into a sum,
  `\sqrt{x^2+9}` read as `x+3`. Two curves separating on screen argues harder than
  a derivation, because the student sees that the answers are not merely written
  differently but are different numbers. Name the two functions and the x window
  in `on_screen`.
- `areamodel` -- a square of side (a+b) split into `a^2`, two `ab` rectangles and
  `b^2`, drawn to scale. Right for `(a+b)^2 -> a^2 + b^2`: the two `ab` rectangles
  are visibly present, so the missing middle term stops being something the student
  has to take on trust. It **counts** what is missing; `surface` below **measures**
  it, and on this family the two beats together argue better than either alone.
- `numberline` -- solutions marked on a line. Right when the error is an *absence*
  rather than a wrong step, above all a lost root (`x^2 = 16 -> x = 4`). Nothing
  is there to cross out, so a comparison beat cannot work, but a second dot
  appearing where the student had nothing can.
- `surface` -- the two rules as two surfaces over the same square of inputs, in a
  space the camera turns around. Right whenever the misconception is a claim about
  **two quantities combining**: `(a+b)^2 -> a^2+b^2`, `\log(x+y) -> \log x + \log y`,
  `\sqrt{a^2+b^2} -> a+b`, `1/(a+b) -> 1/a + 1/b`. This is the strongest available
  frame for that family, because the two sheets touch exactly where the rule is
  accidentally correct -- along `a = 0` and `b = 0` -- and separate everywhere else,
  so the student sees both why it felt right and how far wrong it goes. Name both
  two-variable expressions and the input window in `on_screen`.
- `lift` -- a composition given three dimensions: the inner stage as a shadow on
  the floor, the outer stage as a shadow on the wall, the answer as a shadow on the
  back. Right when the error is about a **chain**: a dropped chain-rule factor, a
  substitution done in the wrong order, an inverse taken of only one stage. The
  middle quantity is invisible in flat algebra, which is exactly why the factor
  coming from it goes missing. Name the inner and outer functions in `on_screen`.
- `custom` -- only if none of the above fit.

A run where every beat is `algebra_steps` is sometimes right, for a misconception
that genuinely is purely symbolic. Often it is a missed opportunity: if the error
can be shown as a picture or as a pair of curves, at least the beat that targets
it should be.

**Spatial beats come in pairs, and the pair is the argument.** A `surface` beat
draws the two sheets; the beat after it marks the gap at the student's own numbers,
which is the moment the picture becomes a number they can check. A `lift` beat
draws the composition; the beat after it shows equal steps in `x` becoming unequal
steps in the middle quantity, which is the dropped factor made visible. Plan the
second beat as `surface` or `lift` too, and say in its `on_screen` which point, or
how many steps, to mark. One spatial beat on its own states the situation without
finishing the argument.

Do not make every beat spatial. Two or three are a centrepiece; six are a
fairground ride, and a student cannot read a symbolic derivation off a turning
box. Open flat, argue in space, land flat.

Return only JSON matching the schema.
