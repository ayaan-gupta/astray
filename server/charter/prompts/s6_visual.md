Plan the animation as an ordered sequence of BEATS.

A beat is one addressable moment. The tutor chat cites beats by id to point the
student at a specific point in the video, so a beat must be a coherent thing worth
pointing at -- not an arbitrary time slice.

Rules, all enforced downstream:

- 4 to 8 beats. Ids are `b1`, `b2`, ... in order, no gaps.
- At least one beat MUST have `targets_misconception: true` -- the moment the
  student's own rule is shown to fail. This is the beat the whole animation exists
  for. Usually it is the side-by-side comparison.
- `title`: short, shown on the beat rail under the player. Write it so a student
  scanning the rail knows what they would be jumping to.
- `teaching_purpose`: why this beat exists. If you cannot say what it does that its
  neighbours do not, cut it.
- `on_screen`: what the viewer literally sees.
- `primitive`: one of `numberline`, `areamodel`, `algebra_steps`, `graph`,
  `balance`, `custom`. Pick `algebra_steps` for symbolic derivations, `areamodel`
  for area/expansion arguments, `numberline` for magnitude/sign, `graph` for
  functions, `balance` for equation solving, `custom` only if none fit.
- `total_estimated_seconds`: realistic total, typically 45-120.

Return only JSON matching the schema.
