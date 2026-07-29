Choose the concrete mathematics the animation will show.

The animation must do two things: show the correct method, AND show what the
student's own rule produces on the same input. Showing only the correct method
leaves their rule untouched -- they watch, agree, and keep the misconception.

- `worked_example`: the correct derivation, one line per step, as LaTeX math
  WITHOUT surrounding $ delimiters. Use the student's own problem where possible;
  a familiar problem is worth more than a cleaner one.
- `counter_example`: the SAME starting expression carried through the student's
  buggy rule, so the two can sit side by side and disagree. End on the value their
  rule produces.
- `key_identity`: the one line that, if they remember nothing else, fixes this.
- `concrete_numbers`: a specific numeric substitution that makes the disagreement
  undeniable -- e.g. at x=1, one side gives 26 and the other 36. Numbers beat
  symbols for convincing someone their rule is wrong.

Return only JSON matching the schema.
