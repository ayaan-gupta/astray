Map the prerequisite concepts needed to close one specific knowledge gap.

Keep the graph SMALL and load-bearing: 2-5 nodes. This feeds an animation of a few
minutes, not a course. A node earns its place only if the student cannot understand
the correction without it.

- `nodes`: each with a short `id` (`p1`, `p2`, ...), the `concept`, and `why_needed`
  stated in terms of this student's error.
- `edges`: `[a, b]` means a depends on b. Only real dependencies.
- `entry_point`: the id of the node to start from -- the deepest one the student
  does NOT already have. If they have all the prerequisites and the error is a
  slip in applying the rule, the entry point is the rule itself.

Return only JSON matching the schema.
