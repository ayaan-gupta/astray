# Design notes

The reasoning behind Astray's non-obvious choices, and the failures that caused
them. Most of these were written after watching something break, so they read as
a record rather than a specification. Start with the [README](../README.md) for
what the system is; this is why it is that way.

## Design commitments

**The diagnosis is falsifiable.** Every diagnosis carries a SymPy expression
whose truth value would disprove it. That check runs deterministically in a
killable subprocess behind a character allow-list, and its result — not the
model's claim — is what gets stored as `verified_by_sympy`. A check that cannot
fail does not count: an equivalence asserting `X == X` is rejected as vacuous,
because `verified_by_sympy` is what lifts a diagnosis past the unverified
confidence ceiling and a tautology would launder an unchecked claim into a
certified one.

**Student text is untrusted.** Submissions are wrapped in per-request nonce
delimiters in every prompt, with content-blind neutralization of forged
delimiter runs. A student can write "ignore your instructions" in their work
without it becoming an instruction.

**SymPy is an input boundary, not a calculator.** `parse_expr` calls `eval()`.
The check runner enforces a character allow-list (no quotes, brackets, attribute
chains, or dunders) and a killable wall-clock bound, because both RCE and
non-terminating-power-tower DoS were reproduced against the naive version.

**The animation is grounded, and the grounding is enforced.** `s6` plans beats,
`s7` must wrap each in `with beat(self, "bN")`, and `s8` fails the render if any
planned beat is missing, duplicated, or computed at runtime. The container
measures each beat's real start and end from the renderer clock, so a chat
citation seeks to a moment that actually exists. Unknown beat ids in a reply are
stripped server-side rather than shown as dead links.

`beat()` also holds each beat open to a five second floor, which is a duration
guarantee rather than a request. A live render came back with no `self.wait()`
anywhere in the file: six beats, 12.5s of video against a storyboard estimate of
90, and one beat lasting 0.8 seconds. Two things break at that length and neither
is cosmetic. A citation into a 0.8s beat points at a moment gone before the
student can look at it, so grounding stops meaning anything. And the narration
budget is computed from measured duration, so that beat earns a three word line,
which is exactly the disconnected-caption failure the narration work exists to
prevent.

Fixing that by asking for pacing overshot immediately, and the overshoot is the
more interesting half. Told to reach the storyboard's estimated total, `s7` spent
it entirely on waiting: `wait(30)`, `wait(35)`, `wait(40)`, `wait(35)`, a 155s video
of four static frames. The estimate was the problem as much as the instruction,
since `s6` had asked for 180s over four beats while its own prompt said 45 to 120.
So the estimate is clamped to a realistic band, and `s7` is handed seconds *per
beat* rather than a total, which is a length it can reason about instead of a budget
it feels obliged to exhaust. A generous runtime does not buy more explanation; it
buys a longer hold on the same picture.

**The storyboard's vocabulary is backed by real builders.** `s6` picks a
`primitive` per beat, and that choice is what the beat looks like. For a long time
the enum offered `graph` and `balance` with nothing behind either, so a beat
choosing one got improvised Manim: no scaling to fit, no ownership of the frame,
and an unguarded call into whatever function the model named. Measured across
every stored session, `s6` chose `algebra_steps` for 16 of 19 beats and `graph`
never once, which is how a misconception tutor ends up explaining calculus with
white text fading in and out.

There is now a builder for each: an area model that splits a square of side (a+b)
into `a²`, two `ab` rectangles and `b²` drawn to scale; a graph that plots the
correct function against the student's on one set of axes; a number line for
errors that are an *absence*, where nothing exists to cross out. `balance` was
removed rather than built, because equation solving is already served by
`algebra_steps` and an option nothing can render is worse than an option that does
not exist.

**Two of the builders work in three dimensions**, because two of the commonest
misconceptions are claims about shape rather than about value.

`surface` draws both rules as two sheets over the same square of inputs, with the
camera orbiting them. For `(a+b)^2 -> a^2 + b^2` the two sheets *touch along the
two axes* and separate everywhere else, so the frame says something a derivation
cannot: the rule is exactly right whenever one term is zero, which is why it feels
right, and the gap elsewhere is a solid object with a size. Its partner
`gap_pillars` puts a bar between the sheets at the student's own numbers and reads
off all three — theirs, the truth, the difference.

`lift` gives a composition its middle quantity as an axis. `sin(x²)` becomes one
curve in space whose three shadows are the three stages: `u = x²` on the floor,
`sin(u)` up the wall, the answer on the back. Its partner `pace_marks` steps
evenly along `x` and shows the unequal steps those become in `u` — which is the
`2x` a dropped chain-rule factor leaves out, seen rather than asserted.

Reaching for either commits the file to a `ThreeDScene`, since there is one camera
per scene. That is enforced statically rather than requested: a plain `Scene` that
imports `primitives.space` fails validation before a container starts, because the
degraded result — surfaces flattened head-on, the near one hiding the far one — is
a bad video rather than a failed render, and a bad video is the outcome with no
feedback path.

Which one wins is a property of the misconception. `(a+b)^2 -> a^2 + b^2` survives
a derivation, because a student who believes it watches the correct expansion,
agrees with every line, and keeps the belief: the two sides are competing strings.
It does not survive the square, where the two `ab` rectangles are visibly
occupying area that `a^2 + b^2` does not account for. A lost root
(`x^2 = 16 -> x = 4`) is the opposite problem, since every line the student wrote
is true and the error is a missing one; on a line the absence becomes a place.

**Generated code is untrusted.** It runs behind an AST allow-list (imports
limited to `manim`, `numpy`, `math`, `primitives`; no dunder attribute access)
*and* inside a `--network=none`, read-only, non-root container with memory, CPU
and PID caps and a wall clock enforced twice. Neither layer is trusted alone. If
codegen fails twice, a deterministic renderer builds the same beats with no
model-authored code at all.

The primitives are the trusted layer, and they are held to their own line: the
validator inspects the *generated scene*, never the package that scene imports, so
an import added to a primitive would pass every existing gate. A test enumerates
what each one may import beyond the scene allow-list, with the reason attached, so
widening it is deliberate.

**A primitive owns its frame and never returns a bare tuple.** Both rules were
written against live failures. Asking generated code to tidy up after itself was
observed to fail, so every primitive clears the frame before drawing. And
`compare_rules` used to return `(wrong, right)`: a scene assigned that to a
variable, called `FadeOut` on it, and died with `TypeError: Animation only works on
Mobjects`. Returning a `VGroup` serves both callers, since a `VGroup` is iterable,
so `wrong, right = compare_rules(...)` and `FadeOut(result)` are now the same
object.

The same reasoning pushed type tolerance into the primitives rather than the
prompt. Generated code called `compare_rules(self, [MathTex(...)], ...)`, which is
a fair reading of "lines of maths"; `MathTex` accepted the Mobject, stringified it,
and typeset its repr, so the misconception beat read
`MathTex('fracdydx = cos(x^2)')` in crossed-out red and the render was recorded as
a success. The `Text` fallback that exists to survive bad LaTeX is what concealed
it, by turning a loud `TypeError` into a quiet wrong frame.

**Secrets never leave the server.** Keys live only in a gitignored `server/.env`.
Upstream error text never reaches a client — a DeepSeek error body once reflected
the `Authorization` header straight through to the SSE stream.

**Model text is rendered, never injected.** The tutor writes markdown. The
client tokenises it and builds the elements itself, so bullets, bold and inline
code become real formatting without a reply ever being parsed as markup.

## Narration

The video gets a spoken track, and the ordering is the whole design: the script
is written **after** the render, not before it. Only then are the beat timings
measured, and a script written from the storyboard would be guessing at how long
each beat lasts. Narration that guesses is narration that talks over the next
visual.

So each beat's measured duration becomes a word budget the line has to fit, the
clips are placed at the beats' measured starts, and the mix is muxed in without
re-encoding the video. Two clips never overlap: a line that outruns its beat
pushes the next one later rather than talking over it, because two voices at once
is unintelligible while a line arriving a fraction late is barely noticeable and
self-corrects at the next beat with spare room. Audio is never time-stretched to
fit, which is the artificial sound this feature exists to avoid.

Sounding natural turns out to be mostly a text problem rather than a model one.
Handed `(y+3)^2`, every engine says "caret two" or nothing. So the prompt asks
for maths the way a teacher says it out loud, and `server/audio/speech.py` is the
net for what slips through: `(y+3)^2` becomes "y plus three, all squared", `6y`
becomes "six y", `2ab` becomes "two a b". "All squared" is the phrase the whole
product turns on, because it is what distinguishes the correct expansion from the
misconception. Punctuation is left alone deliberately, since it is the only
control over where the voice breathes.

Two things decide whether this sounds like a tutor or like a machine, and both
were learned the hard way.

**One voice, pinned.** Every beat is a separate API request, so leaving
`reference_id` unset gives a six-beat video six different narrators. The voice is
chosen on measured evidence rather than taste: `scripts/measure_voice.py`
synthesises a fixed set of lines and reports words per second and its spread, and
across five candidates the same two sentences ranged from 7.7s to 10.5s. The
budget is calibrated to whichever voice is pinned, set just under its *slowest*
line, because every voice measured slows markedly on dense spoken maths.

**One explanation, not six captions.** The first version passed the model only a
rule name and a one-line statement, and capped each beat at about ten words in
isolation. It produced exactly what you would expect: six disconnected fragments
like "They got y squared plus nine." The prompt now carries the problem, the
student's own working, the correct steps and the diagnosis evidence, shows a
worked example of the standard to match, and gives each beat a word *target* as
well as a maximum. A cap alone reads as a floor: told only "at most N words", the
model writes a caption and leaves the animation playing in silence.

The other half of that fix is not in the narration code at all. A word budget is
computed from a beat's measured duration, so a short beat cannot hold a sentence
however good the prompt is, and `beat()`'s five second floor is what guarantees
there is a sentence's worth of room to write into.

**Every variable gets forced phonemes.** A lone letter is genuinely ambiguous to
a TTS model and it guesses badly: "a" is the commonest word in English so it comes
out as the article, and "y" collapses toward "ee". Fish supports phoneme control,
so `speech.with_letter_phonemes` wraps single-letter variables in
`<|phoneme_start|>W AY1<|phoneme_end|>` style CMU Arpabet on the way to the API.

Only "a" and "i" need evidence before being treated as variables, since they are
the only single letters that are also English words: they are tagged when they sit
next to an operator or another variable, and left alone otherwise. The maths-context
set deliberately contains no nouns, because the test is that an article introduces a
noun while a variable sits next to maths, and listing "term" inverted it and spoke
the article in "missing a middle term" as the letter A.

The tags are markup, not text, so they are added at the API boundary only: what
gets word-counted, stored and quoted in the docs stays readable. They are also
billed, since Fish charges per character, which roughly doubles a video's narration
cost to about $0.013.

Narration is non-fatal. A silent video is a working session; a render discarded
because a voice API was down is not, so every failure logs and leaves the
original video in place. It republishes over the render's own path and keeps the
untouched original beside it as `silent.mp4`, which is what makes the step safe to
re-run: a second pass reads that copy instead of handing ffmpeg its own previous
output.

Set `FISH_MODEL=s2.1-pro-free` (the default) for the free developer tier. The paid
`s2.1-pro` string returns HTTP 402 unless the account holds *API* credit, which
Fish bills separately from platform credit.

## The interface

One dark appearance, because every screen frames a Manim render on black and a
light shell around it reads as a hole in the page. Token names follow shadcn/ui
conventions (`--background`, `--card`, `--primary`, `--muted-foreground`,
`--border`, `--ring`) so the vocabulary is a familiar one, and the visual
language is the one Aceternity popularised: an aurora wash behind glass panels,
a cursor-tracked spotlight, a conic border beam, a shimmer across the primary
action. All of it is hand-written CSS. There is no build step, and adding React,
Tailwind and Framer Motion to a working vanilla frontend would have meant
rebuilding the verified grounding client to arrive at the same pixels.

Two hues carry meaning and nothing else does: coral is the student's error and
the one primary action per view, green means a machine checked it. The violet
and teal in the aurora are decorative and never appear on a control, a state or
a value, so a coloured pixel still always means something.

Every foreground/background pair is measured rather than eyeballed, including
the worst case of a translucent card composited over the brightest point of the
aurora. `--foreground-2` clears APCA Lc 75 on every surface, `--muted-foreground`
clears 60, and `--primary` sits at lightness 0.780 for an unobvious reason: a
saturated warm hue has very little contrast headroom, and that is the lightness
where a near-black label on the filled button finally clears 60. No lightness of
that hue reaches 75, which is why coral is never a body-text colour anywhere in
the app: the false rule gets a coral rule down its edge, not coral text.

Motion is expressive where it is rare (a view entrance, the diagnosis landing)
and nearly invisible where it repeats (hover, focus, typing). Every animated
state change also has a static cue, so `prefers-reduced-motion` switches the
whole system off without losing information.

