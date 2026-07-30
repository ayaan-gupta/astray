# Demo notes

Everything below is already in `data/tutor.db`. Nothing here needs the pipeline
to run, so the demo does not depend on a live model call.

```bash
uv run uvicorn server.app:create_app --factory --port 8000
```

## The golden case

**`(y + 3)^2` → `y^2 + 9`** — session `389e4eaf-58d3-4a79-aafc-037adcfbb26f`

<http://localhost:8000/#/session/389e4eaf-58d3-4a79-aafc-037adcfbb26f>

Status `ready`, 61.1s narrated video, rendered from generated code on the **first
attempt** with no repair round, for $0.0081. Five beats, timings measured by the
container's own clock:

| Beat | Title | Primitive | Start |
|---|---|---|---|
| b1 | Area model of (a+b)² | `areamodel` | 0:00 |
| **b2** | **Missing middle terms** | `areamodel` | **0:07** |
| b3 | Two curves diverge | `graph` | 0:14 |
| b4 | Full expansion | `algebra_steps` | 0:28 |
| b5 | The formula | `algebra_steps` | 0:43 |

Three different kinds of visual, which is the point. **b2** is the beat flagged
`targets_misconception`, so it carries the coral border in the rail, and it is the
strongest frame in the run: the square of side (y+3) drawn to scale, `y²` and `9`
dimmed to show what the student's rule accounts for, and the two `3y` rectangles
held in yellow above the caption *"The middle terms are missing!"*.

**b3 is the one to linger on.** `(y+3)²` and `y²+9` plotted on one set of axes,
separating visibly, with a dashed line at y=1 marking **16.00** against **10.00**.
It is the same arithmetic the narration speaks and the diagnosis states, shown a
third way, and it answers the objection a derivation invites: *how far off is it,
really.*

**Why this replaced the previous golden case.** The old `(y+3)^2` session
(`12a959d5-…`) is still in the database and still works, and it is six beats of
white maths text on black, all `algebra_steps`, because until recently that was
the only primitive with a builder behind it. It also took three render attempts
and two repair rounds to get there. Worth opening side by side if there is time.

## The golden video is narrated

One voice throughout, five lines, on `s2.1-pro-free`. **Play it with sound on.**
The untouched render is kept beside it as `silent.mp4`.

Read end to end, which is how it was written:

> A square of side a plus b is split into a squared, two a b rectangles, and b
> squared. Your rule drops the two a b rectangles, so you get just a squared plus b
> squared. Check with y equals one: the correct value is sixteen, but your answer
> gives ten. The missing six y makes a big difference. Multiply y plus three by
> itself: y times y gives y squared, y times three gives three y, three times y
> gives another three y, and three times three gives nine. The correct formula is a
> plus b, all squared, equals a squared plus two a b plus b squared. You wrote a
> squared plus b squared, which leaves out the two a b. That middle term is
> essential.

Note that it **describes what is on screen** rather than restating the algebra:
"a square of side a plus b is split into…" over the area model, "your rule drops
the two a b rectangles" over the beat that dims them, "sixteen but your answer
gives ten" over the graph marking exactly those two values. That is a consequence
of the beats being longer, not of better prompting: a word budget comes from a
beat's measured duration, and beats of five to eighteen seconds have room for a
sentence where beats of one second had room for a fragment.

Every variable is spoken with forced phonemes, so "y" is the letter and not "ee",
and "a" is the letter and not the article. 51.6s of speech in 61.1s of video, and
every line lands inside its own beat: no overlap, and no overrun on the last one.

The voice is pinned to `ba1cd26ca87b42b2bf7d60c1f65f9242` ("Adam - Calm, Smart").
That is not cosmetic: every beat is a separate API request, so an unset voice
gives one video five different narrators.

To change voice, set `FISH_VOICE_ID` and then re-measure the rate, because the
word budget is calibrated to the voice:

```bash
uv run python scripts/measure_voice.py <voice-id>
```

Candidates measured on the same five lines, for reference:

| Voice | Mean w/s | Slowest | Spread |
|---|---|---|---|
| Adam - Calm, Smart | 2.84 | 2.57 | 0.74 |
| calm storyteller male | 2.73 | 2.50 | 0.66 |
| Jon - Warm & Grounded | 2.33 | 2.04 | 0.65 |
| Nathan - Audiobook | 2.21 | 2.04 | 0.59 |
| CALM- NORMAL | 2.14 | 1.80 | 0.73 |

Every voice slows on dense spoken maths, so the budget is set just under the
*slowest* line rather than the mean.

To re-narrate after any change:

```bash
uv run python scripts/narrate_session.py 389e4eaf-58d3-4a79-aafc-037adcfbb26f
```

Safe to run repeatedly: it reads `silent.mp4` and republishes over `video.mp4`,
so the URL the page already serves keeps working.

## Before recording

Adopt the golden session's anonymous handle so **Insights → Your patterns**
shows history rather than an empty state. Paste into the browser console once:

```js
localStorage.setItem("astray.handle", "judging-v2"); location.reload()
```

The handle is a random per-browser id with no other meaning; the page never
displays it and `/api/insights` never returns it.

## The run of show

1. **Submit** — `/#/`. Type a problem and working to show the input, or run a
   live diagnosis if you want one on camera (~20s to the diagnosis card, the
   animation keeps building behind it).
2. **Diagnosis card** — falsifiable rule `(a+b)^2 -> a^2 + b^2`, the plain
   statement, and the badges: *✓ checked with SymPy · confidence 95%*. The SymPy
   badge is the measured result, not the model's claim.
3. **The animation and the rail** — click **0:07 Missing middle terms**; the
   player seeks and the chip fills coral. Then **0:14 Two curves diverge** for the
   graph. The rail scrolls the active beat into view, and the trailing fade shows
   there are more beats past the edge.
4. **Chat** — two exchanges are already in the database, and both replies carry
   chips. The one to read aloud is *"Is my answer ever right, or is it always
   wrong?"*: the tutor answers the actual question (only when one term is zero)
   and cites three beats, including the area model by name. Click a chip and the
   player jumps to that beat's measured start with the rail's active chip
   following.
5. **Insights** — `/#/insights`. The peers are other students whose different
   problems canonicalised onto the same misconception, including one written in
   different letters (`(x + 5)^2`). That is cross-session pattern tracking visible
   in the UI.

## The other sessions

Kept for variety, each showing a different primitive doing the work no other one
could:

| Problem | Misconception | The beat that argues |
|---|---|---|
| `d/dx sin(x²)` → `cos(x²)` | dropped chain-rule factor | `graph`: `2x cos(x²)` against `cos(x²)`, two unrelated curves |
| `x² = 16` → `x = 4` | lost the negative root | `numberline`: a second dot lands where the student had nothing |

The lost-root case is the clearest argument for having more than one primitive.
Every line the student wrote is true, so there is nothing to cross out and a
side-by-side comparison has no content; the error is an *absence*, and on a line
an absence is a place.

## Do not use this session on camera

`e4c561a6-402e-442d-ad2a-e6f5508b6dc5` (`(x + 5)^2`) is a peer that makes the
insights count real, and it has chat history, but it was rendered before the
primitives existed and its comparison beat is unreadable: the student's own
expression is set smaller than the correct one with a thick red **X** drawn
straight over it. Frames at 31s, 33s, 35s and 37s all look the same, so it is not
a transient mid-animation state. It exists to be counted, not to be shown.
