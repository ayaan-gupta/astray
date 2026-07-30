# Demo notes

Everything below is already in `data/tutor.db`. Nothing here needs the pipeline
to run, so the demo does not depend on a live model call.

```bash
uv run uvicorn server.app:create_app --factory --port 8000
```

## The golden case

**`(y + 3)^2` → `y^2 + 9`** — session `84b52e28-05dd-4a33-a295-e3a3b6c167e7`

<http://localhost:8000/#/session/84b52e28-05dd-4a33-a295-e3a3b6c167e7>

Status `ready`, 33.8s narrated video, five beats with timings measured by the
container's own clock:

| Beat | Title | Primitive | Start |
|---|---|---|---|
| b1 | Student's rule | `algebra_steps` | 0:00 |
| b2 | Number check | `algebra_steps` | 0:07 |
| **b3** | **Area model** | **`areamodel`** | **0:13** |
| b4 | Correct expansion | `algebra_steps` | 0:19 |
| b5 | Comparison | `algebra_steps` | 0:27 |

**b3 is the frame to hold on.** Two squares of side (1+3) side by side. The left
one is tiled completely, `1²` and `3²` in blue with the two `1·3` rectangles in
yellow, and totals `(1+3)² = 16`. The right one is the same size with the two
rectangles left as dashed holes, and totals `1² + 3² = 10`. The student's rule does
not produce a *different* square; it produces a square with something missing, and
the six units of missing area are visible.

The cells are drawn to their real proportions, so `1²` is a small corner and `3²`
is a large one. That is the whole reason the picture argues anything: a schematic
diagram with arbitrary sizes would show the same four labels while giving up the
only thing a picture adds over a derivation.

## The golden video is narrated

One voice throughout, five lines, on `s2.1-pro-free`. **Play it with sound on.**
The untouched render is kept beside it as `silent.mp4`.

Read end to end, which is how it was written:

> You thought a plus b, all squared, equals a squared plus b squared. For y equals
> one, your rule gives ten, but the true answer is sixteen. An area model shows the
> missing terms: two rectangles of three y each. Multiply y plus three by itself
> gives y squared, plus two times three y, plus nine. So your answer is y squared
> plus nine, missing the middle term six y.

It **names what is on screen** rather than restating the algebra: "two rectangles
of three y each" over the beat that draws exactly those two rectangles. That is a
consequence of the beats being long enough to say something, not of better
prompting: a word budget comes from a beat's measured duration, and every beat now
has a five second floor.

Every variable is spoken with forced phonemes, so "y" is the letter and not "ee",
and "a" is the letter and not the article.

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

To re-narrate after any change, which costs one model call and a few seconds of
TTS against a render that already exists:

```bash
uv run python scripts/narrate_session.py 84b52e28-05dd-4a33-a295-e3a3b6c167e7
```

Safe to run repeatedly: it reads `silent.mp4` and republishes over `video.mp4`,
so the URL the page already serves keeps working.

## Before recording

Adopt the golden session's anonymous handle so **Insights → Your patterns**
shows history rather than an empty state. Paste into the browser console once:

```js
localStorage.setItem("astray.handle", "astray-final"); location.reload()
```

The handle is a random per-browser id with no other meaning; the page never
displays it and `/api/insights` never returns it.

## The run of show

1. **Submit** — `/#/`. Type a problem and working to show the input, or run a
   live diagnosis if you want one on camera (~20s to the diagnosis card, the
   animation keeps building behind it).
2. **Diagnosis card** — falsifiable rule `(a+b)^2 -> a^2 + b^2`, the plain
   statement, and four badges: *✓ checked with SymPy · diverges at step 1 ·
   confidence 95% · 2 other students made this error*. The SymPy badge is the
   measured result, not the model's claim.
3. **The animation and the rail** — click **0:13 Area model**; the player seeks to
   13.85s against that beat's measured start of 13.8s, and the rail's active chip
   follows the playhead as the video runs. Every chip is also a real accessible
   control ("Jump to 0:13, Area model"), so the rail is keyboard-reachable.
4. **Chat** — three exchanges are already in the database and every reply carries
   chips. The one to read aloud is *"Is my answer ever right, or is it always
   wrong?"*: the tutor answers the actual question, cites four beats, and describes
   the area model by its parts ("split into four pieces, `y^2`, `3y`, `3y`, and
   `9`"). It can talk about the geometry because the geometry is there.
5. **Insights** — `/#/insights`. *Your patterns* lists three different
   misconceptions, one per demo session: *freshman's dream*,
   *square-root-single-sign*, *chain-rule-omitted*. *Across everyone* shows
   **3 students** on the binomial one, and the diagnosis card said *2 others* for
   the same reason: one of those three wrote `(x + 5)^2`, a different problem in
   different letters that canonicalised onto the same entry. The canonical
   statements are the taxonomy's own words, not the model's phrasing for this
   session, which is what makes them countable.

## The other two, each arguing a different way

| Session | Problem | The beat that argues |
|---|---|---|
| `5901c1da-fe6b-4e55-a4f6-31bd5834cb38` | `x^2 = 16` → `x = 4` | `numberline`: a second dot lands at −4 where the student had nothing |
| `224c344d-7e25-47d1-959d-a204edd29232` | `d/dx sin(x²)` → `cos(x²)` | `graph`: `2x cos(x²)` against `cos(x²)`, marked at x=0 as 0 against 1 |

The lost-root case is the clearest argument for having more than one primitive.
Every line the student wrote is true, so there is nothing to cross out and a
side-by-side comparison has no content; the error is an *absence*, and on a line
an absence is a place.

## Worth knowing before you record

**Per-run variation is real, and it has one cause.** Beats built from primitives
render cleanly; beats where the model positions its own mobjects sometimes do not.
Across a dozen runs the failures were all the same shape: a diagram at a quarter of
its intended size jammed against the frame edge, or three lines of text stacked
past the bottom. `layout.fit` and the `caption` stacking exist for exactly this and
are now in the s7 prompt, but a hand-built beat can still come out worse than a
primitive one. If a fresh render looks wrong, that is where to look first.

## Do not use this session on camera

`e4c561a6-402e-442d-ad2a-e6f5508b6dc5` (`(x + 5)^2`) is a peer that makes the
insights count real, and it has chat history, but it was rendered before the
primitives existed and its comparison beat is unreadable: the student's own
expression is set smaller than the correct one with a thick red **X** drawn
straight over it. It exists to be counted, not to be shown.

`12a959d5-3a99-4343-a360-b681dd2aebbc` is the same `(y+3)^2` problem from before
the primitives, and it is worth opening *deliberately* beside the golden case: six
beats of white maths text on black, 34.8s, three render attempts and two repair
rounds. Same pipeline, same problem, no picture.
