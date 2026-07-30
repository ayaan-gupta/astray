# Demo notes

Everything below is already in `data/tutor.db`. Nothing here needs the pipeline
to run, so the demo does not depend on a live model call.

```bash
uv run uvicorn server.app:create_app --factory --port 8000
```

## The golden case

**`(y + 3)^2` → `y^2 + 9`** — session `12a959d5-3a99-4343-a360-b681dd2aebbc`

<http://localhost:8000/#/session/12a959d5-3a99-4343-a360-b681dd2aebbc>

Status `ready`, 34.8s video rendered from generated code (attempt 3, after two
repair rounds), 6 beats with timings measured by the container's own clock:

| Beat | Title | Start |
|---|---|---|
| b1 | Problem: (y+3)² | 0:00 |
| b2 | Correct Expansion | 0:04 |
| b3 | Buggy Rule Applied | 0:11 |
| **b4** | **Side-by-Side Comparison** | **0:16** |
| b5 | Concrete Check with y=1 | 0:23 |
| b6 | Correct Identity | 0:29 |

b4 is the beat flagged `targets_misconception`, so it carries the coral border
in the rail. It is also the strongest single frame in the run: the student's
`= y² + 3²` struck through in red beside `= y² + 6y + 9` in green, both at the
same size.

**Why this one and not `(x + 5)^2`.** The other finished session renders its
equivalent comparison beat badly — the student's own expression is set smaller
than the correct one and a thick red **X** is drawn straight over it, so the
thing the beat exists to show is unreadable. Frames extracted at 31s, 33s, 35s
and 37s all look the same, so it is not a transient mid-animation state.

## The golden video is narrated

It has a spoken track, six lines, one per beat, on the `s2.1-pro-free` model.
**Play it with sound on.** The untouched render is kept beside it as `silent.mp4`
if you want the quiet version on camera instead.

| Beat | Narration | Speech | Beat window |
|---|---|---|---|
| b1 | "Expand y plus three, all squared." | 3.5s | 5.0s |
| b2 | "y squared plus three y plus three y plus nine." | 3.2s | 6.4s |
| b3 | "They got y squared plus nine." | 2.3s | 4.8s |
| **b4** | "The correct expansion has six y, the buggy does not." | 3.7s | 7.0s |
| b5 | "Let y equal one, correct gives sixteen, buggy gives ten." | 6.3s | 6.6s |
| b6 | "The formula a squared plus two a b plus b squared." | 5.1s | 5.0s |

Measured in the published file, speech begins at 5.17, 11.57, 16.38, 23.38 and
29.91 seconds against measured beat starts of 5.0, 11.4, 16.2, 23.2 and 29.8. The
~0.17s offset is the mp3's own lead-in, which is the head-room the word budget
reserves. Duration is unchanged at 34.8s and nothing is truncated.

b6 is the one line that had to be pulled 0.07s earlier to fit, which is the
tail-fit rule doing its job. It is also the one line worth knowing is terse: a
5.0s final beat cannot hold "a plus b, all squared, is a squared plus two a b plus
b squared" (13 words, about 7s spoken), so it states the right-hand side only
while the screen carries the identity. If you want the full sentence spoken, the
render's final beat needs to be longer, not the budget looser.

To re-narrate after any change:

```bash
uv run python scripts/narrate_session.py 12a959d5-3a99-4343-a360-b681dd2aebbc
```

That is safe to run repeatedly. It reads `silent.mp4` and republishes over
`video.mp4`, so the URL the page already serves keeps working.

## Before recording

Adopt the golden session's anonymous handle so **Insights → Your patterns**
shows history rather than an empty state. Paste into the browser console once:

```js
localStorage.setItem("astray.handle", "other-student"); location.reload()
```

The handle is a random per-browser id with no other meaning; the page never
displays it and `/api/insights` never returns it.

## The run of show

1. **Submit** — `/#/`. Type a problem and working to show the input, or run a
   live diagnosis if you want one on camera (~20s to the diagnosis card, the
   animation keeps building behind it).
2. **Diagnosis card** — falsifiable rule `(a+b)^2 -> a^2 + b^2`, the plain
   statement, and the badges: *✓ checked with SymPy · diverges at step 2 ·
   confidence 95% · 1 other student made this error*. The SymPy badge is the
   measured result, not the model's claim.
3. **The animation and the rail** — click **0:16 Side-by-Side Comparison**; the
   player seeks and the chip fills coral. The rail scrolls the active beat into
   view, and the trailing fade shows there are more beats past the edge.
4. **Chat** — three exchanges are already in the database, and every reply
   carries chips. The strongest one to read aloud is *"Where exactly does the
   animation show my rule failing?"*: the tutor separates where the rule is
   **shown** (0:11 Buggy Rule Applied) from where it is **refuted**
   (0:16 Side-by-Side Comparison, 0:23 Concrete Check). Click
   **▶ 0:23 · Concrete Check with y=1** and the player jumps to 23.25s against
   that beat's measured start of 23.2s, with the rail's active chip following.
5. **Insights** — `/#/insights`. *Your patterns* 1×, *Across everyone*
   **2 students**. The peer is a different student who submitted `(x + 5)^2`,
   a different problem in different letters that canonicalised onto the same
   misconception. That is requirement 6 visible in the UI.

## Do not use the other session on camera

`e4c561a6-402e-442d-ad2a-e6f5508b6dc5` (`(x + 5)^2`) is the peer that makes the
count real, and it also has chat history — but its comparison beat is the
broken one described above. It exists to be counted, not to be shown.
