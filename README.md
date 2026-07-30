# Astray

**Finds where your math reasoning went astray — then shows you.**

Most math tools tell you the answer is wrong. Astray finds the exact step where
your reasoning left the correct path, names the false rule you were actually
applying, and builds an animated explanation of that specific misconception.

Status: **complete end to end.** Submit typed or photographed work, get a
falsifiable diagnosis, watch an animation generated for your specific error, and
ask a tutor that cites moments in that animation by timestamp.

## What it does today

A student submits a problem and their own attempted solution (typed, or as a
photo transcribed by a vision model). Astray:

1. Solves the problem correctly, independently of the student's work.
2. Finds the **divergence index** — the first step where the two solutions part.
3. Names the **false rule** the student appears to be applying, as a rewrite
   (`(a+b)^2 -> a^2 + b^2`), not as a vague topic label.
4. Emits a SymPy check that would falsify its own diagnosis, runs it in a
   sandbox, and **overwrites** its confidence claim with the measured result.
   The model does not get to certify itself.
5. Canonicalizes the rule against a growing misconception taxonomy, so the same
   error made with different letters lands on the same entry — which is what
   makes cross-session pattern tracking possible.

When the work is already correct, Astray says so and stops: the session ends at
`correct` with no misconception attached. Inventing an error the student did not
make is treated as exactly as bad as missing a real one, and a "you were right"
result must never enter their misconception history.

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

**Generated code is untrusted.** It runs behind an AST allow-list (imports
limited to `manim`, `numpy`, `math`, `primitives`; no dunder attribute access)
*and* inside a `--network=none`, read-only, non-root container with memory, CPU
and PID caps and a wall clock enforced twice. Neither layer is trusted alone. If
codegen fails twice, a deterministic renderer builds the same beats with no
model-authored code at all.

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

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp server/.env.example server/.env   # then add your DeepSeek key
uv run uvicorn server.app:create_app --factory --port 8000
```

Set `FAKE_LLM=1` to run against canned responses with no network and no cost.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness, plus whether vision input is configured |
| `POST` | `/api/sessions` | Create a session from typed problem + work |
| `POST` | `/api/sessions/{id}/photo` | Transcribe handwritten work into the session |
| `PUT` | `/api/sessions/{id}/submission` | Confirm/correct the transcription before diagnosis |
| `GET` | `/api/sessions/{id}` | Session state and diagnosis, if ready |
| `GET` | `/api/sessions/{id}/stream` | SSE: diagnosis, then s2–s8 and the render |
| `GET` | `/api/sessions/{id}/beats` | Beat rail: plan plus measured timings |
| `GET` | `/media/{id}/video.mp4` | The rendered animation (range requests) |
| `POST` | `/api/sessions/{id}/chat` | Grounded reply citing `[beat:bN]` |
| `GET` | `/api/sessions/{id}/peers` | "N other students made this error" |
| `GET` | `/api/insights` | Misconception frequency and personal history |

`/stream` claims a session with a compare-and-swap, so concurrent connections
cannot double-bill the same run; reconnecting to a finished session replays the
stored result rather than re-running it.

Photo input is never diagnosed until the student confirms it. Transcribed steps
are delimiter-free LaTeX (`x^{2} + 25 = 36`, not `$x^{2} + 25 = 36$`), normalized
server-side so the format does not vary between calls, and meant to be rendered
with KaTeX in the review field.

## Development

```bash
uv run pytest          # 340 tests; no network and no Docker — both are mocked
uv run ruff check .
uv run ruff format --check .
uv run python -m evals.diagnosis.run   # 20 labelled cases against the real model
```

The eval harness scores rule match, topic match, and SymPy verification rate.
Rule match is currently **not** a trustworthy gate — see the plan's Definition
of Done for why the scorer rejects substantively correct diagnoses on notation.
Topic match and verification rate are reliable.

Rendering needs Docker and `manimcommunity/manim:stable`. Set `RENDER_ENABLED=0`
to plan animations without running containers — the pipeline still produces beats,
so chat stays grounded by title.

Design and plan documents live in `docs/superpowers/`. `DEMO.md` names the
prepared session to open, and what is and is not live in it.
