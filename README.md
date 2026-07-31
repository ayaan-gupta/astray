<div align="center">

# Astray

### Find where your reasoning went astray — then see it.

Every math tool tells you the answer is wrong. **Astray tells you what you believe.**
It finds the exact step your reasoning left the correct path, names the false rule
you were actually applying, and builds an animation of *that misconception*.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/ayaan-gupta/astray)

</div>

---

## The problem

A student writes `(y+3)² = y² + 9` and gets a red cross. They now know this
answer was wrong. They do **not** know that they hold a rule — *distribute the
exponent across a sum* — that will break `log(x+y)`, `√(a²+b²)` and `1/(a+b)` in
exactly the same way for the rest of the year.

Worked solutions don't fix this. A student who believes that rule reads the
correct expansion, agrees with every line, and keeps the belief. The two sides
are competing strings, and nothing forces a choice between them.

So Astray doesn't show the correct answer. It shows the student's own rule
**failing**, in a picture built for their specific error.

## What it does

<div align="center">

<img src="docs/assets/surfaces.gif" width="720" alt="Two surfaces over the same square of inputs: (a+b)² above a²+b², touching along both axes and separating everywhere else, with the camera orbiting.">

**`(a+b)² → a² + b²`** · both rules as surfaces, orbiting

</div>

The sheets **touch along the two axes** and separate everywhere else. That says
something a derivation cannot: the rule is exactly right whenever a term is zero
— which is *why it feels right*, because nearly every case the student checked
was one of those — and the gap elsewhere is a solid object with a size. A second
beat then measures that gap at their own numbers: `16` against `10`, missing `6`.

<div align="center">

<img src="docs/assets/lift.gif" width="720" alt="A composition drawn in three dimensions: one curve whose three shadows are the inner stage on the floor, the outer stage up the wall, and the answer on the back. Evenly spaced marks along x become visibly unequal along u.">

**`d/dx sin(x²) → cos(x²)`** · the hidden middle stage, given an axis

</div>

A dropped chain-rule factor goes missing because the middle quantity is invisible
in flat algebra. So give it an axis: `sin(x²)` becomes one curve in space whose
three shadows are its three stages — `u = x²` on the floor, `sin(u)` up the wall,
the answer on the back. Step evenly along `x`, and the steps in `u` come out
unequal. That spacing ratio **is** the `2x` the student left out.

<div align="center">

<img src="docs/assets/numberline.gif" width="560" alt="A number line with a solution marked at 4, then a second solution appearing at -4.">

**`x² = 16 → x = 4`** · deliberately flat

</div>

Not every error deserves a camera. This one is an *absence* — every line the
student wrote is true, and the mistake is a missing one. There's nothing to cross
out, so a second dot arriving where they had nothing is the whole argument. A
surface here would be spectacle with no content.

**Then the animation becomes addressable.** A tutor chat answers follow-up
questions and cites the moment it's talking about — a chip you click to seek the
player to that beat's measured start. Ask "show me the part with actual numbers"
and it lands on the frame with the gap pillar.

Every frame above came out of the real pipeline, from a student's wrong answer.
Nothing is hand-drawn.

## How it works

Nine stages, each a typed artifact persisted with its own token count and cost.
The first two are what make the rest student-specific: every later stage is told
what *this* student believes, not what the topic is.

| Stage | Does | Artifact |
|---|---|---|
| `s0_ingest` | Normalize typed work, or transcribe a photo | `Submission` |
| `s1_diagnose` | Solve independently, find the divergence, name the false rule | `Diagnosis` |
| `s2_intent` | What must change in this student's head | `IntentAnalysis` |
| `s3_prereq` | What they must already know for the argument to land | `PrerequisiteGraph` |
| `s4_curriculum` | Order it into a teaching path | `CurriculumPlan` |
| `s5_math` | The exact expressions the animation will show | `MathContent` |
| `s6_visual` | Beats, each choosing a visual primitive | `Storyboard` |
| `s7_scene` | Manim code for the whole scene | `SceneCode` |
| `s8_validate` | Static gate before any container starts | `ValidationReport` |

Then a sandboxed render, a bounded repair loop, and narration written **after**
the render — because only then are the beat durations *measured*, and a script
written from the storyboard would be guessing at how long each beat lasts.

### Four things that make it hold up

**The diagnosis is falsifiable.** Every diagnosis ships a SymPy expression whose
truth value would disprove it, run deterministically in a killable subprocess.
That result — not the model's claim — is what gets stored. A check that cannot
fail is rejected as vacuous, because a tautology would launder an unchecked claim
into a certified one. **The model does not get to certify itself.**

**Generated code is untrusted, twice.** An AST allow-list *and* a
`--network=none`, read-only, non-root container with memory, CPU and PID caps.
Neither layer is trusted alone. If codegen fails twice, a deterministic renderer
builds the same beats with no model-authored code at all.

**Student text is untrusted.** Submissions are wrapped in per-request nonce
delimiters in every prompt. A student can write "ignore your instructions" in
their working without it becoming one.

**The grounding is enforced, not hoped for.** `s6` plans beats, `s7` must wrap
each in `with beat(self, "bN")`, `s8` fails the render if any is missing or
computed at runtime, and the container measures every beat's real start and end
from the renderer clock. That's why a citation lands on a frame that exists.

The failures behind each of these — and the ones that cost a render before they
were caught — are in **[docs/DESIGN.md](docs/DESIGN.md)**.

## Stack

Python 3.12 · FastAPI · Pydantic v2 · SQLite · Manim CE in Docker · SymPy ·
DeepSeek (reasoning + fast) · Gemini (vision) · Fish Audio (TTS) ·
vanilla JS, no build step

## Try it

**Deployed demo** — the three sessions above, with their animations, measured
timings and tutor transcripts, served as static files. Beat seeking and citation
chips all work; diagnosing new work needs the full stack.

```bash
uv run python scripts/export_demo.py   # writes public/
vercel deploy --prod                   # or use the button above
```

**The whole thing**, locally:

```bash
uv sync
cp server/.env.example server/.env     # add your DeepSeek key
uv run uvicorn server.app:create_app --factory --port 8000
```

`FAKE_LLM=1` runs the entire chain against canned responses — no network, no
cost. `RENDER_ENABLED=0` plans animations without containers. Rendering itself
needs Docker and `manimcommunity/manim:stable`.

## Development

```bash
uv run pytest          # 597 tests; no network and no Docker — both are mocked
uv run ruff check . && uv run ruff format --check .
```

The interesting failures here are *visual*, and the primitives import `manim`,
which only exists inside the render image — so three scripts cover what a test
cannot reach:

```bash
uv run python scripts/check_primitives.py            # every primitive, one frame each
uv run python scripts/check_primitives.py --pass space   # just the 3D ones
uv run python scripts/run_session.py --preset binomial   # a whole session, real calls
```

`check_primitives.py` deliberately includes the abuse the live pipeline produced
as well as the correct usage: a Mobject where a string belongs, a function with a
pole inside the plot window, a label too wide for its cell. Each cost a render or
a frame before it was handled.

```bash
uv run python -m evals.diagnosis.run   # 20 labelled cases against the real model
```

Rule match is **not** a trustworthy gate — the scorer rejects substantively
correct diagnoses on notation. That number is left honest rather than tuned.
Topic match and verification rate are reliable.
