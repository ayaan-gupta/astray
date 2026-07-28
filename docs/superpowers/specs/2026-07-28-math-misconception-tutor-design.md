# Math Misconception Tutor — Design

**Date:** 2026-07-28
**Status:** Approved for planning

## 1. Product

A student submits a math problem plus their own attempted solution. The system:

1. Diagnoses the **specific buggy rule** behind their error, not just right/wrong.
2. Runs a multi-stage planning pipeline (adapted from
   [Math-To-Manim](https://github.com/HarleyCoops/Math-To-Manim)) that turns the diagnosis into a
   Manim animation targeting *their* gap.
3. Opens a chat tutor whose answers cite **specific moments in that animation**, clickable to seek
   the video.
4. Closes with a generated checkpoint that tests whether the misconception is actually gone.
5. Logs every diagnosis against a canonical taxonomy so patterns surface across students.

The unifying idea: **the diagnosis is the spine.** It seeds the animation's focus, the chat's system
prompt, and the checkpoint's distractors. Animation and chat are two views of one diagnosis, not two
features side by side.

### Success criteria

- A student goes from "I got this wrong" to a verified "I see where I went wrong" in one flow.
- Chat citations resolve to real timestamps in the rendered video, always.
- A failed checkpoint routes back into tutoring rather than dead-ending.
- The insights page shows frequency *and resolution rate* per misconception.

### Non-goals

- Grading, gradebooks, assignments, LMS integration.
- Accounts, passwords, teacher roles (anonymous handle only).
- Real-time collaboration or multi-student sessions.
- Being right about every domain. Open-domain is accepted with mitigations (§7, §17).

## 2. Verified platform constraints

These were established by probing the live APIs on 2026-07-28. **Do not re-litigate; do re-verify if
behavior changes.**

### DeepSeek

Models: `deepseek-v4-flash`, `deepseek-v4-pro`. 1M context, 384K max output. Thinking mode is **on by
default**.

| Approach | Result |
|---|---|
| `tool_choice: {type: function, ...}` + thinking on | **400** — `"Thinking mode does not support this tool_choice"` |
| `tool_choice: "required"` + thinking on | **400** — same error |
| `tool_choice: "auto"` + thinking on | Returns prose, ignores the tool. Unreliable. |
| `thinking: {type: disabled}` + forced `tool_choice` | Works. Guaranteed tool call, no reasoning. |
| `response_format: {type: json_object}` + thinking on | **Works.** Valid JSON in `content`, plus `reasoning_content`. |

**Consequence:** structured output uses **JSON mode**, not tool-use, as the primary path — the only
way to keep thinking enabled for the stages that need reasoning. Schema is injected into the prompt;
enforcement comes from Pydantic validation plus a retry that feeds the validation error back.

A `strict=True` mode using `thinking: disabled` + forced `tool_choice` is available for stages where
schema adherence matters more than reasoning (taxonomy slugging, checkpoint grading).

**Bonus:** `reasoning_content` is a genuine per-stage chain of thought. Persist it — it powers the
live "watch the tutor think" progress view and mirrors Math-To-Manim's inspectable run artifacts.

Pricing (per 1M tokens):

| Model | Input (miss) | Input (cache hit) | Output |
|---|---|---|---|
| `deepseek-v4-flash` | $0.14 | $0.0028 | $0.28 |
| `deepseek-v4-pro` | $0.435 | $0.003625 | $0.87 |

Cache hits are ~50× cheaper, so the shared pipeline preamble goes **first** in every prompt to
maximize prefix reuse.

### Gemini (vision only)

`gemini-3.5-flash-lite` — $0.30/1M in, $2.50/1M out. Verified: correctly transcribed handwritten-style
work, returned structured JSON via `responseMimeType: application/json`, and — critically — **did not
correct the student's error** when instructed not to. One photo ≈ 1,200 tokens ≈ $0.0005.

`gemini-2.5-flash-lite` ($0.10/$0.40) is the configured budget alternative. Do not economize below
this: a bad transcription makes the system diagnose a misconception the student never made, which is
worse than having no photo support.

Auth header is `x-goog-api-key`. Endpoint
`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`.

### Host

macOS. Node 26, `uv`, ffmpeg, Docker present. **No LaTeX and no Manim on the host** — rendering is
Docker-only, non-negotiable.

## 3. Architecture

One Python backend owns the pipeline, LLM calls, persistence, and render orchestration. One React
client. No duplicated model layer.

```
┌────────────────┐   SSE + REST   ┌──────────────────────────────┐
│  React client  │◄──────────────►│  FastAPI backend             │
│  Vite/TS/Tail  │                │  charter chain · tutor · db  │
└────────────────┘                └───────────┬──────────────────┘
                                              │
                    ┌─────────────────────────┼──────────────────────┐
                    ▼                         ▼                      ▼
            DeepSeek (reason)         Gemini (vision)      Docker: manimcommunity
            v4-pro / v4-flash        3.5-flash-lite        → video.mp4 + manifest.json
                                              │
                                              ▼
                                     SQLite (WAL) + media/
```

**Stack:** FastAPI on Python 3.12 (`uv`), Pydantic v2, SQLite WAL, SymPy, httpx. Frontend Vite +
React + TS + Tailwind, KaTeX, native `<video>`.

**Why one Python backend:** the pipeline, Manim primitives, Docker orchestration, and SymPy
verification are all Python. Splitting the LLM layer into a Node BFF would duplicate every stage
contract for no gain.

### Repo layout

```
server/
  app.py                  FastAPI app, routers, SSE endpoints
  config.py               Settings: keys, model routing, timeouts, feature flags
  llm/
    deepseek.py           JSON-mode + strict-tool-call client, retries, cache-friendly prompts
    vision.py             VisionProvider interface + GeminiVision impl
    accounting.py         Per-call token/cost ledger
  charter/
    contracts.py          Pydantic models for every stage in/out — the typed spine
    chain.py              Orchestrator: run stages, persist artifacts, emit progress events
    prompts/              One prompt template per stage (versioned)
    stages/
      s0_ingest.py        Normalize input; vision transcription when photo
      s1_diagnose.py      Misconception diagnosis  ← the core
      s2_intent.py        Learner Intent Analysis
      s3_prereq.py        Prerequisite Mapping
      s4_curriculum.py    Curriculum Building
      s5_math.py          Mathematics Selection
      s6_visual.py        Visual Planning → beats
      s7_scene.py         Scene Composition → Manim code
      s8_validate.py      Static validation gate
  render/
    primitives/           Vetted Manim scene library (mounted into container)
      beats.py            beat() context manager → timing manifest
      numberline.py  areamodel.py  algebra_steps.py  graph.py  balance.py
    validator.py          AST deny-list, import allow-list, beat-coverage check
    runner.py             docker run orchestration, timeout, log capture
    repair.py             Bounded render-error → LLM repair loop
    storyboard.py         Deterministic no-LLM fallback renderer
  tutor/
    chat.py               Grounded chat: manifest + diagnosis → [beat:id] citations
    checkpoint.py         Verification item generation + grading
  store/
    db.py  models.py  migrations/
    taxonomy.py           Canonicalization: free-text diagnosis → stable misconception id
    seed_taxonomy.py      ~40 documented misconceptions
    insights.py           Aggregate queries
web/
  src/routes/    Submit  Session  Insights
  src/components/  ProblemForm PhotoDrop LatexField PipelineProgress DiagnosisCard
                   VideoTheater BeatRail ChatPanel Checkpoint InsightsBoard
media/<session_id>/   scene.py  video.mp4  manifest.json  render.log
tests/
docs/superpowers/specs/
```

## 4. Pipeline — the adapted charter chain

Math-To-Manim's six-stage chain, with ingest and diagnosis prepended. Diagnosis is what makes every
downstream stage student-specific instead of a generic topic explainer.

| # | Stage | In | Out | Model |
|---|---|---|---|---|
| s0 | Ingest | raw text or photo | `StudentSubmission` | gemini-3.5-flash-lite *(photo only)* |
| **s1** | **Diagnose** | submission | **`Diagnosis`** | **v4-pro** |
| s2 | Learner Intent | submission + diagnosis | `IntentAnalysis` | v4-flash |
| s3 | Prerequisite Mapping | intent | `PrereqGraph` | v4-flash |
| s4 | Curriculum Building | prereqs | `Curriculum` | v4-flash |
| s5 | Mathematics Selection | curriculum | `MathContent` | v4-flash |
| s6 | Visual Planning | math content | `Storyboard` (**beats**) | v4-flash |
| s7 | Scene Composition | storyboard | `SceneCode` | **v4-pro** |
| s8 | Validate | scene code | `ValidationReport` | *deterministic* |

Every stage: a Pydantic contract, JSON-mode call, validation, bounded retry with the error fed back,
and an artifact row written to `run_artifacts` (payload + `reasoning_content` + model + tokens + ms).
Stage prompts are versioned so artifacts stay interpretable as prompts evolve.

The chain emits SSE progress events throughout: `stage_started`, `stage_reasoning` (streamed
`reasoning_content`), `stage_completed`, `render_progress`, `done`.

## 5. Contracts (abridged)

```python
class StudentSubmission(BaseModel):
    problem: str                     # LaTeX-ish
    steps: list[str]                 # student's work, in order
    prose: str | None                # free-text explanation of reasoning
    source: Literal["typed", "photo"]
    transcription_confidence: float | None
    student_corrected: bool          # did the student edit the transcription

class Diagnosis(BaseModel):
    correct_solution: list[str]
    verified_by_sympy: bool
    divergence_index: int | None     # first student step that departs
    buggy_rule: str                  # e.g. "(a+b)^2 -> a^2 + b^2"
    misconception_statement: str     # student-facing, one sentence
    evidence: list[str]              # cites specific student steps
    confidence: float
    competing_hypotheses: list[str]
    is_unclear: bool                 # true -> tutor asks instead of asserting
    clarifying_question: str | None

class Beat(BaseModel):
    id: str                          # "b1", "b2", ...
    title: str                       # shown on the beat rail
    teaching_purpose: str            # why this beat exists
    on_screen: str                   # what the viewer sees
    targets_misconception: bool      # at least one must be True
    primitive: Literal["numberline","areamodel","algebra_steps","graph","balance","custom"]

class Storyboard(BaseModel):
    beats: list[Beat]                # 4-8
    total_estimated_seconds: int

class BeatTiming(BaseModel):
    id: str; start: float; end: float   # measured at render time
```

## 6. Diagnosis — falsifiable, not vibes

The weak version asks an LLM "what's wrong?" and receives fluent guesswork. Instead, `s1_diagnose`
runs four sub-steps in one reasoning pass:

1. **Solve correctly first.** Produce the correct solution independently of the student's work, and
   **verify with SymPy** wherever the problem is symbolically checkable (`sympy.simplify(lhs - rhs)
   == 0`, `solveset`, `diff`, `integrate`). Sets `verified_by_sympy`. The diagnosis is anchored to a
   checked solution rather than to the model's impression of one.
2. **Align and locate divergence.** Match student steps against correct steps; find the first index
   where they depart. Everything after the divergence is downstream noise, not separate errors.
3. **State the buggy rule explicitly** — `(a+b)^2 -> a^2 + b^2`, not "confused about exponents." A
   rule is falsifiable; a vague description is not. This is the central move of the whole product.
4. **Falsify.** Generate probe instances where the buggy rule predicts a specific value, and check
   whether the student's *other* steps are consistent with it. Consistent → confidence up. Contradicted
   → emit competing hypotheses, or set `is_unclear` and hand a clarifying question to the chat rather
   than asserting a wrong diagnosis.

`is_unclear` is a first-class outcome. Confidently misdiagnosing a student is the worst failure this
product can produce — worse than admitting uncertainty — so the pipeline is allowed to say "I'm not
sure yet, tell me more" and open chat before generating an animation.

### SymPy scope

SymPy verification covers algebraic manipulation, equation solving, differentiation, integration, and
simplification. It does **not** cover word problems, proofs, geometry reasoning, or combinatorics
setup. When SymPy can't check the domain, `verified_by_sympy` is `false`, the confidence ceiling drops,
and the UI does not present the diagnosis with the same certainty. This is the honest mitigation for
the open-domain choice, not a claim that open-domain is fully solved.

### SymPy is an untrusted-input boundary

Discovered during implementation, and corrected: **`sympy.parse_expr` calls Python's `eval`.** With
the naive implementation this spec originally implied, `__import__('os').system(...)` executed and
`().__class__.__bases__[0]` returned `<class 'object'>` — both reproduced, not theorized. The
expressions reaching the parser are emitted by a model whose prompt carries untrusted student text,
so a prompt-injected submission is a plausible path to them.

Two layers, because a filter alone is not enough:

1. **A character allow-list before parsing.** Banning quotes removes every string-literal vector,
   banning brackets removes subscripting, banning dot-before-letter removes attribute chains, and
   dunder names are rejected outright. Adversarial review — unicode homoglyphs, fullwidth digits,
   hex escapes, RTL overrides, comment smuggling, `lambda`, walrus, whitespace-split dunders,
   `globals()` — found no bypass.
2. **A hard wall-clock bound in a killable process.** The allow-list cannot stop resource
   exhaustion: `2**2**2**2**2**2` is 16 allow-listed characters that never return, and
   `factorial(2000000)` burns minutes of CPU. Expression-shape heuristics lose this game; a
   process-level timeout ends it, and incidentally contains any allow-list bypass still unfound.

The related constraint that keeps this surface small: **no LaTeX parsing.** SymPy's LaTeX parser
needs the `antlr` runtime, so the model is instructed to emit SymPy syntax (`**`, not `^`, no
backslash commands) and LaTeX input is rejected by the allow-list rather than parsed.

## 7. Taxonomy canonicalization

Free-text diagnoses cannot be aggregated — "other students make this error" needs stable identity.
Open diagnosis is preserved; identity is added afterward:

1. Normalize the buggy rule into a canonical form (strip variable names: `(a+b)^2 -> a^2+b^2`
   regardless of whether the student used `x` or `t`).
2. Retrieve candidate existing misconceptions by trigram similarity on canonical form + topic tags.
3. A cheap `strict=True` DeepSeek call adjudicates: *same as an existing entry, or new?*
4. Same → attach that `misconception_id`. New → mint an entry with slug, canonical statement, topic.

Seeded with ~40 documented misconceptions (freshman's dream, fraction addition across denominators,
negative-sign distribution, log of a sum, cancelling terms across a sum, chain-rule omission, etc.) so
insights are not cold-start-empty and common errors match a curated entry rather than minting
near-duplicates.

## 8. Grounding — a contract, not a prompt instruction

Beats are the unit of grounding, and the pipeline **enforces** them:

- `s6_visual` emits ordered `Beat`s, at least one with `targets_misconception: true`.
- `s7_scene` must wrap each beat using the container-side helper:

  ```python
  with beat(self, "b3"):      # records start/end into /out/manifest.json
      ...animations...
  ```

- **`s8_validate` hard-fails if any planned beat id is missing, duplicated, or unknown** → repair
  loop. Grounding cannot silently degrade into a chatbot next to a video.
- Render emits `video.mp4` **and** `manifest.json` with *measured* start/end per beat, taken from the
  renderer clock rather than estimated.
- `chat.py` receives the beat manifest (id, title, purpose, real timestamps) plus the diagnosis, and
  is instructed to cite as `[beat:b3]`. The client rewrites those into chips —
  `▶ 0:42 — Why (a+b)² isn't a²+b²` — that seek the player.
- **Reverse direction:** the player tracks the active beat on the beat rail and offers "ask about this
  moment," injecting that beat into chat. Bidirectional, so the two halves genuinely reinforce.

Citations are validated server-side before streaming to the client: a `[beat:bX]` naming a beat not in
the manifest is stripped rather than shown as a dead link.

## 9. Rendering

**Container:** `manimcommunity/manim`, one `docker run` per render.

```
--network=none           no egress from LLM-authored code
--read-only              writable tmpfs only at /out
--user <non-root>
--memory=2g --cpus=2
--pids-limit=256
timeout 300s             hard wall-clock kill
```

Mounts: `render/primitives/` read-only, `media/<session_id>/` as `/out`.

**Static validation before any execution** (`validator.py`, deterministic, no LLM):

- AST parse; reject on syntax error.
- Import allow-list: `manim`, `numpy`, `math`, and the local `primitives` package. Everything else
  rejected.
- Deny-list on names/attributes: `os`, `sys`, `subprocess`, `socket`, `open`, `eval`, `exec`,
  `compile`, `__import__`, `globals`, `builtins`, any dunder attribute access.
- Beat-coverage check (§8).
- Exactly one `Scene` subclass, named as the contract specifies.

Defense in depth: the sandbox is the container, the validator is the second layer. Neither is trusted
alone.

**Repair loop:** on render failure, feed the Manim traceback, the failing line, and the original
storyboard back to `v4-pro` for a corrected file. **Maximum 2 attempts**, then degrade.

**Storyboard fallback:** if codegen fails twice, `storyboard.py` renders the beats deterministically
from the primitives library with **no LLM-authored code at all** — titles, the math content from `s5`,
and beat markers as an animated sequence. Lower production value, still correct, still fully grounded
(same beat ids, same manifest). A student mid-flow never dead-ends on a spinner.

## 10. Latency strategy

A full run is 60–180s. The wait is filled with the most valuable content rather than a progress bar:

- `s1_diagnose` completes in ~10–20s and its card renders **immediately**.
- **Chat opens as soon as the diagnosis exists** — the student can start talking while the animation
  is still planning and rendering. Chat pre-render is grounded in the diagnosis and the storyboard's
  beat *titles*; it gains timestamp citations when the manifest lands.
- Beats appear on the rail as `s6` plans them, greyed, filling in as the render completes.
- Streamed `reasoning_content` is shown per stage — the pipeline's actual thinking is the loading state.

## 11. Checkpoint

Generated from `Diagnosis.buggy_rule`, three items:

1. **Transfer** — the same misconception, new surface (different numbers/context). The distractor set
   **contains the exact answer the buggy rule produces**. Choosing it is direct evidence the
   misconception persists.
2. **Discrimination** — a near-miss problem that superficially resembles the buggy pattern but
   requires the correct rule. Catches overcorrection (a student who now refuses to ever distribute).
3. **Explain** — free response: "Why is X wrong?" Graded by a `strict=True` call against a rubric
   derived from the diagnosis.

Outcome: `resolved | partial | persists`.

- `resolved` → summary card, log, done.
- `partial` / `persists` → chat reopens **on the specific failed item**, and the student can request a
  narrower animation targeting only the sticking point (a re-run of s4–s8 scoped to that beat).

This is what makes the log meaningful: `checkpoints` records outcomes, so insights report **resolution
rate per misconception**, not just frequency. "Common *and* hard to fix" is the signal a teacher wants.

## 12. Data model (SQLite, WAL)

```sql
sessions(id TEXT PK, handle TEXT, created_at, input_mode, problem, student_work_json, status)
run_artifacts(id PK, session_id FK, stage, payload_json, reasoning_text,
              model, prompt_tokens, completion_tokens, cached_tokens, cost_usd, ms, attempt)
diagnoses(id PK, session_id FK, buggy_rule, canonical_rule, statement, confidence,
          divergence_index, verified_by_sympy, is_unclear, misconception_id FK)
misconceptions(id PK, slug UNIQUE, canonical_statement, topic, aliases_json, is_seed, first_seen_at)
beats(session_id FK, beat_id, idx, title, purpose, targets_misconception, primitive,
      start_s, end_s, PRIMARY KEY(session_id, beat_id))
chat_messages(id PK, session_id FK, role, content, cited_beats_json, created_at)
checkpoints(id PK, session_id FK, items_json, responses_json, per_item_json, outcome, created_at)
renders(id PK, session_id FK, attempt, status, duration_s, error_text, video_path, mode)
```

`handle` is a browser-persisted anonymous id (localStorage), enabling per-student history without auth.
`renders.mode` distinguishes `generated` from `storyboard_fallback` so quality is measurable.

## 13. API surface

```
POST   /api/sessions                 create; body: problem, steps/prose, handle
POST   /api/sessions/{id}/photo      multipart; → transcription for student review
GET    /api/sessions/{id}/stream     SSE: stage + render progress events
GET    /api/sessions/{id}            full state: diagnosis, beats, video url, status
POST   /api/sessions/{id}/chat       SSE-streamed grounded reply
GET    /api/sessions/{id}/checkpoint generate (idempotent per session)
POST   /api/sessions/{id}/checkpoint submit responses → graded outcome
POST   /api/sessions/{id}/refocus    narrower re-render scoped to one beat
GET    /api/insights                 taxonomy frequency, resolution rate, co-occurrence
GET    /media/{session_id}/video.mp4 static, range-request enabled
```

## 14. Frontend

Three routes, one visual language.

- **Submit** — problem field + work field, live KaTeX preview, or a photo drop zone. Photo path shows
  the transcription in editable fields with low-confidence spans highlighted; the student confirms
  before anything is diagnosed.
- **Session (the theater)** — the main surface. Diagnosis card at top. Below, a two-pane split: video
  player with a beat rail beneath it (segmented, labeled, click to seek, active beat highlighted) on
  the left; chat on the right. Pipeline progress occupies the video pane until the render lands. The
  checkpoint slides in beneath once the student has watched the animation and exchanged at least one
  chat turn.
- **Insights** — misconception frequency, resolution rate, co-occurrence, and a "you're not alone"
  count surfaced back into the session view.

## 15. Error handling and degradation

| Failure | Response |
|---|---|
| Vision transcription poor/low confidence | Fields flagged; student corrects before submit. Never auto-proceeds below threshold. |
| No `GEMINI_API_KEY` | Photo drop zone disabled with explanation; typed path unaffected. |
| Stage returns invalid JSON | Retry ×2 with validation error appended. Then fail the run with a clear message. |
| Diagnosis low-confidence | `is_unclear` → chat opens with a clarifying question; animation deferred until resolved. |
| SymPy can't verify domain | `verified_by_sympy: false`; confidence ceiling lowered; UI hedges accordingly. |
| Validator rejects generated code | Repair loop (≤2), then storyboard fallback. |
| Render timeout / container failure | Same path: repair, then storyboard fallback. |
| Chat cites a nonexistent beat | Citation stripped server-side before streaming. |
| DeepSeek 429 / 5xx | Exponential backoff ×3; surface a retry affordance rather than a dead session. |
| Insufficient DeepSeek balance | Detected at startup and on 402; banner shown, session creation blocked with a clear reason. |

## 16. Configuration and secrets

`server/.env`, gitignored, `chmod 600`, loaded through `config.py`. **Never logged, never in artifacts,
never returned by the API.** The token ledger stores counts and cost, never prompts containing keys.

```
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_MODEL_REASONING=deepseek-v4-pro
DEEPSEEK_MODEL_FAST=deepseek-v4-flash
GEMINI_MODEL_VISION=gemini-3.5-flash-lite
RENDER_TIMEOUT_S=300
RENDER_MAX_REPAIRS=2
FAKE_LLM=0
```

Both keys supplied during design were pasted into a chat transcript and should be rotated once this
work is done.

## 17. Testing

- **Stage contract tests** — recorded fixture responses per stage, offline. Assert Pydantic validation,
  retry-on-invalid behavior, and artifact persistence.
- **`FAKE_LLM=1` mode** — canned stage outputs for the whole chain, so the full flow and the entire
  frontend are testable and demoable with **zero network and zero cost**. Also the CI path.
- **Validator tests** — malicious code rejected (`os`, `subprocess`, dunder access, disallowed
  imports), missing/duplicate beats rejected, valid code accepted.
- **Diagnosis eval set** — ~20 `(problem, wrong work, expected buggy rule)` cases spanning arithmetic,
  algebra, and calculus, scored on whether the diagnosed rule matches. **This is how we know the core
  feature works**, and it gates prompt changes to `s1`.
- **Grounding integration test** — every beat id in the storyboard appears in the manifest with a
  monotonic, non-overlapping time range.
- **One end-to-end smoke test** with a real Docker render, run manually, not in CI.

## 18. Cost model

Per session, with prefix caching and the flash/pro split:

| Component | Est. |
|---|---|
| Photo transcription (optional) | $0.0005 |
| s1 diagnose (pro, thinking) | ~$0.008 |
| s2–s6 planning (flash) | ~$0.004 |
| s7 codegen (pro) | ~$0.006 |
| Chat + checkpoint (flash) | ~$0.003 |
| **Total** | **~$0.02** |

The $5.67 DeepSeek balance is roughly **250 full sessions**. Rendering is local CPU, free.

## 19. Risks

1. **Open-domain diagnosis accuracy** — the chosen tradeoff. Mitigated by SymPy verification, the
   falsification step, `is_unclear` as a real outcome, and the eval set as a regression gate. Weakest
   on word problems, proofs, and geometry, where SymPy cannot check.
2. **LLM-generated Manim code failing to render** — mitigated by the primitives library narrowing what
   codegen must invent, static validation, a bounded repair loop, and the deterministic storyboard
   fallback.
3. **Latency vs. the "one smooth flow" goal** — mitigated by early diagnosis, chat opening pre-render,
   and streamed reasoning as the loading state.
4. **Taxonomy drift** — near-duplicate entries diluting aggregate counts. Mitigated by canonical-form
   normalization, retrieval before minting, and the 40-entry seed. A periodic merge pass is possible
   later if drift is observed.
5. **Prompt injection via student input** — a student could paste instructions into the "explanation"
   field. Student content is delimited and labeled untrusted in every prompt; generated code passes
   the validator regardless of origin; the container has no network.

## 20. Out of scope for v1

Accounts and auth. Teacher dashboards and class rollups. Voice input. Canvas handwriting input.
Multi-problem worksheets. Spaced-repetition follow-up. Animation editing by the student.
Non-English input.
