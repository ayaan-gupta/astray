# Astray

**Finds where your math reasoning went astray — then shows you.**

Most math tools tell you the answer is wrong. Astray finds the exact step where
your reasoning left the correct path, names the false rule you were actually
applying, and builds an animated explanation of that specific misconception.

Status: **Phase 1 complete** — ingestion and the diagnosis engine. Animation
rendering (Phase 2) and the grounded chat tutor (Phase 3) are specified but not
yet built.

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

## Design commitments

**The diagnosis is falsifiable.** Every diagnosis carries a SymPy expression
whose truth value would disprove it. That check runs deterministically in a
killable subprocess behind a character allow-list, and its result — not the
model's claim — is what gets stored as `verified_by_sympy`.

**Student text is untrusted.** Submissions are wrapped in per-request nonce
delimiters in every prompt, with content-blind neutralization of forged
delimiter runs. A student can write "ignore your instructions" in their work
without it becoming an instruction.

**SymPy is an input boundary, not a calculator.** `parse_expr` calls `eval()`.
The check runner enforces a character allow-list (no quotes, brackets, attribute
chains, or dunders) and a killable wall-clock bound, because both RCE and
non-terminating-power-tower DoS were reproduced against the naive version.

**Secrets never leave the server.** Keys live only in a gitignored `server/.env`.
Upstream error text never reaches a client — a DeepSeek error body once reflected
the `Authorization` header straight through to the SSE stream.

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
| `POST` | `/api/sessions/{id}/photo` | Transcribe handwritten work into a submission |
| `GET` | `/api/sessions/{id}` | Session state and diagnosis, if ready |
| `GET` | `/api/sessions/{id}/stream` | SSE: run the diagnosis, stream progress |

`/stream` claims a session with a compare-and-swap, so concurrent connections
cannot double-bill the same run; reconnecting to a finished session replays the
stored result rather than re-running it.

## Development

```bash
uv run pytest          # 280 tests, no network — all HTTP via MockTransport
uv run ruff check .
uv run ruff format --check .
uv run python -m evals.diagnosis.run   # 20 labelled cases against the real model
```

The eval harness scores rule match, topic match, and SymPy verification rate.
Rule match is currently **not** a trustworthy gate — see the plan's Definition
of Done for why the scorer rejects substantively correct diagnoses on notation.
Topic match and verification rate are reliable.

Design and plan documents live in `docs/superpowers/`.
