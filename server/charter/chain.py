"""Pipeline orchestrator.

Phase 1 runs s0->s1 only: given a ``StudentSubmission`` already ingested by
``s0_ingest``, ``run_diagnosis`` drives the s1 diagnose stage, persists its
payload and reasoning trace as a ``run_artifacts`` row, resolves the
misconception id against the taxonomy, saves the ``diagnoses`` row, and
updates the session's status -- while yielding ``ProgressEvent``s a caller
(an HTTP handler) can stream to the frontend as they occur. Phase 2 extends
this into a full ``run`` that continues through s2-s8 and rendering.

Design notes, answering the questions this module invites:

* **Failure mid-run vs. a crash vs. never started.** ``sessions.status`` must
  already be ``in_progress`` *before* the diagnose stage's LLM call, not just
  on the way out -- otherwise a process killed mid-call (which legitimately
  runs tens of seconds against a reasoning model) leaves the row at
  ``created``, bit-for-bit identical to a session nobody ever picked up,
  since ``created`` is also the row's initial value from ``create_session``.
  That transition is the caller's responsibility, made via
  ``repo.try_start_session``'s compare-and-swap (``UPDATE ... WHERE status =
  'created'``), *before* ``run_diagnosis`` is ever invoked -- not a write this
  method performs itself. It has to happen there rather than here: the
  natural caller (``server/app.py``'s ``/stream`` route) hands this generator
  to ``asyncio.create_task``, which defers the generator's first line to the
  next event-loop iteration -- so a write at the top of this method would
  leave a window where a second concurrent request reads the same
  pre-transition ``created`` status and also starts a run. Doing the
  compare-and-swap synchronously, before the task is even created, closes
  that window; this method repeating the write afterward would be redundant
  at best (the route already made it true) and, worse, would reintroduce a
  plain write for any caller that bypasses the route's CAS. With the
  transition already made, the three cases are distinguishable by reading
  ``sessions.status`` alone: ``created`` means genuinely never started,
  ``in_progress`` means started and either still running or dead mid-run
  (Phase 1 has no heartbeat/staleness column to tell those two apart yet --
  see the note below), and ``failed``/``diagnosed``/``needs_clarification``
  are the terminal outcomes. Any ``LlmError`` (or subclass, e.g.
  ``SchemaRetryExhausted``) raised by the diagnose stage is caught here, the
  session status is set to ``failed``, an ``error`` ``ProgressEvent`` is
  yielded, and the generator returns -- no ``done`` event follows a failure.
* **Partial-write ordering.** The ``s1_diagnose`` artifact is written only
  after the stage's LLM call and SymPy verification have both completed
  successfully, and only the session's *status* column is updated after
  that -- never before. If the process dies between the artifact write and
  the diagnosis row, or between the diagnosis row and the final status
  update, the surviving record is still coherent: an artifact with no
  diagnoses row means "s1 produced an answer, bookkeeping did not finish",
  never a diagnoses row with a missing artifact, and the session's status
  column stays at whatever non-terminal value it was already at (rather
  than ever being flipped to a terminal status before the corresponding
  data exists). ``server/store/db.py`` autocommits every statement
  (``isolation_level=None``), so each ``INSERT``/``UPDATE`` here is durable
  the instant it runs; there is no multi-statement transaction to roll back,
  which is exactly why the *order* of these statements is what provides
  coherence, not a transaction boundary.
* **Taxonomy fallback never fails a session.** ``resolve_misconception``
  (Task 9) already swallows its own ``LlmError`` internally and falls back to
  minting a misconception from the raw ``buggy_rule`` -- it always returns an
  ``int``. This module does not wrap that call in its own try/except: there
  is nothing here that needs to catch, since the callee's contract is "never
  raises". A session with a diagnosis that only matched the taxonomy via
  fallback still reaches ``diagnosed``/``needs_clarification`` normally.
* **Correct work terminates at ``correct``, with no misconception.** When the
  diagnose stage reports ``no_error_found``, the taxonomy is skipped entirely
  and the ``diagnoses`` row is written with a null ``misconception_id`` and an
  empty ``canonical_rule``. This is the only success path that produces a
  ``diagnoses`` row not pointing at a ``misconceptions`` row, and it is what
  keeps "the student was right" out of the misconception statistics that
  cross-session pattern tracking is built on.
* **Late subscribers can miss events.** ``run_diagnosis`` is a plain async
  generator: it is not a broadcast/pub-sub stream, so a second consumer that
  starts iterating after the first has already advanced the generator would
  not "replay" earlier events (and in practice cannot: an async generator can
  only be iterated by one consumer at a time). For Phase 1 this is
  acceptable because the caller is expected to be the single HTTP handler
  that both starts the run and streams its own events to the frontend in the
  same request/response cycle -- there is exactly one consumer, and it is
  present from the first ``yield``. This does mean the terminal state
  (``sessions.status`` and the persisted ``diagnoses``/``run_artifacts``
  rows) is the durable source of truth for anyone who reconnects later or
  polls out-of-band; the event stream itself is not replayable and is not
  meant to be.
* **Cost/token accounting covers every call, not just the diagnose stage.**
  ``LlmCallMeta.cost_usd``/``prompt_tokens``/``completion_tokens`` for the
  diagnose stage's *winning* attempt are persisted verbatim onto the
  ``run_artifacts`` row by ``repo.record_artifact`` (``attempt=meta.attempts``
  records which attempt in DeepSeekClient's internal schema-retry loop
  finally succeeded -- a failed retry's tokens are not separately billed by
  the client, so there is nothing more to accumulate here). The taxonomy
  adjudication call inside ``resolve_misconception`` (Task 9) returns
  ``tuple[int, LlmCallMeta | None]``: ``None`` exactly when the exact
  canonical-rule fast path fired and no HTTP call was made at all, or when
  the call itself raised ``LlmError`` (no billable response was received).
  Whenever that ``meta`` is not ``None``, this module persists it as its own
  ``run_artifacts`` row under ``_TAXONOMY_STAGE`` -- so a session whose
  ``buggy_rule`` needs real adjudication (the common case early on, before
  many exact matches exist) has that call's real cost in the ledger too, not
  silently dropped.
"""

import sqlite3
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel

from server.charter.contracts import LlmCallMeta, StageName, StudentSubmission
from server.charter.stages.s1_diagnose import diagnose
from server.config import Settings
from server.llm.deepseek import DeepSeekClient, LlmError
from server.store import repo, taxonomy

# Not a StageName member: this is a bookkeeping sub-step of s1_diagnose (matching a
# diagnosis onto a stable misconception id), not one of the s0-s8 pipeline stages
# StageName enumerates. A plain string is enough -- repo.record_artifact only ever
# does str(stage) with it -- and keeps this local to the one place that needs it
# instead of widening a contract other stages/tasks rely on.
_TAXONOMY_STAGE = "s1_diagnose_taxonomy"


class ProgressEvent(BaseModel):
    type: Literal["stage_started", "stage_completed", "diagnosis_ready", "error", "done"]
    stage: str | None = None
    payload: dict | None = None
    message: str | None = None


class Chain:
    """Runs the diagnosis pipeline for one session, one stage at a time."""

    def __init__(
        self, conn: sqlite3.Connection, client: DeepSeekClient, *, settings: Settings
    ) -> None:
        self._conn = conn
        self._client = client
        self._settings = settings

    async def run_diagnosis(
        self, session_id: str, submission: StudentSubmission
    ) -> AsyncIterator[ProgressEvent]:
        """Run s1 diagnose for ``session_id``, persist results, yield progress.

        On any ``LlmError`` from the diagnose stage, marks the session
        ``failed`` and yields a terminal ``error`` event -- no ``done``
        event follows. On success, always reaches a terminal
        ``diagnosed``/``needs_clarification`` status (taxonomy resolution
        never raises) and yields ``diagnosis_ready`` then ``done``.

        Does NOT mark the session ``in_progress`` itself -- see the module
        docstring for why that transition must be the caller's
        compare-and-swap, made before this generator is ever handed to
        ``asyncio.create_task``, rather than a write at the top of this
        method.
        """
        yield ProgressEvent(type="stage_started", stage=StageName.DIAGNOSE)

        try:
            diagnosis, meta = await diagnose(
                self._client,
                submission=submission,
                model=self._settings.deepseek_model_reasoning,
            )
        except LlmError as exc:
            repo.set_session_status(self._conn, session_id, "failed")
            yield ProgressEvent(
                type="error", stage=StageName.DIAGNOSE, message=f"diagnosis failed: {exc}"
            )
            return

        repo.record_artifact(
            self._conn,
            session_id=session_id,
            stage=StageName.DIAGNOSE,
            payload=diagnosis.model_dump(),
            meta=meta,
            attempt=meta.attempts,
        )
        yield ProgressEvent(
            type="stage_completed",
            stage=StageName.DIAGNOSE,
            payload={"reasoning": meta.reasoning, "cost_usd": meta.cost_usd},
        )

        misconception_id: int | None
        taxonomy_meta: LlmCallMeta | None
        if diagnosis.no_error_found:
            # Correct work is not a misconception, so it must not be resolved
            # against the taxonomy at all. Doing so minted a row from the prose
            # in `buggy_rule` and then, via the exact-canonical fast path,
            # collapsed every later correct submission -- in any topic -- onto
            # that same row, where it showed up in students' misconception
            # histories as a diagnosed error. A null misconception_id is the
            # honest record: there is no misconception here to point at.
            misconception_id, taxonomy_meta = None, None
        else:
            misconception_id, taxonomy_meta = await taxonomy.resolve_misconception(
                self._conn,
                self._client,
                diagnosis=diagnosis,
                model=self._settings.deepseek_model_fast,
            )
        if taxonomy_meta is not None:
            # Only persisted when a real adjudication call happened -- the exact
            # canonical-match fast path and a failed call both report None, and
            # neither should produce a synthesized zero-cost ledger row.
            repo.record_artifact(
                self._conn,
                session_id=session_id,
                stage=_TAXONOMY_STAGE,
                payload={"misconception_id": misconception_id},
                meta=taxonomy_meta,
                attempt=taxonomy_meta.attempts,
            )
        repo.save_diagnosis(
            self._conn,
            session_id=session_id,
            diagnosis=diagnosis,
            misconception_id=misconception_id,
            # No rule was diagnosed, so there is nothing to canonicalize --
            # canonicalizing the "none" placeholder would write a junk
            # canonical_rule that later queries could group on.
            canonical_rule=(
                "" if diagnosis.no_error_found else taxonomy.canonicalize_rule(diagnosis.buggy_rule)
            ),
        )

        if diagnosis.no_error_found:
            status = "correct"
        elif diagnosis.is_unclear:
            status = "needs_clarification"
        else:
            status = "diagnosed"
        repo.set_session_status(self._conn, session_id, status)

        yield ProgressEvent(
            type="diagnosis_ready",
            stage=StageName.DIAGNOSE,
            payload={**diagnosis.model_dump(), "misconception_id": misconception_id},
        )
        yield ProgressEvent(type="done", payload={"status": status})
