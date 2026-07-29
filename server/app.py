"""FastAPI surface. Phase 1 exposes ingest, diagnosis, and the progress stream.

Two structural decisions here are not obvious from the route bodies alone:

* **MaxBodySizeMiddleware.** ``UploadFile``/Starlette's multipart parser caps the
  size of individual non-file form fields, but a *file* part is streamed straight
  into a ``SpooledTemporaryFile`` with no total-size limit -- a route handler only
  ever sees the fully-parsed ``UploadFile``, by which point an unbounded upload has
  already been spooled (to memory, then disk) in full. Capping ``len(data)`` after
  ``await file.read()`` (as a naive handler would) is too late to prevent that work;
  the guard has to sit outside routing entirely, at the ASGI layer, before Starlette
  ever starts parsing the body. See that class's docstring for the two layers this
  implements (declared Content-Length, and actual bytes received).
* **Streaming decoupled from the diagnosis run.** ``GET .../stream`` starts
  ``chain.run_diagnosis`` as its own ``asyncio.Task`` writing into a queue, rather
  than iterating the async generator directly inside the SSE response body. If the
  client disconnects mid-stream, Starlette cancels the task driving the response
  body (see ``StreamingResponse``'s disconnect handling) -- but that cancellation
  only reaches the queue-reading loop, not the separate task producing into the
  queue, so ``run_diagnosis`` keeps running to completion and its terminal state
  (``sessions.status``, ``diagnoses``, ``run_artifacts``) is still persisted even
  though nobody is left to receive the SSE frames. Without this, a dropped
  connection mid-run would leave the session stuck at ``in_progress`` forever --
  indistinguishable from a genuinely crashed process (see chain.py's module
  docstring on that ambiguity).
* **``GET .../stream`` is not safely repeatable, so it claims the session with a
  compare-and-swap, not a read-then-check.** A plain ``EventSource`` (the natural
  frontend client for SSE) auto-reconnects on any transient network hiccup by
  re-issuing the exact same ``GET`` -- and multiple concurrent requests for the
  same session are also a real, not just theoretical, shape (a double-clicked
  button, two open tabs). Reading ``status`` and then separately deciding whether
  to start a run leaves a window between the two for a second concurrent request
  to read the identical pre-transition status and also start one -- especially
  since the run itself starts via ``asyncio.create_task``, which defers
  ``run_diagnosis``'s first line to the next event-loop iteration, giving a
  same-tick request ample room to land in exactly that window. ``repo.
  try_start_session`` closes it: ``UPDATE sessions SET status = 'in_progress'
  WHERE id = ? AND status = 'created'`` performs the check and the transition as
  one atomic statement, so at most one concurrent caller for a given session ever
  observes success (``rowcount == 1``) -- everyone else observes ``0`` and must
  not start a run. A loser then re-reads the (now up to date) status to tell
  "already running" (``409``) apart from "already terminal" (replay the
  persisted result instead of re-running -- and re-billing -- the whole diagnose
  stage). Only the single winner of the compare-and-swap actually starts
  ``run_diagnosis``.
* **Shutdown drains ``background_tasks`` before closing the client/connection.**
  Because the diagnosis run is deliberately decoupled from the request (previous
  bullet), it is *also* invisible to uvicorn's normal connection-draining on
  shutdown -- once nobody is reading the SSE response there is no connection left
  to drain, so an ordinary redeploy mid-run would otherwise close the shared LLM
  client and DB connection out from under a still-running ``produce()`` task.
  Lifespan shutdown waits (bounded by ``shutdown_drain_timeout_s``) for tracked
  tasks to finish; anything still running past that timeout is cancelled and its
  session marked ``failed`` -- never left at ``in_progress`` with no way to ever
  resolve, since nothing can complete the run after the connection closes.
* **Upstream-derived error text is never forwarded to a client.** ``LlmError`` and
  ``VisionUnavailable`` messages can embed arbitrary text from an upstream HTTP
  response body (DeepSeek's or Gemini's own error responses) -- this app has no
  way to know that text is free of secrets (e.g. a misconfigured proxy reflecting
  the outbound ``Authorization``/``x-goog-api-key`` header back in a non-JSON
  error body). This module is the trust boundary to the client, so it never
  relays that text verbatim: the SSE ``error`` event's ``message`` and the photo
  route's 503 ``detail`` are both replaced with a fixed, generic message before
  leaving this process; the real detail is only logged server-side. The one
  exception is ``NullVision``'s specific "no GEMINI_API_KEY configured" message,
  which is a fixed string literal in this codebase, never upstream-derived, and
  is forwarded verbatim on purpose so a frontend can tell "not configured" apart
  from a transient failure.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from server.charter.chain import Chain, ProgressEvent
from server.charter.contracts import Diagnosis, StageName, StudentSubmission
from server.charter.pipeline import Pipeline
from server.charter.stages.s0_ingest import ingest_photo, ingest_typed, needs_review
from server.config import Settings, get_settings
from server.deps import build_llm_client, build_vision
from server.llm.deepseek import DeepSeekClient, LlmError
from server.llm.vision import NullVision, VisionProvider, VisionUnavailable
from server.store import insights, repo
from server.store.db import connect
from server.store.seed_taxonomy import seed
from server.tutor import chat

logger = logging.getLogger(__name__)

# The photo route's own domain limit, checked against the fully-parsed file content.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# The whole HTTP request's hard cap, enforced by MaxBodySizeMiddleware before any
# multipart parsing happens. A little above MAX_IMAGE_BYTES for multipart
# boundary/header overhead around the file part.
MAX_UPLOAD_BYTES = MAX_IMAGE_BYTES + 1024 * 1024
# How long shutdown waits for in-flight diagnosis runs to finish before cutting
# them off. A little under common infra SIGTERM grace periods (e.g. 30s).
SHUTDOWN_DRAIN_TIMEOUT_S = 25.0
# Stable, generic client-facing messages -- never the real upstream/exception text.
_GENERIC_DIAGNOSIS_ERROR = "diagnosis failed; please try again"
_GENERIC_VISION_ERROR = "photo transcription is temporarily unavailable"
_GENERIC_CHAT_ERROR = "the tutor is temporarily unavailable"


class _NoCacheStatic(StaticFiles):
    """Serve the app shell with revalidation forced.

    StaticFiles' default ETag/Last-Modified handling lets a browser keep serving
    a cached app.js from memory without revalidating, so a deployed frontend fix
    does not reach anyone who already has the page open or in cache. This cost
    real debugging time during development -- a corrected script was being served
    by the server and ignored by the browser, which reads exactly like a code bug.
    The shell is a few KB; correctness beats the saved round-trip.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


class _BodyTooLarge(Exception):
    """Internal sentinel, caught only within MaxBodySizeMiddleware."""


class MaxBodySizeMiddleware:
    """Reject oversized request bodies before Starlette's multipart parser sees them.

    Two independent checks, since either one alone leaves a gap:

    * A declared ``Content-Length`` over ``max_bytes`` is rejected immediately,
      before a single byte of the body is read -- this is the common case (every
      normal browser/curl multipart upload declares its length upfront) and it is
      the one that matters most, since it stops Starlette from ever starting the
      spool.
    * For a request with a missing or understated ``Content-Length`` (e.g.
      chunked transfer-encoding), the wrapped ``receive`` callable counts actual
      bytes as they arrive and aborts as soon as the running total crosses
      ``max_bytes`` -- before that chunk reaches the multipart parser.

    Raising ``_BodyTooLarge`` out of the wrapped ``receive`` and catching it around
    ``self._app(...)`` is safe here specifically because every route behind this
    middleware only starts producing a response *after* its request body has been
    fully parsed (FastAPI resolves ``File(...)``/body parameters before calling a
    route function) -- so no ``send()`` call has happened yet when the exception is
    raised, and this middleware's own 413 is the first and only response sent.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                declared_too_big = int(declared) > self._max_bytes
            except ValueError:
                declared_too_big = False
            if declared_too_big:
                await self._reject(send)
                return

        received = 0

        async def guarded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body") or b"")
                if received > self._max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self._app(scope, guarded_receive, send)
        except _BodyTooLarge:
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps({"detail": "request body too large"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# Per-field caps on typed input. MaxBodySizeMiddleware bounds the whole request
# at MAX_UPLOAD_BYTES (11 MB, sized for image uploads), which is far too generous
# for text: without these, a 10 MB `work` string is accepted and forwarded
# verbatim into the diagnose prompt, costing millions of tokens on a single
# request. These bounds are well above any real submission -- 400 steps of 50
# characters still fits in `work` -- and are enforced by Pydantic, so an
# oversized field is a 422 at the edge rather than an upstream bill.
MAX_PROBLEM_CHARS = 4_000
MAX_WORK_CHARS = 20_000
MAX_PROSE_CHARS = 4_000
MAX_HANDLE_CHARS = 100


class CreateSessionRequest(BaseModel):
    handle: str = Field(default="anon", max_length=MAX_HANDLE_CHARS)
    problem: str = Field(max_length=MAX_PROBLEM_CHARS)
    work: str = Field(default="", max_length=MAX_WORK_CHARS)
    prose: str | None = Field(default=None, max_length=MAX_PROSE_CHARS)

    @field_validator("problem")
    @classmethod
    def problem_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("problem must not be blank")
        return value


# Not a StageName member: chat is a product surface, not a pipeline stage. It
# still belongs in the same cost ledger, so it gets a plain string label.
_CHAT_STAGE = "chat"

MAX_CHAT_CHARS = 2_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_CHARS)


class ConfirmSubmissionRequest(BaseModel):
    """The student's reviewed version of their work, as edited in the UI.

    Same shape and same caps as CreateSessionRequest minus `handle`, since this
    replaces the same three fields on an existing session.
    """

    problem: str = Field(max_length=MAX_PROBLEM_CHARS)
    work: str = Field(default="", max_length=MAX_WORK_CHARS)
    prose: str | None = Field(default=None, max_length=MAX_PROSE_CHARS)

    @field_validator("problem")
    @classmethod
    def problem_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("problem must not be blank")
        return value


def create_app(
    *,
    settings: Settings | None = None,
    client_factory: Callable[[], DeepSeekClient] | None = None,
    vision_factory: Callable[[], VisionProvider] | None = None,
    shutdown_drain_timeout_s: float = SHUTDOWN_DRAIN_TIMEOUT_S,
) -> FastAPI:
    resolved = settings or get_settings()
    make_client = client_factory or (lambda: build_llm_client(resolved))
    make_vision = vision_factory or (lambda: build_vision(resolved))

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.conn = connect(resolved.db_path)
        seed(app.state.conn)
        app.state.client = make_client()
        app.state.vision = make_vision()
        # session_id -> Task, not a bare set: strong references for
        # asyncio.create_task()'d diagnosis runs (the event loop only holds a
        # *weak* reference to a task, so a run whose stream response nobody is
        # reading anymore must be kept alive here or it can be garbage-collected
        # before it finishes) -- keyed by session_id so shutdown can attribute an
        # unfinished task back to the session it must mark `failed`.
        app.state.background_tasks: dict[str, asyncio.Task] = {}
        try:
            yield
        finally:
            tasks = dict(app.state.background_tasks)
            if tasks:
                _done, pending = await asyncio.wait(
                    tasks.values(), timeout=shutdown_drain_timeout_s
                )
                if pending:
                    for task in pending:
                        task.cancel()
                    # return_exceptions=True: a cancelled task raises CancelledError,
                    # which must not propagate out of shutdown and abort the rest of
                    # it (closing the client/connection still has to happen below).
                    await asyncio.gather(*pending, return_exceptions=True)
                    # Nothing can complete these runs after the client/connection
                    # below are closed -- leaving them at `in_progress` would be
                    # indistinguishable from a crash with no way to ever resolve.
                    for session_id, task in tasks.items():
                        if task in pending:
                            repo.set_session_status(app.state.conn, session_id, "failed")
            await app.state.client.aclose()
            aclose = getattr(app.state.vision, "aclose", None)
            if aclose is not None:
                await aclose()
            app.state.conn.close()

    app = FastAPI(title="Astray", lifespan=_lifespan)
    # MaxBodySizeMiddleware is added first so CORSMiddleware ends up outermost
    # (Starlette's user-middleware list is built by inserting each new middleware
    # at the front, so the *last*-added one wraps the others). CORSMiddleware
    # never touches the request body, so nothing is lost by letting it sit above
    # the size guard -- and it must, or a rejected request's response (e.g. this
    # middleware's own 413) never passes through CORSMiddleware's `send` wrapper
    # and reaches the browser with no `access-control-allow-origin` header, which
    # makes it an opaque, unreadable network error to frontend JS instead of the
    # `{"detail": ...}` body.
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def conn_of(request: Request):
        return request.app.state.conn

    @app.get("/api/health")
    def health(request: Request) -> dict:
        return {"ok": True, "vision_enabled": request.app.state.settings.vision_enabled}

    @app.post("/api/sessions", status_code=201)
    def create_session(body: CreateSessionRequest, request: Request) -> dict:
        submission = ingest_typed(problem=body.problem, work=body.work, prose=body.prose)
        session_id = repo.create_session(
            conn_of(request), handle=body.handle, submission=submission
        )
        return {"session_id": session_id, "status": "created"}

    @app.post("/api/sessions/{session_id}/photo")
    async def upload_photo(
        session_id: str, request: Request, file: Annotated[UploadFile, File()]
    ) -> dict:
        if repo.get_session(conn_of(request), session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413, detail=f"image too large (max {MAX_IMAGE_BYTES} bytes)"
            )
        try:
            submission, meta = await ingest_photo(
                request.app.state.vision, data, file.content_type or "image/png"
            )
        except VisionUnavailable as exc:
            logger.warning("photo transcription unavailable for session %s: %s", session_id, exc)
            if isinstance(request.app.state.vision, NullVision):
                # A fixed string literal from this codebase (server/llm/vision.py),
                # never upstream-derived -- safe to forward verbatim, and a frontend
                # needs "GEMINI_API_KEY" in it to tell "not configured" apart from a
                # transient failure below.
                detail = str(exc)
            else:
                # exc may embed arbitrary text from Gemini's own error response body,
                # which this app has no way to sanitize -- never forward it.
                detail = _GENERIC_VISION_ERROR
            raise HTTPException(status_code=503, detail=detail) from exc
        # The Gemini transcription call is billable like any other LLM call -- it must
        # land in the same run_artifacts ledger the diagnose/taxonomy calls do, not be
        # discarded. This is the only s0_ingest artifact a photo-input session gets
        # (typed input never calls a model), so it is its own row per upload.
        repo.record_artifact(
            conn_of(request),
            session_id=session_id,
            stage=StageName.INGEST,
            payload=submission.model_dump(),
            meta=meta,
        )
        # The transcription becomes the session's work. Without this the route was a
        # dead end: it transcribed correctly, logged an artifact, returned JSON, and
        # left `student_work_json` at whatever was typed at create time -- so /stream,
        # which reads exactly that column, diagnosed empty work and told a student who
        # had just photographed their solution that they had not shown any work.
        if not repo.update_submission(conn_of(request), session_id, submission):
            # Not `created` any more: a run already claimed this session, and the
            # submission it is diagnosing must not change underneath it.
            raise HTTPException(
                status_code=409, detail="session already started; cannot replace its work"
            )
        return {
            "transcription": submission.model_dump(),
            "needs_review": needs_review(submission),
            # Photo work is never diagnosed until confirmed via PUT .../submission,
            # regardless of how confident the vision model was. `needs_review` says
            # whether the UI must *highlight* problems; this says the confirm step is
            # required either way.
            "confirmation_required": True,
        }

    @app.put("/api/sessions/{session_id}/submission")
    def confirm_submission(session_id: str, body: ConfirmSubmissionRequest, request: Request):
        """Accept the student's reviewed/corrected work and mark it confirmed.

        This is the step that makes a photo submission diagnosable. `ingest_photo`
        deliberately sets `student_corrected=False` and documents that only the
        student-confirmation flow may set it True -- this is that flow. A bad
        transcription diagnosed unchallenged would pin a misconception on a student
        for an error the vision model invented, which is the single worst failure
        this product can produce.
        """
        connection = conn_of(request)
        row = repo.get_session(connection, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="session not found")

        previous = StudentSubmission.model_validate_json(row["student_work_json"])
        corrected = ingest_typed(problem=body.problem, work=body.work, prose=body.prose)
        submission = corrected.model_copy(
            update={
                # `source` records how the work was *captured*, which confirming does
                # not change -- a corrected photo transcription is still photo input,
                # and flattening it to "typed" would erase that from the record.
                "source": previous.source,
                "student_corrected": True,
                "transcription_confidence": previous.transcription_confidence,
                # The student has now read every line, so nothing remains unreadable;
                # whatever they could not make out they have either fixed or removed.
                "unreadable": [],
            }
        )
        if not repo.update_submission(connection, session_id, submission):
            raise HTTPException(
                status_code=409, detail="session already started; cannot replace its work"
            )
        return {"submission": submission.model_dump(), "needs_review": needs_review(submission)}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, request: Request) -> dict:
        connection = conn_of(request)
        row = repo.get_session(connection, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="session not found")
        diagnosis_row = repo.get_diagnosis(connection, session_id)
        diagnosis = None
        if diagnosis_row is not None:
            diagnosis = json.loads(diagnosis_row["payload_json"])
            diagnosis["misconception_id"] = diagnosis_row["misconception_id"]
        return {
            "session_id": row["id"],
            "status": row["status"],
            "problem": row["problem"],
            "submission": json.loads(row["student_work_json"]),
            "diagnosis": diagnosis,
        }

    def _replay_terminal_state(connection, session_id: str, status: str) -> StreamingResponse:
        """Report an already-finished session's result without re-running it.

        Used when a client (re)connects to a session that is not ``created`` --
        most commonly an ``EventSource`` auto-reconnecting after the run already
        completed. Sourced from the durable ``diagnoses`` row rather than a fresh
        (and separately billed) call to the diagnose stage.

        ``status == "failed"`` is the one terminal status with no ``diagnoses``
        row (chain.py never persists one on ``LlmError``) -- this replays the
        same wire shape a live failure produces: a terminal ``error`` event with
        no ``done`` afterward, so the same failure looks identical whether the
        client was connected live or reconnects later. There is no persisted
        failure detail to include (chain.py doesn't keep any either), so this
        uses the same generic message the live path sanitizes down to.
        """

        async def replay() -> AsyncIterator[str]:
            diagnosis_row = repo.get_diagnosis(connection, session_id)
            if diagnosis_row is None:
                error = ProgressEvent(
                    type="error", stage="s1_diagnose", message=_GENERIC_DIAGNOSIS_ERROR
                )
                yield f"event: {error.type}\ndata: {error.model_dump_json()}\n\n"
                return
            diagnosis = json.loads(diagnosis_row["payload_json"])
            diagnosis["misconception_id"] = diagnosis_row["misconception_id"]
            ready = ProgressEvent(type="diagnosis_ready", stage="s1_diagnose", payload=diagnosis)
            yield f"event: {ready.type}\ndata: {ready.model_dump_json()}\n\n"
            done = ProgressEvent(type="done", payload={"status": status})
            yield f"event: {done.type}\ndata: {done.model_dump_json()}\n\n"

        return StreamingResponse(
            replay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{session_id}/stream")
    async def stream(session_id: str, request: Request) -> StreamingResponse:
        connection = conn_of(request)
        row = repo.get_session(connection, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="session not found")

        # Gate BEFORE the compare-and-swap below: failing after it would leave the
        # session claimed at `in_progress` with no run behind it, permanently
        # unreachable. An unconfirmed photo transcription must never be diagnosed --
        # `ingest_photo` sets student_corrected=False and documents that only the
        # confirmation flow may set it True. Diagnosing a transcription the student
        # never checked risks pinning a misconception on them for an error the vision
        # model invented, which is worse than any latency this gate costs. The gate is
        # unconditional for photo input rather than keyed on `needs_review`: a model
        # can misread a line and still report high confidence, so confidence alone is
        # not evidence the student was ever shown what will be diagnosed.
        if row["status"] == "created":
            pending = StudentSubmission.model_validate_json(row["student_work_json"])
            if pending.source == "photo" and not pending.student_corrected:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "photo transcription must be confirmed before diagnosis; "
                        f"PUT /api/sessions/{session_id}/submission with the reviewed work"
                    ),
                )

        # A compare-and-swap (UPDATE ... WHERE status = 'created'), not a
        # read-then-write: reading `row["status"]` above and separately checking
        # it against "created" leaves a window between that read and starting the
        # run for a second concurrent request to read the very same
        # pre-transition status and also decide it's safe to start -- especially
        # since the run itself starts via asyncio.create_task, which defers its
        # first line to the next event-loop iteration, giving a same-tick
        # concurrent request ample room to land in that window. try_start_session
        # closes it: the check and the transition are one atomic statement, so at
        # most one concurrent caller for a given session ever observes success.
        if not repo.try_start_session(connection, session_id):
            # Lost the compare-and-swap: this session was not `created` at the
            # instant of the UPDATE, so starting a run here would race whoever
            # already claimed it (this handler moments ago, or a concurrent one in
            # the very same event-loop tick) and double-bill the LLM call. Re-read
            # to tell "already running" apart from "already terminal" -- it is the
            # CAS above, not this follow-up read, that closes the race, so this
            # read merely has to report correctly, not itself be race-free.
            current = repo.get_session(connection, session_id)["status"]
            if current == "in_progress":
                # Phase 1 has no way to distinguish "still running" from "crashed
                # mid-run" (see chain.py's module docstring), so a genuinely stuck
                # session surfaces as this same 409 until it's investigated by
                # other means.
                raise HTTPException(
                    status_code=409, detail="diagnosis already in progress for this session"
                )
            return _replay_terminal_state(connection, session_id, current)

        submission = StudentSubmission.model_validate_json(row["student_work_json"])
        chain = Chain(connection, request.app.state.client, settings=request.app.state.settings)

        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                diagnosis: Diagnosis | None = None
                terminal: ProgressEvent | None = None
                async for event in chain.run_diagnosis(session_id, submission):
                    if event.type == "diagnosis_ready" and event.payload:
                        # Captured here rather than re-read from the DB so the
                        # pipeline runs on exactly the object the student was
                        # shown -- no window where the two could differ.
                        diagnosis = Diagnosis.model_validate(
                            {k: v for k, v in event.payload.items() if k != "misconception_id"}
                        )
                    if event.type == "done":
                        # Held back, not forwarded yet. `done` means "the whole run
                        # finished"; forwarding the diagnosis stage's own `done`
                        # here would tell a client to stop listening just as the
                        # animation pipeline starts reporting.
                        terminal = event
                        continue
                    await queue.put(event)

                # The animation pipeline continues on the SAME stream. This is the
                # answer to the latency finding: the diagnosis card renders at
                # ~15-30s and chat opens against it immediately, while s2-s8 and
                # the render (another ~3 minutes, dominated by s7 codegen) report
                # progress into the same connection instead of leaving a blank
                # screen for the rest of it.
                if diagnosis is not None and not diagnosis.no_error_found:
                    pipeline = Pipeline(
                        connection,
                        request.app.state.client,
                        settings=request.app.state.settings,
                    )
                    async for event in pipeline.run(session_id, submission, diagnosis):
                        await queue.put(event)

                if terminal is not None:
                    status = repo.get_session(connection, session_id)["status"]
                    await queue.put(terminal.model_copy(update={"payload": {"status": status}}))
            except Exception:
                # Chain's contract is "an LlmError becomes a terminal `error` event,
                # everything else never raises" (see chain.py) -- this is a last-resort
                # net for a genuine bug that violates that contract anyway. Without marking
                # the session here, a bug in this path would wedge the session at
                # `in_progress` forever: every future reconnect hits the 409 guard in
                # `stream` above, with no way to ever resolve. Marking it `failed` (not just
                # logging) is what turns an unforeseen bug into a recoverable failure
                # instead of a permanently stuck row. Nothing about an exception's str()
                # here is assumed safe to show a client -- this is exactly the path that
                # leaked an upstream secret once before -- so only the fixed, generic
                # message is ever put on the queue; the real detail is logged server-side
                # only.
                logger.exception("run_diagnosis crashed for session %s", session_id)
                repo.set_session_status(connection, session_id, "failed")
                await queue.put(
                    ProgressEvent(
                        type="error", stage="s1_diagnose", message=_GENERIC_DIAGNOSIS_ERROR
                    )
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce())
        request.app.state.background_tasks[session_id] = task
        task.add_done_callback(
            lambda _t, sid=session_id: request.app.state.background_tasks.pop(sid, None)
        )

        async def events() -> AsyncIterator[str]:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event.type == "error":
                    # event.message may embed arbitrary upstream-derived text (an
                    # LlmError built from a misbehaving proxy's response body) --
                    # this is the trust boundary to the client, so it is logged
                    # here, never forwarded verbatim.
                    logger.warning("diagnosis error for session %s: %s", session_id, event.message)
                    event = event.model_copy(update={"message": _GENERIC_DIAGNOSIS_ERROR})
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{session_id}/beats")
    def get_beats(session_id: str, request: Request) -> dict:
        """The beat rail: plan plus measured timings, and the video if ready.

        Beats are returned as soon as s6 plans them, with null timings, so the
        rail can render greyed segments while the animation is still being made.
        """
        connection = conn_of(request)
        if repo.get_session(connection, session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        render = repo.latest_render(connection, session_id)
        return {
            "beats": [
                {
                    "id": row["beat_id"],
                    "title": row["title"],
                    "purpose": row["purpose"],
                    "targets_misconception": bool(row["targets_misconception"]),
                    "start_s": row["start_s"],
                    "end_s": row["end_s"],
                }
                for row in repo.list_beats(connection, session_id)
            ],
            "video_url": f"/media/{session_id}/video.mp4" if render else None,
            "render_mode": render["mode"] if render else None,
        }

    @app.get("/media/{session_id}/video.mp4")
    def get_video(session_id: str, request: Request):
        """Serve the rendered video.

        The path comes from the `renders` row this server wrote, never from the
        request, so a traversal attempt in `session_id` finds no row and 404s
        rather than reaching the filesystem. FileResponse handles range requests,
        which a video element needs to seek -- and seeking is the entire point of
        beat citations.
        """
        connection = conn_of(request)
        render = repo.latest_render(connection, session_id)
        if render is None or not render["video_path"]:
            raise HTTPException(status_code=404, detail="no rendered video for this session")
        path = Path(render["video_path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="rendered video is no longer on disk")
        return FileResponse(path, media_type="video/mp4")

    @app.post("/api/sessions/{session_id}/chat")
    async def post_chat(session_id: str, body: ChatRequest, request: Request) -> dict:
        connection = conn_of(request)
        if repo.get_session(connection, session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        try:
            reply, cited, meta = await chat.answer(
                connection,
                request.app.state.client,
                session_id=session_id,
                question=body.message,
                model=request.app.state.settings.deepseek_model_fast,
            )
        except ValueError as exc:
            # No diagnosis yet: chat is grounded in one, so there is nothing to
            # be grounded in. A client should wait for `diagnosis_ready`.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except LlmError as exc:
            logger.warning("chat failed for session %s: %s", session_id, exc)
            raise HTTPException(status_code=503, detail=_GENERIC_CHAT_ERROR) from exc
        repo.record_artifact(
            connection,
            session_id=session_id,
            stage=_CHAT_STAGE,
            payload={"cited_beats": cited},
            meta=meta,
        )
        return {"reply": reply, "cited_beats": cited}

    @app.get("/api/sessions/{session_id}/chat")
    def get_chat(session_id: str, request: Request) -> dict:
        connection = conn_of(request)
        if repo.get_session(connection, session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "messages": [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "cited_beats": json.loads(row["cited_beats_json"]),
                }
                for row in repo.list_chat(connection, session_id)
            ]
        }

    @app.get("/api/sessions/{session_id}/peers")
    def get_peers(session_id: str, request: Request) -> dict:
        """ "N other students made this error" for this session. Aggregate only."""
        connection = conn_of(request)
        if repo.get_session(connection, session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        return insights.peers_for_session(connection, session_id)

    @app.get("/api/insights")
    def get_insights(request: Request, handle: str | None = None) -> dict:
        connection = conn_of(request)
        return {
            "misconceptions": insights.misconception_frequency(connection),
            "history": insights.student_history(connection, handle) if handle else [],
        }

    _WEB_DIR = Path(__file__).parent.parent / "web"
    if _WEB_DIR.is_dir():
        # Mounted last so it cannot shadow any /api or /media route above.
        app.mount("/", _NoCacheStatic(directory=_WEB_DIR, html=True), name="web")

    return app
