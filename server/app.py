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
* **``GET .../stream`` is not safely repeatable, so it checks status first.** A
  plain ``EventSource`` (the natural frontend client for SSE) auto-reconnects on
  any transient network hiccup by re-issuing the exact same ``GET``. Without a
  guard, that would either race a second ``run_diagnosis`` against the one
  already writing this session's rows (``status == "in_progress"``), or silently
  re-run -- and re-bill -- the whole diagnose stage for a session that already
  reached a terminal status. The route rejects a reconnect while a run is
  already in flight (``409``) and replays the persisted terminal result instead
  of re-running one that already finished; only a session still at ``created``
  actually starts ``run_diagnosis``.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from server.charter.chain import Chain, ProgressEvent
from server.charter.contracts import StudentSubmission
from server.charter.stages.s0_ingest import ingest_photo, ingest_typed, needs_review
from server.config import Settings, get_settings
from server.deps import build_llm_client, build_vision
from server.llm.deepseek import DeepSeekClient
from server.llm.vision import VisionUnavailable
from server.store import repo
from server.store.db import connect
from server.store.seed_taxonomy import seed

logger = logging.getLogger(__name__)

# The photo route's own domain limit, checked against the fully-parsed file content.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# The whole HTTP request's hard cap, enforced by MaxBodySizeMiddleware before any
# multipart parsing happens. A little above MAX_IMAGE_BYTES for multipart
# boundary/header overhead around the file part.
MAX_UPLOAD_BYTES = MAX_IMAGE_BYTES + 1024 * 1024


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


class CreateSessionRequest(BaseModel):
    handle: str = "anon"
    problem: str
    work: str = ""
    prose: str | None = None

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
) -> FastAPI:
    resolved = settings or get_settings()
    make_client = client_factory or (lambda: build_llm_client(resolved))

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.conn = connect(resolved.db_path)
        seed(app.state.conn)
        app.state.client = make_client()
        app.state.vision = build_vision(resolved)
        # Strong references for asyncio.create_task()'d diagnosis runs (see the
        # module docstring): the event loop only holds a *weak* reference to a
        # task, so a run whose stream response nobody is reading anymore must be
        # kept alive here or it can be garbage-collected before it finishes.
        app.state.background_tasks = set()
        try:
            yield
        finally:
            await app.state.client.aclose()
            aclose = getattr(app.state.vision, "aclose", None)
            if aclose is not None:
                await aclose()
            app.state.conn.close()

    app = FastAPI(title="Math Misconception Tutor", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Added after CORSMiddleware so it ends up outermost (Starlette's user-middleware
    # list is built by inserting each new middleware at the front) -- oversized
    # requests must be rejected before any other middleware, routing, or body
    # parsing runs.
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)

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
            submission, _meta = await ingest_photo(
                request.app.state.vision, data, file.content_type or "image/png"
            )
        except VisionUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "transcription": submission.model_dump(),
            "needs_review": needs_review(submission),
        }

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
        completed. Yields the same ``diagnosis_ready``/``done`` shape a live run
        would have produced, sourced from the durable ``diagnoses`` row rather
        than a fresh (and separately billed) call to the diagnose stage.
        """

        async def replay() -> AsyncIterator[str]:
            diagnosis_row = repo.get_diagnosis(connection, session_id)
            if diagnosis_row is not None:
                diagnosis = json.loads(diagnosis_row["payload_json"])
                diagnosis["misconception_id"] = diagnosis_row["misconception_id"]
                ready = ProgressEvent(
                    type="diagnosis_ready", stage="s1_diagnose", payload=diagnosis
                )
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

        status = row["status"]
        if status == "in_progress":
            # A run is already writing this session's rows (by this handler, or one
            # from a moment ago) -- Chain has no notion of "already running" beyond
            # this status column, so starting a second run here would race the
            # first one's writes and double-bill the LLM call. Phase 1 also has no
            # way to distinguish "still running" from "crashed mid-run" (see
            # chain.py's module docstring), so a genuinely stuck session surfaces
            # as this same 409 until it's investigated by other means.
            raise HTTPException(
                status_code=409, detail="diagnosis already in progress for this session"
            )
        if status != "created":
            return _replay_terminal_state(connection, session_id, status)

        submission = StudentSubmission.model_validate_json(row["student_work_json"])
        chain = Chain(connection, request.app.state.client, settings=request.app.state.settings)

        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for event in chain.run_diagnosis(session_id, submission):
                    await queue.put(event)
            except Exception:
                # Chain's contract is "an LlmError becomes a terminal `error` event,
                # everything else never raises" (see chain.py) -- this is a last-resort
                # net so a genuine bug here can't wedge the session at `in_progress`
                # silently or crash an unrelated part of the process. Nothing about an
                # exception's str() here is assumed safe to show a client, so it is only
                # logged, never put on the queue or returned.
                logger.exception("run_diagnosis crashed for session %s", session_id)
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce())
        request.app.state.background_tasks.add(task)
        task.add_done_callback(request.app.state.background_tasks.discard)

        async def events() -> AsyncIterator[str]:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app_factory = create_app
