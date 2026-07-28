import asyncio
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from server.app import MAX_UPLOAD_BYTES, create_app
from server.config import Settings
from server.llm.deepseek import DeepSeekClient
from server.llm.vision import GeminiVision

DIAGNOSIS = {
    "correct_solution": ["x = 2", "x = -8"],
    "sympy_check": {
        "kind": "solution_set",
        "equation": "(x+3)**2 - 25",
        "variable": "x",
        "candidates": ["2", "-8"],
    },
    "verified_by_sympy": False,
    "divergence_index": 0,
    "buggy_rule": "(a+b)^2 -> a^2 + b^2",
    "misconception_statement": "You dropped the cross term.",
    "evidence": [],
    "confidence": 0.9,
    "competing_hypotheses": [],
    "is_unclear": False,
    "clarifying_question": None,
    "topic": "algebra.binomial_expansion",
}


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("tools"):
        args = json.dumps({"same_as_id": None, "new_slug": "fd", "reasoning": "n"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c",
                                    "type": "function",
                                    "function": {"name": "emit_answer", "arguments": args},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(DIAGNOSIS),
                        "reasoning_content": "r",
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        _env_file=None,
        deepseek_api_key="sk-test",
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient("sk-test", transport=httpx.MockTransport(_handler)),
    )
    with TestClient(app) as c:
        yield c


def test_health_reports_vision_disabled(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["vision_enabled"] is False


def test_create_session_returns_id(client):
    r = client.post(
        "/api/sessions",
        json={"handle": "anon-1", "problem": "Solve (x+3)^2 = 25", "work": "x^2+9=25\nx=4"},
    )
    assert r.status_code == 201
    assert r.json()["session_id"]


def test_create_session_rejects_empty_problem(client):
    r = client.post("/api/sessions", json={"handle": "a", "problem": "  ", "work": "x"})
    assert r.status_code == 422


def test_get_unknown_session_404s(client):
    assert client.get("/api/sessions/nope").status_code == 404


def test_stream_yields_diagnosis_then_done(client):
    sid = client.post(
        "/api/sessions", json={"handle": "a", "problem": "Solve (x+3)^2 = 25", "work": "x^2+9=25"}
    ).json()["session_id"]

    with client.stream("GET", f"/api/sessions/{sid}/stream") as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    assert "diagnosis_ready" in text
    assert '"type": "done"' in text or '"type":"done"' in text


def test_get_session_includes_diagnosis_after_stream(client):
    sid = client.post(
        "/api/sessions", json={"handle": "a", "problem": "Solve (x+3)^2 = 25", "work": "x^2+9=25"}
    ).json()["session_id"]
    with client.stream("GET", f"/api/sessions/{sid}/stream") as response:
        list(response.iter_text())

    body = client.get(f"/api/sessions/{sid}").json()
    assert body["status"] == "diagnosed"
    assert body["diagnosis"]["buggy_rule"] == "(a+b)^2 -> a^2 + b^2"
    assert body["diagnosis"]["misconception_id"] is not None


def test_reconnecting_stream_replays_terminal_state_without_rerunning(tmp_path):
    """A plain EventSource auto-reconnects on any transient network hiccup by
    re-issuing the same GET. Once a session has already reached a terminal
    status, a second connection must replay the persisted result instead of
    silently re-running (and re-billing) the whole diagnose stage."""
    calls = {"diagnose": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            args = json.dumps({"same_as_id": None, "new_slug": "fd", "reasoning": "n"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c",
                                        "type": "function",
                                        "function": {"name": "emit_answer", "arguments": args},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        calls["diagnose"] += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(DIAGNOSIS),
                            "reasoning_content": "r",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    settings = Settings(
        _env_file=None,
        deepseek_api_key="sk-test",
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient("sk-test", transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as c:
        sid = c.post(
            "/api/sessions",
            json={"handle": "a", "problem": "Solve (x+3)^2 = 25", "work": "x^2+9=25"},
        ).json()["session_id"]
        with c.stream("GET", f"/api/sessions/{sid}/stream") as response:
            list(response.iter_text())
        assert c.get(f"/api/sessions/{sid}").json()["status"] == "diagnosed"
        assert calls["diagnose"] == 1

        with c.stream("GET", f"/api/sessions/{sid}/stream") as response:
            assert response.status_code == 200
            replayed = "".join(response.iter_text())

        assert "diagnosis_ready" in replayed
        assert '"type": "done"' in replayed or '"type":"done"' in replayed
        # The reconnect must not have triggered a second billable diagnose call.
        assert calls["diagnose"] == 1


def test_second_stream_call_while_in_progress_returns_409(tmp_path):
    """A reconnect (or a second tab) while a run is already writing this
    session's rows must not start a second run.run_diagnosis and race it
    against the first."""

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            args = json.dumps({"same_as_id": None, "new_slug": "fd", "reasoning": "n"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c",
                                        "type": "function",
                                        "function": {"name": "emit_answer", "arguments": args},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        await asyncio.sleep(0.3)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(DIAGNOSIS),
                            "reasoning_content": "r",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    settings = Settings(
        _env_file=None,
        deepseek_api_key="sk-test",
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient(
            "sk-test", transport=httpx.MockTransport(slow_handler)
        ),
    )

    with TestClient(app) as c:
        sid = c.post(
            "/api/sessions",
            json={"handle": "a", "problem": "Solve (x+3)^2 = 25", "work": "x^2+9=25"},
        ).json()["session_id"]

        first_status = {}

        def run_first():
            with c.stream("GET", f"/api/sessions/{sid}/stream") as r:
                first_status["code"] = r.status_code
                list(r.iter_text())

        t = threading.Thread(target=run_first)
        t.start()
        try:
            for _ in range(40):  # poll until the first request marks in_progress
                if c.get(f"/api/sessions/{sid}").json()["status"] == "in_progress":
                    break
                time.sleep(0.01)
            else:
                pytest.fail("session never reached in_progress")

            second = c.get(f"/api/sessions/{sid}/stream")
            assert second.status_code == 409
        finally:
            t.join(timeout=5)

        assert first_status.get("code") == 200
        assert c.get(f"/api/sessions/{sid}").json()["status"] == "diagnosed"


def test_photo_upload_returns_503_when_vision_disabled(client):
    sid = client.post("/api/sessions", json={"handle": "a", "problem": "p", "work": "w"}).json()[
        "session_id"
    ]
    r = client.post(
        f"/api/sessions/{sid}/photo", files={"file": ("w.png", b"\x89PNG", "image/png")}
    )
    assert r.status_code == 503
    assert "GEMINI_API_KEY" in r.json()["detail"]


def test_api_never_returns_secrets(client):
    body = client.get("/api/health").text
    assert "sk-test" not in body


def test_api_never_returns_secrets_from_a_reflecting_upstream(tmp_path):
    """/api/health structurally cannot carry a secret -- the meaningful case is an
    upstream that reflects the configured API key back in its own error body (a
    realistic shape for a misconfigured proxy during an incident). That text must
    never reach the client on either channel that relays upstream-derived error
    text: the SSE `error` event, and the photo route's 503 detail."""
    deepseek_secret = "sk-reflect-me-1234"
    gemini_secret = "AQ.reflect-me-5678"

    def deepseek_handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        return httpx.Response(500, content=f"upstream broke: {auth}".encode())

    async def gemini_handler(request: httpx.Request) -> httpx.Response:
        key = request.headers.get("x-goog-api-key", "")
        return httpx.Response(500, content=f"upstream broke: {key}".encode())

    settings = Settings(
        _env_file=None,
        deepseek_api_key=deepseek_secret,
        gemini_api_key=gemini_secret,
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient(
            deepseek_secret, transport=httpx.MockTransport(deepseek_handler)
        ),
        vision_factory=lambda: GeminiVision(
            gemini_secret, transport=httpx.MockTransport(gemini_handler)
        ),
    )

    with TestClient(app) as c:
        health_text = c.get("/api/health").text
        assert deepseek_secret not in health_text
        assert gemini_secret not in health_text

        sid = c.post(
            "/api/sessions",
            json={"handle": "a", "problem": "Solve (x+3)^2 = 25", "work": "x^2+9=25"},
        ).json()["session_id"]

        with c.stream("GET", f"/api/sessions/{sid}/stream") as response:
            stream_text = "".join(response.iter_text())
        assert deepseek_secret not in stream_text
        assert "event: error" in stream_text

        photo = c.post(
            f"/api/sessions/{sid}/photo", files={"file": ("w.png", b"\x89PNG", "image/png")}
        )
        assert gemini_secret not in photo.text
        assert photo.status_code == 503


def test_reconnecting_stream_after_failure_replays_error_not_done(tmp_path):
    """A live failure emits a terminal `error` event with no `done` afterward.
    A reconnect once the session is already `failed` must produce the identical
    wire shape -- otherwise the same failure looks different depending on when
    the client happens to connect."""

    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    settings = Settings(
        _env_file=None,
        deepseek_api_key="sk-test",
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient(
            "sk-test", transport=httpx.MockTransport(failing_handler)
        ),
    )

    with TestClient(app) as c:
        sid = c.post("/api/sessions", json={"handle": "a", "problem": "p", "work": "w"}).json()[
            "session_id"
        ]

        with c.stream("GET", f"/api/sessions/{sid}/stream") as response:
            live_text = "".join(response.iter_text())
        assert c.get(f"/api/sessions/{sid}").json()["status"] == "failed"
        assert "event: error" in live_text
        assert "event: done" not in live_text

        with c.stream("GET", f"/api/sessions/{sid}/stream") as response:
            replay_text = "".join(response.iter_text())
        assert "event: error" in replay_text
        assert "event: done" not in replay_text


def test_photo_upload_succeeds_when_vision_enabled(tmp_path):
    """Every other photo test runs under the vision-disabled fixture; this is the
    one test that drives the actual 200 path, via a vision_factory hook mirroring
    client_factory."""

    async def gemini_handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "problem": "Solve (x+3)^2 = 25",
            "steps": ["x^2 + 9 = 25", "x = 4"],
            "confidence": 0.97,
            "unreadable": [],
        }
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
            },
        )

    settings = Settings(
        _env_file=None,
        deepseek_api_key="sk-test",
        gemini_api_key="AQ.test",
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient("sk-test", transport=httpx.MockTransport(_handler)),
        vision_factory=lambda: GeminiVision(
            "AQ.test", transport=httpx.MockTransport(gemini_handler)
        ),
    )

    with TestClient(app) as c:
        sid = c.post("/api/sessions", json={"handle": "a", "problem": "p", "work": "w"}).json()[
            "session_id"
        ]
        r = c.post(f"/api/sessions/{sid}/photo", files={"file": ("w.png", b"\x89PNG", "image/png")})

    assert r.status_code == 200
    body = r.json()
    assert body["transcription"]["steps"] == ["x^2 + 9 = 25", "x = 4"]
    assert body["needs_review"] is False


def test_oversized_upload_rejection_carries_cors_headers(client):
    """MaxBodySizeMiddleware's own 413 must still pass through CORSMiddleware --
    otherwise a browser at the app's configured origin sees an opaque network
    error instead of the {"detail": ...} body, since a response with no
    access-control-allow-origin header is not readable by frontend JS."""
    sid = client.post("/api/sessions", json={"handle": "a", "problem": "p", "work": "w"}).json()[
        "session_id"
    ]
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1024)
    r = client.post(
        f"/api/sessions/{sid}/photo",
        files={"file": ("big.png", oversized, "image/png")},
        headers={"Origin": "http://localhost:5173"},
    )
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_concurrent_requests_share_connection_without_corruption(client):
    """FastAPI dispatches `def` routes (create_session, get_session) on a
    threadpool while `async def` routes run on the loop thread, and every
    request shares the one connection opened at startup. server/store/db.py's
    own tests already prove the connection is safe under concurrent raw
    threads; this proves the wiring here doesn't undo that guarantee when
    driven through real concurrent HTTP requests instead."""
    import concurrent.futures

    def create_and_fetch(i: int):
        created = client.post(
            "/api/sessions", json={"handle": f"c{i}", "problem": f"p{i}", "work": "w"}
        )
        sid = created.json()["session_id"]
        fetched = client.get(f"/api/sessions/{sid}")
        return sid, created.status_code, fetched.status_code, fetched.json()["problem"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(create_and_fetch, range(40)))

    assert all(status == 201 for _, status, _, _ in results)
    assert all(status == 200 for _, _, status, _ in results)
    session_ids = [sid for sid, _, _, _ in results]
    assert len(set(session_ids)) == len(session_ids)
    for i, (_, _, _, problem) in enumerate(results):
        assert problem == f"p{i}"


def test_photo_upload_rejects_oversized_body(client):
    """MaxBodySizeMiddleware must reject an oversized upload via its declared
    Content-Length before Starlette's multipart parser (which has no total-size
    cap on file parts -- see server/app.py's module docstring) ever touches it."""
    sid = client.post("/api/sessions", json={"handle": "a", "problem": "p", "work": "w"}).json()[
        "session_id"
    ]
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1024)
    r = client.post(
        f"/api/sessions/{sid}/photo", files={"file": ("big.png", oversized, "image/png")}
    )
    assert r.status_code == 413


def test_unexpected_error_returns_500_without_leaking_detail(tmp_path, monkeypatch):
    """An unhandled exception anywhere in a route must never echo its own str()
    (which could contain internal detail, or in a worse case a secret) back to the
    client -- FastAPI's default handler collapses it to a generic message."""
    settings = Settings(
        _env_file=None,
        deepseek_api_key="sk-test",
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient("sk-test", transport=httpx.MockTransport(_handler)),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("sk-should-never-leak-this-secret")

    monkeypatch.setattr("server.app.repo.get_session", boom)

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/api/sessions/whatever")
        assert r.status_code == 500
        assert "sk-should-never-leak-this-secret" not in r.text
        assert "RuntimeError" not in r.text
        assert "Traceback" not in r.text


async def test_disconnected_stream_still_diagnoses_and_persists(tmp_path):
    """A client that disconnects mid-run must not leave the session stuck at
    ``in_progress`` forever: the diagnosis has to finish and persist in the
    background even though nobody is left to receive the SSE frames.

    TestClient's in-process ASGI transport does not simulate a real client
    disconnect for a streaming response -- closing the stream early still lets
    the server-side generator run to completion, so there is nothing to prove
    against it. This drives the raw ASGI interface directly instead: it sends
    ``{"type": "http.disconnect"}`` from the stream route's own ``receive()``
    partway through a deliberately slow diagnose call, then polls the durable
    session row (not the dead stream) to confirm the run still reached
    ``diagnosed``.
    """

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            args = json.dumps({"same_as_id": None, "new_slug": "fd", "reasoning": "n"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c",
                                        "type": "function",
                                        "function": {"name": "emit_answer", "arguments": args},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        await asyncio.sleep(0.2)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(DIAGNOSIS),
                            "reasoning_content": "r",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    settings = Settings(
        _env_file=None,
        deepseek_api_key="sk-test",
        db_path=tmp_path / "t.db",
        media_root=tmp_path / "media",
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient(
            "sk-test", transport=httpx.MockTransport(slow_handler)
        ),
    )

    def _http_scope(method: str, path: str) -> dict:
        return {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "raw_path": path.encode(),
            "http_version": "1.1",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
        }

    async def _call(scope: dict, body: bytes = b"", *, disconnect_after: float | None = None):
        sent_request = False

        async def receive():
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": body, "more_body": False}
            if disconnect_after is not None:
                await asyncio.sleep(disconnect_after)
                return {"type": "http.disconnect"}
            await asyncio.Event().wait()  # no further messages expected

        events: list[dict] = []

        async def send(message: dict) -> None:
            events.append(message)

        await app(scope, receive, send)
        return b"".join(e["body"] for e in events if e["type"] == "http.response.body")

    startup_complete = asyncio.Event()
    shutdown_requested = asyncio.Event()

    async def lifespan_receive():
        if not startup_complete.is_set():
            return {"type": "lifespan.startup"}
        await shutdown_requested.wait()
        return {"type": "lifespan.shutdown"}

    async def lifespan_send(message: dict) -> None:
        if message["type"] == "lifespan.startup.complete":
            startup_complete.set()

    lifespan_task = asyncio.create_task(app({"type": "lifespan"}, lifespan_receive, lifespan_send))
    await startup_complete.wait()
    try:
        created = json.loads(
            await _call(
                _http_scope("POST", "/api/sessions"),
                json.dumps(
                    {"handle": "a", "problem": "Solve (x+3)^2 = 25", "work": "x^2+9=25"}
                ).encode(),
            )
        )
        session_id = created["session_id"]

        # Disconnect well before the 0.2s slow diagnose call resolves.
        await _call(_http_scope("GET", f"/api/sessions/{session_id}/stream"), disconnect_after=0.02)

        for _ in range(40):  # poll up to ~2s
            row = json.loads(await _call(_http_scope("GET", f"/api/sessions/{session_id}")))
            if row["status"] != "in_progress":
                break
            await asyncio.sleep(0.05)

        assert row["status"] == "diagnosed"
        assert row["diagnosis"]["buggy_rule"] == "(a+b)^2 -> a^2 + b^2"
    finally:
        shutdown_requested.set()
        await lifespan_task


async def test_shutdown_fails_unfinished_background_run_instead_of_leaving_in_progress(tmp_path):
    """Because the diagnosis run is deliberately decoupled from the request (see
    the disconnect test above), it is also invisible to uvicorn's normal
    connection-draining on shutdown -- an ordinary redeploy mid-run would
    otherwise close the shared client/connection out from under a still-running
    background task. Shutdown must wait (bounded by shutdown_drain_timeout_s)
    and mark anything still unfinished `failed`, never leave it `in_progress`
    with the connection already gone underneath it.
    """
    from server.store import repo as repo_module
    from server.store.db import connect as connect_module

    async def very_slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(2)  # much longer than the 0.1s drain timeout below
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(DIAGNOSIS),
                            "reasoning_content": "r",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    db_path = tmp_path / "t.db"
    settings = Settings(
        _env_file=None, deepseek_api_key="sk-test", db_path=db_path, media_root=tmp_path / "media"
    )
    app = create_app(
        settings=settings,
        client_factory=lambda: DeepSeekClient(
            "sk-test", transport=httpx.MockTransport(very_slow_handler)
        ),
        shutdown_drain_timeout_s=0.1,
    )

    def _http_scope(method: str, path: str) -> dict:
        return {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "raw_path": path.encode(),
            "http_version": "1.1",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
        }

    async def _call(scope: dict, body: bytes = b"", *, disconnect_after: float | None = None):
        sent_request = False

        async def receive():
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": body, "more_body": False}
            if disconnect_after is not None:
                await asyncio.sleep(disconnect_after)
                return {"type": "http.disconnect"}
            await asyncio.Event().wait()

        events: list[dict] = []

        async def send(message: dict) -> None:
            events.append(message)

        await app(scope, receive, send)
        return b"".join(e["body"] for e in events if e["type"] == "http.response.body")

    startup_complete = asyncio.Event()
    shutdown_requested = asyncio.Event()

    async def lifespan_receive():
        if not startup_complete.is_set():
            return {"type": "lifespan.startup"}
        await shutdown_requested.wait()
        return {"type": "lifespan.shutdown"}

    async def lifespan_send(message: dict) -> None:
        if message["type"] == "lifespan.startup.complete":
            startup_complete.set()

    lifespan_task = asyncio.create_task(app({"type": "lifespan"}, lifespan_receive, lifespan_send))
    await startup_complete.wait()

    created = json.loads(
        await _call(
            _http_scope("POST", "/api/sessions"),
            json.dumps({"handle": "a", "problem": "p", "work": "w"}).encode(),
        )
    )
    session_id = created["session_id"]

    # Disconnect quickly -- the background task keeps running the 2s slow call.
    await _call(_http_scope("GET", f"/api/sessions/{session_id}/stream"), disconnect_after=0.02)

    # Shut down while the run is still in flight: well past the 0.1s drain
    # timeout, well before the 2s call would ever finish on its own.
    shutdown_requested.set()
    await asyncio.wait_for(lifespan_task, timeout=5)

    # The app's own connection is closed now -- verify durable state with a
    # fresh one, exactly as an operator investigating after a redeploy would.
    verify_conn = connect_module(db_path)
    try:
        row = repo_module.get_session(verify_conn, session_id)
        assert row["status"] == "failed"
    finally:
        verify_conn.close()
