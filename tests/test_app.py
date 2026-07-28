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
