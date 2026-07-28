import json

import httpx

from server.charter.chain import Chain
from server.charter.contracts import StudentSubmission
from server.config import Settings
from server.llm.deepseek import DeepSeekClient
from server.store import repo
from server.store.db import connect
from server.store.seed_taxonomy import seed

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
    "evidence": ["step 0"],
    "confidence": 0.93,
    "competing_hypotheses": [],
    "is_unclear": False,
    "clarifying_question": None,
    "topic": "algebra.binomial_expansion",
}

SUBMISSION = StudentSubmission(
    problem="Solve (x+3)^2 = 25",
    steps=["x^2+9=25", "x=4"],
    source="typed",
    student_corrected=True,
)


def _settings() -> Settings:
    return Settings(_env_file=None, deepseek_api_key="sk-test")


def _client() -> DeepSeekClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):  # taxonomy strict call
            args = json.dumps(
                {"same_as_id": None, "new_slug": "freshmans-dream", "reasoning": "match"}
            )
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
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
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
                            "reasoning_content": "traced the divergence",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 200},
            },
        )

    return DeepSeekClient("sk-test", transport=httpx.MockTransport(handler))


async def test_run_diagnosis_emits_ordered_events(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(conn, _client(), settings=_settings())

    events = [e async for e in chain.run_diagnosis(sid, SUBMISSION)]
    types = [e.type for e in events]
    assert types[0] == "stage_started"
    assert "diagnosis_ready" in types
    assert types[-1] == "done"


async def test_run_diagnosis_persists_artifact_with_reasoning(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(conn, _client(), settings=_settings())
    async for _ in chain.run_diagnosis(sid, SUBMISSION):
        pass

    artifacts = repo.list_artifacts(conn, sid)
    stages = [a["stage"] for a in artifacts]
    assert "s1_diagnose" in stages
    diagnose_row = next(a for a in artifacts if a["stage"] == "s1_diagnose")
    assert diagnose_row["reasoning_text"] == "traced the divergence"
    assert diagnose_row["cost_usd"] > 0


async def test_run_diagnosis_saves_diagnosis_and_links_misconception(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(conn, _client(), settings=_settings())
    async for _ in chain.run_diagnosis(sid, SUBMISSION):
        pass

    row = repo.get_diagnosis(conn, sid)
    assert row["buggy_rule"] == "(a+b)^2 -> a^2 + b^2"
    assert row["misconception_id"] is not None
    assert row["verified_by_sympy"] == 1  # sympy_check genuinely passes
    assert repo.get_session(conn, sid)["status"] == "diagnosed"


async def test_unclear_diagnosis_sets_needs_clarification(tmp_path):
    unclear = {
        **DIAGNOSIS,
        "confidence": 0.2,
        "sympy_check": {"kind": "skip", "skip_reason": "too sparse"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            args = json.dumps({"same_as_id": None, "new_slug": "unclear-x", "reasoning": "n"})
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
                        "message": {"role": "assistant", "content": json.dumps(unclear)},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(
        conn, DeepSeekClient("sk", transport=httpx.MockTransport(handler)), settings=_settings()
    )
    async for _ in chain.run_diagnosis(sid, SUBMISSION):
        pass
    assert repo.get_session(conn, sid)["status"] == "needs_clarification"


async def test_llm_failure_emits_error_event_and_marks_session(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "down"}})

    conn = connect(tmp_path / "t.db")
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(
        conn, DeepSeekClient("sk", transport=httpx.MockTransport(handler)), settings=_settings()
    )
    events = [e async for e in chain.run_diagnosis(sid, SUBMISSION)]
    assert events[-1].type == "error"
    assert repo.get_session(conn, sid)["status"] == "failed"


async def test_llm_failure_leaves_no_artifact_or_diagnosis_rows(tmp_path):
    """A diagnose-stage LlmError must not leave a partial s1_diagnose artifact
    or a diagnoses row behind -- the failure happens before either is ever
    written, so neither should exist afterward."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "down"}})

    conn = connect(tmp_path / "t.db")
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(
        conn, DeepSeekClient("sk", transport=httpx.MockTransport(handler)), settings=_settings()
    )
    async for _ in chain.run_diagnosis(sid, SUBMISSION):
        pass

    assert repo.list_artifacts(conn, sid) == []
    assert repo.get_diagnosis(conn, sid) is None


async def test_session_marked_in_progress_before_diagnose_call(tmp_path):
    """A process killed mid-diagnose-call must be distinguishable from a
    session nobody ever picked up: sessions.status must already be
    ``in_progress`` by the time the first (stage_started) event is observed,
    not just at the end of a successful run."""
    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    assert repo.get_session(conn, sid)["status"] == "created"

    chain = Chain(conn, _client(), settings=_settings())
    events = chain.run_diagnosis(sid, SUBMISSION)
    first = await events.__anext__()
    assert first.type == "stage_started"
    assert repo.get_session(conn, sid)["status"] == "in_progress"

    async for _ in events:
        pass
    assert repo.get_session(conn, sid)["status"] == "diagnosed"


async def test_schema_retry_records_correct_attempt_number(tmp_path):
    """DeepSeekClient.complete_json's schema-retry loop can take more than one
    attempt to produce a valid Diagnosis; the winning attempt number must land
    in run_artifacts.attempt, not always 1."""
    calls = {"diagnose": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            args = json.dumps(
                {"same_as_id": None, "new_slug": "freshmans-dream", "reasoning": "match"}
            )
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
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                },
            )
        calls["diagnose"] += 1
        if calls["diagnose"] == 1:
            # Unparseable content forces DeepSeekClient.complete_json to retry.
            content = "not json at all, no braces here"
        else:
            content = json.dumps(DIAGNOSIS)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": "traced the divergence",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 200},
            },
        )

    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(
        conn, DeepSeekClient("sk", transport=httpx.MockTransport(handler)), settings=_settings()
    )
    async for _ in chain.run_diagnosis(sid, SUBMISSION):
        pass

    assert calls["diagnose"] == 2
    artifacts = repo.list_artifacts(conn, sid)
    diagnose_row = next(a for a in artifacts if a["stage"] == "s1_diagnose")
    assert diagnose_row["attempt"] == 2


async def test_exact_canonical_match_makes_no_taxonomy_call_or_artifact(tmp_path):
    """DIAGNOSIS.buggy_rule is an exact canonical match for the seeded
    'freshmans-dream' entry, so resolve_misconception must take its fast
    path: zero HTTP calls to the adjudication endpoint, and no separate
    run_artifacts row for it (only s1_diagnose)."""
    tool_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            tool_calls["n"] += 1
            raise AssertionError("adjudication must not be called on an exact canonical match")
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
                            "reasoning_content": "traced the divergence",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 200},
            },
        )

    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(
        conn, DeepSeekClient("sk", transport=httpx.MockTransport(handler)), settings=_settings()
    )
    async for _ in chain.run_diagnosis(sid, SUBMISSION):
        pass

    assert tool_calls["n"] == 0
    artifacts = repo.list_artifacts(conn, sid)
    assert [a["stage"] for a in artifacts] == ["s1_diagnose"]


async def test_taxonomy_adjudication_call_is_recorded_as_its_own_artifact(tmp_path):
    """When the buggy_rule does NOT already have an exact canonical match,
    resolve_misconception makes a real (billable) adjudication call. That
    call's tokens/cost must land in the ledger as its own run_artifacts row,
    not be silently dropped."""
    novel = {
        **DIAGNOSIS,
        "buggy_rule": "tan(a+b) -> tan(a) + tan(b)",
        "misconception_statement": "Distributed tan across a sum.",
        "topic": "trigonometry.identities",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            args = json.dumps(
                {"same_as_id": None, "new_slug": "tan-distribute", "reasoning": "novel"}
            )
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
                    "usage": {"prompt_tokens": 5000, "completion_tokens": 5000},
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
                            "content": json.dumps(novel),
                            "reasoning_content": "traced the divergence",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 200},
            },
        )

    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(
        conn, DeepSeekClient("sk", transport=httpx.MockTransport(handler)), settings=_settings()
    )
    async for _ in chain.run_diagnosis(sid, SUBMISSION):
        pass

    artifacts = repo.list_artifacts(conn, sid)
    stages = [a["stage"] for a in artifacts]
    assert "s1_diagnose" in stages
    taxonomy_rows = [a for a in artifacts if a["stage"] != "s1_diagnose"]
    assert len(taxonomy_rows) == 1
    assert taxonomy_rows[0]["cost_usd"] > 0
    assert taxonomy_rows[0]["model"] == "deepseek-v4-flash"
    assert taxonomy_rows[0]["prompt_tokens"] == 5000
    assert taxonomy_rows[0]["completion_tokens"] == 5000


async def test_taxonomy_llm_error_through_chain_still_diagnoses_with_fallback_mint(tmp_path):
    """A taxonomy adjudication failure must never fail the session: the
    diagnose stage succeeded, so the run should still reach 'diagnosed' with
    a fallback-minted misconception_id, and (per the ledger contract) no
    taxonomy artifact row, since no billable call completed."""
    novel = {
        **DIAGNOSIS,
        "buggy_rule": "tan(a+b) -> tan(a) + tan(b)",
        "misconception_statement": "Distributed tan across a sum.",
        "topic": "trigonometry.identities",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            return httpx.Response(500, json={"error": {"message": "adjudication down"}})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(novel),
                            "reasoning_content": "traced the divergence",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 200},
            },
        )

    conn = connect(tmp_path / "t.db")
    seed(conn)
    sid = repo.create_session(conn, handle="anon", submission=SUBMISSION)
    chain = Chain(
        conn, DeepSeekClient("sk", transport=httpx.MockTransport(handler)), settings=_settings()
    )
    events = [e async for e in chain.run_diagnosis(sid, SUBMISSION)]

    assert events[-1].type == "done"
    assert repo.get_session(conn, sid)["status"] == "diagnosed"
    row = repo.get_diagnosis(conn, sid)
    assert row["misconception_id"] is not None
    artifacts = repo.list_artifacts(conn, sid)
    assert [a["stage"] for a in artifacts] == ["s1_diagnose"]
