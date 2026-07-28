import json

import httpx

from server.charter.contracts import Diagnosis, SympyCheck
from server.llm.deepseek import DeepSeekClient
from server.store import taxonomy
from server.store.db import connect
from server.store.seed_taxonomy import seed


def _diagnosis(rule: str, topic: str = "algebra.binomial_expansion") -> Diagnosis:
    return Diagnosis(
        correct_solution=["x=2"],
        sympy_check=SympyCheck(kind="skip", skip_reason="n/a"),
        buggy_rule=rule,
        misconception_statement="dropped cross term",
        confidence=0.9,
        topic=topic,
    )


def _strict_client(decision: dict) -> DeepSeekClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "emit_answer",
                                        "arguments": json.dumps(decision),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    return DeepSeekClient("sk-test", transport=httpx.MockTransport(handler))


def test_canonicalize_strips_variable_identity():
    a = taxonomy.canonicalize_rule("(x+3)^2 -> x^2 + 9")
    b = taxonomy.canonicalize_rule("(t + 5)^2  ->  t^2+25")
    assert a == b


def test_canonicalize_is_stable_and_lowercase():
    assert taxonomy.canonicalize_rule("(A+B)^2 -> A^2+B^2") == taxonomy.canonicalize_rule(
        "(a+b)^2 -> a^2+b^2"
    )


def test_seed_is_idempotent(tmp_path):
    conn = connect(tmp_path / "t.db")
    first = seed(conn)
    assert first >= 20
    assert seed(conn) == 0
    total = conn.execute("SELECT COUNT(*) FROM misconceptions").fetchone()[0]
    assert total == first


async def test_exact_canonical_match_skips_the_llm(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn)

    # A client that would explode if called proves the fast path is taken.
    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM must not be called on an exact canonical match")

    client = DeepSeekClient("sk-test", transport=httpx.MockTransport(boom))
    existing = conn.execute(
        "SELECT id, canonical_rule FROM misconceptions WHERE slug = 'freshmans-dream'"
    ).fetchone()
    diagnosis = _diagnosis("(a+b)^2 -> a^2 + b^2")
    assert taxonomy.canonicalize_rule(diagnosis.buggy_rule) == existing["canonical_rule"]
    got = await taxonomy.resolve_misconception(
        conn, client, diagnosis=diagnosis, model="deepseek-v4-flash"
    )
    assert got == existing["id"]


async def test_llm_says_same_as_existing(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn)
    target = conn.execute(
        "SELECT id FROM misconceptions WHERE slug = 'freshmans-dream'"
    ).fetchone()["id"]
    client = _strict_client({"same_as_id": target, "new_slug": None, "reasoning": "same error"})
    got = await taxonomy.resolve_misconception(
        conn,
        client,
        diagnosis=_diagnosis("exponent distributes over a sum"),
        model="deepseek-v4-flash",
    )
    assert got == target


async def test_llm_mints_new_entry(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn)
    before = conn.execute("SELECT COUNT(*) FROM misconceptions").fetchone()[0]
    client = _strict_client(
        {"same_as_id": None, "new_slug": "invented-tensor-rule", "reasoning": "novel"}
    )
    got = await taxonomy.resolve_misconception(
        conn,
        client,
        diagnosis=_diagnosis("tensor index lowering is commutative", topic="linear_algebra"),
        model="deepseek-v4-flash",
    )
    after = conn.execute("SELECT COUNT(*) FROM misconceptions").fetchone()[0]
    assert after == before + 1
    assert (
        conn.execute("SELECT slug FROM misconceptions WHERE id = ?", (got,)).fetchone()["slug"]
        == "invented-tensor-rule"
    )


async def test_duplicate_slug_reuses_existing_row(tmp_path):
    conn = connect(tmp_path / "t.db")
    seed(conn)
    client = _strict_client(
        {"same_as_id": None, "new_slug": "freshmans-dream", "reasoning": "collides"}
    )
    got = await taxonomy.resolve_misconception(
        conn, client, diagnosis=_diagnosis("something"), model="deepseek-v4-flash"
    )
    row = conn.execute("SELECT slug FROM misconceptions WHERE id = ?", (got,)).fetchone()
    assert row["slug"] == "freshmans-dream"
