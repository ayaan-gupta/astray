import json
import logging
import sqlite3

import httpx
import pytest

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


async def test_slug_truncation_collision_mints_distinct_rows(tmp_path):
    """Slug construction truncates to 64 characters. Two long, genuinely different
    buggy rules that share the same 64-character prefix must not be merged
    into one row just because their auto-derived slugs collide after
    truncation — that would silently discard the second misconception's own
    statement/rule/topic and misattribute every future occurrence of it to
    the first (the over-merging failure the brief singles out as the costlier
    error direction). Confirmed offline that these two rules share a 64-char
    slug prefix (both truncate to the same 64 characters) while their
    canonical_rule values differ.
    """
    conn = connect(tmp_path / "t.db")
    seed(conn)

    common = (
        "student treats the differential operator as distributing across "
        "every term of a long sum expression here"
    )
    rule1 = common + " producing wrong derivative outcome alpha"
    rule2 = common + " producing wrong integral outcome beta"

    # decision.new_slug is None for both, so resolve_misconception falls back
    # to slugifying diagnosis.buggy_rule itself for each — the same source of
    # truncation the brief's fallback-mint path uses.
    client1 = _strict_client({"same_as_id": None, "new_slug": None, "reasoning": "new"})
    got1 = await taxonomy.resolve_misconception(
        conn,
        client1,
        diagnosis=_diagnosis(rule1, topic="calculus.derivatives"),
        model="deepseek-v4-flash",
    )
    client2 = _strict_client({"same_as_id": None, "new_slug": None, "reasoning": "new"})
    got2 = await taxonomy.resolve_misconception(
        conn,
        client2,
        diagnosis=_diagnosis(rule2, topic="calculus.integrals"),
        model="deepseek-v4-flash",
    )

    assert got1 != got2, "two distinct misconceptions must not collapse into one row"
    row1 = conn.execute(
        "SELECT slug, canonical_rule FROM misconceptions WHERE id = ?", (got1,)
    ).fetchone()
    row2 = conn.execute(
        "SELECT slug, canonical_rule FROM misconceptions WHERE id = ?", (got2,)
    ).fetchone()
    assert row1["slug"] != row2["slug"]
    assert row1["canonical_rule"] == taxonomy.canonicalize_rule(rule1)
    assert row2["canonical_rule"] == taxonomy.canonicalize_rule(rule2)
    assert row1["canonical_rule"] != row2["canonical_rule"]


async def test_llm_error_fallback_mints_from_raw_rule_and_logs(tmp_path, caplog):
    """The `except LlmError` branch in resolve_misconception (mint from the
    raw rule when the model call itself fails) had no coverage — this is
    exactly the path the truncation-collision bug above lived on. Also pins
    that the fallback logs, so an LLM outage minting a wave of near-duplicates
    can be found and re-triaged later instead of vanishing silently.
    """
    conn = connect(tmp_path / "t.db")
    seed(conn)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "service unavailable"}})

    client = DeepSeekClient("sk-test", transport=httpx.MockTransport(handler))
    diagnosis = _diagnosis("a brand new rule the llm never gets to see", topic="new.topic")

    with caplog.at_level(logging.WARNING, logger="server.store.taxonomy"):
        got = await taxonomy.resolve_misconception(
            conn, client, diagnosis=diagnosis, model="deepseek-v4-flash"
        )

    row = conn.execute(
        "SELECT slug, canonical_rule, is_seed FROM misconceptions WHERE id = ?", (got,)
    ).fetchone()
    assert row["is_seed"] == 0
    assert row["canonical_rule"] == taxonomy.canonicalize_rule(diagnosis.buggy_rule)
    assert row["slug"]
    assert any("LLM adjudication failed" in record.message for record in caplog.records)


def test_is_slug_unique_violation_classifies_correctly():
    """Only a UNIQUE-on-slug violation should be treated as a lost mint race;
    any other IntegrityError (a future FK or CHECK constraint on
    misconceptions) must surface as real corruption, not be misreported."""
    slug_violation = sqlite3.IntegrityError("UNIQUE constraint failed: misconceptions.slug")
    not_null_violation = sqlite3.IntegrityError("NOT NULL constraint failed: misconceptions.topic")
    fk_violation = sqlite3.IntegrityError("FOREIGN KEY constraint failed")
    assert taxonomy._is_slug_unique_violation(slug_violation) is True
    assert taxonomy._is_slug_unique_violation(not_null_violation) is False
    assert taxonomy._is_slug_unique_violation(fk_violation) is False


async def test_lost_slug_race_reuses_the_winning_row(tmp_path, monkeypatch):
    """Two concurrent resolve_misconception() calls minting the same
    brand-new slug must resolve to one row, not crash and not duplicate.
    Simulated deterministically (no real threads) by making the slug lookup
    insert the competitor's row itself, right before the INSERT that would
    otherwise hit the UNIQUE constraint.
    """
    conn = connect(tmp_path / "t.db")
    seed(conn)

    real_resolve = taxonomy._resolve_slug_for_mint

    def racing_resolve(conn_, source_text, canonical):
        slug, existing_id = real_resolve(conn_, source_text, canonical)
        if existing_id is None:
            conn_.execute(
                """INSERT INTO misconceptions
                   (slug, canonical_statement, canonical_rule, topic, is_seed)
                   VALUES (?, ?, ?, ?, 0)""",
                (slug, "raced in first", canonical, "race.topic"),
            )
        return slug, existing_id

    monkeypatch.setattr(taxonomy, "_resolve_slug_for_mint", racing_resolve)

    client = _strict_client(
        {"same_as_id": None, "new_slug": "brand-new-race-slug", "reasoning": "novel"}
    )
    got = await taxonomy.resolve_misconception(
        conn,
        client,
        diagnosis=_diagnosis("something totally new for the race"),
        model="deepseek-v4-flash",
    )
    row = conn.execute(
        "SELECT slug, canonical_statement FROM misconceptions WHERE id = ?", (got,)
    ).fetchone()
    assert row["slug"] == "brand-new-race-slug"
    assert row["canonical_statement"] == "raced in first"


async def test_non_slug_integrity_error_is_not_swallowed(tmp_path, monkeypatch):
    """A non-slug IntegrityError on the mint INSERT (e.g. a future NOT NULL or
    FK constraint) must propagate, not be misreported as a harmless slug race."""
    conn = connect(tmp_path / "t.db")
    seed(conn)

    real_execute = conn.execute

    def boom_execute(sql, parameters=()):
        if sql.strip().startswith("INSERT INTO misconceptions"):
            raise sqlite3.IntegrityError("NOT NULL constraint failed: misconceptions.topic")
        return real_execute(sql, parameters)

    monkeypatch.setattr(conn, "execute", boom_execute)

    client = _strict_client(
        {"same_as_id": None, "new_slug": "totally-fresh-slug", "reasoning": "novel"}
    )
    with pytest.raises(sqlite3.IntegrityError):
        await taxonomy.resolve_misconception(
            conn,
            client,
            diagnosis=_diagnosis("brand new rule text nobody has seen"),
            model="deepseek-v4-flash",
        )


async def test_alias_appended_on_confident_match(tmp_path):
    """A confident LLM 'same' verdict should record the new phrasing as an
    alias of the existing row, not just return its id silently."""
    conn = connect(tmp_path / "t.db")
    seed(conn)
    target = conn.execute(
        "SELECT id FROM misconceptions WHERE slug = 'freshmans-dream'"
    ).fetchone()["id"]
    client = _strict_client({"same_as_id": target, "new_slug": None, "reasoning": "same error"})
    new_rule = "exponent distributes over a sum, alias variant"
    got = await taxonomy.resolve_misconception(
        conn, client, diagnosis=_diagnosis(new_rule), model="deepseek-v4-flash"
    )
    assert got == target
    aliases = json.loads(
        conn.execute("SELECT aliases_json FROM misconceptions WHERE id = ?", (target,)).fetchone()[
            "aliases_json"
        ]
    )
    assert new_rule in aliases


async def test_hallucinated_same_as_id_falls_through_to_mint(tmp_path):
    """A same_as_id with no matching row (model hallucination) must not
    crash — it should fall through to the normal new_slug minting path.
    This currently works only because `if row:` is False and control falls
    past the block; a well-meaning simplification of that check could
    silently reintroduce a crash, so it is pinned here."""
    conn = connect(tmp_path / "t.db")
    seed(conn)
    client = _strict_client(
        {"same_as_id": 999999, "new_slug": "hallucinated-id-fallback", "reasoning": "??"}
    )
    got = await taxonomy.resolve_misconception(
        conn,
        client,
        diagnosis=_diagnosis("something new entirely"),
        model="deepseek-v4-flash",
    )
    row = conn.execute("SELECT slug FROM misconceptions WHERE id = ?", (got,)).fetchone()
    assert row["slug"] == "hallucinated-id-fallback"
