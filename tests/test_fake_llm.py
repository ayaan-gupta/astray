import json

import pytest
from pydantic import BaseModel

from server.charter.contracts import Diagnosis, SympyCheck, Transcription
from server.llm.deepseek import DeepSeekClient, LlmError
from server.llm.fake import FIXTURES, fake_transport
from server.store import taxonomy
from server.store.db import connect
from server.store.seed_taxonomy import seed
from server.store.taxonomy import MatchDecision


class Answer(BaseModel):
    buggy_rule: str


async def test_fake_transport_matches_on_prompt_substring():
    transport = fake_transport({"expand": '{"buggy_rule": "(a+b)^2 -> a^2+b^2"}'})
    client = DeepSeekClient("sk-fake", transport=transport)
    answer, meta = await client.complete_json(
        messages=[{"role": "user", "content": "please expand this"}],
        schema=Answer,
        model="deepseek-v4-flash",
    )
    assert answer.buggy_rule == "(a+b)^2 -> a^2+b^2"
    assert meta.cost_usd == 0.0


async def test_fake_transport_unmatched_prompt_raises():
    client = DeepSeekClient("sk-fake", transport=fake_transport({"expand": "{}"}))
    with pytest.raises(LlmError):
        await client.complete_json(
            messages=[{"role": "user", "content": "unrelated"}],
            schema=Answer,
            model="deepseek-v4-flash",
        )


def test_fixtures_diagnosis_is_valid():
    """Validate that the Diagnosis fixture is properly formed."""
    payload = json.loads(FIXTURES["Diagnosis"])
    diagnosis = Diagnosis.model_validate(payload)
    assert diagnosis.buggy_rule == "(a+b)^2 -> a^2 + b^2"
    assert diagnosis.confidence == 0.94
    assert diagnosis.topic == "algebra.binomial_expansion"


def test_fixtures_diagnosis_is_pydantic_clean():
    """Confirm Diagnosis fixture has no extra fields (extra='forbid')."""
    payload = json.loads(FIXTURES["Diagnosis"])
    Diagnosis.model_validate(payload)
    # If extra="forbid" is violated, model_validate would raise. No fields added.
    # 13 since `no_error_found` was added; bump deliberately, never to silence a
    # failure -- a field added to Diagnosis without a matching prompt instruction
    # is a field the model will never populate.
    assert len(Diagnosis.model_fields) == 13


def test_fixtures_match_decision_is_valid():
    """Validate that the MatchDecision fixture is properly formed."""
    payload = json.loads(FIXTURES["MatchDecision"])
    decision = MatchDecision.model_validate(payload)
    assert decision.same_as_id is None
    assert decision.new_slug == "offline-fake-new-misconception"


async def test_novel_rule_resolves_via_fake_transport_not_the_outage_fallback(tmp_path):
    """Before the MatchDecision fixture existed, FAKE_LLM=1 (or any caller driving
    resolve_misconception through fake_transport(FIXTURES)) 500'd on the taxonomy
    adjudication call for any buggy_rule that wasn't an exact canonical-rule match
    against an already-seeded misconception -- fake_transport's own "no fixture
    matched" 500, which complete_strict/DeepSeekClient surfaces as LlmError, which
    resolve_misconception's own `except LlmError` then silently swallows into its
    raw-buggy_rule fallback mint. That made offline mode quietly exercise the
    taxonomy-outage path on every novel rule, with no visible failure. This drives
    exactly that path (a buggy_rule with no seeded match) through the real
    FIXTURES dict and asserts the adjudication call actually succeeded (meta is
    not None) rather than silently falling back."""
    conn = connect(tmp_path / "t.db")
    seed(conn)
    client = DeepSeekClient("sk-fake", transport=fake_transport(FIXTURES))
    diagnosis = Diagnosis(
        correct_solution=["irrelevant"],
        sympy_check=SympyCheck(kind="skip", skip_reason="n/a"),
        buggy_rule="a genuinely novel rule no seed entry matches",
        misconception_statement="something new",
        confidence=0.9,
        topic="a.brand.new.topic",
    )
    misconception_id, meta = await taxonomy.resolve_misconception(
        conn, client, diagnosis=diagnosis, model="deepseek-v4-flash"
    )
    assert meta is not None, (
        "a novel rule under FAKE_LLM must resolve via the MatchDecision fixture, "
        "not silently fall back to the taxonomy-outage path"
    )
    row = conn.execute(
        "SELECT slug FROM misconceptions WHERE id = ?", (misconception_id,)
    ).fetchone()
    assert row["slug"] == "offline-fake-new-misconception"


async def test_diagnosis_through_complete_json_with_real_model():
    """Verify the Diagnosis fixture is reachable through complete_json."""
    transport = fake_transport(FIXTURES)
    client = DeepSeekClient("sk-fake", transport=transport)
    diagnosis, meta = await client.complete_json(
        messages=[{"role": "user", "content": "diagnose this misconception"}],
        schema=Diagnosis,
        model="deepseek-v4-flash",
    )
    assert diagnosis.buggy_rule == "(a+b)^2 -> a^2 + b^2"
    assert diagnosis.confidence == 0.94
    assert meta.cost_usd == 0.0


async def test_complete_strict_returns_tool_calls():
    """Verify complete_strict gets tool_calls, not content."""
    transport = fake_transport(FIXTURES)
    client = DeepSeekClient("sk-fake", transport=transport)
    diagnosis, meta = await client.complete_strict(
        messages=[{"role": "user", "content": "diagnose this"}],
        schema=Diagnosis,
        model="deepseek-v4-flash",
    )
    assert diagnosis.buggy_rule == "(a+b)^2 -> a^2 + b^2"
    assert diagnosis.confidence == 0.94


async def test_complete_text_through_fake():
    """Verify complete_text works through the fake transport."""
    transport = fake_transport({"text_response": "This is a text answer."})
    client = DeepSeekClient("sk-fake", transport=transport)
    text, meta = await client.complete_text(
        messages=[{"role": "user", "content": "please text_response with text"}],
        model="deepseek-v4-flash",
    )
    assert text == "This is a text answer."
    assert meta.cost_usd == 0.0


async def test_case_insensitive_key_matching():
    """Verify keys match case-insensitively."""
    transport = fake_transport({"UPPERCASE": '{"buggy_rule": "case insensitive"}'})
    client = DeepSeekClient("sk-fake", transport=transport)
    answer, _ = await client.complete_json(
        messages=[{"role": "user", "content": "ask for uppercase"}],
        schema=Answer,
        model="deepseek-v4-flash",
    )
    assert answer.buggy_rule == "case insensitive"


async def test_first_matching_key_wins():
    """Verify first matching key wins when multiple keys match."""
    transport = fake_transport(
        {
            "key": '{"buggy_rule": "first"}',
            "key_extended": '{"buggy_rule": "second"}',
        }
    )
    client = DeepSeekClient("sk-fake", transport=transport)
    answer, _ = await client.complete_json(
        messages=[{"role": "user", "content": "ask for key_extended"}],
        schema=Answer,
        model="deepseek-v4-flash",
    )
    # First key "key" also matches "key_extended", so it wins
    assert answer.buggy_rule == "first"


async def test_meta_reasoning_is_synthetic():
    """Verify meta.reasoning contains the synthetic reasoning content."""
    transport = fake_transport({"test_key": '{"buggy_rule": "test"}'})
    client = DeepSeekClient("sk-fake", transport=transport)
    _, meta = await client.complete_json(
        messages=[{"role": "user", "content": "test_key"}],
        schema=Answer,
        model="deepseek-v4-flash",
    )
    assert meta.reasoning == "[fake reasoning for test_key]"


def test_transcription_model_accepts_minimal_payload():
    """Validate that Transcription model accepts minimal valid payload (not a fixture test)."""
    payload = {
        "problem": "Solve x + 2 = 5",
        "steps": ["x = 3"],
        "confidence": 0.95,
    }
    transcription = Transcription.model_validate(payload)
    assert transcription.confidence == 0.95


async def test_fixture_key_does_not_collide_with_model_id():
    """Regression test: fixture keyed with a model ID must not match unrelated schema.

    This test prevents silent false matches when a fixture is accidentally keyed
    with a string that appears in the request metadata (e.g., a model ID).
    """
    transport = fake_transport({"deepseek-v4-pro": '{"buggy_rule": "wrong"}'})
    client = DeepSeekClient("sk-fake", transport=transport)
    with pytest.raises(LlmError) as exc_info:
        await client.complete_json(
            messages=[{"role": "user", "content": "unrelated photosynthesis question"}],
            schema=Answer,
            model="deepseek-v4-pro",
        )
    assert "no fake fixture matched" in str(exc_info.value)
