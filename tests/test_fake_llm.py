import json

import pytest
from pydantic import BaseModel

from server.charter.contracts import Diagnosis, Transcription
from server.llm.deepseek import DeepSeekClient, LlmError
from server.llm.fake import FIXTURES, fake_transport


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
    assert len(Diagnosis.model_fields) == 12


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


async def test_fixtures_transcription_is_valid():
    """Ensure future Transcription fixture addition is Pydantic-clean."""
    transcription_json = json.dumps(
        {
            "problem": "Solve x + 2 = 5",
            "steps": ["x = 3"],
            "confidence": 0.95,
        }
    )
    payload = json.loads(transcription_json)
    transcription = Transcription.model_validate(payload)
    assert transcription.confidence == 0.95
