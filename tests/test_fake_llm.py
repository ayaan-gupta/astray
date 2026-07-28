import pytest
from pydantic import BaseModel

from server.llm.deepseek import DeepSeekClient, LlmError
from server.llm.fake import fake_transport


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
