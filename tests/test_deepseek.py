import json

import httpx
import pytest
from pydantic import BaseModel

from server.llm.deepseek import DeepSeekClient, SchemaRetryExhausted


class Answer(BaseModel):
    buggy_rule: str
    confidence: float


def _reply(content: str, *, reasoning: str | None = None, prompt=100, completion=20, cached=0):
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def _client(handler) -> DeepSeekClient:
    return DeepSeekClient("sk-test", transport=httpx.MockTransport(handler))


async def test_complete_json_parses_and_reports_meta():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "thinking" not in body or body["thinking"]["type"] == "enabled"
        return httpx.Response(
            200,
            json=_reply(
                '{"buggy_rule": "(a+b)^2 -> a^2+b^2", "confidence": 0.9}',
                reasoning="student dropped the cross term",
                prompt=1000,
                completion=500,
                cached=400,
            ),
        )

    answer, meta = await _client(handler).complete_json(
        messages=[{"role": "user", "content": "diagnose"}],
        schema=Answer,
        model="deepseek-v4-pro",
    )
    assert answer.buggy_rule == "(a+b)^2 -> a^2+b^2"
    assert meta.reasoning == "student dropped the cross term"
    assert meta.cached_tokens == 400
    assert meta.attempts == 1
    assert meta.cost_usd == pytest.approx(0.00069745, rel=1e-6)


async def test_schema_injected_into_prompt():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_reply('{"buggy_rule": "r", "confidence": 0.5}'))

    await _client(handler).complete_json(
        messages=[{"role": "user", "content": "diagnose"}],
        schema=Answer,
        model="deepseek-v4-flash",
    )
    blob = json.dumps(seen[0]["messages"])
    assert "buggy_rule" in blob and "confidence" in blob


async def test_retries_on_invalid_json_then_succeeds():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json=_reply("not json at all"))
        return httpx.Response(200, json=_reply('{"buggy_rule": "r", "confidence": 0.5}'))

    answer, meta = await _client(handler).complete_json(
        messages=[{"role": "user", "content": "go"}], schema=Answer, model="deepseek-v4-flash"
    )
    assert answer.confidence == 0.5
    assert meta.attempts == 2
    # the retry must tell the model what was wrong
    assert len(calls[1]["messages"]) > len(calls[0]["messages"])
    assert "json" in json.dumps(calls[1]["messages"]).lower()


async def test_retries_on_schema_violation():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=_reply('{"buggy_rule": "r"}'))  # missing confidence
        return httpx.Response(200, json=_reply('{"buggy_rule": "r", "confidence": 0.1}'))

    answer, meta = await _client(handler).complete_json(
        messages=[{"role": "user", "content": "go"}], schema=Answer, model="deepseek-v4-flash"
    )
    assert meta.attempts == 2
    assert answer.confidence == 0.1


async def test_raises_after_retries_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply("still not json"))

    with pytest.raises(SchemaRetryExhausted):
        await _client(handler).complete_json(
            messages=[{"role": "user", "content": "go"}],
            schema=Answer,
            model="deepseek-v4-flash",
            max_retries=1,
        )


async def test_strict_mode_disables_thinking_and_forces_tool_call():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # This combination is the ONLY one DeepSeek accepts for forced tool use.
        assert body["thinking"] == {"type": "disabled"}
        assert body["tool_choice"]["function"]["name"] == "emit_answer"
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
                                        "arguments": '{"buggy_rule": "r", "confidence": 0.7}',
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

    answer, _ = await _client(handler).complete_strict(
        messages=[{"role": "user", "content": "go"}], schema=Answer, model="deepseek-v4-flash"
    )
    assert answer.confidence == 0.7


async def test_markdown_fenced_json_is_recovered():
    def handler(request: httpx.Request) -> httpx.Response:
        fenced = '```json\n{"buggy_rule": "r", "confidence": 0.3}\n```'
        return httpx.Response(200, json=_reply(fenced))

    answer, meta = await _client(handler).complete_json(
        messages=[{"role": "user", "content": "go"}], schema=Answer, model="deepseek-v4-flash"
    )
    assert answer.confidence == 0.3
    assert meta.attempts == 1


async def test_http_error_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    from server.llm.deepseek import LlmError

    with pytest.raises(LlmError):
        await _client(handler).complete_text(
            messages=[{"role": "user", "content": "go"}], model="deepseek-v4-flash"
        )


async def test_non_dict_error_body_raises_llm_error():
    """A proxy/gateway in front of the real API can return a JSON array or bare string
    as its error body instead of a dict. That must not surface as AttributeError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=["not", "a", "dict"])

    from server.llm.deepseek import LlmError

    with pytest.raises(LlmError):
        await _client(handler).complete_text(
            messages=[{"role": "user", "content": "go"}], model="deepseek-v4-flash"
        )


@pytest.mark.parametrize("body", [{}, {"choices": []}])
async def test_malformed_success_body_raises_llm_error_via_complete_text(body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    from server.llm.deepseek import LlmError

    with pytest.raises(LlmError):
        await _client(handler).complete_text(
            messages=[{"role": "user", "content": "go"}], model="deepseek-v4-flash"
        )


@pytest.mark.parametrize("body", [{}, {"choices": []}])
async def test_malformed_success_body_raises_llm_error_via_complete_json(body):
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    from server.llm.deepseek import LlmError

    with pytest.raises(LlmError) as exc_info:
        await _client(handler).complete_json(
            messages=[{"role": "user", "content": "go"}], schema=Answer, model="deepseek-v4-flash"
        )
    # a malformed response shape is a protocol failure, not a schema violation: it must
    # fail immediately as the base LlmError, not get folded into the retry-exhaustion path.
    assert not isinstance(exc_info.value, SchemaRetryExhausted)
    assert len(calls) == 1


async def test_thinking_false_disables_thinking_in_complete_json():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        return httpx.Response(200, json=_reply('{"buggy_rule": "r", "confidence": 0.5}'))

    await _client(handler).complete_json(
        messages=[{"role": "user", "content": "go"}],
        schema=Answer,
        model="deepseek-v4-flash",
        thinking=False,
    )


async def test_thinking_false_disables_thinking_in_complete_text():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        return httpx.Response(200, json=_reply("plain text reply"))

    await _client(handler).complete_text(
        messages=[{"role": "user", "content": "go"}],
        model="deepseek-v4-flash",
        thinking=False,
    )


async def test_connect_error_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    from server.llm.deepseek import LlmError

    with pytest.raises(LlmError):
        await _client(handler).complete_text(
            messages=[{"role": "user", "content": "go"}], model="deepseek-v4-flash"
        )


async def test_timeout_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    from server.llm.deepseek import LlmError

    with pytest.raises(LlmError):
        await _client(handler).complete_text(
            messages=[{"role": "user", "content": "go"}], model="deepseek-v4-flash"
        )


async def test_three_attempt_retry_pins_conversation_growth():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) < 3:
            return httpx.Response(200, json=_reply("not json at all"))
        return httpx.Response(200, json=_reply('{"buggy_rule": "r", "confidence": 0.9}'))

    answer, meta = await _client(handler).complete_json(
        messages=[{"role": "user", "content": "go"}],
        schema=Answer,
        model="deepseek-v4-flash",
        max_retries=2,
    )
    assert answer.confidence == 0.9
    assert meta.attempts == 3
    assert len(calls) == 3
    # each failed attempt appends exactly one assistant turn + one corrective user turn
    assert len(calls[1]["messages"]) == len(calls[0]["messages"]) + 2
    assert len(calls[2]["messages"]) == len(calls[1]["messages"]) + 2
