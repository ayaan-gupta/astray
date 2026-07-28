"""DeepSeek client.

Structured output uses JSON mode, not tool-use: DeepSeek returns HTTP 400
("Thinking mode does not support this tool_choice") for forced tool_choice while
thinking is enabled, and tool_choice="auto" is unreliable — it returns prose and
ignores the tool. JSON mode is the only path that keeps reasoning on, so schema
adherence is enforced here by Pydantic validation plus a retry that feeds the
validation error back to the model.
"""

import json
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from server.charter.contracts import LlmCallMeta
from server.llm.accounting import cost_usd

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LlmError(Exception):
    """Transport, HTTP, or protocol failure."""


class SchemaRetryExhausted(LlmError):
    """The model never produced output matching the schema."""


def _extract_json(text: str) -> Any:
    """Parse JSON, tolerating markdown fences and leading prose."""
    candidates = [text]
    if match := _FENCE.search(text):
        candidates.insert(0, match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no parseable JSON object in response")


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else. No prose, no markdown "
        "fences. It must validate against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


def _message(body: dict) -> dict:
    """Validate and return ``choices[0].message`` from a decoded response body.

    Raises ``LlmError`` (never a bare ``KeyError``/``IndexError``/``AttributeError``) for
    any malformed shape, so a caller following this module's contract can catch one type.
    """
    if not isinstance(body, dict):
        raise LlmError(f"deepseek response body is not a JSON object: {type(body).__name__}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmError(f"deepseek response missing non-empty 'choices': {body!r}"[:500])
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise LlmError(f"deepseek response choice missing 'message': {first!r}"[:500])
    return message


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: int = 240,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict) -> tuple[dict, int]:
        started = time.monotonic()
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            # str(exc) for httpx errors never includes headers/auth, only URL + message.
            raise LlmError(f"deepseek transport failure: {exc}") from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if response.status_code >= 400:
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError):
                data = None
            # The error body may not be a dict at all (proxy/gateway error pages return
            # arrays or bare strings) — never assume shape before `.get`.
            error = data.get("error") if isinstance(data, dict) else None
            detail = error.get("message", "") if isinstance(error, dict) else ""
            if not detail:
                detail = response.text[:200]
            raise LlmError(f"deepseek HTTP {response.status_code}: {detail}")
        return response.json(), elapsed_ms

    @staticmethod
    def _meta(body: dict, model: str, elapsed_ms: int, attempts: int) -> LlmCallMeta:
        usage = body.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        message = _message(body)
        return LlmCallMeta(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached,
            cost_usd=cost_usd(model, prompt_tokens, completion_tokens, cached),
            ms=elapsed_ms,
            attempts=attempts,
            reasoning=message.get("reasoning_content"),
        )

    async def complete_text(
        self, *, messages: list[dict], model: str, thinking: bool = True
    ) -> tuple[str, LlmCallMeta]:
        payload: dict = {"model": model, "messages": messages}
        if not thinking:
            payload["thinking"] = {"type": "disabled"}
        body, elapsed_ms = await self._post(payload)
        content = _message(body).get("content") or ""
        return content, self._meta(body, model, elapsed_ms, attempts=1)

    async def complete_json(
        self,
        *,
        messages: list[dict],
        schema: type[T],
        model: str,
        thinking: bool = True,
        max_retries: int = 2,
    ) -> tuple[T, LlmCallMeta]:
        convo = [*messages, {"role": "user", "content": _schema_instruction(schema)}]
        last_error = "unknown"

        for attempt in range(1, max_retries + 2):
            payload: dict = {
                "model": model,
                "messages": convo,
                "response_format": {"type": "json_object"},
            }
            if not thinking:
                payload["thinking"] = {"type": "disabled"}

            body, elapsed_ms = await self._post(payload)
            content = _message(body).get("content") or ""
            try:
                parsed = schema.model_validate(_extract_json(content))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:1500]
                convo = [
                    *convo,
                    {"role": "assistant", "content": content[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "That response was rejected. Fix it and return ONLY a valid "
                            f"JSON object matching the schema.\nError:\n{last_error}"
                        ),
                    },
                ]
                continue
            return parsed, self._meta(body, model, elapsed_ms, attempts=attempt)

        raise SchemaRetryExhausted(
            f"{model} failed schema {schema.__name__} after {max_retries + 1} attempts: "
            f"{last_error}"
        )

    async def complete_strict(
        self, *, messages: list[dict], schema: type[T], model: str
    ) -> tuple[T, LlmCallMeta]:
        """Forced tool call. Requires thinking disabled — DeepSeek rejects the pair otherwise."""
        tool = {
            "type": "function",
            "function": {
                "name": "emit_answer",
                "description": f"Emit a {schema.__name__}",
                "parameters": schema.model_json_schema(),
            },
        }
        payload = {
            "model": model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": "emit_answer"}},
        }
        body, elapsed_ms = await self._post(payload)
        calls = _message(body).get("tool_calls") or []
        if not calls:
            raise SchemaRetryExhausted(f"{model} returned no tool call in strict mode")
        try:
            parsed = schema.model_validate(json.loads(calls[0]["function"]["arguments"]))
        except (ValueError, ValidationError) as exc:
            raise SchemaRetryExhausted(f"strict-mode arguments invalid: {exc}") from exc
        return parsed, self._meta(body, model, elapsed_ms, attempts=1)
