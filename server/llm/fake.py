"""Offline transport for FAKE_LLM=1 and tests. Zero network, zero cost."""

import json

import httpx

FIXTURES: dict[str, str] = {
    "Diagnosis": json.dumps(
        {
            "correct_solution": ["(x+3)^2 = 25", "x+3 = \\pm 5", "x = 2 or x = -8"],
            "sympy_check": {
                "kind": "equivalence",
                "lhs": "(x+3)**2",
                "rhs": "x**2+6*x+9",
            },
            "verified_by_sympy": False,
            "divergence_index": 0,
            "buggy_rule": "(a+b)^2 -> a^2 + b^2",
            "misconception_statement": (
                "You distributed the exponent across the sum, which drops the 2ab cross term."
            ),
            "evidence": ["Step 1 wrote x^2 + 9 instead of x^2 + 6x + 9"],
            "confidence": 0.94,
            "competing_hypotheses": [],
            "is_unclear": False,
            "clarifying_question": None,
            "topic": "algebra.binomial_expansion",
        }
    ),
    # server/store/taxonomy.py's resolve_misconception() calls complete_strict with
    # this schema for any buggy_rule that isn't an exact canonical-rule match against
    # an already-seeded misconception. Without a fixture for it, FAKE_LLM=1 (and any
    # test using FIXTURES directly) 500s on that call -- fake_transport's own
    # "no fake fixture matched" fallback -- which complete_strict/DeepSeekClient
    # surfaces as LlmError, which resolve_misconception's `except LlmError` then
    # silently swallows into its raw-buggy_rule fallback mint. That makes offline
    # mode quietly exercise the taxonomy-outage fallback path on every novel rule
    # instead of the real adjudication path, with no visible failure -- exactly the
    # gap this fixture closes. `new_slug` (not `same_as_id`) so a novel rule mints a
    # fresh row rather than claiming to match one of the ~20 seeded misconceptions
    # that may or may not exist under this slug.
    "MatchDecision": json.dumps(
        {
            "same_as_id": None,
            "new_slug": "offline-fake-new-misconception",
            "reasoning": "offline fake transport: no existing entry to match, minting new",
        }
    ),
}


def fake_transport(script: dict[str, str]) -> httpx.MockTransport:
    """Return a transport that replies with the value whose key appears in the request.

    Keys are matched as case-insensitive substrings of intent-bearing parts of the
    request body (messages and tools arrays), which carry schema definitions.
    The model field is excluded to prevent silent collisions with model IDs.
    An unmatched request returns HTTP 500 so missing fixtures fail loudly.

    Detects tool_choice to distinguish complete_json (content) from complete_strict
    (tool_calls). For complete_strict, the fixture content is embedded in arguments.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        search_parts = {
            "messages": body.get("messages", []),
            "tools": body.get("tools", []),
        }
        blob = json.dumps(search_parts).lower()
        for key, content in script.items():
            if key.lower() in blob:
                is_strict = "tool_choice" in body
                if is_strict:
                    message = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "emit_answer",
                                    "arguments": content,
                                },
                            }
                        ],
                        "reasoning_content": f"[fake reasoning for {key}]",
                    }
                else:
                    message = {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": f"[fake reasoning for {key}]",
                    }
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    },
                )
        return httpx.Response(
            500, json={"error": {"message": f"no fake fixture matched; keys={list(script)}"}}
        )

    return httpx.MockTransport(handler)
