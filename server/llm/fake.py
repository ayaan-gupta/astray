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
    )
}


def fake_transport(script: dict[str, str]) -> httpx.MockTransport:
    """Return a transport that replies with the value whose key appears in the request.

    Keys are matched as case-insensitive substrings of the serialized request body.
    This allows matching on messages (for complete_json) and schema names in tools
    (for complete_strict). An unmatched request returns HTTP 500 so missing fixtures
    fail loudly.

    Detects tool_choice to distinguish complete_json (content) from complete_strict
    (tool_calls). For complete_strict, the fixture content is embedded in arguments.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        blob = json.dumps(body).lower()
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
