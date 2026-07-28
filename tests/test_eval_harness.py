import json
from pathlib import Path

import httpx
import pytest
import yaml

from evals.diagnosis import run as eval_run
from evals.diagnosis.run import load_cases, main, score_case
from server.charter.contracts import Diagnosis, SympyCheck
from server.llm.deepseek import DeepSeekClient

CASES = Path("evals/diagnosis/cases.yaml")


def _diagnosis(rule: str, topic: str = "algebra.binomial_expansion", verified=True) -> Diagnosis:
    return Diagnosis(
        correct_solution=["x=2"],
        sympy_check=SympyCheck(kind="skip", skip_reason="n/a"),
        verified_by_sympy=verified,
        buggy_rule=rule,
        misconception_statement="s",
        confidence=0.9,
        topic=topic,
    )


def test_cases_file_has_at_least_twenty_cases():
    cases = load_cases(CASES)
    assert len(cases) >= 20
    assert len({c.id for c in cases}) == len(cases)  # ids unique


def test_case_ids_and_fields_are_populated():
    for case in load_cases(CASES):
        assert case.problem and case.steps and case.expected_rule
        assert case.expected_topic_prefix


def test_score_exact_canonical_rule_match():
    case = next(c for c in load_cases(CASES) if c.id == "freshmans-dream-solve")
    score = score_case(case, _diagnosis("(a+b)^2 -> a^2 + b^2"))
    assert score.rule_match is True
    assert score.topic_match is True


def test_score_matches_on_variable_rename():
    case = next(c for c in load_cases(CASES) if c.id == "freshmans-dream-solve")
    assert score_case(case, _diagnosis("(p+q)^2 -> p^2 + q^2")).rule_match is True


def test_score_matches_on_alias_phrase():
    case = next(c for c in load_cases(CASES) if c.id == "freshmans-dream-solve")
    score = score_case(case, _diagnosis("student dropped cross term when squaring"))
    assert score.rule_match is True


def test_score_rejects_unrelated_rule():
    case = next(c for c in load_cases(CASES) if c.id == "freshmans-dream-solve")
    assert score_case(case, _diagnosis("log(a+b) -> log a + log b")).rule_match is False


def test_topic_mismatch_detected():
    case = next(c for c in load_cases(CASES) if c.id == "chain-rule-omitted")
    score = score_case(case, _diagnosis("d/dx f(g(x)) -> f'(g(x))", topic="algebra.exponents"))
    assert score.topic_match is False


# --- Fix round 1, reviewer Critical finding 1: _token_overlap (the loosest of the three
# match routes) let a diagnosis naming a completely unrelated misconception "pass" by
# incidentally repeating the one or two words a case's expected_rule reduces to once
# punctuation/short-word filtering is applied. 8 of the 20 cases reduce to <=2
# discriminative tokens (negative-distribute, cancel-across-sum, power-of-power,
# single-root, divide-by-variable, zero-product-misuse, like-terms, product-rule); three
# are exercised below as concrete regressions for the >=3-token guard now in
# evals/diagnosis/run.py's _token_overlap. Each wrong_rule below explicitly disclaims or
# contradicts the real misconception while incidentally quoting the case's own notation --
# exactly the shape that passed, incorrectly, before the guard existed.


@pytest.mark.parametrize(
    ("case_id", "wrong_rule"),
    [
        (
            "cancel-across-sum",
            "(a+b)/b really is not the issue here -- the student actually misplaced the "
            "decimal point when converting a fraction to a percentage, an unrelated "
            "place-value error",
        ),
        (
            "power-of-power",
            "The student expanded (a^m)^n = a^(m+n) correctly -- the actual error is "
            "failing to rationalize the denominator afterward, an unrelated "
            "simplification mistake",
        ),
        (
            "product-rule",
            "d/dx (f*g) -> f' * g' is exactly what the student wrote and it's correct "
            "here -- the actual mistake is an unrelated arithmetic slip in evaluating "
            "sin(0)",
        ),
    ],
)
def test_token_overlap_no_longer_passes_an_unrelated_diagnosis(case_id, wrong_rule):
    case = next(c for c in load_cases(CASES) if c.id == case_id)
    score = score_case(case, _diagnosis(wrong_rule))
    assert score.rule_match is False


def test_token_overlap_still_matches_a_legitimate_paraphrase():
    # A real paraphrase of sqrt-of-sum (3 discriminative tokens, above the guard's floor)
    # that shares no words with its curated alias ("root of a sum as sum of roots"), so
    # this exercises the overlap route specifically, not alias matching -- confirming the
    # guard didn't also kill the legitimate loose-match coverage it's meant to preserve.
    case = next(c for c in load_cases(CASES) if c.id == "sqrt-of-sum")
    got = (
        "the student wrote sqrt(a+b) equals sqrt(a) plus sqrt(b) by splitting the "
        "radical across addition"
    )
    score = score_case(case, _diagnosis(got))
    assert score.rule_match is True
    assert score.notes == "overlap"


# --- Fix round 1, reviewer Important finding 3: two correct-but-differently-phrased
# diagnoses that the scorer wrongly rejected. Fixed with accept_aliases entries (not a
# threshold change -- a false positive is worse than a false negative for a regression
# gate, since a false negative still surfaces in the `regressed:` list for a human to see).


def test_new_alias_matches_transposition_sign_paraphrase():
    case = next(c for c in load_cases(CASES) if c.id == "transposition-sign")
    got = (
        "When moving the 5 to the other side of the equation, the student kept the "
        "same sign instead of negating it"
    )
    assert score_case(case, _diagnosis(got)).rule_match is True


def test_new_alias_matches_zero_product_misuse_paraphrase():
    case = next(c for c in load_cases(CASES) if c.id == "zero-product-misuse")
    got = (
        "The student set one factor equal to the constant 6 directly, as if the "
        "product being 6 meant a factor could equal 6, but that shortcut only works "
        "when the product is zero"
    )
    assert score_case(case, _diagnosis(got)).rule_match is True


# --- Fix round 2, reviewer Critical finding (residual): the same disclaiming attack the
# expected_rule route was fixed against in round 1 still works through the alias ratio
# path for any alias with >=3 discriminative tokens -- round 1's guard only zeroed the
# ratio for aliases at or below that floor (2-token aliases like "product of derivatives").
#
# The reviewer asked for the root-cause fix: make _token_overlap symmetric, requiring the
# shared tokens to be a meaningful fraction of the *got* text too, not just of the
# reference string. I implemented and measured this (min(shared/len(ta), shared/len(tb)))
# against every currently-required legitimate ratio match and against reconstructed
# disclaiming attacks on the same aliases. Result: not just "hard to threshold" but
# provably non-separable for at least one existing case --
#
#   alias "zero product property applied to nonzero" (5 tokens):
#     legitimate paraphrase -> symmetric score 0.105
#     disclaiming attack    -> symmetric score 0.263   (the ATTACK outscores the legitimate
#                                                        match -- an inversion, not just an
#                                                        overlapping range)
#
# A real pytest run against the symmetric formula confirmed this: it broke
# test_token_overlap_still_matches_a_legitimate_paraphrase (sqrt-of-sum) and
# test_new_alias_matches_zero_product_misuse_paraphrase (both above) while still not
# rejecting the reconstructed alias attack. No threshold value can fix an inversion --
# per the reviewer's own instruction ("if a symmetric ratio can't satisfy both, tell me
# what you measured rather than tuning the threshold"), _token_overlap is UNCHANGED from
# round 1 (still the asymmetric min(len(ta),len(tb)) denominator, still guarded by
# _MIN_DISCRIMINATIVE_TOKENS). See "Fix round 2" in the task report for the full
# measurement across more formulas (Jaccard, Dice) -- all showed the same non-separability
# on this route. The three tests below pin exactly what does and doesn't hold today:
# the ratio path still legitimately matches (unchanged, still needed), the *short*-alias
# half of the guard still rejects an attack that avoids literal substring, and the
# >=3-token-alias exploit is captured as a strict xfail -- a known, open residual, not a
# silently accepted one -- so it fails loudly the moment anyone actually closes it (a
# future fix should remove the xfail marker, not leave it stacked on top of a real fix).


def test_alias_ratio_path_matches_a_legitimate_paraphrase():
    # Deliberately avoids every literal accept_aliases substring, so this can only pass
    # through the ratio fallback, not the substring check -- isolates the same mechanism
    # test_token_overlap_still_matches_a_legitimate_paraphrase isolates for expected_rule.
    case = next(c for c in load_cases(CASES) if c.id == "zero-product-misuse")
    got = (
        "the shortcut of setting a factor equal to the constant only works when the "
        "product is actually zero"
    )
    assert not any(alias.lower() in got.lower() for alias in case.accept_aliases)
    score = score_case(case, _diagnosis(got))
    assert score.rule_match is True
    assert score.notes == "alias"


def test_alias_ratio_path_rejects_unrelated_diagnosis_for_short_alias():
    # product-rule's only alias, "product of derivatives", is 2 discriminative tokens --
    # at or below the guard's floor, so its ratio fallback is always 0.0. This diagnosis
    # avoids the literal substring "product of derivatives" (it says "derivatives
    # multiplied as a product" instead) while disclaiming the real misconception, so a
    # match here could only come from the ratio path the guard is supposed to zero out.
    case = next(c for c in load_cases(CASES) if c.id == "product-rule")
    got = (
        "This mistake has nothing to do with derivatives multiplied as a product -- the "
        "actual issue is an unrelated sign error copying down the final answer"
    )
    assert "product of derivatives" not in got.lower()
    score = score_case(case, _diagnosis(got))
    assert score.rule_match is False


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known, open residual (Fix round 2): the alias ratio path still accepts a "
        "disclaiming diagnosis for any alias with >=3 discriminative tokens. Measured "
        "non-separable from the legitimate match even with a symmetric forward/reverse "
        "ratio (the attack scores 0.263 vs the legitimate match's 0.105 against this same "
        "alias -- an inversion, not a threshold problem). strict=True so this flips to a "
        "hard failure -- not a silent pass -- the moment a real fix lands; remove this "
        "marker then, don't leave it stacked on top of the fix."
    ),
)
def test_alias_ratio_path_residual_still_accepts_a_disclaiming_attack():
    case = next(c for c in load_cases(CASES) if c.id == "zero-product-misuse")
    got = (
        "This is an unrelated transcription mistake -- the photo was blurry and misread "
        "a digit, nothing to do with the zero product property being applied to a "
        "nonzero product"
    )
    score = score_case(case, _diagnosis(got))
    assert score.rule_match is False


# --- Fix round 1, reviewer Important finding 4: main()'s CLI behavior beyond the brief's
# reference code (the FAKE_LLM refusal, the unknown-`--case`-id path, LlmError
# catch-and-continue, and the `regressed:` summary line) had no tests of its own. All run
# fully offline: the unknown-case path never builds a client at all, the FAKE_LLM refusal
# fires before any client is built, and the LlmError test drives a real DeepSeekClient
# wired to httpx.MockTransport (same technique as tests/test_s1_diagnose.py), never the
# network.


async def test_unknown_case_id_returns_nonzero_and_never_builds_a_client(capsys):
    def _must_not_be_called():
        raise AssertionError("client must not be built when --case matches no case")

    exit_code = await main(["--case", "not-a-real-id"], client_factory=_must_not_be_called)
    assert exit_code == 1
    assert "no case with id" in capsys.readouterr().out


def test_build_client_refuses_when_fake_llm_enabled(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("FAKE_LLM", "1")
    with pytest.raises(SystemExit, match="FAKE_LLM"):
        eval_run._build_client()


def _mock_diagnosis_response(buggy_rule: str) -> httpx.Response:
    payload = {
        "correct_solution": ["x=1"],
        "sympy_check": {"kind": "skip", "skip_reason": "n/a"},
        "verified_by_sympy": False,
        "divergence_index": None,
        "buggy_rule": buggy_rule,
        "misconception_statement": "s",
        "evidence": [],
        "confidence": 0.9,
        "competing_hypotheses": [],
        "is_unclear": False,
        "clarifying_question": None,
        "topic": "algebra.x",
    }
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload),
                        "reasoning_content": "ok",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        },
    )


async def test_main_continues_after_llm_error_and_reports_regressed(tmp_path, capsys):
    cases = [
        {
            "id": "ok-case",
            "problem": "UNIQUE_OK_PROBLEM: solve x=1",
            "steps": ["x=1"],
            "expected_rule": "r1 -> r1",
            "expected_topic_prefix": "algebra",
            "accept_aliases": [],
        },
        {
            "id": "bad-case",
            "problem": "UNIQUE_BAD_PROBLEM: solve x=2",
            "steps": ["x=2"],
            "expected_rule": "r2 -> r2",
            "expected_topic_prefix": "algebra",
            "accept_aliases": [],
        },
    ]
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(yaml.safe_dump(cases))

    def handler(request: httpx.Request) -> httpx.Response:
        blob = json.dumps(json.loads(request.content)).lower()
        if "unique_bad_problem" in blob:
            return httpx.Response(500, json={"error": {"message": "upstream exploded"}})
        return _mock_diagnosis_response("r1 -> r1")

    client = DeepSeekClient("sk-test", transport=httpx.MockTransport(handler))
    exit_code = await main(["--cases", str(cases_path)], client_factory=lambda: client)

    out = capsys.readouterr().out
    assert "[ERR ]" in out
    assert "[PASS] ok-case" in out
    assert "regressed (1): bad-case" in out
    assert exit_code == 1  # 1/2 pass rate is below the 80% gate
