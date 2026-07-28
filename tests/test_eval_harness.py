from pathlib import Path

from evals.diagnosis.run import load_cases, score_case
from server.charter.contracts import Diagnosis, SympyCheck

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
