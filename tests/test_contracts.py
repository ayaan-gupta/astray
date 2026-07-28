import pytest
from pydantic import ValidationError

from server.charter.contracts import (
    Diagnosis,
    LlmCallMeta,
    StageName,
    StudentSubmission,
    SympyCheck,
    Transcription,
)


def test_submission_requires_problem():
    with pytest.raises(ValidationError):
        StudentSubmission(steps=["x=4"], source="typed")


def test_submission_defaults():
    s = StudentSubmission(problem="(x+3)^2=25", steps=["x^2+9=25"], source="typed")
    assert s.prose is None
    assert s.student_corrected is False
    assert s.transcription_confidence is None


def test_submission_literal_rejection():
    with pytest.raises(ValidationError):
        StudentSubmission(problem="x+1=2", source="voice")


def test_submission_extra_key_allowed():
    s = StudentSubmission(problem="x+1=2", steps=[], source="typed", extra_field="ignored")
    assert s.problem == "x+1=2"


def test_diagnosis_confidence_bounded():
    with pytest.raises(ValidationError):
        Diagnosis(
            correct_solution=["x=2"],
            buggy_rule="(a+b)^2 -> a^2+b^2",
            misconception_statement="Missing cross term.",
            confidence=1.4,
            sympy_check=SympyCheck(kind="skip", skip_reason="n/a"),
        )


def test_diagnosis_confidence_lower_bound():
    with pytest.raises(ValidationError):
        Diagnosis(
            correct_solution=["x=2"],
            buggy_rule="(a+b)^2 -> a^2+b^2",
            misconception_statement="Missing cross term.",
            confidence=-0.1,
            sympy_check=SympyCheck(kind="skip", skip_reason="n/a"),
        )


def test_diagnosis_unclear_defaults_false():
    d = Diagnosis(
        correct_solution=["x=2", "x=-8"],
        buggy_rule="(a+b)^2 -> a^2+b^2",
        misconception_statement="Missing the cross term.",
        confidence=0.9,
        sympy_check=SympyCheck(kind="skip", skip_reason="n/a"),
    )
    assert d.is_unclear is False
    assert d.verified_by_sympy is False
    assert d.evidence == []
    assert d.competing_hypotheses == []


def test_diagnosis_extra_key_forbidden():
    with pytest.raises(ValidationError):
        Diagnosis(
            correct_solution=["x=2"],
            buggy_rule="(a+b)^2 -> a^2+b^2",
            misconception_statement="Missing cross term.",
            confidence=0.8,
            sympy_check=SympyCheck(kind="skip", skip_reason="n/a"),
            toipc="typo",
        )


def test_sympy_check_literal_rejection():
    with pytest.raises(ValidationError):
        SympyCheck(kind="bogus")


def test_sympy_check_equivalence_kind():
    c = SympyCheck(kind="equivalence", lhs="(x+3)**2", rhs="x**2+6*x+9")
    assert c.kind == "equivalence"
    assert c.lhs == "(x+3)**2"
    assert c.rhs == "x**2+6*x+9"


def test_transcription_confidence_upper_bound():
    with pytest.raises(ValidationError):
        Transcription(problem="1+1=?", confidence=1.5)


def test_transcription_confidence_lower_bound():
    with pytest.raises(ValidationError):
        Transcription(problem="1+1=?", confidence=-0.1)


def test_transcription_defaults():
    t = Transcription(problem="1+1=?", confidence=0.95)
    assert t.steps == []
    assert t.unreadable == []


def test_transcription_extra_key_forbidden():
    with pytest.raises(ValidationError):
        Transcription(problem="1+1=?", confidence=0.95, extra_field="bad")


def test_llm_call_meta_defaults():
    m = LlmCallMeta(model="gpt-4")
    assert m.model == "gpt-4"
    assert m.prompt_tokens == 0
    assert m.completion_tokens == 0
    assert m.cached_tokens == 0
    assert m.cost_usd == 0.0
    assert m.ms == 0
    assert m.attempts == 1
    assert m.reasoning is None


def test_stage_name_serialization():
    assert StageName.INGEST == "s0_ingest"
    assert StageName.DIAGNOSE == "s1_diagnose"
    assert StageName.INTENT == "s2_intent"
    assert StageName.PREREQ == "s3_prereq"
    assert StageName.CURRICULUM == "s4_curriculum"
    assert StageName.MATH == "s5_math"
    assert StageName.VISUAL == "s6_visual"
    assert StageName.SCENE == "s7_scene"
    assert StageName.VALIDATE == "s8_validate"
