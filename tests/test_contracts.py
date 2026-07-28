import pytest
from pydantic import ValidationError

from server.charter.contracts import Diagnosis, StudentSubmission, SympyCheck


def test_submission_requires_problem():
    with pytest.raises(ValidationError):
        StudentSubmission(steps=["x=4"], source="typed")


def test_submission_defaults():
    s = StudentSubmission(problem="(x+3)^2=25", steps=["x^2+9=25"], source="typed")
    assert s.prose is None
    assert s.student_corrected is False
    assert s.transcription_confidence is None


def test_diagnosis_confidence_bounded():
    with pytest.raises(ValidationError):
        Diagnosis(
            correct_solution=["x=2"],
            buggy_rule="(a+b)^2 -> a^2+b^2",
            misconception_statement="Missing cross term.",
            confidence=1.4,
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


def test_sympy_check_equivalence_shape():
    c = SympyCheck(kind="equivalence", lhs="(x+3)**2", rhs="x**2+6*x+9")
    assert c.lhs == "(x+3)**2"
