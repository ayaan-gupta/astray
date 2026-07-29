from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StageName(StrEnum):
    INGEST = "s0_ingest"
    DIAGNOSE = "s1_diagnose"
    INTENT = "s2_intent"
    PREREQ = "s3_prereq"
    CURRICULUM = "s4_curriculum"
    MATH = "s5_math"
    VISUAL = "s6_visual"
    SCENE = "s7_scene"
    VALIDATE = "s8_validate"


class LlmCallMeta(BaseModel):
    """Provenance for one model call. Never contains prompt text or secrets."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    ms: int = 0
    attempts: int = 1
    reasoning: str | None = None


class Transcription(BaseModel):
    """Raw vision output. Must preserve the student's errors verbatim."""

    model_config = ConfigDict(extra="forbid")

    problem: str
    steps: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    unreadable: list[str] = Field(default_factory=list)


class StudentSubmission(BaseModel):
    problem: str
    steps: list[str] = Field(default_factory=list)
    prose: str | None = None
    source: Literal["typed", "photo"] = "typed"
    transcription_confidence: float | None = None
    student_corrected: bool = False
    unreadable: list[str] = Field(default_factory=list)


class SympyCheck(BaseModel):
    """A deterministic verification the model asks us to run on its own solution.

    The model emits SymPy-parseable syntax (``**`` not ``^``), never LaTeX.
    ``kind="skip"`` means the domain is not symbolically checkable.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["equivalence", "solution_set", "skip"]
    lhs: str | None = None
    rhs: str | None = None
    equation: str | None = None
    variable: str | None = None
    candidates: list[str] = Field(default_factory=list)
    skip_reason: str | None = None


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct_solution: list[str]
    sympy_check: SympyCheck
    verified_by_sympy: bool = False
    # True when the student's work is already correct and there is no error to
    # diagnose. This is a first-class outcome, not a degenerate diagnosis: the
    # prompt has always told the model to say so rather than invent an error,
    # but without an explicit flag the downstream pipeline could not tell that
    # answer apart from a real one and minted a taxonomy entry from the prose
    # in `buggy_rule` ("none -- the student's solution is correct"). That row
    # then became a cross-topic magnet, since every later correct submission
    # canonicalized onto it, and it surfaced in students' misconception
    # histories as if it were a diagnosed error. `divergence_index is None`
    # cannot substitute for this flag -- it is also null when the student
    # supplied no steps at all.
    no_error_found: bool = False
    divergence_index: int | None = None
    buggy_rule: str
    misconception_statement: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    competing_hypotheses: list[str] = Field(default_factory=list)
    is_unclear: bool = False
    clarifying_question: str | None = None
    topic: str = "unknown"
