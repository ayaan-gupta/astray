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
    # Runs after the render rather than before it: the narration script is
    # budgeted against each beat's measured duration, so it cannot be written
    # until the container has reported its clock.
    NARRATE = "s9_narrate"


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


# --- s2..s8: the planning and rendering half of the chain -------------------
#
# Every stage below is a JSON-mode call validated against one of these models,
# so a stage that drifts fails loudly at its own boundary rather than passing
# malformed data downstream. `extra="forbid"` throughout: a model inventing a
# field is a contract violation we want surfaced, not silently dropped.
#
# The beat is the unit that ties this half of the pipeline to the tutor. s6
# plans beats, s7 must wrap each one in the container-side `beat()` helper, s8
# hard-fails if any planned beat is missing or duplicated, and the renderer
# measures each beat's real start/end. Chat then cites `[beat:b3]` against that
# measured manifest. Break the chain anywhere and grounding degrades into a
# chatbot sitting next to a video, which is the thing this design exists to
# prevent -- hence the validation gate rather than a prompt instruction.

PRIMITIVES = ("numberline", "areamodel", "algebra_steps", "graph", "balance", "custom")


class IntentAnalysis(BaseModel):
    """s2 -- what this student actually needs to understand, given their error."""

    model_config = ConfigDict(extra="forbid")

    learner_goal: str
    assumed_knowledge: list[str] = Field(default_factory=list)
    knowledge_gap: str
    tone: Literal["encouraging", "neutral", "direct"] = "encouraging"


class PrereqNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    concept: str
    why_needed: str


class PrereqGraph(BaseModel):
    """s3 -- the concepts the explanation may lean on, ordered by dependency."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[PrereqNode] = Field(default_factory=list)
    # "a depends on b" as [a_id, b_id]. Kept as plain pairs rather than nested
    # structures so a model cannot express a shape the renderer cannot walk.
    edges: list[list[str]] = Field(default_factory=list)
    entry_point: str


class CurriculumStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int
    concept: str
    objective: str


class Curriculum(BaseModel):
    """s4 -- the teaching sequence, shortest path from what they know to the fix."""

    model_config = ConfigDict(extra="forbid")

    steps: list[CurriculumStep] = Field(default_factory=list)
    target_misconception: str


class MathContent(BaseModel):
    """s5 -- the concrete math the animation shows.

    `worked_example` and `counter_example` are the two halves of the correction:
    what the correct rule does, and what the student's buggy rule produces on the
    same input. Showing only the correct method leaves the student's own rule
    unchallenged, which is the failure mode this product exists to fix.
    """

    model_config = ConfigDict(extra="forbid")

    worked_example: list[str] = Field(default_factory=list)
    counter_example: list[str] = Field(default_factory=list)
    key_identity: str
    concrete_numbers: list[str] = Field(default_factory=list)


class Beat(BaseModel):
    """s6 -- one addressable moment of the animation, and the unit of grounding."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^b[0-9]+$")
    title: str
    teaching_purpose: str
    on_screen: str
    targets_misconception: bool = False
    primitive: Literal["numberline", "areamodel", "algebra_steps", "graph", "balance", "custom"]


class Storyboard(BaseModel):
    """s6 -- the ordered beats. 4-8, at least one targeting the misconception."""

    model_config = ConfigDict(extra="forbid")

    beats: list[Beat] = Field(min_length=1)
    total_estimated_seconds: int = Field(ge=1, le=600)


class SceneCode(BaseModel):
    """s7 -- LLM-authored Manim source. Untrusted until s8 and the sandbox clear it."""

    model_config = ConfigDict(extra="forbid")

    scene_class_name: str
    code: str
    beats_covered: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["syntax", "import", "name", "beat", "structure"]
    detail: str
    line: int | None = None


class ValidationReport(BaseModel):
    """s8 -- deterministic gate. No LLM: a model cannot be asked to police itself."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    def failure_text(self) -> str:
        """One string suitable for feeding back into the repair loop."""
        return "\n".join(
            f"[{issue.kind}]" + (f" line {issue.line}:" if issue.line else "") + f" {issue.detail}"
            for issue in self.issues
        )


class BeatTiming(BaseModel):
    """Measured at render time by the container-side `beat()` helper -- never estimated."""

    model_config = ConfigDict(extra="forbid")

    id: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)


class RenderResult(BaseModel):
    """Outcome of one `docker run`. `mode` records whether an LLM authored the scene."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    mode: Literal["generated", "storyboard_fallback"] = "generated"
    video_path: str | None = None
    timings: list[BeatTiming] = Field(default_factory=list)
    duration_s: float = 0.0
    error_text: str | None = None
