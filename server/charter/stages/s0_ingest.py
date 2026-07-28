"""s0 -- normalize typed or photographed input into a StudentSubmission.

The product-critical property for this stage: preserve the student's errors exactly.
Everything downstream diagnoses the mistake, so any normalization that "tidies up" a
wrong step destroys the thing being analysed. Only whitespace/formatting normalization
happens here (stripping surrounding blank space, dropping blank lines); mathematical
content -- digits, operators, variable names -- is never altered.
"""

from server.charter.contracts import LlmCallMeta, StudentSubmission
from server.llm.vision import VisionProvider

LOW_CONFIDENCE_THRESHOLD = 0.75


def ingest_typed(*, problem: str, work: str, prose: str | None) -> StudentSubmission:
    """Build a StudentSubmission from typed input.

    ``work`` is split into steps on newlines; blank/whitespace-only lines are dropped
    and each retained line has its surrounding whitespace stripped. This is pure
    formatting normalization -- it never touches the mathematical content of a line,
    so a wrong step (e.g. "x^2+9=25") passes through unchanged.

    Typed input is authored directly by the student (no OCR risk), so it is always
    marked ``student_corrected=True`` -- there is nothing to confirm.
    """
    if not problem.strip():
        raise ValueError("problem must not be empty")
    steps = [line.strip() for line in work.splitlines() if line.strip()]
    return StudentSubmission(
        problem=problem.strip(),
        steps=steps,
        prose=(prose.strip() if prose and prose.strip() else None),
        source="typed",
        student_corrected=True,
    )


async def ingest_photo(
    provider: VisionProvider, image_bytes: bytes, mime_type: str
) -> tuple[StudentSubmission, LlmCallMeta]:
    """Transcribe a photo via ``provider`` and wrap the result as a StudentSubmission.

    The result MUST be shown to the student for confirmation before diagnosis -- a bad
    transcription would diagnose an error they never made. ``student_corrected`` is
    therefore always False here; only the student-confirmation flow (outside this
    stage) is allowed to set it True. Nothing in ``transcription.problem``/``.steps``
    is reformatted or stripped -- the vision model's own output is trusted verbatim,
    unlike the typed path where we perform the line-splitting ourselves.

    If ``provider`` has no working Gemini key (``NullVision``), this raises
    ``VisionUnavailable``. That is deliberately NOT caught here: mapping it to an HTTP
    503 is a route's job (Task 12/13), not this stage's.

    NOTE: ``transcription.unreadable`` (the list of regions the model couldn't read) is
    not carried onto ``StudentSubmission`` -- that contract (Task 2) has no field for
    it, and adding one is out of scope for this stage. ``needs_review`` below therefore
    relies on ``confidence`` alone as the review signal; ``TRANSCRIBE_PROMPT`` instructs
    the model to set confidence to its *overall* accuracy, which should already be
    dragged down by any unreadable region. A completely blank/unreadable transcription
    is still caught explicitly below, independent of what confidence value comes back.
    """
    transcription, meta = await provider.transcribe(image_bytes, mime_type)
    submission = StudentSubmission(
        problem=transcription.problem,
        steps=transcription.steps,
        prose=None,
        source="photo",
        transcription_confidence=transcription.confidence,
        student_corrected=False,
    )
    return submission, meta


def needs_review(submission: StudentSubmission) -> bool:
    """True when the UI must block on student confirmation of the transcription.

    Typed input and anything the student has already confirmed/corrected never need
    review. For an unconfirmed photo, review is required when either:
      - the transcription is entirely blank (no problem, no non-blank step) -- a
        malformed or unreadable photo can still come back with a high confidence
        value, so this is checked independent of the confidence score; or
      - the model's reported confidence is missing or below ``LOW_CONFIDENCE_THRESHOLD``.
    """
    if submission.student_corrected:
        return False
    if submission.source != "photo":
        return False
    transcribed_nothing = not submission.problem.strip() and not any(
        step.strip() for step in submission.steps
    )
    if transcribed_nothing:
        return True
    confidence = submission.transcription_confidence
    return confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD
