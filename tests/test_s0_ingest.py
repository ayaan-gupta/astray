import json

import httpx
import pytest

from server.charter.contracts import StudentSubmission
from server.charter.stages.s0_ingest import (
    LOW_CONFIDENCE_THRESHOLD,
    ingest_photo,
    ingest_typed,
    needs_review,
)
from server.llm.vision import GeminiVision, NullVision, VisionUnavailable


def test_ingest_typed_splits_steps_on_newlines():
    s = ingest_typed(problem="(x+3)^2=25", work="x^2+9=25\nx^2=16\nx=4", prose=None)
    assert s.steps == ["x^2+9=25", "x^2=16", "x=4"]
    assert s.source == "typed"
    assert s.student_corrected is True  # typed input is authored by the student


def test_ingest_typed_drops_blank_lines():
    s = ingest_typed(problem="p", work="a\n\n  \nb\n", prose="I squared each part")
    assert s.steps == ["a", "b"]
    assert s.prose == "I squared each part"


def test_ingest_typed_requires_nonempty_problem():
    with pytest.raises(ValueError):
        ingest_typed(problem="   ", work="a", prose=None)


def test_ingest_typed_allows_empty_work():
    """A student may submit a problem before showing any work -- that's not an error,
    it just means there are no steps to diagnose yet."""
    s = ingest_typed(problem="p", work="\n  \n", prose=None)
    assert s.steps == []


def test_ingest_typed_blank_prose_becomes_none():
    s = ingest_typed(problem="p", work="a", prose="   ")
    assert s.prose is None


async def test_ingest_photo_marks_source_and_confidence():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"problem":"(x+3)^2=25","steps":["x^2+9=25"],'
                                    '"confidence":0.6,"unreadable":["line 3"]}'
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
            },
        )

    provider = GeminiVision("AQ.test", transport=httpx.MockTransport(handler))
    submission, meta = await ingest_photo(provider, b"png", "image/png")
    assert submission.source == "photo"
    assert submission.transcription_confidence == pytest.approx(0.6)
    assert submission.student_corrected is False
    assert meta.model == "gemini-3.5-flash-lite"


async def test_ingest_photo_preserves_wrong_step_verbatim():
    """The one property that matters most: a wrong step must survive ingestion unchanged,
    not get "corrected" into the right answer."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"problem":"(x+3)^2=25","steps":["x^2+9=25",'
                                    '"x^2=16","x=4"],"confidence":0.95,"unreadable":[]}'
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
            },
        )

    provider = GeminiVision("AQ.test", transport=httpx.MockTransport(handler))
    submission, _ = await ingest_photo(provider, b"png", "image/png")
    # (x+3)^2 = x^2 + 6x + 9, not x^2 + 9 -- the wrong step must NOT be silently fixed.
    assert submission.steps[0] == "x^2+9=25"


async def test_ingest_photo_that_transcribes_to_nothing_needs_review():
    """A blank or unreadable photo can come back with an empty problem and no steps.
    That must be flagged for student confirmation even if the model happens to report
    high confidence, since diagnosing nothing is meaningless."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"problem":"","steps":[],"confidence":0.9,'
                                    '"unreadable":["entire image"]}'
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            },
        )

    provider = GeminiVision("AQ.test", transport=httpx.MockTransport(handler))
    submission, _ = await ingest_photo(provider, b"png", "image/png")
    assert needs_review(submission) is True


async def test_ingest_photo_carries_unreadable_and_forces_review():
    """Reviewer-found gap: a model that reads most of the work but guesses at one blurry
    line can still self-report high overall confidence. `unreadable` must be carried onto
    the submission and must force `needs_review` regardless of that confidence score --
    otherwise a wrong guess on the blurry line reaches diagnosis unconfirmed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "problem": "(x+3)^2=25",
                                            "steps": ["x^2+9=25"],
                                            "confidence": 0.9,
                                            "unreadable": ["middle step blurry"],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
            },
        )

    provider = GeminiVision("AQ.test", transport=httpx.MockTransport(handler))
    submission, _ = await ingest_photo(provider, b"png", "image/png")
    assert submission.unreadable == ["middle step blurry"]
    assert submission.transcription_confidence == pytest.approx(0.9)
    assert needs_review(submission) is True


def test_needs_review_true_for_unreadable_regions_despite_high_confidence():
    flagged = StudentSubmission(
        problem="p",
        steps=["a"],
        source="photo",
        transcription_confidence=0.99,
        unreadable=["line 2"],
    )
    assert needs_review(flagged) is True


def test_needs_review_true_below_threshold():
    low = StudentSubmission(problem="p", steps=["a"], source="photo", transcription_confidence=0.5)
    assert needs_review(low) is True
    assert LOW_CONFIDENCE_THRESHOLD > 0.5


def test_needs_review_false_at_exact_threshold():
    """Pins the comparison as strictly less-than: a confidence exactly equal to the
    threshold must NOT be treated as low."""
    at_threshold = StudentSubmission(
        problem="p",
        steps=["a"],
        source="photo",
        transcription_confidence=LOW_CONFIDENCE_THRESHOLD,
    )
    assert needs_review(at_threshold) is False


def test_needs_review_false_for_corrected_photo():
    corrected = StudentSubmission(
        problem="p",
        steps=["a"],
        source="photo",
        transcription_confidence=0.4,
        student_corrected=True,
    )
    assert needs_review(corrected) is False


def test_needs_review_false_for_typed():
    assert needs_review(ingest_typed(problem="p", work="a", prose=None)) is False


def test_needs_review_true_for_missing_confidence():
    """Defense in depth: a photo submission with no confidence recorded at all must be
    treated as needing review, not silently trusted."""
    no_confidence = StudentSubmission(problem="p", steps=["a"], source="photo")
    assert needs_review(no_confidence) is True


async def test_ingest_photo_without_provider_raises():
    with pytest.raises(VisionUnavailable):
        await ingest_photo(NullVision(), b"png", "image/png")


# Regression guard for the "preserve the student's errors exactly" property: unusual step
# shapes that a future "helpful" refactor of the stripping/splitting logic could plausibly
# mangle. Reviewer verified these by hand for round 1; pinned here so a regression fails CI
# instead of requiring another manual pass.
UNUSUAL_STEP_SHAPES = [
    pytest.param("x−9=16", id="unicode-minus"),  # U+2212, not an ASCII hyphen
    pytest.param("\\frac{x}{2}=4", id="latex-fragment"),
    pytest.param("x+3=", id="trailing-operator"),
    pytest.param("=", id="equals-only"),
    pytest.param("x    =    4", id="internal-alignment-spaces"),
]


@pytest.mark.parametrize("step", UNUSUAL_STEP_SHAPES)
def test_ingest_typed_preserves_unusual_step_content(step):
    """`work` is split into one step per non-blank line; only the line's *surrounding*
    whitespace is stripped (that's sanctioned formatting normalization -- see
    test_ingest_typed_drops_blank_lines). Everything inside the line, however unusual,
    must come out identical."""
    s = ingest_typed(problem="p", work=step, prose=None)
    assert s.steps == [step]


@pytest.mark.parametrize(
    "step",
    [
        *UNUSUAL_STEP_SHAPES,
        pytest.param("   x=4", id="leading-alignment-whitespace"),
        pytest.param("x=4   ", id="trailing-alignment-whitespace"),
    ],
)
async def test_ingest_photo_preserves_unusual_step_content(step):
    """Unlike the typed path, ingest_photo applies zero transformation to a transcribed
    step -- not even whitespace stripping -- since the vision model's own formatting
    (including leading whitespace used as handwritten alignment) is trusted verbatim."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "problem": "p",
                                            "steps": [step],
                                            "confidence": 0.9,
                                            "unreadable": [],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            },
        )

    provider = GeminiVision("AQ.test", transport=httpx.MockTransport(handler))
    submission, _ = await ingest_photo(provider, b"png", "image/png")
    assert submission.steps == [step]
