"""Bounded render-error -> LLM repair loop.

Two attempts maximum, then the caller degrades to the deterministic storyboard
fallback. The bound is the point: an unbounded repair loop against a model that
keeps producing the same class of error burns the reasoning-model budget and the
student's patience simultaneously, and the fallback always produces a correct,
grounded video, so there is a good floor to degrade to.

The repair prompt is deliberately given the validator's own findings rather than
a paraphrase. `ValidationReport.failure_text()` names the exact rule broken and
the line, which is far more actionable than "your code was rejected".
"""

import logging

from server.charter.contracts import LlmCallMeta, MathContent, SceneCode, Storyboard
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient

logger = logging.getLogger(__name__)

MAX_REPAIRS = 2


async def repair_scene(
    client: DeepSeekClient,
    *,
    scene: SceneCode,
    storyboard: Storyboard,
    math: MathContent,
    failure: str,
    model: str,
) -> tuple[SceneCode, LlmCallMeta]:
    """Ask for a corrected scene file given the failure text.

    Reuses the s7 template so the contract the code must satisfy is stated once.
    The previous attempt and its failure are appended as stage input, which the
    shared helper wraps in nonce markers like any other untrusted-derived text --
    a Manim traceback can echo student-supplied strings back at us.
    """
    beats = "\n".join(
        f"{b.id} [{b.primitive}] {b.title} -- {b.teaching_purpose}"
        + (" -- TARGETS THE MISCONCEPTION" if b.targets_misconception else "")
        for b in storyboard.beats
    )
    return await call_stage(
        client,
        template="s7_scene",
        schema=SceneCode,
        model=model,
        scene_class_name=scene.scene_class_name,
        beats=beats,
        total_estimated_seconds=str(storyboard.total_estimated_seconds),
        worked_example="\n".join(math.worked_example),
        counter_example="\n".join(math.counter_example),
        key_identity=math.key_identity,
        previous_attempt=scene.code,
        why_it_failed=failure,
        instruction=(
            "The previous attempt above FAILED for the stated reason. Return a corrected "
            "complete file that fixes it without introducing new violations."
        ),
    )
