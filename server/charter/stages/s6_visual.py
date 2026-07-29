"""s6 -- Visual Planning. Emits the beats that everything downstream grounds on.

`plan_beats` post-validates what the prompt asks for, because the beat contract
is enforced, not requested: ids must be b1..bN in order with no gaps, and at
least one beat must target the misconception. A storyboard that violates either
is rejected here so the failure names the real problem, rather than surfacing
later as a confusing s8 coverage error against a malformed plan.
"""

from server.charter.contracts import Diagnosis, LlmCallMeta, MathContent, Storyboard
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient, LlmError

MIN_BEATS = 3
MAX_BEATS = 8


class StoryboardInvalid(LlmError):
    """The storyboard violated the beat contract the whole pipeline depends on."""


def check_storyboard(storyboard: Storyboard) -> None:
    """Raise StoryboardInvalid unless the beat contract holds."""
    beats = storyboard.beats
    if not MIN_BEATS <= len(beats) <= MAX_BEATS:
        raise StoryboardInvalid(
            f"storyboard has {len(beats)} beats; expected {MIN_BEATS}-{MAX_BEATS}"
        )
    expected = [f"b{i}" for i in range(1, len(beats) + 1)]
    actual = [b.id for b in beats]
    if actual != expected:
        raise StoryboardInvalid(f"beat ids must be {expected}, got {actual}")
    if not any(b.targets_misconception for b in beats):
        # Without this beat the animation explains the topic but never contradicts
        # the student's rule, which is the one thing it exists to do.
        raise StoryboardInvalid("no beat has targets_misconception=true")


async def plan_beats(
    client: DeepSeekClient, *, math: MathContent, diagnosis: Diagnosis, model: str
) -> tuple[Storyboard, LlmCallMeta]:
    storyboard, meta = await call_stage(
        client,
        template="s6_visual",
        schema=Storyboard,
        model=model,
        buggy_rule=diagnosis.buggy_rule,
        misconception=diagnosis.misconception_statement,
        worked_example="\n".join(math.worked_example),
        counter_example="\n".join(math.counter_example),
        key_identity=math.key_identity,
        concrete_numbers="; ".join(math.concrete_numbers),
    )
    check_storyboard(storyboard)
    return storyboard, meta
