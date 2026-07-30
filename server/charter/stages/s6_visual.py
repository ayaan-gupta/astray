"""s6 -- Visual Planning. Emits the beats that everything downstream grounds on.

`plan_beats` post-validates what the prompt asks for, because the beat contract
is enforced, not requested: ids must be b1..bN in order with no gaps, and at
least one beat must target the misconception. A storyboard that violates either
is rejected here so the failure names the real problem, rather than surfacing
later as a confusing s8 coverage error against a malformed plan.
"""

import logging

from server.charter.contracts import Diagnosis, LlmCallMeta, MathContent, Storyboard
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient, LlmError

logger = logging.getLogger(__name__)

MIN_BEATS = 3
MAX_BEATS = 8

# Seconds per beat the runtime estimate is held to. The estimate is the only pacing
# signal s7 gets, and s7 satisfies it with `self.wait()` -- so an inflated estimate
# does not buy more explanation, it buys dead air. Measured across three runs, s6
# asked for 130s over 5 beats, 180s over 4, and 240s over 6, all well outside the
# "typically 45-120" its own prompt states; the 180s one rendered as four beats
# holding a static frame for 30 to 40 seconds each.
#
# 6 to 14 seconds is a beat that has time to be read and seeked to without becoming
# a still image. Clamped rather than rejected, because unlike the beat ids this is
# advisory metadata: a bad estimate is a pacing problem, not a grounding one, and
# failing a whole session's animation over it would be wildly out of proportion.
MIN_SECONDS_PER_BEAT = 6
MAX_SECONDS_PER_BEAT = 14


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


def clamp_runtime(storyboard: Storyboard) -> Storyboard:
    """Hold the runtime estimate to something a diagram can actually fill.

    Returns the storyboard unchanged when the estimate is already sensible, so the
    common case is not rewritten.
    """
    count = len(storyboard.beats)
    low, high = MIN_SECONDS_PER_BEAT * count, MAX_SECONDS_PER_BEAT * count
    asked = storyboard.total_estimated_seconds
    clamped = max(low, min(high, asked))
    if clamped == asked:
        return storyboard
    logger.info("clamped storyboard runtime from %ss to %ss for %d beats", asked, clamped, count)
    return storyboard.model_copy(update={"total_estimated_seconds": clamped})


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
    return clamp_runtime(storyboard), meta
