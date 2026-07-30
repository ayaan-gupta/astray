"""s7 -- Scene Composition. The only stage that emits executable code.

Uses the reasoning model rather than the fast one: this is the hardest generation
in the chain and a failure here costs a container round-trip plus a repair call,
which is far more expensive than the model-tier difference.
"""

from server.charter.contracts import LlmCallMeta, MathContent, SceneCode, Storyboard
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient

SCENE_CLASS_NAME = "AstrayScene"


async def compose_scene(
    client: DeepSeekClient, *, storyboard: Storyboard, math: MathContent, model: str
) -> tuple[SceneCode, LlmCallMeta]:
    beats = "\n".join(
        f"{b.id} [{b.primitive}] {b.title} -- purpose: {b.teaching_purpose} "
        f"-- on screen: {b.on_screen}"
        + (" -- THIS BEAT TARGETS THE MISCONCEPTION" if b.targets_misconception else "")
        for b in storyboard.beats
    )
    # Per beat, not just the total. Handed only a total, the model divides by the
    # beat count itself and then spends the quotient entirely on `self.wait()`: a
    # 180s estimate over four beats came back as `wait(30)`, `wait(35)`, `wait(40)`,
    # `wait(35)` and a 155s video of static frames. Doing the division here lets the
    # prompt talk about one beat's pacing, which is a length it can reason about,
    # rather than a budget it feels obliged to exhaust.
    per_beat = max(1, round(storyboard.total_estimated_seconds / len(storyboard.beats)))
    return await call_stage(
        client,
        template="s7_scene",
        schema=SceneCode,
        model=model,
        scene_class_name=SCENE_CLASS_NAME,
        beats=beats,
        seconds_per_beat=str(per_beat),
        total_estimated_seconds=str(storyboard.total_estimated_seconds),
        worked_example="\n".join(math.worked_example),
        counter_example="\n".join(math.counter_example),
        key_identity=math.key_identity,
    )
