"""s5 -- Mathematics Selection.

Produces both halves of the correction: the correct derivation and the same
input carried through the student's buggy rule. The second half is what makes
the animation argue with the student rather than talk past them.
"""

from server.charter.contracts import Curriculum, Diagnosis, LlmCallMeta, MathContent
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient


async def select_math(
    client: DeepSeekClient,
    *,
    curriculum: Curriculum,
    diagnosis: Diagnosis,
    problem: str,
    model: str,
) -> tuple[MathContent, LlmCallMeta]:
    return await call_stage(
        client,
        template="s5_math",
        schema=MathContent,
        model=model,
        problem=problem,
        curriculum="; ".join(f"{s.order}. {s.concept}: {s.objective}" for s in curriculum.steps),
        buggy_rule=diagnosis.buggy_rule,
        correct_solution="\n".join(diagnosis.correct_solution),
    )
