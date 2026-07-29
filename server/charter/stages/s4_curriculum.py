"""s4 -- Curriculum Building. Shortest path that ends on the misconception."""

from server.charter.contracts import Curriculum, Diagnosis, LlmCallMeta, PrereqGraph
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient


async def build_curriculum(
    client: DeepSeekClient, *, prereqs: PrereqGraph, diagnosis: Diagnosis, model: str
) -> tuple[Curriculum, LlmCallMeta]:
    return await call_stage(
        client,
        template="s4_curriculum",
        schema=Curriculum,
        model=model,
        nodes="; ".join(f"{n.id}={n.concept} ({n.why_needed})" for n in prereqs.nodes),
        entry_point=prereqs.entry_point,
        buggy_rule=diagnosis.buggy_rule,
        correct_solution="\n".join(diagnosis.correct_solution),
    )
