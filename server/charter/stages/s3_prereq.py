"""s3 -- Prerequisite Mapping. Small, load-bearing graph, not a course outline."""

from server.charter.contracts import Diagnosis, IntentAnalysis, LlmCallMeta, PrereqGraph
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient


async def map_prerequisites(
    client: DeepSeekClient, *, intent: IntentAnalysis, diagnosis: Diagnosis, model: str
) -> tuple[PrereqGraph, LlmCallMeta]:
    return await call_stage(
        client,
        template="s3_prereq",
        schema=PrereqGraph,
        model=model,
        learner_goal=intent.learner_goal,
        knowledge_gap=intent.knowledge_gap,
        assumed_knowledge="; ".join(intent.assumed_knowledge),
        buggy_rule=diagnosis.buggy_rule,
        topic=diagnosis.topic,
    )
