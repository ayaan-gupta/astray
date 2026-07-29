"""s2 -- Learner Intent Analysis.

Turns "what did they get wrong" into "what does this person need". Every stage
after this one reads the intent rather than the raw diagnosis, which is what
keeps the animation specific to the student instead of a generic topic explainer.
"""

from server.charter.contracts import Diagnosis, IntentAnalysis, LlmCallMeta, StudentSubmission
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient


async def analyse_intent(
    client: DeepSeekClient, *, submission: StudentSubmission, diagnosis: Diagnosis, model: str
) -> tuple[IntentAnalysis, LlmCallMeta]:
    return await call_stage(
        client,
        template="s2_intent",
        schema=IntentAnalysis,
        model=model,
        problem=submission.problem,
        student_steps="\n".join(submission.steps),
        buggy_rule=diagnosis.buggy_rule,
        misconception=diagnosis.misconception_statement,
        topic=diagnosis.topic,
    )
