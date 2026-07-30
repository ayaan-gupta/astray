"""Deciding what a spoken sentence was for, and repairing how it was heard.

Two problems that look separate and are not. When a student says "hey Astray, I
was solving dy/dx equals 2x minus 5 over y squared and I couldn't figure it out",
the microphone delivers *"I was solving DUI over DX = 2x - 5 over y squared and I
couldn't really figure it out can you help me out here"* -- observed verbatim in
this product -- and the chat tutor answers it as a question about the bracket
expansion the student was previously looking at. Both failures are one failure:
nothing between the recogniser and the tutor knew that sentence was a new
problem, and nothing repaired `DUI over DX` into `dy/dx`.

So one call does both. The classification needs the utterance cleaned up to
classify it correctly, and the cleanup needs to know which fields it is filling,
so splitting them into two calls would double the latency to answer strictly
worse. The transcript is student-supplied text and is nonce-wrapped like every
other stage's input.

The bias is deliberate and asymmetric. `followup` is the fallback for anything
ambiguous and for every failure mode here, because a new problem answered in the
existing conversation is a mildly wrong answer the student can correct in one
sentence, while a followup routed to `new_problem` spends three minutes and a
render building an animation for a problem nobody asked about.
"""

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from server.charter.contracts import LlmCallMeta
from server.charter.stages._planning import call_stage
from server.llm.deepseek import DeepSeekClient, LlmError

logger = logging.getLogger(__name__)

# Long enough for a student to state a problem and what they tried; short enough
# that a recogniser stuck open on a room full of talking cannot turn into a large
# prompt. A 60-second Chrome session of continuous speech is around 900
# characters, so this is roughly two of them.
MAX_TRANSCRIPT_CHARS = 2_000

# Below this there is nothing to classify. "Hey Astray" on its own lands here,
# and it must not become a new problem with an empty statement.
MIN_TRANSCRIPT_CHARS = 8


class RoutedUtterance(BaseModel):
    """Where one spoken sentence goes, and its contents repaired into notation."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["new_problem", "followup"]
    problem: str = Field(default="", max_length=MAX_TRANSCRIPT_CHARS)
    work: str = Field(default="", max_length=MAX_TRANSCRIPT_CHARS)
    question: str = Field(default="", max_length=MAX_TRANSCRIPT_CHARS)
    topic: str = Field(default="", max_length=120)


def followup(question: str) -> RoutedUtterance:
    """The safe answer: send it to the conversation already in progress."""
    return RoutedUtterance(kind="followup", question=question.strip())


def coerce(routed: RoutedUtterance, transcript: str, *, has_session: bool) -> RoutedUtterance:
    """Hold the model's answer to what the caller can actually act on.

    Three ways a well-formed reply is still unusable, all of which have to resolve
    to something rather than raise, because this sits between a student's voice and
    a reply they are waiting for:

    A `new_problem` with no problem statement cannot create a session -- that is
    what the prompt asks for when speech was too garbled to reconstruct, so it
    falls back to the conversation.

    A `followup` with an empty question would post a blank message. The raw
    transcript is a worse question than a repaired one but a far better one than
    nothing, so it stands in.

    A `followup` with no session to follow up on happens on the submit page, where
    there is no conversation yet. There the only useful reading of anything spoken
    is a new problem, so the transcript becomes the problem statement verbatim --
    unrepaired, which is visible to the student in the field and editable, unlike a
    silent drop.
    """
    if routed.kind == "new_problem" and not routed.problem.strip():
        logger.info("routed as new_problem with no problem statement; treating as followup")
        return followup(routed.question or transcript)

    if routed.kind == "followup":
        if not has_session:
            return RoutedUtterance(
                kind="new_problem", problem=transcript.strip()[:MAX_TRANSCRIPT_CHARS]
            )
        if not routed.question.strip():
            return followup(transcript)

    return routed


async def route(
    client: DeepSeekClient,
    *,
    transcript: str,
    model: str,
    current_problem: str = "",
    current_misconception: str = "",
) -> tuple[RoutedUtterance, LlmCallMeta | None]:
    """Classify and repair one spoken utterance. Never raises.

    `current_problem` empty means there is no session to follow up on, so every
    utterance is a new problem. Returns `(routed, meta)`; `meta` is None when no
    model call was made, so the caller bills only what was actually spent.
    """
    text = transcript.strip()[:MAX_TRANSCRIPT_CHARS]
    has_session = bool(current_problem.strip())

    if len(text) < MIN_TRANSCRIPT_CHARS:
        # "Hey Astray" and nothing else. There is no question here and no problem
        # here; the caller shows the state and waits rather than asking the model
        # to route silence.
        return followup(text), None

    try:
        routed, meta = await call_stage(
            client,
            template="voice_route",
            schema=RoutedUtterance,
            model=model,
            transcript=text,
            current_problem=current_problem or "(none: the student has no problem open)",
            current_misconception=current_misconception or "(none diagnosed)",
        )
    except (LlmError, ValueError) as exc:
        # Routing is an optimisation over "send it to chat". Losing it must not
        # lose the student's sentence.
        logger.warning("voice routing failed, falling back to followup: %s", exc)
        return (
            followup(text) if has_session else RoutedUtterance(kind="new_problem", problem=text),
            None,
        )

    return coerce(routed, text, has_session=has_session), meta
