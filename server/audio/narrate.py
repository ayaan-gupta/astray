"""Writing the narration script, after the render rather than before it.

The ordering is the whole design. A script written from the storyboard would be
guessing at how long each beat lasts, and narration that guesses is narration
that talks over the next visual. By the time this runs the container has reported
its own clock, so every line is written against a real measured duration and gets
a word budget it has to fit.

The script is also written to be *spoken*, which is a different job from writing
prose. Symbols get spelled out, sentences stay short, and punctuation is left
where the voice needs to breathe. `speech.py` is the net for anything the model
still writes in notation.
"""

import json
import logging
import sqlite3

from pydantic import BaseModel, ConfigDict, Field

from server.audio import speech
from server.charter.contracts import LlmCallMeta
from server.charter.stages.s1_diagnose import _neutralize_markers
from server.llm.deepseek import DeepSeekClient
from server.store import repo

logger = logging.getLogger(__name__)


class NarrationLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_id: str
    line: str


class NarrationScript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lines: list[NarrationLine] = Field(default_factory=list)


PROMPT = """You are the voice of a maths tutor, narrating a short animation made \
for one student's specific mistake. Your words are read aloud over it.

Write ONE continuous explanation, then divide it across the beats listed below.
This is the most important instruction here. Read end to end, your lines must
sound like a single person explaining one idea from start to finish, each sentence
following from the last. What you must not produce is a list of disconnected
fragments, one per beat, each restating a label. "They got y squared plus nine."
is a caption, not an explanation. Explain WHY the mistake happens and what the
student should do instead.

Here is the standard to match, for a student who wrote (y+3)^2 = y^2 + 9:

  "Squaring a bracket isn't the same as squaring each piece separately."
  "y plus three, all squared, just means y plus three times itself."
  "Multiply that out and every term meets every other term."
  "So you get y squared, then three y twice over, then nine."
  "Those two middle terms are what you dropped. Together they make six y."
  "Try y equals one. The real answer is sixteen, and yours gives ten."

Notice what that does. It gives a reason, walks through it, names the dropped
term, and checks it with a number. Each line is a complete sentence, and the
lines join into one argument.

Each beat is synthesised separately, so every line must be complete sentences on
its own. Never let a sentence run from one beat into the next.

Say maths the way a teacher says it out loud. Never write a symbol.
  (y+3)^2      -> "y plus three, all squared"
  y^2 + 6y + 9 -> "y squared plus six y plus nine"
  2ab          -> "two a b"
  y=1          -> "y equals one"
The phrase "all squared" is not optional. Say "a plus b, all squared". Never say \
"a plus b squared", because a listener hears that as a plus b-squared, which is \
the exact mistake this animation exists to correct. Use "all squared" every time \
a bracket is raised to a power, including when stating the correct rule.

Use contractions and short sentences. Talk, do not recite. Punctuation is your
only control over the voice, so put a comma where you want a breath and a full
stop where you want a stop. Never use an em dash, an en dash, brackets, or a colon.

Do not describe the animation ("here we see", "on the left"). Say the maths.

Never tell the student their wrong answer was right, and never address them as
though they are watching someone else's mistake.

Each beat gives a word target and a hard maximum, both derived from how long that
beat is actually on screen. Aim for the target: coming in far under it leaves the
animation playing in silence. Never exceed the maximum, or you will still be
talking when the next beat starts.

The problem they were given: {problem}
What they wrote: {work}
The correct working: {solution}
The false rule they used: {rule}
Why it is wrong, in their terms: {statement}
The specific evidence: {evidence}

Beats, in order:
{beats}

Return one line per beat id, in this order: {ids}"""


def _beat_brief(
    row: sqlite3.Row, words_per_second: float, *, final: bool = False
) -> tuple[str, int]:
    duration = float(row["end_s"]) - float(row["start_s"])
    budget = speech.budget_words(duration, words_per_second, final=final)
    target = speech.target_words(budget)
    marker = "  <- THIS BEAT SHOWS THE MISTAKE ITSELF" if row["targets_misconception"] else ""
    brief = (
        f'- {row["beat_id"]} "{_neutralize_markers(row["title"])}" '
        f"({duration:.1f}s on screen; aim for {target} words, hard maximum {budget}): "
        f"{_neutralize_markers(row['purpose'])}{marker}"
    )
    return brief, budget


def build_prompt(
    diagnosis_row: sqlite3.Row,
    beat_rows: list[sqlite3.Row],
    words_per_second: float,
    *,
    session_row: sqlite3.Row | None = None,
) -> str:
    """Assemble the narration prompt.

    Everything the diagnosis already knows goes in. The first version of this
    passed only the rule name and a one-line statement, which is why it produced
    captions rather than an explanation: it had nothing concrete to explain with.
    The problem, the student's own working, the correct steps and the evidence are
    all already persisted, and all of them are student-influenced text, so all of
    them are neutralised before going anywhere near a prompt.
    """
    briefs, ids = [], []
    for index, row in enumerate(beat_rows):
        brief, _ = _beat_brief(row, words_per_second, final=index == len(beat_rows) - 1)
        briefs.append(brief)
        ids.append(row["beat_id"])

    payload = json.loads(diagnosis_row["payload_json"])
    solution = _neutralize_markers(" ".join(payload.get("correct_solution", [])))
    evidence = _neutralize_markers(" ".join(payload.get("evidence", []))) or "(none recorded)"

    problem, work = "(not recorded)", "(not recorded)"
    if session_row is not None:
        problem = _neutralize_markers(session_row["problem"] or problem)
        try:
            steps = json.loads(session_row["student_work_json"] or "[]")
        except (TypeError, ValueError):
            steps = []
        if steps:
            work = _neutralize_markers(" ".join(str(s) for s in steps))

    return PROMPT.format(
        problem=problem,
        work=work,
        solution=solution or "(not recorded)",
        rule=_neutralize_markers(diagnosis_row["buggy_rule"]),
        statement=_neutralize_markers(diagnosis_row["statement"]),
        evidence=evidence,
        beats="\n".join(briefs),
        ids=", ".join(ids),
    )


def trim_to_budget(line: str, budget: int) -> str:
    """Cut an over-long line at a sentence boundary, never mid-clause.

    The model is asked for a budget and mostly respects it. When it does not,
    dropping whole sentences from the end keeps the line speakable; truncating
    at the budget'th word would leave the voice reading a fragment that stops
    dead, which sounds far worse than a shorter line.
    """
    if speech.word_count(line) <= budget:
        return line

    kept: list[str] = []
    for sentence in _sentences(line):
        candidate = " ".join([*kept, sentence])
        if kept and speech.word_count(candidate) > budget:
            break
        kept.append(sentence)
    return " ".join(kept) if kept else line


def _sentences(text: str) -> list[str]:
    out, current = [], ""
    for char in text:
        current += char
        if char in ".!?":
            out.append(current.strip())
            current = ""
    if current.strip():
        out.append(current.strip())
    return out


async def write_script(
    conn: sqlite3.Connection,
    client: DeepSeekClient,
    *,
    session_id: str,
    model: str,
    words_per_second: float,
) -> tuple[list[tuple[str, str]], LlmCallMeta]:
    """Write one speakable line per timed beat.

    Returns `[(beat_id, spoken_text), ...]` in beat order, already passed through
    `speech.speakable` and trimmed to budget, plus the call's cost provenance.

    Beats the renderer never timed are skipped rather than narrated at a guessed
    offset, for the same reason an untimed beat is not citable in chat.
    """
    diagnosis_row = repo.get_diagnosis(conn, session_id)
    if diagnosis_row is None:
        raise ValueError("session has no diagnosis to narrate")

    timed = [r for r in repo.list_beats(conn, session_id) if r["start_s"] is not None]
    if not timed:
        raise ValueError("session has no measured beat timings to narrate against")

    prompt = build_prompt(
        diagnosis_row, timed, words_per_second, session_row=repo.get_session(conn, session_id)
    )
    script, meta = await client.complete_json(
        messages=[{"role": "user", "content": prompt}],
        schema=NarrationScript,
        model=model,
    )

    by_id = {line.beat_id: line.line for line in script.lines}
    budgets = {
        row["beat_id"]: _beat_brief(row, words_per_second, final=index == len(timed) - 1)[1]
        for index, row in enumerate(timed)
    }

    out: list[tuple[str, str]] = []
    for row in timed:
        beat_id = row["beat_id"]
        raw = by_id.get(beat_id)
        if not raw or not raw.strip():
            logger.info("no narration line for beat %s; leaving it silent", beat_id)
            continue
        spoken = speech.speakable(trim_to_budget(raw, budgets[beat_id]))
        if spoken:
            out.append((beat_id, spoken))
    return out, meta
