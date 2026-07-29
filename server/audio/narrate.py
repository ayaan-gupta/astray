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


PROMPT = """You are writing the spoken narration for a short maths animation. A \
student made a specific mistake, the animation explains that mistake, and your \
words are read aloud over it by a text-to-speech voice.

You are writing for the ear, not the page. Follow these rules exactly.

Say maths as a teacher says it out loud. Never write a symbol.
  (y+3)^2      -> "y plus three, all squared"
  y^2 + 6y + 9 -> "y squared plus six y plus nine"
  2ab          -> "two a b"
  y=1          -> "y equals one"
The phrase "all squared" is important. It is the difference between the correct \
expansion and the mistake this student made, so use it whenever a bracket is \
raised to a power.

One idea per sentence. Short sentences. Use contractions. Write the way a person \
talks, not the way a textbook reads.

Punctuation is your only control over the voice, so put a comma where you want a \
breath and a full stop where you want a stop. Never use an em dash, an en dash, \
brackets, or a colon.

Do not narrate the animation ("here we see", "on the left"). Say the maths.

Never tell the student their wrong answer was right.

Each beat below gives you a word budget. It is derived from how long that beat \
is actually on screen, so a line that runs over will be talking during the next \
beat. Stay at or under the budget. Shorter is always safe.

Their mistake: {rule}
Stated for them as: {statement}

Beats, in order:
{beats}

Return one line per beat id, in this order: {ids}"""


def _beat_brief(row: sqlite3.Row, words_per_second: float) -> tuple[str, int]:
    duration = float(row["end_s"]) - float(row["start_s"])
    budget = speech.budget_words(duration, words_per_second)
    marker = "  <- THIS BEAT SHOWS THE MISTAKE ITSELF" if row["targets_misconception"] else ""
    brief = (
        f'- {row["beat_id"]} "{_neutralize_markers(row["title"])}" '
        f"({duration:.1f}s on screen, at most {budget} words): "
        f"{_neutralize_markers(row['purpose'])}{marker}"
    )
    return brief, budget


def build_prompt(
    diagnosis_row: sqlite3.Row, beat_rows: list[sqlite3.Row], words_per_second: float
) -> str:
    briefs, ids = [], []
    for row in beat_rows:
        brief, _ = _beat_brief(row, words_per_second)
        briefs.append(brief)
        ids.append(row["beat_id"])
    return PROMPT.format(
        rule=_neutralize_markers(diagnosis_row["buggy_rule"]),
        statement=_neutralize_markers(diagnosis_row["statement"]),
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

    prompt = build_prompt(diagnosis_row, timed, words_per_second)
    script, meta = await client.complete_json(
        messages=[{"role": "user", "content": prompt}],
        schema=NarrationScript,
        model=model,
    )

    by_id = {line.beat_id: line.line for line in script.lines}
    budgets = {row["beat_id"]: _beat_brief(row, words_per_second)[1] for row in timed}

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
