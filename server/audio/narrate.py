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

**Never state the student's rule without contradicting it in the same breath.**
This is the failure that matters most, and it is subtle, because the sentence
looks like it is doing its job. "You thought a plus b, all squared, equals a
squared plus b squared." is a sentence that agrees with them. Said aloud, with
nothing after it, the student hears their own rule read back as fact and the
animation has taught them the mistake. Every time you state what they did, the
correction is attached to it:

  NOT  "You thought a plus b, all squared, equals a squared plus b squared."
  BUT  "You thought a plus b, all squared, equals a squared plus b squared. It
        doesn't, and here's the piece that goes missing."

The same holds for a number check. "Your rule gives ten, but the true answer is
sixteen" states two numbers and explains neither. Say what the gap *is*: "Your
rule gives ten, the real answer is sixteen, and those six are the term you
dropped."

**Join each line to the one before it.** The animation cuts from section to
section, and your voice is the only thing carrying the student across the cut. So
open lines after the first by picking up where the last one stopped: "So", "But",
"That's why", "Now watch what happens when", "Those two pieces are". A line that
starts a fresh topic makes the video feel like five separate videos.

Here is the standard to match, for a student who wrote (y+3)^2 = y^2 + 9:

  "You thought squaring a bracket means squaring each piece. It doesn't, and
   that's the whole mistake."
  "y plus three, all squared, just means y plus three times itself."
  "So multiply it out, and every term meets every other term."
  "That gives you y squared, then three y twice over, then nine."
  "Those two middle terms are the ones you dropped, and together they make six y."
  "Check it with y equals one. The real answer is sixteen, yours gives ten, and
   the six missing is exactly that middle term."

Notice what that does. It names the mistake AND denies it in the first breath,
gives the reason, walks through it, names the dropped term, then checks it with a
number and ties the number back to the term. Every line after the first begins by
reaching back to the one before.

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

Name what is in the picture, but never narrate the medium. "The two small corner
squares" and "the square of side y plus three" are the explanation, because the
beat has a square in it and those cells are the argument. "Here we see", "on the
left of the screen", "this animation shows" are filler that spends words without
teaching anything.

Say "you" and "your answer". Never "the student", and never write as though the
student were watching someone else's mistake. Never tell them their wrong answer
was right.

Each beat gives a word target and a hard maximum. Aim for the target. Coming in
far under it is the more common failure and the worse one: it leaves the animation
playing in silence and it is what turns an explanation into a caption, so use the
room you are given. If a line needs a few words past the target to finish its
thought properly, the picture waits for you rather than cutting away. What it
cannot absorb is a line at twice its maximum, which freezes the frame while you
talk over a still image.

The problem they were given: {problem}
What they wrote: {work}
The correct working: {solution}
The false rule they used: {rule}
Why it is wrong, in their terms: {statement}
The specific evidence: {evidence}

Beats, in order:
{beats}

Return one line per beat id, in this order: {ids}"""


# The floor under every beat's word budget, and the reason `pad.py` exists.
#
# Before padding, a budget was a hard constraint: the render was a fixed length,
# so a 6-second beat got sixteen words and there was nothing to be done about it.
# Sixteen words is a caption. Stating the student's rule, contradicting it, and
# giving the reason does not fit, so the script wrote the label instead and the
# animation explained nothing -- which is exactly the complaint this floor
# answers.
#
# Now a line longer than its beat holds the picture still until the sentence
# finishes, so the budget is a preference rather than a wall, and the floor is set
# where a complete explanatory sentence fits: about ten seconds of speech at this
# product's measured rate. `pad.MAX_HOLD_S` is what stops it running away.
MIN_BEAT_WORDS = 24


# The narration prompt for a session with no working shown.
#
# A separate prompt rather than a flag inside the main one, because almost every
# instruction in that one is about the student's own mistake -- state it, deny it,
# name the term they dropped -- and none of it applies to someone who never said
# what they tried. This is the one place a wrong framing does direct harm: telling
# a student "you thought a plus b, all squared, equals a squared plus b squared"
# when they never said any such thing invents a mistake and attributes it to them.
EXPLAIN_PROMPT = """You are the voice of a maths tutor, narrating a short animation \
made for one student who is stuck on a problem. Your words are read aloud over it.

This student has NOT shown any working. They asked how to do it. So there is no
mistake to correct here, and this is the one rule you must not break: never tell
them what they thought, never say they got something wrong, and never imply they
made an error. There is no error. They are stuck, which is a different thing, and
being told you made a mistake you did not make is worse than not being helped.

Write ONE continuous explanation, then divide it across the beats listed below.
Read end to end, your lines must sound like a single person working through one
problem from start to finish, each sentence following from the last. What you must
not produce is a list of disconnected fragments, one per beat, each restating a
label. "The integral of y squared." is a caption, not an explanation.

Teach the method. Say what kind of problem it is and how you recognise that, then
take the steps in order, and for each one say WHY that step is the one to take.
The reason is the part worth hearing: a student who only sees what was done has to
guess when to do it again.

Here is the standard to match, for a student stuck on dy/dx = (2x-5)/y squared:

  "This one separates, because every y is on one side and every x on the other."
  "So multiply both sides by y squared, and by d x."
  "That gives y squared, d y, equals two x minus five, d x."
  "Now integrate each side on its own, which is the whole point of separating."
  "You get y cubed over three, equals x squared minus five x, plus a constant."
  "Multiply through by three, and y cubed equals three x squared minus fifteen x
   plus C."

Notice what that does. It names the method and the reason it applies, then each
line does one step and joins on to the one before it. Nowhere does it suggest the
student did anything.

**Join each line to the one before it.** The animation cuts from section to
section and your voice is the only thing carrying the student across the cut. Open
lines after the first with "So", "Now", "That gives", "Which means". A line that
starts a fresh topic makes the video feel like five separate videos.

**Say only what the problem gives you.** If it states no initial or boundary
condition, there is none: give the general solution and name the constant. Never
announce a value that was not in the problem. A beat may be titled as though a
condition were given; if the problem does not give one, narrate what is actually
true instead of reading the title. Inventing the question is the same failure as
inventing the mistake.

Each beat is synthesised separately, so every line must be complete sentences on
its own. Never let a sentence run from one beat into the next.

Say maths the way a teacher says it out loud. Never write a symbol.
  (y+3)^2      -> "y plus three, all squared"
  y^2 + 6y + 9 -> "y squared plus six y plus nine"
  dy/dx        -> "d y by d x"
  ∫ y^2 dy      -> "the integral of y squared, d y"
The phrase "all squared" is not optional. Say "a plus b, all squared". Never say \
"a plus b squared", because a listener hears that as a plus b-squared.

Use contractions and short sentences. Talk, do not recite. Punctuation is your
only control over the voice, so put a comma where you want a breath and a full
stop where you want a stop. Never use an em dash, an en dash, brackets, or a colon.

Name what is in the picture, but never narrate the medium. "Here we see", "on the
left of the screen" and "this animation shows" are filler that spends words
without teaching anything.

Say "you" and "your". Talk to them, not about them.

Each beat gives a word target and a hard maximum. Aim for the target. Coming in
far under it is the more common failure and the worse one: it leaves the animation
playing in silence and it is what turns an explanation into a caption, so use the
room you are given. If a line needs a few words past the target to finish its
thought properly, the picture waits for you rather than cutting away.

The problem they are stuck on: {problem}
What they wrote: {work}
The correct working: {solution}
The method, stated for them: {statement}
Anything else recorded: {evidence}

Beats, in order:
{beats}

Return one line per beat id, in this order: {ids}"""


def _beat_brief(
    row: sqlite3.Row,
    words_per_second: float,
    *,
    duration: float,
    final: bool = False,
    explaining: bool = False,
) -> tuple[str, int]:
    budget = max(speech.budget_words(duration, words_per_second, final=final), MIN_BEAT_WORDS)
    target = speech.target_words(budget)
    # Suppressed when explaining. A storyboard for a student with no working still
    # sometimes plans a beat about a *common* pitfall, which is fair teaching -- but
    # labelling it "the mistake itself" to the narrator invites a line about the
    # mistake *they* made, which is the one thing the explainer prompt forbids.
    mistake = row["targets_misconception"] and not explaining
    marker = "  <- THIS BEAT SHOWS THE MISTAKE ITSELF" if mistake else ""
    brief = (
        f'- {row["beat_id"]} "{_neutralize_markers(row["title"])}" '
        f"({duration:.1f}s on screen; aim for {target} words, hard maximum {budget}): "
        f"{_neutralize_markers(row['purpose'])}{marker}"
    )
    return brief, budget


def span_duration(row: sqlite3.Row, spans: dict[str, tuple[float, float]] | None) -> float:
    """How long this beat is on screen in the *render*, not in what was published.

    `spans` carries the render's own measured timings. It matters on a re-narration:
    the beats table holds the published video's timings, which include the holds a
    previous narration added, so budgeting against it would grant a longer line
    because the last line was long -- and each pass would compound. Budgeting
    against the render keeps every pass computing the same answer from the same
    input.
    """
    if spans is not None and row["beat_id"] in spans:
        start, end = spans[row["beat_id"]]
        return end - start
    return float(row["end_s"]) - float(row["start_s"])


def build_prompt(
    diagnosis_row: sqlite3.Row,
    beat_rows: list[sqlite3.Row],
    words_per_second: float,
    *,
    session_row: sqlite3.Row | None = None,
    spans: dict[str, tuple[float, float]] | None = None,
    explaining: bool = False,
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
        brief, _ = _beat_brief(
            row,
            words_per_second,
            duration=span_duration(row, spans),
            final=index == len(beat_rows) - 1,
            explaining=explaining,
        )
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

    return (EXPLAIN_PROMPT if explaining else PROMPT).format(
        problem=problem,
        work=work,
        solution=solution or "(not recorded)",
        rule=_neutralize_markers(diagnosis_row["buggy_rule"]),
        statement=_neutralize_markers(diagnosis_row["statement"]),
        evidence=evidence,
        beats="\n".join(briefs),
        ids=", ".join(ids),
    )


def budgets_for(
    beat_rows: list[sqlite3.Row],
    words_per_second: float,
    *,
    spans: dict[str, tuple[float, float]] | None = None,
    explaining: bool = False,
) -> dict[str, int]:
    """Each beat's hard word maximum, keyed by beat id."""
    return {
        row["beat_id"]: _beat_brief(
            row,
            words_per_second,
            duration=span_duration(row, spans),
            final=index == len(beat_rows) - 1,
            explaining=explaining,
        )[1]
        for index, row in enumerate(beat_rows)
    }


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
    spans: dict[str, tuple[float, float]] | None = None,
    explaining: bool = False,
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
        diagnosis_row,
        timed,
        words_per_second,
        session_row=repo.get_session(conn, session_id),
        spans=spans,
        explaining=explaining,
    )
    script, meta = await client.complete_json(
        messages=[{"role": "user", "content": prompt}],
        schema=NarrationScript,
        model=model,
    )

    by_id = {line.beat_id: line.line for line in script.lines}
    budgets = budgets_for(timed, words_per_second, spans=spans, explaining=explaining)

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
