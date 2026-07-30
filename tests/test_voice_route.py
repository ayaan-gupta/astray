"""Where a spoken sentence goes, and what it sounds like when read back.

The transcripts here are real. They were pulled out of a live session's chat
history after the first round of voice testing, which is where the bug was found:
every one of them had been answered as a question about the animation on screen,
including the one that was plainly a different problem.

No test here makes a model call. `route.route` is exercised through its own
failure and coercion paths, and the LLM path is covered by `tests/test_app.py`
through the fake transport -- the interesting logic is what happens to a
well-formed answer that the caller still cannot act on.
"""

import pytest

from server.charter.contracts import BeatTiming
from server.llm.deepseek import LlmError
from server.store import repo
from server.store.db import connect
from server.tutor import route, say
from tests.test_audio import _board, _session

# Verbatim from a live session. "DUI over DX" is how Chrome heard "dy/dx".
GARBLED_DE = (
    "I was solving DUI over DX = 2x - 5 over y squared and I couldn't really "
    "figure it out can you help me out here"
)


class _Broken:
    """A client whose every call fails, for the paths that must survive that."""

    async def complete_json(self, **_):
        raise LlmError("upstream is down")


async def test_a_failed_route_still_reaches_the_tutor():
    """Routing is a convenience over "send it to chat". Losing it must not lose
    the sentence the student just said."""
    routed, meta = await route.route(
        _Broken(), transcript=GARBLED_DE, model="m", current_problem="(y+3)^2"
    )
    assert routed.kind == "followup"
    assert routed.question == GARBLED_DE, "the words are kept, unrepaired"
    assert meta is None, "no call was billed"


async def test_a_failed_route_with_no_session_open_becomes_a_new_problem():
    """On the submit page there is no conversation to fall back into, so the only
    reading left is a problem."""
    routed, _ = await route.route(_Broken(), transcript=GARBLED_DE, model="m")
    assert routed.kind == "new_problem"
    assert routed.problem == GARBLED_DE


async def test_the_wake_phrase_alone_is_not_a_problem():
    """ "Hey Astray" and nothing after it. There is nothing to classify, and it must
    not become a session with an empty problem statement."""
    routed, meta = await route.route(
        _Broken(), transcript="hey", model="m", current_problem="(y+3)^2"
    )
    assert routed.kind == "followup"
    assert meta is None, "too short to be worth a model call"


@pytest.mark.parametrize("transcript", ["", "   ", "hi"])
async def test_nothing_short_reaches_the_model(transcript):
    routed, meta = await route.route(_Broken(), transcript=transcript, model="m")
    assert meta is None
    assert routed.kind == "followup"


def test_a_new_problem_with_nothing_in_it_falls_back_to_the_conversation():
    """What the prompt asks for when the speech was too garbled to reconstruct. A
    session cannot be created from an empty problem statement."""
    empty = route.RoutedUtterance(kind="new_problem", problem="   ")
    coerced = route.coerce(empty, "something mumbled", has_session=True)
    assert coerced.kind == "followup"
    assert coerced.question == "something mumbled"


def test_a_followup_with_no_question_falls_back_to_the_raw_transcript():
    """A blank message would post an empty chat turn. The unrepaired transcript is
    a worse question than a repaired one and a far better one than nothing."""
    blank = route.RoutedUtterance(kind="followup", question="")
    coerced = route.coerce(blank, GARBLED_DE, has_session=True)
    assert coerced.question == GARBLED_DE


def test_a_followup_with_no_session_becomes_a_new_problem():
    """The submit page. Nothing to follow up on, so what was said is the problem."""
    followup = route.RoutedUtterance(kind="followup", question="why is this wrong")
    coerced = route.coerce(followup, "expand x plus five all squared", has_session=False)
    assert coerced.kind == "new_problem"
    assert coerced.problem == "expand x plus five all squared"


def test_an_oversized_field_is_rejected_rather_than_quietly_truncated():
    """The bound is enforced by the contract, not patched up afterwards: a reply
    this far outside spec is a broken reply, and a truncated equation is worse than
    no equation because it looks like something the student said."""
    with pytest.raises(ValueError):
        route.RoutedUtterance(kind="new_problem", problem="x" * 5000)


async def test_an_oversized_transcript_is_cut_before_it_reaches_a_prompt():
    """A recogniser left open on a room full of talking must not become a large
    prompt. Trimmed rather than rejected, because the student still said something."""
    routed, _ = await route.route(_Broken(), transcript="why " * 2000, model="m")
    assert len(routed.problem) <= route.MAX_TRANSCRIPT_CHARS


# --------------------------------------------------------------- spoken replies
def _spoken_session(tmp_path):
    conn = connect(tmp_path / "t.db")
    sid = _session(conn)
    repo.save_beats(conn, sid, _board())
    repo.save_beat_timings(
        conn,
        sid,
        [BeatTiming(id="b1", start=0.0, end=6.0), BeatTiming(id="b2", start=6.0, end=12.0)],
    )
    return conn, sid


def test_a_citation_is_spoken_as_the_beat_it_points_at(tmp_path):
    """Deleting it leaves "Watch to see that it fails", a sentence with a hole in
    it: the prose is written around the chip, so the chip has to become words."""
    conn, sid = _spoken_session(tmp_path)
    titles = say.beat_titles(conn, sid)
    spoken = say.for_speech("Watch [beat:b1] to see it fail.", titles)
    assert "[beat:" not in spoken, "the citation markup must never be read aloud"
    assert spoken == f"Watch {titles['b1']} to see it fail."


def test_a_beat_title_is_lowercased_so_it_reads_mid_sentence(tmp_path):
    conn, sid = _spoken_session(tmp_path)
    titles = say.beat_titles(conn, sid)
    assert all(value.startswith("the ") for value in titles.values())
    assert all(value[4].islower() for value in titles.values())


def test_a_camel_cased_title_keeps_its_spelling():
    """ "SymPy" is spelled that way rather than merely capitalised, so lowercasing
    the first letter changes the word."""
    assert say.for_speech("See [beat:b1] now.", {"b1": "the SymPy check"}) == (
        "See the SymPy check now."
    )


def test_a_citation_to_an_unknown_beat_still_leaves_a_readable_sentence(tmp_path):
    spoken = say.for_speech("Look at [beat:b9] again.", {})
    assert spoken == f"Look at {say.UNKNOWN_BEAT} again."


def test_notation_and_markdown_do_not_reach_the_voice(tmp_path):
    conn, sid = _spoken_session(tmp_path)
    spoken = say.for_speech(
        "- **Careful:** `(y+3)^2` is not `y^2 + 9`.", say.beat_titles(conn, sid)
    )
    assert "`" not in spoken and "*" not in spoken
    assert not spoken.startswith("-")
    assert "squared" in spoken, "the notation is spoken, not stripped"


def test_a_long_reply_is_cut_at_a_sentence_end():
    """A spoken answer that stops mid-sentence sounds like the connection dropped,
    which is worse than a shorter answer."""
    reply = "One sentence here. " * 40
    spoken = say.for_speech(reply, {})
    assert len(spoken) <= say.MAX_SPOKEN_CHARS
    assert spoken.endswith(".")


def test_a_single_sentence_over_the_cap_is_kept_whole():
    """One long sentence is still a complete thought. Cutting it at the cap would
    leave a fragment that stops dead."""
    reply = "This one sentence just keeps going and going " * 20 + "and then it ends."
    spoken = say.for_speech(reply, {})
    assert spoken.endswith("ends.")


def test_an_empty_reply_produces_nothing_to_say():
    assert say.for_speech("   ", {}) == ""


# ------------------------------------------------------ stuck, with no attempt
def test_a_student_with_no_working_is_taught_rather_than_corrected():
    """The framing every stage from s2 to s7 reads. Left as a real diagnosis, a
    live run on a spoken differential equation with no working storyboarded a beat
    titled "Buggy Method": it invented a mistake and argued against it."""
    from server.charter.contracts import Diagnosis
    from server.charter.pipeline import as_explainer

    stuck = Diagnosis(
        correct_solution=["y^2 dy = (2x-5) dx", "y^3/3 = x^2 - 5x + C"],
        sympy_check={"kind": "skip", "skip_reason": "no attempt to check"},
        verified_by_sympy=True,
        buggy_rule="unknown",
        misconception_statement="I can't identify a specific mistake because you showed no steps.",
        confidence=0.1,
        is_unclear=True,
    )
    explained = as_explainer(stuck)
    assert "unknown" not in explained.buggy_rule
    assert "Teach the method" in explained.misconception_statement
    assert "never suggest they got something wrong" in explained.misconception_statement
    assert explained.correct_solution == stuck.correct_solution, "the solved maths is kept"


def test_a_storyboard_with_nothing_to_contradict_is_still_valid():
    """The beat contract requires a beat targeting the misconception, which is the
    right rule for a diagnosis and impossible for an explainer. Enforcing it here
    would fail the whole session over a contract that does not apply."""
    from server.charter.contracts import Storyboard
    from server.charter.stages.s6_visual import StoryboardInvalid, check_storyboard

    board = Storyboard.model_validate(
        {
            "beats": [
                {
                    "id": f"b{i}",
                    "title": f"Step {i}",
                    "teaching_purpose": "p",
                    "on_screen": "o",
                    "targets_misconception": False,
                    "primitive": "algebra_steps",
                }
                for i in (1, 2, 3)
            ],
            "total_estimated_seconds": 40,
        }
    )
    check_storyboard(board, require_target=False)
    with pytest.raises(StoryboardInvalid):
        check_storyboard(board)


def test_the_explainer_narration_never_tells_them_what_they_thought():
    """The one place a wrong framing does direct harm: "you thought a plus b, all
    squared, equals..." said to someone who never claimed it invents a mistake and
    attributes it to them."""
    from server.audio.narrate import EXPLAIN_PROMPT

    assert "never say they got something wrong" in EXPLAIN_PROMPT
    assert "There is no error" in EXPLAIN_PROMPT
    assert "You thought" not in EXPLAIN_PROMPT


async def test_the_explainer_prompt_is_the_one_used_when_nothing_was_attempted(tmp_path):
    from server.audio import narrate

    conn, sid = _spoken_session(tmp_path)
    timed = [r for r in repo.list_beats(conn, sid) if r["start_s"] is not None]
    args = (repo.get_diagnosis(conn, sid), timed, 2.32)
    teaching = narrate.build_prompt(*args, explaining=True)
    correcting = narrate.build_prompt(*args, explaining=False)

    assert "has NOT shown any working" in teaching
    assert "has NOT shown any working" not in correcting
    assert "Never state the student's rule without contradicting it" in correcting


def test_the_explainer_framing_forbids_inventing_the_question():
    """Observed on the first live explainer run: with no student working to anchor
    on, the storyboard added an "Apply Initial Condition" beat to a problem that
    states no initial condition, and the narration then asserted x=0, y=1 as though
    it had been given. Inventing the question is the same failure as inventing the
    mistake."""
    from server.audio.narrate import EXPLAIN_PROMPT
    from server.charter.pipeline import _NO_ATTEMPT_BRIEF

    assert "never add a condition" in _NO_ATTEMPT_BRIEF
    # Matched on unwrapped fragments: the prompt is hard-wrapped, so any assertion
    # long enough to be specific would straddle a newline.
    assert "Say only what the problem gives you" in EXPLAIN_PROMPT
    assert "announce a value that was not in the problem" in EXPLAIN_PROMPT.replace("\n", " ")


async def test_no_working_mints_no_taxonomy_entry(tmp_path):
    """A session with nothing attempted has no misconception, so nothing may be
    resolved against the taxonomy.

    Found live. Resolving it anyway minted the row `no-work-provided-error`, whose
    canonical statement was s1's own apology for finding no mistake, and then that
    row was *counted*: it showed up as the diagnosis in the student's problem list
    and in /api/insights beside real misconceptions. Every later no-working session
    in any topic would have collapsed onto it via the exact-canonical fast path --
    the same failure correct work already had a guard for.
    """
    from server.charter.chain import Chain
    from server.charter.contracts import StudentSubmission
    from server.config import Settings

    conn = connect(tmp_path / "t.db")
    settings = Settings(_env_file=None, deepseek_api_key="sk-test", db_path=tmp_path / "t.db")
    chain = Chain(conn, _Broken(), settings=settings)

    stuck = StudentSubmission(
        problem="dy/dx = (2x-5)/y^2", steps=[], source="typed", student_corrected=True
    )
    sid = repo.create_session(conn, handle="h", submission=stuck)
    # The diagnose call fails, which is enough: what matters is that the guard is
    # keyed on the submission having no steps, not on anything the model returns.
    events = [e async for e in chain.run_diagnosis(sid, stuck)]
    assert events, "the chain must always report something"

    minted = conn.execute("SELECT COUNT(*) AS n FROM misconceptions WHERE is_seed = 0").fetchone()[
        "n"
    ]
    assert minted == 0, "a stuck student must not become a taxonomy entry"
