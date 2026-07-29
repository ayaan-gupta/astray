"""Typed persistence helpers. All SQL lives here or in db.MIGRATIONS."""

import json
import sqlite3
import uuid

from server.charter.contracts import Diagnosis, LlmCallMeta, StageName, StudentSubmission


def create_session(conn: sqlite3.Connection, *, handle: str, submission: StudentSubmission) -> str:
    session_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO sessions (id, handle, input_mode, problem, student_work_json, status)
           VALUES (?, ?, ?, ?, ?, 'created')""",
        (
            session_id,
            handle,
            submission.source,
            submission.problem,
            submission.model_dump_json(),
        ),
    )
    return session_id


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()


def set_session_status(conn: sqlite3.Connection, session_id: str, status: str) -> None:
    conn.execute("UPDATE sessions SET status = ? WHERE id = ?", (status, session_id))


def try_start_session(conn: sqlite3.Connection, session_id: str) -> bool:
    """Atomically transition ``session_id`` from ``created`` to ``in_progress``.

    A compare-and-swap, not a read-then-write: the transition and the check that
    it is legal to make happen in the same statement, so there is no window
    between "read the status" and "write in_progress" for a second concurrent
    caller to read the same pre-transition status and also decide it may start a
    run. ``UPDATE ... WHERE status = 'created'`` either matches this session's one
    row (it was still ``created``) or matches nothing (some other caller already
    won the race, or it was already terminal) -- ``rowcount`` distinguishes the
    two outcomes without a second query.

    Returns ``True`` if this call performed the transition (the caller may now
    safely start a run), ``False`` if it did not (the session was already
    ``in_progress`` or a terminal status -- the caller must not start a run and
    should inspect the current status to decide between a 409 and a replay).
    """
    cur = conn.execute(
        "UPDATE sessions SET status = 'in_progress' WHERE id = ? AND status = 'created'",
        (session_id,),
    )
    return cur.rowcount == 1


def update_submission(
    conn: sqlite3.Connection, session_id: str, submission: StudentSubmission
) -> bool:
    """Replace a session's stored submission, but only while it is still ``created``.

    Every write of ``student_work_json`` goes through here so the three columns
    derived from a submission (``input_mode``, ``problem``, ``student_work_json``)
    can never drift apart -- a photo transcription carries its own ``problem``
    text, which is usually not the placeholder typed at create time.

    The ``status = 'created'`` predicate is the important part, and it is in the
    UPDATE rather than a preceding read for the same reason ``try_start_session``
    is: a read-then-write leaves a window in which a concurrent ``/stream`` claims
    the session and begins diagnosing while this call is rewriting the very work
    being diagnosed. Once a run has started, the submission is frozen -- otherwise
    the persisted diagnosis would describe work that no longer exists in the row.

    Returns ``True`` if the row was updated, ``False`` if the session was missing
    or no longer ``created`` (the caller decides whether that is a 404 or a 409).
    """
    cur = conn.execute(
        """UPDATE sessions
              SET input_mode = ?, problem = ?, student_work_json = ?
            WHERE id = ? AND status = 'created'""",
        (submission.source, submission.problem, submission.model_dump_json(), session_id),
    )
    return cur.rowcount == 1


def record_artifact(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    stage: StageName,
    payload: dict,
    meta: LlmCallMeta,
    attempt: int = 1,
) -> None:
    conn.execute(
        """INSERT INTO run_artifacts (session_id, stage, attempt, payload_json, reasoning_text,
                                      model, prompt_tokens, completion_tokens, cached_tokens,
                                      cost_usd, ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            str(stage),
            attempt,
            json.dumps(payload),
            meta.reasoning,
            meta.model,
            meta.prompt_tokens,
            meta.completion_tokens,
            meta.cached_tokens,
            meta.cost_usd,
            meta.ms,
        ),
    )


def list_artifacts(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM run_artifacts WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()


def save_diagnosis(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    diagnosis: Diagnosis,
    misconception_id: int | None,
    canonical_rule: str = "",
) -> int:
    cursor = conn.execute(
        """INSERT INTO diagnoses (session_id, misconception_id, buggy_rule, canonical_rule,
                                  statement, topic, confidence, divergence_index,
                                  verified_by_sympy, is_unclear, payload_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            misconception_id,
            diagnosis.buggy_rule,
            canonical_rule,
            diagnosis.misconception_statement,
            diagnosis.topic,
            diagnosis.confidence,
            diagnosis.divergence_index,
            int(diagnosis.verified_by_sympy),
            int(diagnosis.is_unclear),
            diagnosis.model_dump_json(),
        ),
    )
    return int(cursor.lastrowid)


def get_diagnosis(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM diagnoses WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()


def save_beats(conn: sqlite3.Connection, session_id: str, storyboard) -> None:
    """Persist the planned beats for a session.

    Written as soon as s6 plans them, before any render exists, so the beat rail
    can show the upcoming beats greyed while the video is still rendering --
    start_s/end_s stay NULL until a render measures them.

    INSERT OR REPLACE keyed on (session_id, beat_id): a re-plan after a failed
    render replaces the previous plan rather than accumulating orphan rows.
    """
    for index, item in enumerate(storyboard.beats):
        conn.execute(
            """INSERT OR REPLACE INTO beats
                 (session_id, beat_id, idx, title, purpose, on_screen,
                  targets_misconception, primitive, start_s, end_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                       (SELECT start_s FROM beats WHERE session_id=? AND beat_id=?),
                       (SELECT end_s   FROM beats WHERE session_id=? AND beat_id=?))""",
            (
                session_id,
                item.id,
                index,
                item.title,
                item.teaching_purpose,
                item.on_screen,
                int(item.targets_misconception),
                item.primitive,
                session_id,
                item.id,
                session_id,
                item.id,
            ),
        )


def save_beat_timings(conn: sqlite3.Connection, session_id: str, timings) -> int:
    """Write measured start/end onto already-planned beats. Returns rows updated.

    Only UPDATEs: a timing for a beat that was never planned is discarded rather
    than inserted, because an unplanned beat has no title or purpose and would
    appear on the rail as a blank, unciteable segment. s8 already rejects scenes
    that wrap unplanned beats, so reaching here means something upstream broke.
    """
    updated = 0
    for timing in timings:
        cur = conn.execute(
            "UPDATE beats SET start_s = ?, end_s = ? WHERE session_id = ? AND beat_id = ?",
            (timing.start, timing.end, session_id, timing.id),
        )
        updated += cur.rowcount
    return updated


def list_beats(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM beats WHERE session_id = ? ORDER BY idx", (session_id,)
    ).fetchall()


def record_render(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    attempt: int,
    status: str,
    duration_s: float = 0.0,
    error_text: str | None = None,
    video_path: str | None = None,
    mode: str = "generated",
) -> int:
    """Append one render attempt to the ledger.

    Every attempt is a row, including failures: `renders.mode` distinguishes
    generated from storyboard_fallback so the fallback rate is measurable, and
    keeping failed attempts is what makes "how often does codegen work" a
    question the data can answer.
    """
    cur = conn.execute(
        """INSERT INTO renders
             (session_id, attempt, status, duration_s, error_text, video_path, mode)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, attempt, status, duration_s, error_text, video_path, mode),
    )
    return int(cur.lastrowid)


def latest_render(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM renders WHERE session_id = ? AND status = 'ok' ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
