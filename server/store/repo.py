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
