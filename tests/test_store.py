import threading

from server.charter.contracts import (
    Diagnosis,
    LlmCallMeta,
    StageName,
    StudentSubmission,
    SympyCheck,
)
from server.store import repo
from server.store.db import connect


def _conn(tmp_path):
    return connect(tmp_path / "t.db")


def _diagnosis() -> Diagnosis:
    return Diagnosis(
        correct_solution=["x=2", "x=-8"],
        sympy_check=SympyCheck(kind="equivalence", lhs="(x+3)**2", rhs="x**2+6*x+9"),
        verified_by_sympy=True,
        divergence_index=0,
        buggy_rule="(a+b)^2 -> a^2+b^2",
        misconception_statement="You dropped the cross term.",
        confidence=0.93,
        topic="algebra.binomial_expansion",
    )


def test_wal_enabled(tmp_path):
    conn = _conn(tmp_path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "t.db"
    connect(path).close()
    conn = connect(path)  # second connect must not re-run migrations
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 1


def test_create_and_get_session(tmp_path):
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn,
        handle="anon-1",
        submission=StudentSubmission(problem="(x+3)^2=25", steps=["x^2+9=25"], source="typed"),
    )
    row = repo.get_session(conn, sid)
    assert row["handle"] == "anon-1"
    assert row["status"] == "created"
    assert "x^2+9=25" in row["student_work_json"]


def test_record_and_list_artifacts(tmp_path):
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    repo.record_artifact(
        conn,
        session_id=sid,
        stage=StageName.DIAGNOSE,
        payload={"buggy_rule": "r"},
        meta=LlmCallMeta(model="deepseek-v4-pro", cost_usd=0.001, reasoning="because"),
    )
    rows = repo.list_artifacts(conn, sid)
    assert len(rows) == 1
    assert rows[0]["stage"] == "s1_diagnose"
    assert rows[0]["reasoning_text"] == "because"
    assert rows[0]["cost_usd"] == 0.001


def test_save_and_get_diagnosis(tmp_path):
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    repo.save_diagnosis(conn, session_id=sid, diagnosis=_diagnosis(), misconception_id=None)
    row = repo.get_diagnosis(conn, sid)
    assert row["buggy_rule"] == "(a+b)^2 -> a^2+b^2"
    assert row["verified_by_sympy"] == 1


def test_artifact_requires_valid_session(tmp_path):
    import sqlite3

    import pytest

    conn = _conn(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        repo.record_artifact(
            conn,
            session_id="does-not-exist",
            stage=StageName.INGEST,
            payload={},
            meta=LlmCallMeta(model="deepseek-v4-flash"),
        )


def test_set_session_status(tmp_path):
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    repo.set_session_status(conn, sid, "diagnosed")
    assert repo.get_session(conn, sid)["status"] == "diagnosed"


def test_deleting_session_cascades_to_artifacts_and_diagnoses(tmp_path):
    """FK ON DELETE CASCADE actually fires, not just declared."""
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    repo.record_artifact(
        conn,
        session_id=sid,
        stage=StageName.INGEST,
        payload={},
        meta=LlmCallMeta(model="deepseek-v4-flash"),
    )
    repo.save_diagnosis(conn, session_id=sid, diagnosis=_diagnosis(), misconception_id=None)

    conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))

    assert repo.list_artifacts(conn, sid) == []
    assert repo.get_diagnosis(conn, sid) is None


def test_diagnosis_json_roundtrip_reconstructs_identical_model(tmp_path):
    """payload_json must reconstruct a Diagnosis equal to the one that was saved.

    Note: Diagnosis.model_dump_json() (no exclude_unset) always emits every
    declared field, so there is no reachable "missing key" state for this model
    to round-trip through — this test checks round-trip equality, including
    explicit None values on optional fields, not a missing-vs-None distinction.
    """
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    diagnosis = Diagnosis(
        correct_solution=["x=1"],
        sympy_check=SympyCheck(kind="skip", skip_reason="no symbolic domain"),
        divergence_index=None,
        buggy_rule="r",
        misconception_statement="s",
        confidence=0.5,
        clarifying_question=None,
    )
    repo.save_diagnosis(conn, session_id=sid, diagnosis=diagnosis, misconception_id=None)

    row = repo.get_diagnosis(conn, sid)
    roundtripped = Diagnosis.model_validate_json(row["payload_json"])

    assert roundtripped == diagnosis
    assert row["divergence_index"] is None


def test_migration_numbering_survives_an_appended_migration(tmp_path, monkeypatch):
    """A later task appending MIGRATIONS[1] must apply only the new script, once."""
    from server.store import db

    path = tmp_path / "t.db"
    connect(path).close()  # applies v1 only

    extra_migration = "CREATE TABLE probe_v2 (id INTEGER PRIMARY KEY);"
    monkeypatch.setattr(db, "MIGRATIONS", db.MIGRATIONS + [extra_migration])

    conn = connect(path)  # must apply only the appended v2 script

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    # v1 tables are untouched (no duplicate-table error) and v2 table exists.
    conn.execute("INSERT INTO probe_v2 (id) VALUES (1)")
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_concurrent_writes_from_multiple_threads_are_not_lost(tmp_path):
    """The shared connection is genuinely touched from several threads (FastAPI
    threadpool workers). Without serializing statement execution, pysqlite's
    per-connection statement cache corrupts under concurrent execute() calls
    and writes silently disappear."""
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    writes_per_thread = 20
    thread_count = 8
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            for i in range(writes_per_thread):
                repo.record_artifact(
                    conn,
                    session_id=sid,
                    stage=StageName.INGEST,
                    payload={"i": i, "thread": n},
                    meta=LlmCallMeta(model="deepseek-v4-flash"),
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(repo.list_artifacts(conn, sid)) == writes_per_thread * thread_count


def test_concurrent_writes_via_raw_cursor_are_not_lost(tmp_path):
    """conn.cursor().execute(...) must be as serialized as conn.execute(...).

    _SerializedConnection only overrides execute/executemany/executescript on
    the Connection itself; pysqlite's C-level Connection.execute does not
    dispatch through the overridable cursor() method, so a caller reaching for
    conn.cursor() directly (a very natural way to get cursor.lastrowid) would
    silently bypass the lock unless cursor() itself returns a locking cursor.
    This reproduces the original corruption exactly if that hole reopens.
    """
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    writes_per_thread = 20
    thread_count = 8
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            for _i in range(writes_per_thread):
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO run_artifacts (session_id, stage, payload_json, model)
                       VALUES (?, ?, ?, ?)""",
                    (sid, "s0_ingest", "{}", "deepseek-v4-flash"),
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(repo.list_artifacts(conn, sid)) == writes_per_thread * thread_count


def test_get_diagnosis_returns_most_recent(tmp_path):
    """Re-diagnosis flows depend on get_diagnosis returning the latest row, not
    the first, when a session has been diagnosed more than once."""
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    first = _diagnosis()
    second = Diagnosis(
        correct_solution=["x=-2", "x=8"],
        sympy_check=SympyCheck(kind="equivalence", lhs="(x-3)**2", rhs="x**2-6*x+9"),
        verified_by_sympy=True,
        divergence_index=1,
        buggy_rule="a different buggy rule",
        misconception_statement="Second, corrected diagnosis.",
        confidence=0.5,
        topic="algebra.binomial_expansion",
    )
    repo.save_diagnosis(conn, session_id=sid, diagnosis=first, misconception_id=None)
    repo.save_diagnosis(conn, session_id=sid, diagnosis=second, misconception_id=None)

    row = repo.get_diagnosis(conn, sid)

    assert row["buggy_rule"] == "a different buggy rule"
    assert row["statement"] == "Second, corrected diagnosis."


def test_diagnosis_requires_valid_misconception_id(tmp_path):
    """diagnoses.misconception_id FK must reject an id with no matching row.

    Task 9 starts populating the misconceptions table, so this becomes a live
    path rather than a theoretical one.
    """
    import sqlite3

    import pytest

    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_diagnosis(conn, session_id=sid, diagnosis=_diagnosis(), misconception_id=9999)


def test_concurrent_identical_reads_return_the_correct_row(tmp_path):
    """Locking only execute() (not the fetch that follows) still lets pysqlite's
    statement cache hand two threads the same in-flight prepared statement for
    identical SQL text. The corruption isn't only lost writes or exceptions —
    it includes *silently wrong reads*: None, an empty string, or another
    thread's row, with nothing raised. This must never happen, not merely
    never raise, so this asserts the actual value on every read.
    """
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    diagnosis = _diagnosis()
    repo.save_diagnosis(conn, session_id=sid, diagnosis=diagnosis, misconception_id=None)

    reader_count = 12
    reads_per_thread = 50
    errors: list[BaseException] = []
    wrong: list[object] = []

    def reader() -> None:
        try:
            for _ in range(reads_per_thread):
                row = repo.get_diagnosis(conn, sid)
                if row is None or row["buggy_rule"] != diagnosis.buggy_rule:
                    wrong.append(row["buggy_rule"] if row is not None else None)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(reader_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert wrong == []


def test_concurrent_fetchall_via_cursor_returns_the_exact_row_count(tmp_path):
    """Same statement-cache hazard as above, but for fetchall() reached via
    conn.cursor() rather than conn.execute(). The observed failure mode was
    over-counting (e.g. 502 rows read back from a 200-row table), so this
    pins the exact count on every read, not merely the absence of an
    exception.
    """
    conn = _conn(tmp_path)
    sid = repo.create_session(
        conn, handle="a", submission=StudentSubmission(problem="p", source="typed")
    )
    expected_rows = 40
    for i in range(expected_rows):
        repo.record_artifact(
            conn,
            session_id=sid,
            stage=StageName.INGEST,
            payload={"i": i},
            meta=LlmCallMeta(model="deepseek-v4-flash"),
        )

    reader_count = 9
    reads_per_thread = 15
    errors: list[BaseException] = []
    wrong_counts: list[int] = []

    def reader() -> None:
        try:
            for _ in range(reads_per_thread):
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM run_artifacts WHERE session_id = ? ORDER BY id",
                    (sid,),
                )
                rows = cur.fetchall()
                if len(rows) != expected_rows:
                    wrong_counts.append(len(rows))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(reader_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert wrong_counts == []
