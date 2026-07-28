"""SQLite connection and migrations. Schema version tracked in PRAGMA user_version."""

import sqlite3
import threading
from pathlib import Path

MIGRATIONS: list[str] = [
    # v1 — Phase 1 tables. Phase 2/3 add beats, chat_messages, checkpoints, renders.
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        handle TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        input_mode TEXT NOT NULL,
        problem TEXT NOT NULL,
        student_work_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'created'
    );

    CREATE TABLE misconceptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slug TEXT NOT NULL UNIQUE,
        canonical_statement TEXT NOT NULL,
        canonical_rule TEXT NOT NULL,
        topic TEXT NOT NULL,
        aliases_json TEXT NOT NULL DEFAULT '[]',
        is_seed INTEGER NOT NULL DEFAULT 0,
        first_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE run_artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        attempt INTEGER NOT NULL DEFAULT 1,
        payload_json TEXT NOT NULL,
        reasoning_text TEXT,
        model TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd REAL NOT NULL DEFAULT 0,
        ms INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX idx_artifacts_session ON run_artifacts(session_id, stage);

    CREATE TABLE diagnoses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        misconception_id INTEGER REFERENCES misconceptions(id),
        buggy_rule TEXT NOT NULL,
        canonical_rule TEXT NOT NULL DEFAULT '',
        statement TEXT NOT NULL,
        topic TEXT NOT NULL DEFAULT 'unknown',
        confidence REAL NOT NULL,
        divergence_index INTEGER,
        verified_by_sympy INTEGER NOT NULL DEFAULT 0,
        is_unclear INTEGER NOT NULL DEFAULT 0,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX idx_diagnoses_session ON diagnoses(session_id);
    CREATE INDEX idx_diagnoses_misconception ON diagnoses(misconception_id);
    """,
]


class _SerializedConnection(sqlite3.Connection):
    """A Connection that serializes statement execution across threads.

    ``check_same_thread=False`` (below) only disables Python's same-thread
    guard; it does not make the connection safe for *concurrent* use. Even
    though the underlying SQLite build reports ``sqlite3.threadsafety == 3``
    ("serialized"), pysqlite keeps a per-connection statement cache that is
    not itself protected against concurrent ``execute()`` calls from multiple
    threads — verified empirically: hammering one connection from five
    threads without this lock produced ``sqlite3.InterfaceError: bad
    parameter or other API misuse`` and silently dropped writes. A single
    lock around statement execution fixes it. This is a Python-level
    add-on; it does not reintroduce ``check_same_thread=True`` and does not
    conflict with ``isolation_level=None`` autocommit or WAL.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._write_lock = threading.Lock()

    def execute(self, sql, parameters=()):  # type: ignore[override]
        with self._write_lock:
            return super().execute(sql, parameters)

    def executemany(self, sql, parameters):  # type: ignore[override]
        with self._write_lock:
            return super().executemany(sql, parameters)

    def executescript(self, sql_script):  # type: ignore[override]
        with self._write_lock:
            return super().executescript(sql_script)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is required: FastAPI dispatches `def` routes on a
    # threadpool worker while `async def` routes and startup run on the loop
    # thread, so one connection is touched from several threads. Safe here
    # because isolation_level=None autocommits every statement (no interleaved
    # transactions), WAL + busy_timeout handle contention, and
    # _SerializedConnection (above) serializes concurrent execute() calls.
    conn = sqlite3.connect(
        db_path,
        isolation_level=None,
        check_same_thread=False,
        factory=_SerializedConnection,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for index, script in enumerate(MIGRATIONS, start=1):
        if index <= current:
            continue
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version={index}")
