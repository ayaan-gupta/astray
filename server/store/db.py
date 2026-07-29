"""SQLite connection and migrations. Schema version tracked in PRAGMA user_version."""

import sqlite3
import threading
from pathlib import Path

MIGRATIONS: list[str] = [
    # APPEND-ONLY: scripts are numbered by list position (see _migrate below) and
    # matched against PRAGMA user_version on the target database. Adding a new
    # migration means appending a new string to the end of this list — never
    # edit, remove, or reorder an existing entry, or already-deployed databases
    # will desync from user_version and either re-run a migration or skip one.
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
    # v2 -- Phase 2 rendering tables.
    #
    # `beats` is the grounding index: s6 plans the rows, the renderer fills in
    # start_s/end_s from the container's measured manifest, and chat cites
    # against them. start_s/end_s are nullable because a planned beat exists
    # before it has ever been rendered -- the beat rail shows it greyed until
    # the manifest lands, so "planned but untimed" has to be representable.
    #
    # PRIMARY KEY(session_id, beat_id) makes a duplicate beat id within one
    # session a database error rather than a silently overwritten row, which
    # matters because duplicate ids are exactly what s8's coverage check exists
    # to catch: two beats sharing an id make every citation to that id ambiguous.
    """
    CREATE TABLE beats (
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        beat_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        title TEXT NOT NULL,
        purpose TEXT NOT NULL DEFAULT '',
        on_screen TEXT NOT NULL DEFAULT '',
        targets_misconception INTEGER NOT NULL DEFAULT 0,
        primitive TEXT NOT NULL DEFAULT 'custom',
        start_s REAL,
        end_s REAL,
        PRIMARY KEY (session_id, beat_id)
    );
    CREATE INDEX idx_beats_session_idx ON beats(session_id, idx);

    CREATE TABLE renders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        attempt INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL,
        duration_s REAL NOT NULL DEFAULT 0,
        error_text TEXT,
        video_path TEXT,
        mode TEXT NOT NULL DEFAULT 'generated',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX idx_renders_session ON renders(session_id);
    """,
    # v3 -- Phase 3 tutor tables.
    #
    # `cited_beats_json` stores the beat ids a reply actually cited, AFTER
    # server-side validation against the manifest. Keeping them as their own
    # column rather than re-parsing `content` later means the grounding record
    # survives any change to citation syntax, and makes "how often does the
    # tutor ground its answers" a query rather than a regex over message text.
    """
    CREATE TABLE chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        cited_beats_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX idx_chat_session ON chat_messages(session_id, id);
    """,
]


class _MaterializedRows:
    """A DB-API-like result whose rows were fetched immediately, while the
    connection's write lock was still held, instead of a live cursor.

    pysqlite keeps a per-connection cache of prepared statements keyed by SQL
    text; concurrent ``execute()`` calls with identical SQL can be handed the
    same cached statement object. Locking only the ``execute()`` call closes
    half the hazard — the fetch phase that follows (``fetchone()``/
    ``fetchall()`` stepping through that shared statement) is just as
    exposed, and it bites hardest on exactly the queries this store repeats
    verbatim (``get_session``, ``get_diagnosis``, ``list_artifacts``).
    Verified empirically: fetching outside the lock corrupted concurrent
    ``get_diagnosis``-style reads (``IndexError``, ``InterfaceError``, and
    silently wrong values — ``None`` or ``""`` instead of the stored row) and
    produced wildly wrong row counts under concurrent ``fetchall()``.

    Draining the full result set inside the lock and handing back this
    plain, already-materialized object removes the live statement from the
    picture entirely once the lock is released — there is nothing left for
    another thread to race with. Eager materialization is safe here because
    every result set in this store is a session row, a diagnosis row, or a
    handful of artifacts, never a large streamed query.
    """

    def __init__(self, rows: list, *, lastrowid, rowcount: int, description) -> None:
        self._rows = list(rows)
        self._pos = 0
        self.lastrowid = lastrowid
        self.rowcount = rowcount
        self.description = description

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchmany(self, size: int = 1) -> list:
        end = min(self._pos + size, len(self._rows))
        rows = self._rows[self._pos : end]
        self._pos = end
        return rows

    def fetchall(self) -> list:
        rows = self._rows[self._pos :]
        self._pos = len(self._rows)
        return rows

    def __iter__(self):
        while (row := self.fetchone()) is not None:
            yield row


def _materialize(cur: sqlite3.Cursor) -> _MaterializedRows:
    """Fetch every remaining row off ``cur`` now, while the lock is held.

    Reads ``fetchall``/``lastrowid``/``rowcount`` via the base ``sqlite3.Cursor``
    descriptors explicitly (not ``cur.fetchall()``/``cur.lastrowid`` directly):
    when ``cur`` is a ``_LockingCursor``, those names are themselves overridden
    to read back from ``_materialized`` — calling through the instance would
    just return whatever was materialized last time instead of draining the
    live statement now being executed.
    """
    return _MaterializedRows(
        sqlite3.Cursor.fetchall(cur),
        lastrowid=sqlite3.Cursor.lastrowid.__get__(cur),
        rowcount=sqlite3.Cursor.rowcount.__get__(cur),
        description=cur.description,
    )


class _LockingCursor(sqlite3.Cursor):
    """Cursor whose execute methods materialize rows under the parent
    connection's write lock, so ``conn.cursor().execute(...)`` is exactly as
    safe as ``conn.execute(...)``.

    Returned by ``_SerializedConnection.cursor()``. This is not redundant
    with the overrides on ``_SerializedConnection`` below: pysqlite's
    C-level ``Connection.execute``/``executemany``/``executescript`` do not
    dispatch through the (overridable) ``cursor()`` method, so without this
    class a caller that does ``conn.cursor().execute(...)`` directly would
    bypass the lock entirely — verified empirically, this reproduces the
    exact same corruption as having no lock at all. See
    ``_MaterializedRows`` for why the fetch phase, not just ``execute()``
    itself, must happen under the lock.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._materialized = _MaterializedRows([], lastrowid=None, rowcount=-1, description=None)

    def execute(self, sql, parameters=()):  # type: ignore[override]
        with self.connection._write_lock:
            super().execute(sql, parameters)
            self._materialized = _materialize(self)
        return self

    def executemany(self, sql, parameters):  # type: ignore[override]
        with self.connection._write_lock:
            super().executemany(sql, parameters)
            self._materialized = _materialize(self)
        return self

    def executescript(self, sql_script):  # type: ignore[override]
        # Migrations only (DDL); never produces rows, so nothing to materialize.
        with self.connection._write_lock:
            return super().executescript(sql_script)

    def fetchone(self):
        return self._materialized.fetchone()

    def fetchmany(self, size: int = 1) -> list:
        return self._materialized.fetchmany(size)

    def fetchall(self) -> list:
        return self._materialized.fetchall()

    def __iter__(self):
        return iter(self._materialized)

    @property
    def lastrowid(self):
        return self._materialized.lastrowid

    @property
    def rowcount(self) -> int:
        return self._materialized.rowcount


class _SerializedConnection(sqlite3.Connection):
    """A Connection that serializes statement execution — call *and* fetch —
    across threads.

    ``check_same_thread=False`` (below) only disables Python's same-thread
    guard; it does not make the connection safe for *concurrent* use. Even
    though the underlying SQLite build reports ``sqlite3.threadsafety == 3``
    ("serialized"), pysqlite keeps a per-connection statement cache that is
    not itself protected against concurrent use from multiple threads —
    verified empirically: hammering one connection from several threads
    without this lock produced ``sqlite3.InterfaceError: bad parameter or
    other API misuse``, silently dropped writes, and — when the lock covered
    only ``execute()`` and not the subsequent fetch — silently wrong reads
    and wrong row counts on top of that. Locking ``execute()``/``executemany()``
    together with draining every row (via ``_materialize``, see
    ``_MaterializedRows``) before releasing the lock closes both. This is a
    Python-level add-on; it does not reintroduce ``check_same_thread=True``
    and does not conflict with ``isolation_level=None`` autocommit or WAL.

    Both these direct methods *and* ``cursor()`` (via ``_LockingCursor``) are
    covered — see that class's docstring for why both are needed.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._write_lock = threading.Lock()

    def execute(self, sql, parameters=()) -> _MaterializedRows:  # type: ignore[override]
        with self._write_lock:
            cur = super().execute(sql, parameters)
            return _materialize(cur)

    def executemany(self, sql, parameters) -> _MaterializedRows:  # type: ignore[override]
        with self._write_lock:
            cur = super().executemany(sql, parameters)
            return _materialize(cur)

    def executescript(self, sql_script):  # type: ignore[override]
        # Migrations only (DDL); never produces rows, so nothing to materialize.
        with self._write_lock:
            return super().executescript(sql_script)

    def cursor(self, factory=_LockingCursor):  # type: ignore[override]
        return super().cursor(factory)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is required: FastAPI dispatches `def` routes on a
    # threadpool worker while `async def` routes and startup run on the loop
    # thread, so one connection is touched from several threads. Safe here
    # because isolation_level=None autocommits every statement (no interleaved
    # transactions), WAL + busy_timeout handle contention, and
    # _SerializedConnection (above) serializes concurrent execute()/fetch calls.
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
