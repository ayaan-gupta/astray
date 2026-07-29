"""Aggregate queries over the misconception log.

This is what the taxonomy work was for. Canonicalizing free-text diagnoses onto
stable ids is only worth doing if something reads the result, and these are the
reads: how often each misconception appears, across how many distinct students,
and -- the one a student actually feels -- "N other students made this error."

`peers_for_session` deliberately counts DISTINCT handles other than this
student's own. Counting rows would let one student's three attempts at the same
problem report "3 others made this error", which is both false and the kind of
false that destroys trust in the number.

Anonymity: nothing here returns a handle. Handles are browser-local anonymous
ids, but a count keyed by misconception is aggregate by construction and a list
of handles would not be.
"""

import sqlite3


def misconception_frequency(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Most common diagnosed misconceptions, with distinct-student counts."""
    rows = conn.execute(
        """SELECT m.id, m.slug, m.canonical_statement, m.topic,
                  COUNT(*)                    AS occurrences,
                  COUNT(DISTINCT s.handle)    AS students
             FROM diagnoses d
             JOIN sessions s       ON s.id = d.session_id
             JOIN misconceptions m ON m.id = d.misconception_id
            GROUP BY m.id
            ORDER BY occurrences DESC, m.id
            LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def peers_for_session(conn: sqlite3.Connection, session_id: str) -> dict:
    """ "N other students made this error" for one session.

    Returns `{"misconception_id", "slug", "statement", "others"}`, or
    `{"others": 0}` when the session has no misconception -- correct work has a
    null misconception_id by design, and must not be reported as shared with
    anyone.
    """
    row = conn.execute(
        """SELECT d.misconception_id AS mid, s.handle AS handle,
                  m.slug AS slug, m.canonical_statement AS statement
             FROM diagnoses d
             JOIN sessions s       ON s.id = d.session_id
             LEFT JOIN misconceptions m ON m.id = d.misconception_id
            WHERE d.session_id = ?
            ORDER BY d.id DESC LIMIT 1""",
        (session_id,),
    ).fetchone()
    if row is None or row["mid"] is None:
        return {"misconception_id": None, "others": 0}

    others = conn.execute(
        """SELECT COUNT(DISTINCT s.handle) AS n
             FROM diagnoses d
             JOIN sessions s ON s.id = d.session_id
            WHERE d.misconception_id = ? AND s.handle != ?""",
        (row["mid"], row["handle"]),
    ).fetchone()["n"]

    return {
        "misconception_id": row["mid"],
        "slug": row["slug"],
        "statement": row["statement"],
        "others": int(others),
    }


def student_history(conn: sqlite3.Connection, handle: str) -> list[dict]:
    """One student's diagnosed misconceptions over time, most repeated first.

    The point of the product's memory: a misconception seen three times across
    three different problems is a pattern, not a slip.
    """
    rows = conn.execute(
        """SELECT m.id, m.slug, m.canonical_statement, m.topic,
                  COUNT(*) AS times, MAX(d.created_at) AS last_seen
             FROM diagnoses d
             JOIN sessions s       ON s.id = d.session_id
             JOIN misconceptions m ON m.id = d.misconception_id
            WHERE s.handle = ?
            GROUP BY m.id
            ORDER BY times DESC, last_seen DESC""",
        (handle,),
    ).fetchall()
    return [dict(row) for row in rows]
