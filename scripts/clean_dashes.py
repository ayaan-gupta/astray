"""Apply the chat dash sanitiser to messages already in the database.

`strip_em_dashes` runs on the way in, so every reply written from now on is
clean. Rows persisted before it existed are not, and chat history is replayed to
the client verbatim on reload, so the seeded demo conversation would still show
em dashes on camera. This rewrites those rows in place.

Idempotent: sanitising clean text is a no-op, so re-running changes nothing.

    uv run python scripts/clean_dashes.py [--apply]

Without --apply it only reports what would change.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.tutor.chat import strip_em_dashes  # noqa: E402

DB = Path(__file__).resolve().parent.parent / "data" / "tutor.db"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--db", type=Path, default=DB)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"no database at {args.db}")
        return 1

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, role, content FROM chat_messages ORDER BY id").fetchall()

    changed = []
    for row in rows:
        cleaned = strip_em_dashes(row["content"])
        if cleaned != row["content"]:
            changed.append((row["id"], row["role"], row["content"], cleaned))

    if not changed:
        print(f"{len(rows)} messages, none need changing")
        return 0

    for mid, role, before, after in changed:
        print(f"\n--- id {mid} ({role})")
        for b, a in zip(before.split("\n"), after.split("\n"), strict=False):
            if b != a:
                print(f"  - {b}")
                print(f"  + {a}")

    if not args.apply:
        print(f"\n{len(changed)} of {len(rows)} messages would change; re-run with --apply")
        return 0

    with conn:
        conn.executemany(
            "UPDATE chat_messages SET content = ? WHERE id = ?",
            [(after, mid) for mid, _, _, after in changed],
        )
    print(f"\nrewrote {len(changed)} messages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
