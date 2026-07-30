"""Remove sessions from the demo database, with their rows and their media.

Iterating on the render pipeline means running the same problem repeatedly, and
every run leaves a full session behind: rows in six tables plus a media directory
holding every attempt's workspace and video. Left alone they distort the one thing
in the app that counts across sessions -- "N other students made this error" -- so
a dozen of my own retries would read as a dozen students.

Dry run by default. `--apply` is the gate, because this deletes rendered video that
cost real model calls to produce.

    uv run python scripts/prune_sessions.py --keep 04300a00 --keep 12a959d5
    uv run python scripts/prune_sessions.py --keep ... --apply

Ids may be given as prefixes, which is how they are read off the console.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import get_settings  # noqa: E402
from server.store import db  # noqa: E402

# Every table holding a session_id. Ordered children-first so a partial run cannot
# leave a diagnosis with no session, only a session with no diagnosis, which is a
# state the app already handles.
CHILD_TABLES = ("chat_messages", "beats", "renders", "diagnoses", "run_artifacts")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="append", default=[], help="session id or prefix to keep")
    parser.add_argument("--handle", action="append", default=[], help="keep every session of this")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    conn = db.connect(settings.db_path)
    rows = list(conn.execute("select id, handle, status from sessions order by created_at"))

    def kept(row) -> bool:
        return row["handle"] in args.handle or any(row["id"].startswith(k) for k in args.keep)

    keep = [r for r in rows if kept(r)]
    drop = [r for r in rows if not kept(r)]

    unmatched = [k for k in args.keep if not any(r["id"].startswith(k) for r in rows)]
    if unmatched:
        # A typo in a --keep id would silently delete the session it was meant to
        # protect, which is exactly the mistake this script must not make quietly.
        print(f"no session matches: {', '.join(unmatched)}")
        return 1
    if not keep:
        print("refusing to empty the database; pass at least one --keep that matches")
        return 1

    print(f"keeping {len(keep)}:")
    for row in keep:
        print(f"  {row['id']}  {row['handle']}")

    def count(table: str, session_id: str) -> int:
        sql = f"select count(*) from {table} where session_id = ?"  # noqa: S608 -- fixed list
        return conn.execute(sql, (session_id,)).fetchone()[0]

    print(f"\ndropping {len(drop)}:")
    for row in drop:
        counts = ", ".join(f"{t}={count(t, row['id'])}" for t in CHILD_TABLES)
        media = Path(settings.media_root) / row["id"]
        print(f"  {row['id']}  {row['handle']:<18} {counts}{'  +media' if media.exists() else ''}")

    # Sessions being dropped count as known, not orphaned. Their media is removed
    # by the drop loop below, and listing it here as well both double-counts it in
    # the dry run and, with --apply, tries to remove the same directory twice --
    # the second `rmtree` raised FileNotFoundError and took the script's summary
    # line with it, after the deletions had already happened.
    known = {row["id"] for row in keep} | {row["id"] for row in drop}
    orphans = _orphan_media(Path(settings.media_root), known)
    if orphans:
        print(f"\norphaned media, no session row ({len(orphans)}):")
        for path in orphans:
            print(f"  {path}")

    if not args.apply:
        print("\ndry run; pass --apply to delete")
        return 0

    for row in drop:
        for table in CHILD_TABLES:
            conn.execute(f"delete from {table} where session_id = ?", (row["id"],))  # noqa: S608
        conn.execute("delete from sessions where id = ?", (row["id"],))
        media = Path(settings.media_root) / row["id"]
        if media.exists():
            shutil.rmtree(media)
    for path in orphans:
        shutil.rmtree(path)

    print(f"\ndeleted {len(drop)} sessions and {len(orphans)} orphaned media directories")
    return 0


def _orphan_media(media_root: Path, live: set[str]) -> list[Path]:
    """Media directories with no session row at all.

    These outlive the sessions they belonged to: a database reset, or a session
    deleted by hand, leaves the workspace and its videos behind with nothing
    pointing at them. Three were sitting in `media/` when this script was written,
    from before it existed.

    Directories whose names begin with an underscore are skipped. They are tooling
    output rather than sessions -- `scripts/check_primitives.py` renders into
    `media/_primitive_check` through the same runner, so it has no session row by
    design and must not be swept away as an orphan.
    """
    if not media_root.exists():
        return []
    return sorted(
        path
        for path in media_root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and path.name not in live
    )


if __name__ == "__main__":
    raise SystemExit(main())
