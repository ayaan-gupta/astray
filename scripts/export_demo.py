"""Freeze finished sessions into a static site that needs no server at all.

The running app needs Python, a writable SQLite file and a Docker daemon to
render inside. A static host has none of those, so a live deployment of the real
thing is not on the table -- but nothing about *showing* it requires them. Every
demo session is finished: the diagnosis is stored, the beats carry measured
timings, the chat transcript is on disk and the video is an mp4.

The export goes through the app's **own endpoints**, not through SQL. Reading the
tables directly means re-implementing every serializer here, and the copy would
drift the first time a response shape changed -- silently, because the client
would still find a file at the path it asked for and just render less. Driving
the real handlers with a test client means the JSON on disk is by construction
the JSON the client already handles. Only GETs are called, so nothing is written
and nothing is billed.

    uv run python scripts/export_demo.py

Writes `public/`, which is committed -- the database and `media/` are gitignored,
so a build step on the host would have nothing to read. Re-run and re-commit
after re-rendering a demo session.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from server.app import create_app  # noqa: E402

# The three sessions the demo is built around, in the order they should appear.
# Ids are full, because this is a committed artifact rather than a console tool
# and a prefix that silently matched the wrong run would ship as the demo.
DEMO_SESSIONS = (
    "bb6a4531-dbde-4d4e-953c-6237dd1e3245",  # d/dx sin(x^2) -- the lifted composition
    "fa8f9e92-220d-4345-b9e5-484a228fb6ff",  # (y+3)^2 -- the two surfaces
    "5901c1da-fe6b-4e55-a4f6-31bd5834cb38",  # x^2 = 16 -- deliberately flat
)

DEMO_HANDLE = "astray-final"

# Route -> file, matching `demoPath()` in web/app.js. Kept as one list so the two
# sides can be compared by eye: every entry here is a fetch the client makes.
PER_SESSION = (
    ("", "session.json"),
    ("/beats", "beats.json"),
    ("/chat", "chat.json"),
    ("/peers", "peers.json"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="public", help="output directory")
    args = parser.parse_args()
    out = Path(args.out)

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree("web", out)
    _mark_demo(out / "index.html")
    (out / "demo").mkdir()

    app = create_app()
    with TestClient(app) as client:
        for sid in DEMO_SESSIONS:
            into = out / "demo" / sid
            into.mkdir(parents=True)
            for suffix, name in PER_SESSION:
                _write(into / name, _get(client, f"/api/sessions/{sid}{suffix}"))
            _copy_video(client, sid, into)
            # The url the client is handed has to move with the file. Rewritten
            # here rather than in the client, because where the video sits is a
            # fact about this export and not something the player should know.
            beats = json.loads((into / "beats.json").read_text())
            beats["video_url"] = f"demo/{sid}/video.mp4"
            _write(into / "beats.json", beats)

        # Filtered to what was actually exported. The live endpoint returns every
        # session this handle owns, and the extras are real rows -- earlier runs
        # of the same problems -- so an unfiltered list looks correct and gives a
        # row that 404s the moment it is clicked.
        listing = _get(client, f"/api/sessions?handle={DEMO_HANDLE}")
        listing["sessions"] = [s for s in listing["sessions"] if s["session_id"] in DEMO_SESSIONS]
        _write(out / "demo" / "sessions.json", listing)
        _write(out / "demo" / "insights.json", _get(client, f"/api/insights?handle={DEMO_HANDLE}"))

    print(f"wrote {out}/ ({len(DEMO_SESSIONS)} sessions, {_size(out)})")
    return 0


def _get(client, path):
    response = client.get(path)
    if response.status_code != 200:
        raise SystemExit(f"{path} returned {response.status_code}: {response.text[:200]}")
    return response.json()


def _copy_video(client, sid: str, into: Path) -> None:
    """Pull the mp4 through the same handler that serves it live.

    Going through the endpoint rather than reading `renders.video_path` means the
    export cannot ship a video the running app would not have served -- a stale
    row pointing at a deleted file 404s here instead of silently exporting
    nothing.
    """
    response = client.get(f"/media/{sid}/video.mp4")
    if response.status_code != 200:
        raise SystemExit(f"no video for {sid}: {response.status_code}")
    (into / "video.mp4").write_bytes(response.content)


def _mark_demo(index_html: Path) -> None:
    """Set the flag ahead of app.js.

    It has to be a real script tag rather than a query string or a hostname
    check: app.js reads `window.ASTRAY_DEMO` at module scope, and `defer` runs
    scripts in document order, so an inline tag placed before it is guaranteed to
    have executed first.
    """
    html = index_html.read_text()
    marker = '<script defer src="/app.js"></script>'
    if marker not in html:
        raise SystemExit("index.html no longer loads /app.js the way export_demo expects")
    html = html.replace(marker, f"<script>window.ASTRAY_DEMO=true;</script>\n{marker}")
    index_html.write_text(html)


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1))


def _size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1_048_576:.1f} MB"


if __name__ == "__main__":
    raise SystemExit(main())
