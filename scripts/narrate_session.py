"""Narrate an already-rendered session, without re-running the pipeline.

The pipeline narrates automatically after a successful render. This exists for the
session that was rendered before narration did, or one whose narration was skipped
because the TTS API was unreachable at the time: it picks up an existing video and
its already-measured beat timings and only does the audio half.

    uv run python scripts/narrate_session.py <session-id>

It is also how a narration prompt change is tried out, since the script costs one
model call and a few seconds of TTS against a render that already exists.

Publishes over the render's own path and keeps the untouched original beside it as
`silent.mp4`. That is what makes it safe to re-run: a second pass reads the silent
copy rather than handing ffmpeg its own previous output, and the URL the page
already serves keeps working.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.audio.pipeline import add_narration  # noqa: E402
from server.config import get_settings  # noqa: E402
from server.deps import build_llm_client  # noqa: E402
from server.store import repo  # noqa: E402
from server.store.db import connect  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.narration_available:
        print("narration is unavailable: set FISH_API_KEY in server/.env")
        return 1

    conn = connect(settings.db_path)
    render = repo.latest_render(conn, args.session_id)
    if render is None or not render["video_path"]:
        print(f"no successful render for {args.session_id}")
        return 1

    timed = [r for r in repo.list_beats(conn, args.session_id) if r["start_s"] is not None]
    print(f"video : {render['video_path']}")
    print(f"beats : {len(timed)} with measured timings")

    llm = build_llm_client(settings)
    try:
        result = await add_narration(
            conn,
            session_id=args.session_id,
            video_path=render["video_path"],
            settings=settings,
            llm=llm,
        )
    finally:
        await llm.aclose()

    if not result.ok:
        print(f"not narrated: {result.skipped_reason}")
        return 1

    conn.commit()
    print(f"\nnarrated -> {result.video_path}")
    print(f"lines    : {result.lines}")
    print(f"cost     : ${result.cost_usd:.4f} ({result.characters} characters)")
    print(f"drift    : {result.worst_drift_s:.2f}s worst case")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
