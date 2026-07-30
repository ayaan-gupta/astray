"""Ask a finished session a few questions, so a demo opens with chat already in it.

Chat history is what shows that grounding works: a reply citing `[beat:b3]`
becomes a chip that seeks the player to that beat's measured start. A session with
an empty chat panel cannot demonstrate it, and typing questions live on camera
spends twenty seconds each on a model call.

These are real calls against the real tutor, with citations validated exactly as
the HTTP route validates them, so the stored replies are the ones the product
would actually produce. Nothing here is fabricated: an invented transcript would
show chips pointing at beats no one checked.

    uv run python scripts/seed_chat.py <session-id> [--question Q]...
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import get_settings  # noqa: E402
from server.deps import build_llm_client  # noqa: E402
from server.store import db, repo  # noqa: E402
from server.tutor import chat  # noqa: E402

# Ordered so the transcript builds: what went wrong, where the animation proves
# it, then how to avoid repeating it. The middle one is the question worth reading
# aloud, because answering it well requires distinguishing the beat that *shows*
# the student's rule from the beat that *refutes* it.
DEFAULT_QUESTIONS = [
    "I still do not see what I did wrong. Can you explain it simply?",
    "Where exactly does the animation show my rule failing?",
    "How do I stop making this mistake next time?",
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("--question", action="append", dest="questions")
    args = parser.parse_args()

    settings = get_settings()
    conn = db.connect(settings.db_path)

    if repo.get_session(conn, args.session_id) is None:
        print("no such session")
        return 1
    existing = repo.list_chat(conn, args.session_id)
    if existing:
        print(f"session already has {len(existing)} messages; not adding more")
        return 1

    client = build_llm_client(settings)
    spent = 0.0
    try:
        for question in args.questions or DEFAULT_QUESTIONS:
            reply, cited, meta = await chat.answer(
                conn,
                client,
                session_id=args.session_id,
                question=question,
                model=settings.deepseek_model_fast,
            )
            spent += meta.cost_usd or 0.0
            print(f"\nQ: {question}\nA: {reply}\n   cites {cited or 'nothing'}")
    finally:
        await client.aclose()

    print(f"\n{len(repo.list_chat(conn, args.session_id))} messages stored, ${spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
