"""Run one whole session from the command line: ingest, diagnose, plan, render, narrate.

The same sequence `/api/sessions/{id}/stream` drives, minus HTTP. This exists
because the interesting failures in this pipeline are visual, and chasing them
through an SSE stream in a browser means re-uploading a problem by hand every
time. Here the problem is an argument and the progress is stdout.

    uv run python scripts/run_session.py "d/dx[sin(x^2)]" "= cos(x^2)"
    uv run python scripts/run_session.py --preset chain-rule

It is a real run: real model calls, real cost, a real container. Nothing is
mocked, because a mocked render cannot tell you whether the frame is readable.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.charter.chain import Chain  # noqa: E402
from server.charter.contracts import Diagnosis  # noqa: E402
from server.charter.pipeline import Pipeline  # noqa: E402
from server.charter.stages.s0_ingest import ingest_typed  # noqa: E402
from server.config import get_settings  # noqa: E402
from server.deps import build_llm_client  # noqa: E402
from server.store import db, repo  # noqa: E402

# Problems chosen to stress the parts of the renderer that plain algebra never
# reaches: a function graph, a signed number line, an area decomposition. Each is
# also a real, common misconception rather than a contrived error.
PRESETS = {
    # The demo's flagship misconception. Kept here because it is the one case with
    # a purpose-built primitive (`area.binomial_square`), so it is the regression
    # test for whether s6 actually reaches for a picture when a picture exists.
    "binomial": (
        r"Expand (y + 3)^{2}",
        r"(y + 3)^{2} = y^{2} + 9",
    ),
    "chain-rule": (
        r"Differentiate y = \sin(x^{2})",
        r"\frac{dy}{dx} = \cos(x^{2})",
    ),
    "log-sum": (
        r"Expand \log(x + y)",
        r"\log(x + y) = \log x + \log y",
    ),
    "sqrt-sum": (
        r"Simplify \sqrt{x^{2} + 9}",
        r"\sqrt{x^{2} + 9} = x + 3",
    ),
    "lost-root": (
        r"Solve x^{2} = 16",
        r"x = 4",
    ),
    "power-rule-int": (
        r"Evaluate \int 2x \cos(x^{2})\, dx",
        r"\int 2x \cos(x^{2})\, dx = x^{2} \sin(x^{2}) + C",
    ),
}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", nargs="?")
    parser.add_argument("work", nargs="?")
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--handle", default="cli")
    args = parser.parse_args()

    if args.preset:
        problem, work = PRESETS[args.preset]
    elif args.problem and args.work:
        problem, work = args.problem, args.work
    else:
        parser.error("give a problem and work, or --preset")

    settings = get_settings()
    conn = db.connect(settings.db_path)
    client = build_llm_client(settings)

    submission = ingest_typed(problem=problem, work=work, prose=None)
    session_id = repo.create_session(conn, handle=args.handle, submission=submission)
    print(f"session {session_id}\n  problem: {problem}\n  work:    {work}\n")

    if not repo.try_start_session(conn, session_id):
        print("could not claim the session")
        return 1

    spent = 0.0
    diagnosis: Diagnosis | None = None
    try:
        async for event in Chain(conn, client, settings=settings).run_diagnosis(
            session_id, submission
        ):
            spent += _report(event)
            if event.type == "diagnosis_ready" and event.payload:
                diagnosis = Diagnosis.model_validate(
                    {k: v for k, v in event.payload.items() if k != "misconception_id"}
                )

        if diagnosis is None or diagnosis.no_error_found:
            print("\nno misconception to animate")
            return 0

        async for event in Pipeline(conn, client, settings=settings).run(
            session_id, submission, diagnosis
        ):
            spent += _report(event)
    finally:
        await client.aclose()

    status = repo.get_session(conn, session_id)["status"]
    render = repo.latest_render(conn, session_id)
    print(f"\nstatus {status}  cost ${spent:.4f}")
    if render is not None:
        print(f"render {render['mode']} attempt {render['attempt']}: {render['video_path']}")
    print(f"\n  http://localhost:8000/#/session/{session_id}")
    return 0


def _report(event) -> float:
    payload = event.payload or {}
    cost = float(payload.get("cost_usd") or 0.0)
    detail = {k: v for k, v in payload.items() if k not in ("cost_usd", "reasoning")}
    print(f"[{event.type:>16}] {event.stage or '':<12} {str(detail)[:150]}")
    if event.message:
        print(f"                   {event.message}")
    return cost


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
