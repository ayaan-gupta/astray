"""s2-s8 plus rendering, as one resumable sequence with per-stage progress.

Kept separate from `chain.py` rather than bolted onto `run_diagnosis` for one
reason that matters at runtime: the diagnosis must reach the student the moment
it exists. It completes in ~10-40s; the rest of this pipeline adds a minute or
more. A single generator that produced the diagnosis and then kept going would
be fine, but a caller could not easily choose to stream only the first part --
and the frontend does exactly that, rendering the diagnosis card and opening
chat while the animation is still planning.

Progress events are emitted at the completion of every stage, even though the
consuming UI does not exist yet. They are the pipeline's only externally visible
account of a run that takes minutes; adding them later would mean revisiting
every stage boundary, and their absence is what makes a long pipeline feel
broken. Each carries the stage's cost so a client can show real spend, and its
`reasoning` so the loading state can be the model's actual thinking rather than
a spinner.

Failure policy is uniform and deliberate: any `LlmError` from a planning stage
ends the run with a terminal `error` event, because every later stage depends on
the one before it and there is nothing useful to render from a missing plan. A
*render* failure is different -- the plan is intact, so it degrades to the
deterministic storyboard fallback rather than failing the session.
"""

import logging
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

from server.charter.chain import ProgressEvent
from server.charter.contracts import Diagnosis, StageName, StudentSubmission
from server.charter.stages.s2_intent import analyse_intent
from server.charter.stages.s3_prereq import map_prerequisites
from server.charter.stages.s4_curriculum import build_curriculum
from server.charter.stages.s5_math import select_math
from server.charter.stages.s6_visual import plan_beats
from server.charter.stages.s7_scene import compose_scene
from server.config import Settings
from server.llm.deepseek import DeepSeekClient, LlmError
from server.render import repair, runner
from server.render import storyboard as fallback
from server.render.validator import validate_scene
from server.store import repo

logger = logging.getLogger(__name__)


class Pipeline:
    """Runs s2-s8 and the render for one already-diagnosed session."""

    def __init__(
        self, conn: sqlite3.Connection, client: DeepSeekClient, *, settings: Settings
    ) -> None:
        self._conn = conn
        self._client = client
        self._settings = settings

    async def run(
        self, session_id: str, submission: StudentSubmission, diagnosis: Diagnosis
    ) -> AsyncIterator[ProgressEvent]:
        """Plan, generate, validate and render. Never raises."""
        fast = self._settings.deepseek_model_fast
        pro = self._settings.deepseek_model_reasoning

        try:
            intent, meta = await self._stage(
                session_id,
                StageName.INTENT,
                analyse_intent(
                    self._client, submission=submission, diagnosis=diagnosis, model=fast
                ),
            )
            yield self._completed(StageName.INTENT, meta, {"knowledge_gap": intent.knowledge_gap})

            prereqs, meta = await self._stage(
                session_id,
                StageName.PREREQ,
                map_prerequisites(self._client, intent=intent, diagnosis=diagnosis, model=fast),
            )
            yield self._completed(
                StageName.PREREQ, meta, {"nodes": [n.concept for n in prereqs.nodes]}
            )

            curriculum, meta = await self._stage(
                session_id,
                StageName.CURRICULUM,
                build_curriculum(self._client, prereqs=prereqs, diagnosis=diagnosis, model=fast),
            )
            yield self._completed(
                StageName.CURRICULUM, meta, {"steps": [s.concept for s in curriculum.steps]}
            )

            math, meta = await self._stage(
                session_id,
                StageName.MATH,
                select_math(
                    self._client,
                    curriculum=curriculum,
                    diagnosis=diagnosis,
                    problem=submission.problem,
                    model=fast,
                ),
            )
            yield self._completed(StageName.MATH, meta, {"key_identity": math.key_identity})

            board, meta = await self._stage(
                session_id,
                StageName.VISUAL,
                plan_beats(self._client, math=math, diagnosis=diagnosis, model=fast),
            )
            # Persisted before the render so the beat rail can show the plan while
            # the video is still rendering -- untimed beats are a real UI state,
            # not a placeholder.
            repo.save_beats(self._conn, session_id, board)
            yield self._completed(
                StageName.VISUAL,
                meta,
                {"beats": [{"id": b.id, "title": b.title} for b in board.beats]},
            )

            scene, meta = await self._stage(
                session_id,
                StageName.SCENE,
                compose_scene(self._client, storyboard=board, math=math, model=pro),
            )
            yield self._completed(StageName.SCENE, meta, {"scene_class": scene.scene_class_name})
        except LlmError as exc:
            logger.warning("pipeline planning failed for %s: %s", session_id, exc)
            repo.set_session_status(self._conn, session_id, "failed")
            # Upstream text is never forwarded (it has carried an auth header
            # before); the detail is logged server-side only.
            yield ProgressEvent(
                type="error", stage=StageName.SCENE, message="could not plan the animation"
            )
            return

        async for event in self._render_loop(session_id, scene, board, math):
            yield event

    async def _render_loop(self, session_id, scene, board, math) -> AsyncIterator[ProgressEvent]:
        """Validate -> render -> repair, bounded, then the deterministic fallback."""
        attempt = 0
        max_attempts = self._settings.render_max_repairs + 1

        while attempt < max_attempts:
            attempt += 1
            report = validate_scene(scene, board)
            yield ProgressEvent(
                type="stage_completed",
                stage=StageName.VALIDATE,
                payload={"ok": report.ok, "attempt": attempt, "issues": len(report.issues)},
            )

            if report.ok:
                result = await self._render(session_id, scene, attempt, mode="generated")
                if result.ok:
                    yield self._render_done(result, attempt)
                    return
                failure = result.error_text or "render produced no video"
            else:
                # A scene that fails static validation is never executed: the
                # container is the second layer, not the first line of defence.
                failure = report.failure_text()

            if attempt >= max_attempts:
                break
            try:
                scene, meta = await self._stage(
                    session_id,
                    StageName.SCENE,
                    repair.repair_scene(
                        self._client,
                        scene=scene,
                        storyboard=board,
                        math=math,
                        failure=failure,
                        model=self._settings.deepseek_model_reasoning,
                    ),
                    attempt=attempt + 1,
                )
                yield self._completed(
                    StageName.SCENE, meta, {"repair_attempt": attempt, "reason": failure[:200]}
                )
            except LlmError:
                break

        # Bounded attempts exhausted. The plan is still good, so render it with
        # no model-authored code at all: same beat ids, same manifest, so chat
        # grounding survives a codegen failure intact.
        logger.info("falling back to deterministic storyboard render for %s", session_id)
        deterministic = fallback.build_scene(board, math)
        result = await self._render(
            session_id, deterministic, attempt + 1, mode="storyboard_fallback"
        )
        if result.ok:
            yield self._render_done(result, attempt + 1)
            return

        repo.set_session_status(self._conn, session_id, "render_failed")
        yield ProgressEvent(
            type="error", stage=StageName.VALIDATE, message="could not render the animation"
        )

    async def _render(self, session_id: str, scene, attempt: int, *, mode: str):
        paths = runner.prepare(
            Path(self._settings.media_root), session_id, scene.code, attempt=attempt
        )
        result = await runner.run(
            paths,
            scene.scene_class_name,
            timeout_s=self._settings.render_timeout_s,
            mode=mode,
        )
        repo.record_render(
            self._conn,
            session_id=session_id,
            attempt=attempt,
            status="ok" if result.ok else "failed",
            duration_s=result.duration_s,
            error_text=None if result.ok else (result.error_text or "")[:4000],
            video_path=result.video_path,
            mode=mode,
        )
        if result.ok:
            # Measured timings, from the container's own clock. A citation that
            # seeks to an estimated timestamp points at the wrong moment, which
            # is worse than not citing.
            repo.save_beat_timings(self._conn, session_id, result.timings)
            repo.set_session_status(self._conn, session_id, "ready")
        return result

    def _render_done(self, result, attempt: int) -> ProgressEvent:
        return ProgressEvent(
            type="render_complete",
            stage=StageName.VALIDATE,
            payload={
                "mode": result.mode,
                "attempt": attempt,
                "duration_s": round(result.duration_s, 1),
                "video_path": result.video_path,
                "timings": [t.model_dump() for t in result.timings],
            },
        )

    async def _stage(self, session_id: str, stage: StageName, coro, attempt: int = 1):
        """Await one stage and persist its artifact with full cost provenance."""
        value, meta = await coro
        repo.record_artifact(
            self._conn,
            session_id=session_id,
            stage=stage,
            payload=value.model_dump(),
            meta=meta,
            attempt=attempt,
        )
        return value, meta

    def _completed(self, stage: StageName, meta, payload: dict) -> ProgressEvent:
        return ProgressEvent(
            type="stage_completed",
            stage=stage,
            payload={**payload, "cost_usd": meta.cost_usd, "reasoning": meta.reasoning},
        )
