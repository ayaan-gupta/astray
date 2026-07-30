"""s6's runtime estimate, and the per-beat number s7 is given instead of a total.

The estimate is the only pacing signal `s7` receives, and `s7` satisfies it with
`self.wait()`, which makes an inflated estimate actively harmful: it buys dead air
rather than explanation. Measured across three live runs, `s6` asked for 130s over
5 beats, 180s over 4, and 240s over 6, all outside the "typically 45-120" its own
prompt states. The 180s one rendered as four beats each holding a static frame for
30 to 40 seconds, a 155s video.

So the estimate is clamped, and the number handed to `s7` is per beat.
"""

import httpx
import pytest

from server.charter.contracts import Beat, MathContent, SceneCode, Storyboard
from server.charter.stages.s6_visual import (
    MAX_SECONDS_PER_BEAT,
    MIN_SECONDS_PER_BEAT,
    clamp_runtime,
)
from server.charter.stages.s7_scene import compose_scene
from server.llm.deepseek import DeepSeekClient


def _board(count: int, seconds: int) -> Storyboard:
    return Storyboard(
        beats=[
            Beat(
                id=f"b{i}",
                title=f"title {i}",
                teaching_purpose="p",
                on_screen="o",
                targets_misconception=i == 1,
                primitive="algebra_steps",
            )
            for i in range(1, count + 1)
        ],
        total_estimated_seconds=seconds,
    )


class TestClampRuntime:
    def test_a_sensible_estimate_is_returned_unchanged(self):
        board = _board(5, 5 * 10)
        assert clamp_runtime(board) is board

    @pytest.mark.parametrize(("count", "asked"), [(4, 180), (6, 240), (5, 130), (3, 600)])
    def test_an_inflated_estimate_is_brought_down(self, count, asked):
        """The first three are estimates live runs actually produced."""
        clamped = clamp_runtime(_board(count, asked)).total_estimated_seconds
        assert clamped == MAX_SECONDS_PER_BEAT * count
        assert clamped < asked

    def test_a_starved_estimate_is_brought_up(self):
        clamped = clamp_runtime(_board(6, 10)).total_estimated_seconds
        assert clamped == MIN_SECONDS_PER_BEAT * 6

    def test_the_beats_themselves_are_never_touched(self):
        board = _board(4, 180)
        assert clamp_runtime(board).beats == board.beats

    def test_the_band_allows_a_beat_the_render_will_not_override(self):
        """A ceiling below the render's own five second hold would be unreachable."""
        assert MIN_SECONDS_PER_BEAT >= 5
        assert MIN_SECONDS_PER_BEAT < MAX_SECONDS_PER_BEAT


async def test_s7_is_given_seconds_per_beat_not_just_a_total():
    """The division happens here, not in the model.

    Handed only a total, the model divides by the beat count itself and then spends
    the quotient entirely on waits. A per-beat figure is a length it can reason
    about rather than a budget it feels obliged to exhaust.
    """
    scene = SceneCode(scene_class_name="AstrayScene", code="x = 1\n", beats_covered=["b1"])
    sent: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": scene.model_dump_json()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = DeepSeekClient("sk-test", transport=httpx.MockTransport(handler))
    try:
        await compose_scene(
            client,
            storyboard=_board(4, 48),
            math=MathContent(
                worked_example=["a"],
                counter_example=["b"],
                key_identity="c",
                concrete_numbers=["1"],
            ),
            model="deepseek-v4-pro",
        )
    finally:
        await client.aclose()

    assert "seconds_per_beat: 12" in sent["body"]
    assert "total_estimated_seconds: 48" in sent["body"]
