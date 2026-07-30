"""The arithmetic that decides how long the video becomes.

Nothing here runs ffmpeg. `plan_holds` and `shift` are pure on purpose -- the
decision about where every beat lands in the published video is the part a wrong
answer breaks silently, because a shifted beat still plays, it just plays the
wrong picture when a citation seeks to it. `build_filter` is asserted as a string
for the same reason `mux.build_command` is: it is the one place a boundary
off-by-one turns into a dropped or duplicated segment.
"""

import re

from server.audio import pad


def _beats(*spans: tuple[float, float]) -> list[pad.Beat]:
    return [pad.Beat(f"b{i}", start, end) for i, (start, end) in enumerate(spans, start=1)]


def test_a_line_that_fits_needs_no_hold():
    """The common case before this existed, and it must stay free: no holds means
    no re-encode, so the video stream is still copied rather than rebuilt."""
    beats = _beats((0.0, 8.0))
    assert pad.plan_holds(beats, {"b1": 5.0}) == []


def test_a_line_longer_than_its_beat_holds_the_frame_open():
    beats = _beats((0.0, 6.0))
    (hold,) = pad.plan_holds(beats, {"b1": 9.0}, tail_s=0.5)
    assert hold.beat_id == "b1"
    assert hold.at_s == 6.0, "the cut is at the beat's end, on the original timeline"
    assert hold.seconds == 3.5, "the shortfall, plus the tail silence"


def test_a_beat_with_no_clip_is_left_alone():
    """A line that failed to synthesise leaves silence, and silence needs no room.
    Padding for it would freeze the picture over nothing."""
    beats = _beats((0.0, 6.0), (6.0, 12.0))
    holds = pad.plan_holds(beats, {"b2": 9.0})
    assert [hold.beat_id for hold in holds] == ["b2"]


def test_a_hold_shifts_later_beats_but_not_its_own_start():
    """The property citations depend on. A beat's hold sits at its end, so it
    extends that beat; only the beats after it move."""
    beats = _beats((0.0, 6.0), (6.0, 12.0), (12.0, 18.0))
    holds = pad.plan_holds(beats, {"b1": 9.0}, tail_s=0.0)
    shifted = pad.shift(beats, holds)

    assert (shifted[0].start, shifted[0].end) == (0.0, 9.0), "b1 grows in place"
    assert (shifted[1].start, shifted[1].end) == (9.0, 15.0), "b2 moves by b1's hold"
    assert (shifted[2].start, shifted[2].end) == (15.0, 21.0)


def test_holds_accumulate_across_beats():
    beats = _beats((0.0, 6.0), (6.0, 12.0), (12.0, 18.0))
    holds = pad.plan_holds(beats, {"b1": 8.0, "b2": 10.0}, tail_s=0.0)
    shifted = pad.shift(beats, holds)
    assert (shifted[2].start, shifted[2].end) == (18.0, 24.0), "delayed by 2.0 + 4.0"


def test_one_runaway_line_is_capped_rather_than_freezing_the_frame():
    beats = _beats((0.0, 6.0))
    (hold,) = pad.plan_holds(beats, {"b1": 90.0}, max_hold_s=8.0)
    assert hold.seconds == 8.0


def test_holds_are_scaled_together_when_every_line_overran():
    """Scaled rather than truncated in order: capping each in turn would spend the
    whole allowance on the first beats and leave the last ones unpadded, so the
    video would slow down and then abruptly stop doing so."""
    beats = _beats((0.0, 4.0), (4.0, 8.0), (8.0, 12.0))
    holds = pad.plan_holds(
        beats, {"b1": 14.0, "b2": 14.0, "b3": 14.0}, tail_s=0.0, max_total_s=15.0
    )
    assert sum(hold.seconds for hold in holds) == 15.0
    assert len({hold.seconds for hold in holds}) == 1, "equal overruns, equal holds"


def test_a_sub_frame_shortfall_is_not_worth_a_segment_boundary():
    beats = _beats((0.0, 6.0))
    assert pad.plan_holds(beats, {"b1": 5.8}, tail_s=0.1) == []


def test_the_filter_cuts_at_beat_ends_and_starts_from_zero():
    """Segment one runs from 0 rather than from the first beat's start, so anything
    the scene drew outside a beat is carried through instead of dropped."""
    beats = _beats((1.5, 6.0), (6.0, 12.0))
    holds = pad.plan_holds(beats, {"b1": 9.0}, tail_s=0.0)
    chains = pad.build_filter(beats, holds, duration_s=12.0, fps=15).split(";")

    assert chains[0].startswith("[0:v]trim=start=0.000:end=6.000")
    # b1 is on screen for 4.5s (1.5 to 6.0), so a 9.0s line is 4.5s short. The
    # segment still starts at 0: the hold is sized by the beat, the cut is not.
    assert "tpad=stop_mode=clone:stop_duration=4.500" in chains[0]
    assert chains[1].startswith("[0:v]trim=start=6.000:end=12.000")
    assert "tpad" not in chains[1], "b1 overran, b2 did not"
    assert chains[-1] == "[v0][v1]concat=n=2:v=1:a=0[padded]"


def test_footage_after_the_last_beat_is_kept():
    """A closing FadeOut lives past the final beat's end. Dropping it would cut the
    animation off mid-fade."""
    beats = _beats((0.0, 6.0))
    filtered = pad.build_filter(beats, [], duration_s=9.0, fps=15)
    assert "trim=start=6.000,setpts" in filtered
    assert "concat=n=2" in filtered


def test_a_final_beat_that_ends_on_the_last_frame_adds_no_tail_segment():
    beats = _beats((0.0, 6.0))
    filtered = pad.build_filter(beats, [], duration_s=6.0, fps=15)
    assert "concat=n=1" in filtered


def test_every_chain_pins_the_frame_rate():
    """`concat` requires its inputs to agree on frame rate, and a boundary landing
    between frames can otherwise leave one chain a frame short."""
    beats = _beats((0.0, 6.0), (6.0, 12.0))
    chains = pad.build_filter(beats, [], duration_s=12.0, fps=15)
    assert chains.count("fps=15") == 2


def test_the_encode_keeps_keyframes_dense_enough_to_seek():
    """The beat rail seeks to arbitrary timestamps. x264's default 250-frame GOP is
    one keyframe every 16.7s at this render's 15fps -- measured on a real padded
    render, which came back with three keyframes in 44 seconds."""
    beats = _beats((0.0, 6.0))
    holds = pad.plan_holds(beats, {"b1": 9.0})
    cmd = pad.build_command(
        __import__("pathlib").Path("in.mp4"),
        beats,
        holds,
        __import__("pathlib").Path("out.mp4"),
        duration_s=6.0,
        fps=15,
    )
    assert cmd[cmd.index("-g") + 1] == "15"
    assert "-an" in cmd, "the source is silent and the audio is muxed in afterwards"


def test_the_command_names_the_padded_output_stream():
    beats = _beats((0.0, 6.0))
    holds = pad.plan_holds(beats, {"b1": 9.0})
    cmd = pad.build_command(
        __import__("pathlib").Path("in.mp4"),
        beats,
        holds,
        __import__("pathlib").Path("out.mp4"),
        duration_s=6.0,
        fps=15,
    )
    assert cmd[cmd.index("-map") + 1] == "[padded]"
    assert re.search(r"\[padded\]$", cmd[cmd.index("-filter_complex") + 1])
