"""Step-sequence and rule-comparison primitives. Container side.

`compare_rules` is the one that carries the product's thesis: it puts the
student's buggy rule and the correct rule side by side on the same input, so the
animation contradicts their rule rather than merely demonstrating the right one.
An explanation that only shows the correct method leaves the student's own rule
untouched -- they watch it, agree with it, and keep their misconception.
"""

from manim import (
    DOWN,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    Create,
    Cross,
    FadeIn,
    Text,
    VGroup,
    Write,
)
from primitives.layout import clear_frame, fit, math_lines


def step_sequence(scene, lines: list[str], run_time: float = 0.8):
    """Reveal math lines one at a time, top to bottom. Returns the VGroup.

    Clears the frame first: a beat drawn over the previous beat's leftovers is
    unreadable, and relying on generated code to tidy up was observed to fail.
    """
    clear_frame(scene)
    group = math_lines(lines)
    for line in group:
        scene.play(Write(line), run_time=run_time)
    return group


def compare_rules(
    scene,
    wrong_lines: list[str],
    right_lines: list[str],
    wrong_label: str = "What you did",
    right_label: str = "What the rule actually gives",
    run_time: float = 0.8,
):
    """Show the buggy derivation and the correct one side by side, and cross out
    the buggy one.

    The cross is not decoration: it is the moment the student's rule is
    explicitly rejected on screen. Returns (wrong_group, right_group).
    """
    clear_frame(scene)
    wrong = VGroup(Text(wrong_label, font_size=26, color=RED), math_lines(wrong_lines, 34)).arrange(
        DOWN, buff=0.3
    )
    right = VGroup(
        Text(right_label, font_size=26, color=GREEN), math_lines(right_lines, 34)
    ).arrange(DOWN, buff=0.3)

    columns = VGroup(wrong, right).arrange(RIGHT, buff=1.0, aligned_edge=UP)
    fit(columns)

    scene.play(FadeIn(wrong.shift(LEFT * 0.0)), run_time=run_time)
    scene.play(Create(Cross(wrong[1], color=RED, stroke_width=6)), run_time=run_time)
    scene.play(FadeIn(right), run_time=run_time)
    return wrong, right
