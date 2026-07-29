"""Deterministic fallback renderer -- no LLM-authored code at all.

When codegen fails twice, this builds the scene file itself from the storyboard
and the s5 math content, using only the vetted primitives. Lower production
value, still correct, and critically **still fully grounded**: same beat ids,
same `beat()` instrumentation, so the manifest and every chat citation work
exactly as they do for a generated scene.

That last property is why this exists rather than showing an error card. A
student mid-flow should never dead-end on a spinner, and a session that falls
back must not silently lose the grounding the tutor depends on -- the difference
between generated and fallback is recorded in `renders.mode` so the quality gap
is measurable rather than invisible.

The output is Python source built by string assembly from data that reached us
through a model, so every interpolated value goes through `_py_str`, which emits
a Python string literal via `repr`. Naive quoting here would be an injection hole
that hands arbitrary code to the container -- the fallback path must not be the
weakest link in a design whose whole point is that generated code is untrusted.
"""

from server.charter.contracts import MathContent, SceneCode, Storyboard

SCENE_CLASS_NAME = "AstrayScene"


def _py_str(value: str) -> str:
    """Render `value` as a safe Python string literal.

    `repr` handles quotes, backslashes (LaTeX is full of them), newlines and
    non-printables. Never use f-string interpolation with bare quotes here.
    """
    return repr(value)


def _py_list(values: list[str]) -> str:
    return "[" + ", ".join(_py_str(v) for v in values) + "]"


def build_scene(storyboard: Storyboard, math: MathContent) -> SceneCode:
    """Compose a runnable scene covering every planned beat, deterministically."""
    lines = [
        "from manim import FadeIn, FadeOut, Scene, Write",
        "",
        "from primitives.algebra_steps import compare_rules, step_sequence",
        "from primitives.beats import beat",
        "from primitives.layout import math_lines, title_card",
        "",
        "",
        f"class {SCENE_CLASS_NAME}(Scene):",
        "    def construct(self):",
    ]

    for index, item in enumerate(storyboard.beats):
        lines.append(f"        with beat(self, {_py_str(item.id)}):")
        if item.targets_misconception:
            # The one beat that must argue with the student: their rule crossed
            # out beside the correct one.
            lines += [
                f"            compare_rules(self, {_py_list(math.counter_example or ['?'])},"
                f" {_py_list(math.worked_example or ['?'])})",
                "            self.wait(1.2)",
            ]
        elif index == 0:
            lines += [
                f"            card = title_card({_py_str(item.title)},"
                f" {_py_str(item.teaching_purpose[:80])})",
                "            self.play(FadeIn(card), run_time=1.0)",
                "            self.wait(0.8)",
                "            self.play(FadeOut(card))",
            ]
        elif index == len(storyboard.beats) - 1:
            lines += [
                f"            closing = math_lines([{_py_str(math.key_identity)}])",
                "            self.play(Write(closing), run_time=1.2)",
                "            self.wait(1.5)",
                "            self.play(FadeOut(closing))",
            ]
        else:
            lines += [
                "            group = step_sequence(self, "
                f"{_py_list(math.worked_example or ['?'])})",
                "            self.wait(0.8)",
                "            self.play(FadeOut(group))",
            ]

    return SceneCode(
        scene_class_name=SCENE_CLASS_NAME,
        code="\n".join(lines) + "\n",
        beats_covered=[item.id for item in storyboard.beats],
    )
