"""Turning a model-written sentence into plain text. Container side, no manim.

Split out of `layout.py` for the same reason as `sampling.py`: `manim` exists only
inside the render image, so nothing importing it can be tested on the host, and
the substitutions below are exactly the kind of rule that is quietly wrong on real
input. `\\neq` deleted rather than translated turns "so 16 \\neq 10" into "so 16
10", which reads as a claim rather than as a formatting fault.
"""

import re

# LaTeX a one-line caption actually contains, mapped to the characters a reader
# wants. `Text` has no LaTeX at all, so anything not translated here is typeset
# literally, backslash and all.
_LATEX_PROSE = {
    r"\neq": "≠",
    r"\ne": "≠",
    r"\leq": "≤",
    r"\le": "≤",
    r"\geq": "≥",
    r"\ge": "≥",
    r"\pm": "±",
    r"\mp": "∓",
    r"\times": "×",
    r"\cdot": "·",
    r"\div": "÷",
    r"\sqrt": "√",
    r"\approx": "≈",
    r"\rightarrow": "→",
    r"\to": "→",
    r"\ldots": "...",
    r"\dots": "...",
}

# Superscript digits, so `x^2` reads as `x²` rather than as `x^2`.
_SUPERSCRIPTS = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")


# Words that are mathematics even though they are made of letters. Left alone so
# `\sin` and `\log` keep their upright maths shape rather than being re-wrapped.
_MATH_WORDS = frozenset(
    """sin cos tan sec csc cot sinh cosh tanh arcsin arccos arctan log ln exp lim
    max min sup inf det gcd lcm mod deg dim ker arg""".split()
)

# A run of letters long enough to be a word, not preceded by a backslash (which
# would make it a LaTeX command) and not inside braces we already wrote.
_WORD = re.compile(r"(?<![\\A-Za-z{])([A-Za-z][A-Za-z']{2,})(\s*)")


def mathify(line: str) -> str:
    r"""Wrap prose words in `\text{}` so `MathTex` renders them as words.

    Math mode has no spaces: every gap in the source is discarded and the glyphs
    are set as a product of variables. So a line of prose handed to `MathTex` comes
    out as one run of italics with the spaces gone. Live renders produced `Lety = 1`
    and `Thestudent'srulefails` from "Let y = 1" and "The student's rule fails" --
    both readable in the source, both unreadable on screen.

    Only runs of three or more letters are wrapped. Shorter ones are the variables
    this is protecting: `ab` must stay a product, and a lone `y` must stay italic.
    That is also why the trailing whitespace goes *inside* the braces -- `\text{Let}
    y` renders "Lety", because the space between the group and the variable is math
    mode again.

    `legend` solves the same problem a different way, by choosing `Text` over
    `MathTex` for a whole label. That works for a label, which is one or the other.
    A line is routinely both, and `\text{}` is how LaTeX says so.
    """

    def wrap(match: re.Match) -> str:
        word, trailing = match.group(1), match.group(2)
        if word.lower() in _MATH_WORDS:
            return match.group(0)
        return f"\\text{{{word}{' ' if trailing else ''}}}"

    return _WORD.sub(wrap, line)


def prose(text: str) -> str:
    r"""Render a caller-supplied sentence as plain text a reader can actually read.

    `Text` has no LaTeX, and a model writing a caption *about mathematics* reaches
    for LaTeX by habit however firmly the prompt says not to. Two live captions:
    *"The two $3y$ rectangles are missing!"* with the dollar signs on screen, and
    *"For y=1, (1+3)^2=16, but 1^2+3^2=10, so 16\neq10."* with the caret and the
    command both typeset verbatim.

    Translating beats stripping here. Deleting `\neq` would turn that second caption
    into "so 16 10", which is worse than the raw command: it silently removes the
    claim. So the handful of constructs a one-line caption contains are mapped to
    the characters they mean, and only the leftovers are dropped.

    Longer sequences are matched before their own prefixes (`\neq` before `\ne`),
    which is why the replacement is ordered by length rather than by the dict.
    """
    out = text.replace("$", "")
    out = re.sub(r"\\[()\[\]]", "", out)

    for command in sorted(_LATEX_PROSE, key=len, reverse=True):
        out = out.replace(command, _LATEX_PROSE[command])

    # `^2`, `^{2}`, `^{-1}`, `^n`. Anything else superscripted falls to the sweep
    # below.
    #
    # The sign is allowed only in the leading position, and that is not pedantry:
    # a character class of `[0-9+-n]+` matched `^2+3` in "1^2+3^2=10" and rendered
    # it "1²⁺³²=10", turning a sum of two squares into one number.
    out = re.sub(
        r"\^\{?([+-]?[0-9]+|n)\}?",
        lambda m: m.group(1).translate(_SUPERSCRIPTS),
        out,
    )

    # Whatever LaTeX is left was not worth a mapping; braces and stray backslash
    # commands read as noise, and the words around them still carry the sentence.
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", out).strip()
