"""Turning written maths into speakable maths.

This is the single biggest lever on how natural the narration sounds, and it has
nothing to do with the TTS model. Handed `(y+3)^2 = y^2 + 6y + 9`, every engine
produces something between "y plus three caret two" and an outright skip. Handed
"y plus three, all squared, equals y squared plus six y plus nine", the same
engine sounds like a teacher.

The prompt in `narrate.py` asks for speakable prose directly, because a sentence
composed for the ear beats one transliterated from symbols. Everything here is
the net for what slips through, in the same belt-and-braces shape the LaTeX,
citation and em-dash paths already use.

The other half of sounding natural is punctuation. Fish Audio's models take
their prosody from it, so a comma is a short breath and a full stop is a real
stop. `pace()` exists to keep those marks meaningful rather than decorative.
"""

import re

# Ordinal power names. Beyond cubed, "to the power of n" is what a person says.
POWER_WORDS = {
    "2": "squared",
    "3": "cubed",
}

OPERATORS = {
    "+": " plus ",
    "=": " equals ",
    "*": " times ",
    "/": " over ",
    "<": " is less than ",
    ">": " is greater than ",
}

# The rule arrow, as in `(a+b)^2 -> a^2 + b^2`. It has to resolve before the
# operator pass, which would otherwise read the two characters separately and
# turn it into "minus is greater than".
ARROW = re.compile(r"\s*(?:->|=>|-->|→|⇒)\s*")

# Characters a speech engine either spells out or silently drops. None of them
# carry meaning once the maths is already words.
#
# `*` is deliberately absent. It is markdown emphasis *and* multiplication, and
# stripping it as noise turned `3*4` into "34". Doubled asterisks are unwrapped as
# bold first, so any single one still standing is a multiplication sign, which is
# the reading that matters in a maths tutor.
NOISE = re.compile(r"[`_#~|\\]")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

# A bracketed group raised to a power is the case that matters most: it is
# exactly the shape of the misconception this product exists to explain, and
# "all squared" is the phrase that distinguishes (y+3)^2 from y^2 + 3^2.
GROUP_POWER = re.compile(r"\(([^()]{1,40})\)\s*\^\s*(\w+)")

# A bare power on a single term: y^2, 3^2, ab^2.
TERM_POWER = re.compile(r"([A-Za-z0-9]+)\s*\^\s*(\w+)")

# An implicit product with a leading coefficient: 6y, 2ab, 10x. Spoken as
# "six y" and "two a b", never "sixy" or "two ab". The letters are matched as a
# run rather than singly, because `\b(\d+)([A-Za-z])\b` never matches `2ab` at
# all: there is no word boundary between the `a` and the `b`.
COEFFICIENT = re.compile(r"\b(\d+)([A-Za-z]{1,4})\b")

# A minus sign between operands is subtraction; a leading one is negation.
BINARY_MINUS = re.compile(r"(?<=[\w)])\s*-\s*(?=[\w(])")
LEADING_MINUS = re.compile(r"(?<![\w)])-\s*(?=[\w(])")

DIGITS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
}


def _power(exponent: str) -> str:
    return POWER_WORDS.get(exponent, f"to the power of {DIGITS.get(exponent, exponent)}")


def _coefficient(match: re.Match) -> str:
    """`2ab` -> "two a b". Each variable is its own word or the voice reads them
    as one, and "ab" comes out as a syllable rather than two letters."""
    number = DIGITS.get(match.group(1), match.group(1))
    letters = " ".join(match.group(2))
    return f"{number} {letters}"


def speakable(text: str) -> str:
    """Rewrite leftover maths notation as words a voice can read aloud.

    Order matters. Grouped powers resolve before bare ones so `(y+3)^2` becomes
    "all squared" rather than "three squared", and coefficients resolve before
    the digit pass so `6y` becomes "six y" rather than "6 y".
    """
    out = BOLD.sub(r"\1", text)
    out = NOISE.sub("", out)
    out = ARROW.sub(" becomes ", out)

    # (y+3)^2 -> "y plus 3, all squared". The comma is deliberate: it gives the
    # engine a breath before the qualifier, which is how the grouping is heard.
    out = GROUP_POWER.sub(lambda m: f"{m.group(1)}, all {_power(m.group(2))}", out)
    out = TERM_POWER.sub(lambda m: f"{m.group(1)} {_power(m.group(2))}", out)

    out = COEFFICIENT.sub(_coefficient, out)

    out = BINARY_MINUS.sub(" minus ", out)
    out = LEADING_MINUS.sub("negative ", out)
    for symbol, word in OPERATORS.items():
        out = out.replace(symbol, word)

    # Any brackets still standing were grouping that the rewrites above did not
    # need. Read as a pause rather than as the word "bracket", which no teacher
    # says out loud mid-sentence.
    out = out.replace("(", ", ").replace(")", ", ")

    return pace(out)


def pace(text: str) -> str:
    """Normalise whitespace and punctuation so the marks still drive prosody.

    Doubled commas and comma-then-full-stop runs come out of the rewrites above,
    and an engine reading them produces an audible stumble.
    """
    out = re.sub(r"\s+", " ", text)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r",(\s*,)+", ",", out)
    out = re.sub(r",\s*([.!?])", r"\1", out)
    out = re.sub(r"([.!?])\s*,", r"\1", out)
    return out.strip().strip(",").strip()


def word_count(text: str) -> int:
    """Words as a voice would count them, for budgeting against a beat."""
    return len(re.findall(r"[^\s]+", speakable(text)))


def budget_words(duration_s: float, words_per_second: float) -> int:
    """How many words fit in a beat, with a beat of silence left at each end.

    The trim is not decoration. Speech that begins on a beat's first frame lands
    before the visual it describes has finished being drawn, and speech running
    to the last frame collides with the next beat.
    """
    usable = max(0.0, duration_s - 0.55)
    return max(3, int(usable * words_per_second))
