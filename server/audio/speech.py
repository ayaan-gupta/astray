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

# "y plus three all squared" runs together into something a listener can parse
# either way. The comma is the breath that makes the grouping audible, and it is
# the whole difference between the correct reading and the misconception. The
# prompt asks for it and the model drops it about a third of the time.
UNCOMMAED_ALL = re.compile(r"(?<![,.])\s+all (squared|cubed)\b")

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


# How far the final line may overrun its beat. `mux._fit_last` reclaims this by
# starting the clip earlier, bounded by the previous clip, so it is spendable
# rather than wishful.
FINAL_OVERRUN_ALLOWANCE_S = 2.0


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

    out = UNCOMMAED_ALL.sub(lambda m: f", all {m.group(1)}", out)

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


# Letter names in CMU Arpabet, with the stress digits Fish requires. A lone letter
# in running text is genuinely ambiguous to a TTS model and it guesses badly: "a"
# is the most common word in English so it comes out as the article /ə/ rather than
# the letter /eɪ/, and "y" collapses toward /iː/. Forcing the phonemes is the only
# reliable fix, since the letter is the variable and mispronouncing it makes the
# maths unintelligible.
LETTER_ARPABET = {
    "a": "EY1",
    "b": "B IY1",
    "c": "S IY1",
    "d": "D IY1",
    "e": "IY1",
    "f": "EH1 F",
    "g": "JH IY1",
    "h": "EY1 CH",
    "i": "AY1",
    "j": "JH EY1",
    "k": "K EY1",
    "l": "EH1 L",
    "m": "EH1 M",
    "n": "EH1 N",
    "o": "OW1",
    "p": "P IY1",
    "q": "K Y UW1",
    "r": "AA1 R",
    "s": "EH1 S",
    "t": "T IY1",
    "u": "Y UW1",
    "v": "V IY1",
    "w": "D AH1 B AH0 L Y UW0",
    "x": "EH1 K S",
    "y": "W AY1",
    "z": "Z IY1",
}

# Operators and post-modifiers only. Nouns must NOT be in here: the whole test is
# that an article introduces a noun while a variable sits next to maths, so listing
# "term" or "bracket" inverts it and tags the article in "missing a middle term".
MATHS_CONTEXT = frozenset("plus minus times over equals squared cubed itself".split())

# "a" and "i" are also ordinary English words, so they are the only two letters
# that need evidence before being treated as variables. Every other single letter
# is never a word, so a lone one is always a variable.
AMBIGUOUS_LETTERS = frozenset({"a", "i"})

_TOKEN = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]+|\s+")


def _is_variable(token: str, before: list[str], after: list[str]) -> bool:
    if len(token) != 1 or token.lower() not in LETTER_ARPABET:
        return False
    if token.lower() not in AMBIGUOUS_LETTERS:
        return True
    # An article is followed by the noun it introduces; a variable sits next to
    # maths. Look one word each way, which is enough to separate "a middle term"
    # from "a plus b" and "two a b".
    neighbours = {w.lower() for w in before[-1:] + after[:1]}
    return bool(neighbours & MATHS_CONTEXT) or any(
        len(w) == 1 and w.lower() in LETTER_ARPABET for w in before[-1:] + after[:1]
    )


def with_letter_phonemes(text: str) -> str:
    """Force the pronunciation of single-letter variables, for the API call only.

    Kept separate from `speakable` on purpose: the tags are markup for Fish, not
    text. `speakable` output is what gets word-counted, stored and shown in the
    docs, and it stays readable. This runs last, on the way out.
    """
    tokens = _TOKEN.findall(text)
    words = [t for t in tokens if t.isalpha()]
    out, seen_words = [], 0
    for token in tokens:
        if token.isalpha():
            before, after = words[:seen_words], words[seen_words + 1 :]
            seen_words += 1
            if _is_variable(token, before, after):
                arpabet = LETTER_ARPABET[token.lower()]
                out.append(f"<|phoneme_start|>{arpabet}<|phoneme_end|>")
                continue
        out.append(token)
    return "".join(out)


def word_count(text: str) -> int:
    """Words as a voice would count them, for budgeting against a beat."""
    return len(re.findall(r"[^\s]+", speakable(text)))


def budget_words(duration_s: float, words_per_second: float, *, final: bool = False) -> int:
    """How many words fit in a beat, with a little silence left at each end.

    The trim is not decoration. Speech that begins on a beat's first frame lands
    before the visual it describes has finished being drawn, and speech running
    to the last frame collides with the next beat.

    `final` loosens both allowances for the last beat, which is the one place a
    longer line is safe: there is no next beat to collide with, and `mux._fit_last`
    can pull an overrunning final clip earlier to fit. Without this the closing
    line is the most squeezed in the whole script, and it is usually the one
    stating the correct identity: budgeting the golden render's 5.0s final beat at
    the tighter figure produced "a squared, two a b, plus b squared", a fragment
    that never says what the identity equals.
    """
    if final:
        usable = max(0.0, duration_s - 0.25) + FINAL_OVERRUN_ALLOWANCE_S
    else:
        usable = max(0.0, duration_s - 0.35)
    return max(3, int(usable * words_per_second))


def target_words(budget: int) -> int:
    """What to aim for, given a hard maximum.

    A cap on its own is a floor in practice: told only "at most N words", the model
    writes a five-word caption and the animation plays in silence. Naming a target
    as well is what makes it fill the time. 0.8 leaves room to land under the cap
    without the line being clipped by `trim_to_budget`.
    """
    return max(3, int(budget * 0.8))
