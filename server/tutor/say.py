"""Turning a written tutor reply into something worth hearing.

A chat reply is written to be *read*: backticked notation, markdown bullets, and
`[beat:b3]` citations that the client renders as buttons which seek the player.
Fed to a voice unchanged, all three fail differently -- notation is read out as
punctuation, bullets as the word "asterisk", and a citation as "bracket beat
colon bee three".

The citation is the interesting one. Deleting it leaves "Watch to see that
plugging in y equals one gives the wrong answer", a sentence with a hole in it,
because the tutor's prose is written *around* the chip. So a citation is replaced
by the title of the beat it points at, which is what the chip is labelled with
anyway: "Watch the number check to see ...". The reply is not rewritten, it is
read the way its author would read it aloud.

Length is capped because speech has a cost the written reply does not. A reply
that scrolls is fine to skim and interminable to sit through, and the student is
looking at the text while it plays.
"""

import re
import sqlite3

from server.audio import speech
from server.store import repo

CITATION_RE = re.compile(r"\s*\[beat:(b[0-9]+)\]")

# Markdown that carries no meaning aloud. Bold and code survive as their contents
# (`speech.speakable` handles those); these are the ones with nothing to keep.
BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.MULTILINE)

# About twenty seconds at this product's measured speaking rate. Past that a
# spoken answer stops being an answer and becomes a lecture the student cannot
# skim, skip, or interrupt.
MAX_SPOKEN_CHARS = 480

# What a citation becomes when its beat is not in this session's manifest. Chat
# validates citations before persisting, so this is the belt-and-braces path for
# a reply stored before that validation existed.
UNKNOWN_BEAT = "the animation"


def beat_titles(conn: sqlite3.Connection, session_id: str) -> dict[str, str]:
    """Beat id to a spoken noun phrase. "Area model" becomes "the area model".

    A beat title is written as a rail label, so it is capitalised like one. Dropped
    mid-sentence it has to read as a noun phrase instead, which means losing the
    capital -- unless the title carries another capital further in, which is what
    a name, an acronym or a camel-cased word looks like ("SymPy check"). Those are
    spelled that way rather than merely capitalised, so their case is left alone.
    """
    titles: dict[str, str] = {}
    for row in repo.list_beats(conn, session_id):
        title = (row["title"] or "").strip()
        if not title:
            continue
        spelled = any(char.isupper() for char in title[1:])
        lowered = title if spelled else title[0].lower() + title[1:]
        titles[row["beat_id"]] = lowered if lowered.startswith("the ") else f"the {lowered}"
    return titles


def _truncate(text: str, limit: int) -> str:
    """Cut at the last sentence end inside `limit`, never mid-clause.

    A spoken answer that stops mid-sentence sounds like the connection dropped,
    which is worse than a shorter answer. If the first sentence alone is over the
    limit it is kept whole rather than butchered -- one long sentence is still a
    complete thought.
    """
    if len(text) <= limit:
        return text
    window = text[: limit + 1]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut <= 0:
        end = text.find(". ")
        return text if end < 0 else text[: end + 1]
    return text[: cut + 1]


def for_speech(reply: str, titles: dict[str, str], *, limit: int = MAX_SPOKEN_CHARS) -> str:
    """The spoken form of one written reply. Empty if there is nothing to say."""
    text = CITATION_RE.sub(lambda m: " " + titles.get(m.group(1), UNKNOWN_BEAT), reply)
    text = HEADING_RE.sub("", BULLET_RE.sub("", text))
    # Collapse the blank lines markdown uses for paragraphs: to a voice they are
    # not pauses, they are nothing, and they leave double spaces behind.
    text = re.sub(r"\n{2,}", "\n", text).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    return speech.speakable(_truncate(text, limit))
