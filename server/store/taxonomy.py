"""Maps a free-text diagnosis onto a stable misconception id.

Open-domain diagnosis cannot be aggregated without stable identity, so identity is
added after the fact rather than by constraining what the tutor may diagnose.
"""

import json
import re
import sqlite3

from pydantic import BaseModel

from server.charter.contracts import Diagnosis
from server.llm.deepseek import DeepSeekClient, LlmError

_VAR_RUN = re.compile(r"\b[a-z]\b")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")


class MatchDecision(BaseModel):
    same_as_id: int | None = None
    new_slug: str | None = None
    reasoning: str = ""


def canonicalize_rule(rule: str) -> str:
    """Collapse cosmetic differences so the same error matches across problems.

    Single-letter variables become ``v`` and numerals become ``#``, so
    ``(x+3)^2 -> x^2+9`` and ``(t+5)^2 -> t^2+25`` share one canonical form
    while ``(a+b)^2 -> a^2+b^2`` stays distinct from both.

    The numeral placeholder MUST NOT be a lowercase letter: ``_VAR_RUN``
    runs afterwards and would re-capture it as a variable, collapsing
    digits and variables into the same token.
    """
    text = rule.strip().lower()
    text = text.replace("→", "->").replace("=>", "->")
    text = _NUM.sub("#", text)
    text = _VAR_RUN.sub("v", text)
    text = _WS.sub("", text)
    return text


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:64] or "unnamed-misconception"


def _candidates(
    conn: sqlite3.Connection, diagnosis: Diagnosis, limit: int = 8
) -> list[sqlite3.Row]:
    """Retrieve plausible existing entries by topic, then by rule token overlap."""
    rows = conn.execute(
        """SELECT id, slug, canonical_statement, canonical_rule, topic
           FROM misconceptions
           WHERE topic = ? OR topic LIKE ?
           LIMIT ?""",
        (diagnosis.topic, diagnosis.topic.split(".")[0] + "%", limit),
    ).fetchall()
    if rows:
        return rows
    return conn.execute(
        "SELECT id, slug, canonical_statement, canonical_rule, topic FROM misconceptions LIMIT ?",
        (limit,),
    ).fetchall()


async def resolve_misconception(
    conn: sqlite3.Connection, client: DeepSeekClient, *, diagnosis: Diagnosis, model: str
) -> int:
    canonical = canonicalize_rule(diagnosis.buggy_rule)

    exact = conn.execute(
        "SELECT id FROM misconceptions WHERE canonical_rule = ?", (canonical,)
    ).fetchone()
    if exact:
        return int(exact["id"])

    candidates = _candidates(conn, diagnosis)
    listing = "\n".join(
        f"- id={row['id']} slug={row['slug']} rule={row['canonical_rule']} "
        f"statement={row['canonical_statement']}"
        for row in candidates
    )
    prompt = (
        "Decide whether this newly diagnosed student misconception is the SAME underlying "
        "error as one already in our taxonomy, or genuinely new.\n\n"
        f"New buggy rule: {diagnosis.buggy_rule}\n"
        f"New statement: {diagnosis.misconception_statement}\n"
        f"Topic: {diagnosis.topic}\n\n"
        f"Existing candidates:\n{listing or '(none)'}\n\n"
        "If it matches an existing entry, set same_as_id to that id and leave new_slug null. "
        "If it is genuinely new, leave same_as_id null and propose a short kebab-case new_slug. "
        "Prefer matching an existing entry — near-duplicates dilute our statistics."
    )

    try:
        decision, _ = await client.complete_strict(
            messages=[{"role": "user", "content": prompt}], schema=MatchDecision, model=model
        )
    except LlmError:
        # Never fail a session over taxonomy bookkeeping; mint from the rule itself.
        decision = MatchDecision(new_slug=_slugify(diagnosis.buggy_rule))

    if decision.same_as_id is not None:
        row = conn.execute(
            "SELECT id, aliases_json FROM misconceptions WHERE id = ?", (decision.same_as_id,)
        ).fetchone()
        if row:
            aliases = json.loads(row["aliases_json"])
            if diagnosis.buggy_rule not in aliases:
                aliases.append(diagnosis.buggy_rule)
                conn.execute(
                    "UPDATE misconceptions SET aliases_json = ? WHERE id = ?",
                    (json.dumps(aliases), row["id"]),
                )
            return int(row["id"])

    slug = _slugify(decision.new_slug or diagnosis.buggy_rule)
    existing = conn.execute("SELECT id FROM misconceptions WHERE slug = ?", (slug,)).fetchone()
    if existing:
        return int(existing["id"])

    try:
        cursor = conn.execute(
            """INSERT INTO misconceptions
               (slug, canonical_statement, canonical_rule, topic, is_seed)
               VALUES (?, ?, ?, ?, 0)""",
            (slug, diagnosis.misconception_statement, canonical, diagnosis.topic),
        )
    except sqlite3.IntegrityError:
        # Lost a race against a concurrent resolve_misconception() that minted the
        # same slug between our existence check and this insert (e.g. two sessions
        # hitting the same brand-new rule at once). Reuse the row the other caller
        # just created instead of surfacing a crash for ordinary concurrent use.
        winner = conn.execute("SELECT id FROM misconceptions WHERE slug = ?", (slug,)).fetchone()
        if winner is None:
            raise
        return int(winner["id"])
    return int(cursor.lastrowid)
