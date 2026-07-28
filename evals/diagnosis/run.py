"""Diagnosis eval harness.

The regression gate for the product's core claim: every other test in this repo checks
mechanism (does the pipeline run, does the schema validate); this is the only one that
checks whether the diagnosis is actually *right*. It makes real DeepSeek API calls, so it
is run manually -- see ``pyproject.toml``'s ``testpaths`` -- and is never part of
``uv run pytest``.

Usage: uv run python -m evals.diagnosis.run [--model deepseek-v4-pro] [--case ID]
"""

import argparse
import asyncio
from pathlib import Path

import yaml
from pydantic import BaseModel

from server.charter.contracts import Diagnosis, StudentSubmission
from server.charter.stages.s1_diagnose import diagnose
from server.config import get_settings
from server.llm.deepseek import DeepSeekClient, LlmError
from server.store.taxonomy import canonicalize_rule

_DEFAULT_CASES = Path("evals/diagnosis/cases.yaml")
_PASS_THRESHOLD = 0.8  # gate: >=80% rule-match, i.e. 16/20 on the full set


class EvalCase(BaseModel):
    id: str
    problem: str
    steps: list[str]
    expected_rule: str
    expected_topic_prefix: str
    accept_aliases: list[str] = []


class CaseScore(BaseModel):
    case_id: str
    rule_match: bool
    topic_match: bool
    verified: bool
    got_rule: str = ""
    got_topic: str = ""
    notes: str = ""


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [EvalCase.model_validate(entry) for entry in raw]


def _token_overlap(a: str, b: str) -> float:
    """Fraction of the smaller token set shared between ``a`` and ``b``.

    Deliberately loose (this is the "does it also share plain English words" fallback
    match, behind exact-canonical and alias matching), and deliberately fragile in one
    specific way worth knowing about: because the denominator is ``min(len(ta), len(tb))``
    rather than the union, a short phrase (one or two tokens after the length>2 filter --
    several ``expected_rule`` strings in cases.yaml are this short, e.g. the single-token
    ``-(a+b) -> -a + b``) can be fully "matched" by a got-string that happens to repeat
    that one token while describing a completely different error. See the eval harness
    report for a worked false-positive example found while building this file.
    """
    ta = {t for t in a.lower().replace("->", " ").split() if len(t) > 2}
    tb = {t for t in b.lower().replace("->", " ").split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def score_case(case: EvalCase, diagnosis: Diagnosis) -> CaseScore:
    """Score one diagnosis against its labelled case.

    A match counts via any of three routes, from strictest to loosest:
      1. ``canonical_match`` -- ``canonicalize_rule`` collapses both rules to the same
         normal form (handles pure variable renaming, e.g. ``(x+3)^2`` vs ``(p+q)^2``).
      2. ``alias_match`` -- the model's own phrasing either contains a curated alias
         verbatim, or shares most of an alias's words.
      3. ``overlap_match`` -- the model's phrasing shares most of the words in
         ``expected_rule`` itself, with no alias involved.
    """
    got = diagnosis.buggy_rule
    canonical_match = canonicalize_rule(got) == canonicalize_rule(case.expected_rule)
    alias_match = any(
        alias.lower() in got.lower() or _token_overlap(alias, got) >= 0.6
        for alias in case.accept_aliases
    )
    overlap_match = _token_overlap(case.expected_rule, got) >= 0.7

    rule_match = canonical_match or alias_match or overlap_match
    topic_match = diagnosis.topic.startswith(case.expected_topic_prefix)

    if canonical_match:
        notes = "canonical"
    elif alias_match:
        notes = "alias"
    elif overlap_match:
        notes = "overlap"
    else:
        notes = "none"

    return CaseScore(
        case_id=case.id,
        rule_match=rule_match,
        topic_match=topic_match,
        verified=diagnosis.verified_by_sympy,
        got_rule=got,
        got_topic=diagnosis.topic,
        notes=notes,
    )


def _build_client() -> DeepSeekClient:
    """Build a client straight from ``Settings`` (reads the key from ``server/.env``,
    same as the app), deliberately bypassing ``server.deps.build_llm_client``'s
    ``FAKE_LLM`` branch.

    This harness exists to score real model output; if ``FAKE_LLM=1`` were honored here,
    every case would silently get back the same one canned fixture in
    ``server/llm/fake.py`` (its match key is the ``Diagnosis`` schema name, which appears
    in every diagnose() request regardless of the actual problem), producing a confident
    but meaningless score instead of a loud failure. Refuse instead of guessing.
    """
    settings = get_settings()
    if settings.fake_llm:
        raise SystemExit(
            "evals/diagnosis/run.py requires FAKE_LLM=0 in server/.env -- this harness "
            "makes real API calls to score against, and the fake transport would answer "
            "every case with the same one canned fixture regardless of the problem."
        )
    return DeepSeekClient(
        settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout_s=settings.llm_timeout_s,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-v4-pro", help="model id to diagnose with")
    parser.add_argument("--case", default=None, help="run only this case id")
    parser.add_argument("--cases", default=_DEFAULT_CASES, type=Path, help="path to cases.yaml")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            print(f"no case with id {args.case!r} in {args.cases}")
            return 1

    client = _build_client()
    scores: list[CaseScore] = []
    total_cost = 0.0

    try:
        for case in cases:
            submission = StudentSubmission(
                problem=case.problem, steps=case.steps, source="typed", student_corrected=True
            )
            try:
                diagnosis, meta = await diagnose(client, submission=submission, model=args.model)
            except LlmError as exc:
                # A regression gate that aborts on the first flaky call is worse than one
                # that scores the flake as a miss: the other 19 cases still tell you
                # something, and a real prompt regression won't hide behind one timeout.
                # Every failure surface below diagnose() (schema retries, transport, HTTP)
                # raises LlmError or a subclass -- see server/llm/deepseek.py -- so this
                # catch is scoped to that contract rather than a bare `except Exception`,
                # which would also swallow a genuine bug in this harness as a silent miss.
                scores.append(
                    CaseScore(
                        case_id=case.id,
                        rule_match=False,
                        topic_match=False,
                        verified=False,
                        notes=f"ERROR {type(exc).__name__}",
                    )
                )
                print(f"[ERR ] {case.id:28} {type(exc).__name__}: {exc}"[:110])
                continue

            total_cost += meta.cost_usd
            score = score_case(case, diagnosis)
            scores.append(score)
            mark = "PASS" if score.rule_match else "FAIL"
            topic_flag = "" if score.topic_match else "  [topic mismatch]"
            print(
                f"[{mark}] {case.id:28} verified={score.verified!s:5} "
                f"({score.notes:9}) {score.got_rule[:48]}{topic_flag}"
            )
    finally:
        await client.aclose()

    passed = sum(1 for s in scores if s.rule_match)
    topic_ok = sum(1 for s in scores if s.topic_match)
    verified = sum(1 for s in scores if s.verified)
    failed_ids = [s.case_id for s in scores if not s.rule_match]

    print(
        f"\nrule match     {passed}/{len(scores)}"
        f"\ntopic match    {topic_ok}/{len(scores)}"
        f"\nsympy verified {verified}/{len(scores)}"
        f"\ncost           ${total_cost:.4f}"
    )
    # A bare pass rate doesn't tell a caller which cases regressed -- print the list so a
    # prompt-change PR can be diffed against a known set of ids, not just a percentage.
    if failed_ids:
        print(f"regressed ({len(failed_ids)}): {', '.join(failed_ids)}")

    return 0 if passed >= _PASS_THRESHOLD * len(scores) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
