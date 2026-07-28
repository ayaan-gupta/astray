"""Deterministic symbolic verification of the model's own correct solution.

Covers algebraic manipulation, equation solving, and calculus. Word problems,
proofs, and geometry are out of scope — those return verified=False with a
reason, which lowers the diagnosis confidence ceiling rather than faking rigor.

``run_check`` never raises. Every failure mode — malformed syntax, an
unsupported check kind, a solver that can't find a finite answer, an input
designed to be slow or unsafe — is folded into ``CheckResult(verified=False)``
with a human-readable ``detail``. The caller (Task 11) treats "unverified" as
"cap the confidence and hedge the UI," never as an error to propagate.

Security note: ``sympy.parsing.sympy_parser.parse_expr`` evaluates the input
through Python's ``eval`` under the hood. Handed a raw string it will run
arbitrary Python — ``open('/etc/passwd').read()`` and
``().__class__.__bases__[0].__subclasses__()``-style sandbox escapes both
execute for real, regardless of the ``evaluate=`` flag. Because these check
strings originate from an LLM whose prompt includes untrusted student input,
we cannot assume they are merely "syntactically valid SymPy." Every string is
passed through ``_validate_syntax`` first: it allow-lists a bare arithmetic/
function-call character set (no quotes, no brackets, no LaTeX backslashes)
and separately bans dunder names and attribute access, which closes the
concrete escapes above before any parser code runs.
"""

import re

import sympy
from pydantic import BaseModel
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from server.charter.contracts import SympyCheck

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Above this, a literal integer exponent (e.g. "2**10000000000") computes a
# number with tens of millions of digits and can take the process down; a
# real algebra/calculus problem never needs an exponent this large.
_MAX_EXPONENT = 1000

# Only plain arithmetic / function-call syntax is accepted: digits, letters,
# underscore, whitespace, and the operators/punctuation algebra needs. No
# quotes (blocks string literals), no [ ] { } (blocks subscripting and LaTeX
# braces), no backslash (blocks LaTeX commands).
_SAFE_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/().,\s!=<>^]*$")
_DUNDER_NAME = re.compile(r"__[A-Za-z0-9_]+__")
_ATTRIBUTE_ACCESS = re.compile(r"\.[A-Za-z_]")
_CALL_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# sympy_parser's default namespace is `from sympy import *`. An identifier
# used as a function call that isn't a real sympy name doesn't raise: the
# implicit-multiplication transform silently splits it into a product of
# one-letter symbols (`undefinedfunc(x)` becomes `c*d**2*e**2*f**2*i*n**3*u**2*x`).
# That's a confident, wrong answer, not a crash, so we reject it explicitly.
# Single-letter names are exempt: `x(x+2)` is unambiguous implicit
# multiplication, not a disguised function call.
_SYMPY_NAMES = frozenset(name for name in dir(sympy) if not name.startswith("_"))
_EXTRA_SAFE_NAMES = frozenset({"abs"})


class CheckResult(BaseModel):
    verified: bool
    detail: str


def _validate_syntax(text: str) -> None:
    if not text or not text.strip():
        raise ValueError("empty expression")
    if not _SAFE_CHARS.match(text):
        raise ValueError("expression contains unsupported characters (no quotes/brackets/LaTeX)")
    if _DUNDER_NAME.search(text):
        raise ValueError("expression references a disallowed name")
    if _ATTRIBUTE_ACCESS.search(text):
        raise ValueError("expression uses attribute access, which is not supported")


def _unknown_call_names(text: str) -> set[str]:
    names = {match.group(1) for match in _CALL_NAME.finditer(text)}
    return {
        name
        for name in names
        if len(name) > 1 and name not in _SYMPY_NAMES and name not in _EXTRA_SAFE_NAMES
    }


def _has_dangerous_power(expr: sympy.Basic) -> bool:
    for node in sympy.preorder_traversal(expr):
        if not isinstance(node, sympy.Pow) or not node.exp.is_Integer:
            continue
        if abs(int(node.exp)) > _MAX_EXPONENT:
            return True
    return False


def _parse(text: str, local_dict: dict[str, sympy.Symbol] | None = None) -> sympy.Basic:
    """Parse SymPy-syntax text into an expression. Fails closed, never executes unsafe input.

    Raises (caught by ``run_check``) rather than returning, since this is an internal
    helper: malformed/hostile input is a control-flow signal here, a result at the boundary.
    """
    _validate_syntax(text)
    unknown = _unknown_call_names(text)
    if unknown:
        raise ValueError(f"unknown identifier(s) used as functions: {', '.join(sorted(unknown))}")
    # Probe with evaluate=False first: constructing the expression tree doesn't
    # compute the power, so we can inspect exponents before anything expensive runs.
    probe = parse_expr(text, transformations=_TRANSFORMS, evaluate=False, local_dict=local_dict)
    if _has_dangerous_power(probe):
        raise ValueError(f"exponent exceeds safety limit of {_MAX_EXPONENT}")
    return parse_expr(text, transformations=_TRANSFORMS, evaluate=True, local_dict=local_dict)


def _is_zero(difference: sympy.Basic) -> bool:
    """True if difference is provably zero.

    ``simplify() == 0`` catches most cases, but simplify doesn't always fully
    collapse an expression that is genuinely zero (e.g. some radical/logarithm
    forms). ``.equals(0)`` does extra work, including numeric sampling, to
    confirm equality in those cases. It can return None ("can't determine"),
    which we treat as not verified rather than guessing.
    """
    simplified = sympy.simplify(difference)
    if simplified == 0:
        return True
    return bool(simplified.equals(0))


def run_check(check: SympyCheck) -> CheckResult:
    """Run the model's requested symbolic check. Never raises.

    A failure to verify — malformed syntax, an unsupported kind, a solver that
    can't produce a finite comparable answer, hostile input — is a result,
    not an exception: ``CheckResult(verified=False, detail=...)``.
    """
    if check.kind == "skip":
        return CheckResult(verified=False, detail=check.skip_reason or "not symbolically checkable")

    try:
        if check.kind == "equivalence":
            if not check.lhs or not check.rhs:
                return CheckResult(verified=False, detail="equivalence needs lhs and rhs")
            difference = _parse(check.lhs) - _parse(check.rhs)
            if _is_zero(difference):
                return CheckResult(verified=True, detail=f"{check.lhs} == {check.rhs}")
            return CheckResult(
                verified=False,
                detail=f"not equivalent; difference simplifies to {sympy.simplify(difference)}",
            )

        if check.kind == "solution_set":
            if not check.equation or not check.variable:
                return CheckResult(
                    verified=False, detail="solution_set needs equation and variable"
                )
            symbol = sympy.Symbol(check.variable)
            # Inject the declared variable so it resolves to *this* symbol even
            # if its name collides with a sympy builtin constant (I, E, S, ...).
            local_dict = {check.variable: symbol}
            actual = sympy.solveset(
                _parse(check.equation, local_dict=local_dict), symbol, domain=sympy.S.Reals
            )
            claimed = sympy.FiniteSet(*[_parse(c, local_dict=local_dict) for c in check.candidates])
            # `actual.is_finite_set` (not `isinstance(actual, FiniteSet)`) is the
            # correct test: sympy's EmptySet is finite but is its own singleton
            # class, not a FiniteSet subclass, while a Union of ImageSets or a
            # ConditionSet reports `is_finite_set is None` (undetermined) and
            # must not be treated as a match either.
            if actual.is_finite_set and actual == claimed:
                return CheckResult(verified=True, detail=f"solution set {actual}")
            return CheckResult(
                verified=False,
                detail=f"claimed {claimed} but actual solution set is {actual}",
            )
    except Exception as exc:  # noqa: BLE001 - sympy raises many types; all mean "unverified"
        message = str(exc).strip()
        detail = (
            f"verification failed: {type(exc).__name__}: {message}"
            if message
            else (f"verification failed: {type(exc).__name__}")
        )
        return CheckResult(verified=False, detail=detail)

    return CheckResult(verified=False, detail=f"unsupported check kind {check.kind!r}")
