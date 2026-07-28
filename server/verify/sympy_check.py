"""Deterministic symbolic verification of the model's own correct solution.

Covers algebraic manipulation, equation solving, and calculus. Word problems,
proofs, and geometry are out of scope — those return verified=False with a
reason, which lowers the diagnosis confidence ceiling rather than faking rigor.

``run_check`` never raises and never hangs. Every failure mode — malformed
syntax, an unsupported check kind, a solver that can't find a finite answer,
an input designed to be slow or unsafe — is folded into
``CheckResult(verified=False)`` with a human-readable ``detail``. The caller
(Task 11) treats "unverified" as "cap the confidence and hedge the UI," never
as an error to propagate.

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
concrete escapes above before any parser code runs. This character/attribute
gate has been adversarially tested (unicode homoglyphs, hex escapes, RTL
overrides, comment/newline smuggling, `lambda`/walrus, `globals()`/`vars()`,
huge and deeply-nested inputs) and held.

Hang note: the exponent-magnitude guard (``_MAX_EXPONENT``) is a cheap,
*incomplete* fast-path, not the real defense. It inspects each ``Pow`` node's
own literal exponent, so it catches ``2**10000000000`` but not a composed
tower like ``2**2**2**2**2**2`` (every outer node's exponent is itself an
unevaluated ``Pow``, not an ``Integer``, so the guard skips it) — and it does
nothing at all for expensive non-``Pow`` calls like ``factorial(2000000)`` or
``binomial``/``primorial``/``factorint`` of a large number. Expression-shape
heuristics are whack-a-mole against a Turing-complete-ish surface (any sympy
function is reachable through an allow-listed call). The real defense is
``run_check`` executing the actual computation in a child process with a hard
wall-clock timeout and killing it outright on expiry — see ``run_check`` and
``_run_check_unbounded`` below.
"""

import asyncio
import multiprocessing
import re
from multiprocessing.connection import Connection

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
# real algebra/calculus problem never needs an exponent this large. This is a
# cheap fast-path only — it does not catch composed exponent towers or
# expensive non-Pow functions (see module docstring); the wall-clock timeout
# in run_check is the actual defense for those.
_MAX_EXPONENT = 1000

# Hard wall-clock ceiling on a single run_check call, enforced by running the
# computation in a killable child process. Generous for any real algebra/
# calculus/derivative check; anything that takes longer is either pathological
# input or a sympy performance cliff we haven't characterized, and either way
# "unverified, timed out" is the correct, safe answer.
_DEFAULT_TIMEOUT_SECONDS = 5.0

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


def _run_check_unbounded(check: SympyCheck) -> CheckResult:
    """The actual symbolic computation, with no time bound of its own.

    Only ever called inside the child process spawned by ``run_check`` — that
    process boundary, not anything in here, is what makes a runaway
    computation (a composed exponent tower, an expensive number-theoretic
    function, or some other sympy performance cliff neither of us has found
    yet) terminate instead of hanging the caller forever.
    """
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
            if not check.variable.isidentifier():
                return CheckResult(
                    verified=False, detail=f"{check.variable!r} is not a legal variable name"
                )
            symbol = sympy.Symbol(check.variable)
            # Inject the declared variable so it resolves to *this* symbol even
            # if its name collides with a sympy builtin constant (I, E, S, ...).
            local_dict = {check.variable: symbol}
            parsed_equation = _parse(check.equation, local_dict=local_dict)
            if symbol not in parsed_equation.free_symbols:
                # Catches e.g. variable="5x" (not even a legal name, so it can
                # never appear in the equation) and variable="y" against an
                # equation in x: without this, both sides can independently
                # come out EmptySet (candidates=[] vs. a solver that can't
                # relate an absent symbol to anything) and compare equal —
                # a confidently "verified" answer that never checked anything.
                return CheckResult(
                    verified=False,
                    detail=f"variable {check.variable!r} does not appear in the equation",
                )
            actual = sympy.solveset(parsed_equation, symbol, domain=sympy.S.Reals)
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


def _worker(check_data: dict, conn: Connection) -> None:
    """Entry point for the child process. Runs the unbounded check and sends
    the result back over a Pipe (synchronous send — unlike a Queue, there's no
    background feeder thread, so there's no race between the child exiting and
    the data actually reaching the parent)."""
    check = SympyCheck(**check_data)
    result = _run_check_unbounded(check)
    conn.send(result.model_dump())


def run_check(check: SympyCheck, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> CheckResult:
    """Run the model's requested symbolic check. Never raises. Never hangs.

    **Blocks the calling thread for up to ``timeout`` seconds**
    (``process.start()``/``process.join()`` are synchronous). Fine to call
    directly from sync code or a test. From an ``async`` call site, calling
    this inline stalls the event loop — and every other coroutine scheduled
    on it — for up to the full timeout; use ``run_check_async`` instead.

    A failure to verify — malformed syntax, an unsupported kind, a solver that
    can't produce a finite comparable answer, hostile input, or a computation
    that ran past ``timeout`` seconds — is a result, not an exception:
    ``CheckResult(verified=False, detail=...)``.

    The actual computation runs in a child process so a runaway can be killed
    outright rather than blocking the caller's thread forever:

    - Hard wall-clock limit: ``process.join(timeout)`` bounds the whole check
      (parsing plus solving/simplifying), not each field parsed individually.
    - Actually kills, doesn't orphan: on expiry we call ``process.kill()``
      (SIGKILL) and reap it with ``process.join()``. A ``threading`` timeout
      couldn't do this — Python cannot forcibly stop another thread, so a
      runaway thread would keep burning a core after we "gave up" on it.
    - Works off the main thread: ``multiprocessing.Process`` can be started
      from any thread, unlike ``signal.alarm`` (main-thread-of-main-
      interpreter only), which matters because uvicorn commonly runs sync
      handlers in a worker thread pool.
    - Never raises on timeout: expiry is converted to
      ``CheckResult(verified=False, detail=...)``, same as every other
      failure mode here.

    Secondary benefit: the process boundary also contains any allow-list
    bypass neither of us has found yet, which is worth having in eval-adjacent
    code regardless of the timeout.

    Skipped checks short-circuit before any of this — no computation, so no
    need to pay the subprocess-start cost.
    """
    if check.kind == "skip":
        return CheckResult(verified=False, detail=check.skip_reason or "not symbolically checkable")

    try:
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_worker, args=(check.model_dump(), child_conn), daemon=True)
        process.start()
        child_conn.close()  # the parent only ever reads
        process.join(timeout)

        if process.is_alive():
            process.kill()
            process.join()
            return CheckResult(
                verified=False,
                detail=(
                    f"verification timed out after {timeout}s and was killed "
                    "(e.g. a composed exponent tower or an expensive "
                    "number-theoretic function)"
                ),
            )

        if parent_conn.poll():
            # poll() returns True as soon as the peer's write end closes, which
            # happens whenever the child exits — including when it dies
            # (crash, OOM kill, external signal) without ever calling send().
            # recv() in that case raises EOFError, not a shortage of data to
            # wait for; catch it and fall through to the exit-code detail
            # below instead of surfacing a generic, uninformative EOFError.
            try:
                return CheckResult(**parent_conn.recv())
            except EOFError:
                pass

        return CheckResult(
            verified=False,
            detail=f"verification process exited unexpectedly (exit code {process.exitcode})",
        )
    except Exception as exc:  # noqa: BLE001 - the sandboxing machinery must not raise either
        message = str(exc).strip()
        detail = (
            f"verification failed: {type(exc).__name__}: {message}"
            if message
            else f"verification failed: {type(exc).__name__}"
        )
        return CheckResult(verified=False, detail=detail)


async def run_check_async(
    check: SympyCheck, timeout: float = _DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    """Async entry point for ``run_check``. Use this from async callers.

    ``run_check`` blocks its calling thread for up to ``timeout`` seconds
    (``process.start()``/``process.join()`` are synchronous calls). Called
    inline from a coroutine, that stalls the event loop — and every other
    request being served on that loop — for up to the full timeout. This
    wrapper offloads the blocking call to a worker thread via
    ``asyncio.to_thread``, so the event loop stays free while the check runs.
    Task 11 (and any other async caller) should call this, not ``run_check``,
    from request-handling coroutines. Never raises, never hangs, for the same
    reasons ``run_check`` doesn't — it's the same code, just off-thread.
    """
    return await asyncio.to_thread(run_check, check, timeout)
