import time

from server.charter.contracts import SympyCheck
from server.verify.sympy_check import run_check

# Short timeout for the deliberate-hang tests below: long enough to comfortably
# clear subprocess start/kill overhead (observed ~0.2-0.3s), short enough that
# a regression back to an unbounded hang fails the test suite in seconds
# instead of hanging it, and the wall-clock assertion catches a regression
# even if run_check somehow stopped returning verified=False on timeout.
_HANG_TEST_TIMEOUT = 2.0


def test_equivalence_true():
    result = run_check(SympyCheck(kind="equivalence", lhs="(x+3)**2", rhs="x**2+6*x+9"))
    assert result.verified is True


def test_equivalence_false():
    result = run_check(SympyCheck(kind="equivalence", lhs="(x+3)**2", rhs="x**2+9"))
    assert result.verified is False
    assert "not equivalent" in result.detail.lower()


def test_solution_set_correct():
    result = run_check(
        SympyCheck(
            kind="solution_set",
            equation="(x+3)**2 - 25",
            variable="x",
            candidates=["2", "-8"],
        )
    )
    assert result.verified is True


def test_solution_set_incomplete_is_not_verified():
    result = run_check(
        SympyCheck(kind="solution_set", equation="(x+3)**2 - 25", variable="x", candidates=["4"])
    )
    assert result.verified is False


def test_skip_kind_is_not_verified_but_records_reason():
    result = run_check(SympyCheck(kind="skip", skip_reason="word problem, not symbolic"))
    assert result.verified is False
    assert "word problem" in result.detail


def test_malformed_expression_does_not_raise():
    result = run_check(SympyCheck(kind="equivalence", lhs="((((", rhs="x"))
    assert result.verified is False
    assert result.detail  # a reason is always given


def test_latex_input_is_rejected_gracefully():
    # The model is told to emit SymPy syntax; LaTeX must fail closed, not crash.
    result = run_check(SympyCheck(kind="equivalence", lhs=r"\frac{1}{2}", rhs="0.5"))
    assert result.verified is False


def test_derivative_equivalence():
    result = run_check(
        SympyCheck(kind="equivalence", lhs="diff(sin(x)*x, x)", rhs="sin(x) + x*cos(x)")
    )
    assert result.verified is True


# --- Additional edge cases the reviewer will probe (see task-7-report.md) ---


def test_equivalence_true_for_a_rational_function_identity():
    # (x**3-1)/(x-1) == x**2+x+1 for all x != 1. NOTE: on the currently pinned
    # sympy version (1.14.0), simplify() alone already resolves this to 0, so
    # this test does *not* specifically exercise the .equals(0) fallback in
    # _is_zero() (an earlier version of this test claimed it did — that claim
    # was wrong and has been corrected). It's kept as a general regression
    # test for the equivalence path. test_equivalence_conservatively_unverified_
    # without_domain_assumptions below is the test that actually exercises the
    # .equals(0) code path (it returns None there, correctly staying unverified).
    result = run_check(SympyCheck(kind="equivalence", lhs="(x**3 - 1)/(x - 1)", rhs="x**2 + x + 1"))
    assert result.verified is True


def test_equivalence_conservatively_unverified_without_domain_assumptions():
    # sqrt(x**2) == Abs(x) only holds for real x; with no assumption on the
    # symbol, sympy can't prove it (.equals(0) returns None), so this must
    # fail closed rather than guess true.
    result = run_check(SympyCheck(kind="equivalence", lhs="sqrt(x**2)", rhs="Abs(x)"))
    assert result.verified is False


def test_solution_set_candidates_with_duplicates_and_reversed_order():
    result = run_check(
        SympyCheck(
            kind="solution_set",
            equation="(x+3)**2 - 25",
            variable="x",
            candidates=["-8", "2", "2", "-8"],
        )
    )
    assert result.verified is True


def test_solution_set_infinite_true_solution_is_not_verified():
    # 0*x = 0 is satisfied by every real x; solveset returns S.Reals, which is
    # not a FiniteSet and must never be reported as matching a finite guess.
    result = run_check(
        SympyCheck(kind="solution_set", equation="0*x", variable="x", candidates=["0"])
    )
    assert result.verified is False


def test_solution_set_periodic_solution_is_not_verified():
    # sin(x) = 0 has infinitely many real solutions; solveset returns a Union
    # of ImageSets, not a FiniteSet, so a finite candidate guess can't match.
    result = run_check(
        SympyCheck(kind="solution_set", equation="sin(x)", variable="x", candidates=["0"])
    )
    assert result.verified is False


def test_solution_set_variable_name_collides_with_sympy_builtin():
    # "I" is sympy's imaginary unit constant. Without injecting the declared
    # variable as a local override, `parse_expr` would silently bind "I" to
    # the builtin constant instead of the student's variable, producing a
    # nonsensical empty solution set for a genuinely correct answer.
    result = run_check(
        SympyCheck(kind="solution_set", equation="2*I - 6", variable="I", candidates=["3"])
    )
    assert result.verified is True


def test_skip_kind_without_reason_still_has_a_detail():
    result = run_check(SympyCheck(kind="skip"))
    assert result.verified is False
    assert result.detail


def test_enormous_exponent_does_not_hang_and_is_rejected():
    result = run_check(SympyCheck(kind="equivalence", lhs="2**10000000000", rhs="0"))
    assert result.verified is False
    assert result.detail


def test_undefined_multiletter_function_call_does_not_silently_miscompute():
    # Without a guard, sympy's implicit-multiplication transform silently
    # splits an unrecognized multi-letter function name into a product of
    # one-letter symbols instead of raising - a confident wrong answer.
    result = run_check(
        SympyCheck(kind="equivalence", lhs="notarealfunction(x)", rhs="notarealfunction(x)")
    )
    assert result.verified is False


def test_equation_referencing_undeclared_variable_does_not_raise():
    # Equation doesn't mention the declared variable at all.
    result = run_check(
        SympyCheck(kind="solution_set", equation="y - 5", variable="x", candidates=["5"])
    )
    assert result.verified is False


def test_code_execution_payload_is_rejected_not_executed():
    # Regression test for a real vulnerability found during implementation:
    # sympy.parsing.sympy_parser.parse_expr evaluates through Python's eval()
    # under the hood, so an unsanitized string is arbitrary code execution,
    # not just an unusual math expression. This must fail closed, without
    # ever reaching eval (the module must not touch the filesystem/network).
    result = run_check(
        SympyCheck(
            kind="equivalence",
            lhs="__import__('os').system('echo pwned')",
            rhs="0",
        )
    )
    assert result.verified is False
    assert "unsupported characters" in result.detail


def test_sandbox_escape_via_attribute_chain_is_rejected():
    result = run_check(SympyCheck(kind="equivalence", lhs="().__class__.__bases__[0]", rhs="0"))
    assert result.verified is False


def test_empty_candidates_against_genuinely_empty_solution_set_is_verified():
    # x**2 + 1 = 0 has no real roots. Claiming an empty candidate list is the
    # mathematically correct answer, and must not crash on the empty FiniteSet.
    result = run_check(
        SympyCheck(kind="solution_set", equation="x**2 + 1", variable="x", candidates=[])
    )
    assert result.verified is True


def test_empty_candidates_against_nonempty_solution_set_is_not_verified():
    result = run_check(
        SympyCheck(kind="solution_set", equation="x - 5", variable="x", candidates=[])
    )
    assert result.verified is False


# --- Fix round 1: process-boundary timeout, variable validation (see
# task-7-report.md "Fix round 1" section for the full writeup) ---


def test_composed_exponent_tower_times_out_instead_of_hanging():
    # _has_dangerous_power only inspects each Pow node's own literal exponent.
    # In "2**2**2**2**2**2" every outer Pow's exponent is itself an
    # unevaluated Pow, not an Integer, so that guard doesn't fire on any of
    # them - only the process-level wall-clock timeout in run_check catches
    # this. Asserting elapsed time turns a regression back into a hang into a
    # fast, loud test failure instead of a suite that never finishes.
    start = time.monotonic()
    result = run_check(
        SympyCheck(kind="equivalence", lhs="2**2**2**2**2**2", rhs="0"),
        timeout=_HANG_TEST_TIMEOUT,
    )
    elapsed = time.monotonic() - start
    assert result.verified is False
    assert "timed out" in result.detail
    assert elapsed < _HANG_TEST_TIMEOUT + 5, (
        f"took {elapsed:.1f}s - timeout enforcement appears to not be working"
    )


def test_expensive_non_pow_function_times_out_instead_of_hanging():
    # factorial (and binomial/primorial/factorint of a large number) is a
    # plain allow-listed call needing no brackets or quotes - _has_dangerous_
    # power never even looks at it, since there's no Pow node involved at
    # all. Only the process timeout bounds this class of input.
    start = time.monotonic()
    result = run_check(
        SympyCheck(kind="equivalence", lhs="factorial(2000000)", rhs="0"),
        timeout=_HANG_TEST_TIMEOUT,
    )
    elapsed = time.monotonic() - start
    assert result.verified is False
    assert "timed out" in result.detail
    assert elapsed < _HANG_TEST_TIMEOUT + 5, (
        f"took {elapsed:.1f}s - timeout enforcement appears to not be working"
    )


def test_illegal_variable_identifier_is_rejected_not_silently_wrong():
    # Exact repro of a real spurious-verified=True bug: "5x" is not a legal
    # identifier, so sympy.Symbol("5x") is unrelated to anything in the
    # equation. Before this fix, both sides came out EmptySet (solveset for a
    # symbol absent from the equation, vs. an empty claimed candidate list)
    # and compared equal - a confident "verified" that never actually related
    # the claimed variable to the equation.
    result = run_check(
        SympyCheck(kind="solution_set", equation="x-5", variable="5x", candidates=[])
    )
    assert result.verified is False
    assert "not a legal variable name" in result.detail


def test_whitespace_variable_is_rejected_not_silently_wrong():
    result = run_check(
        SympyCheck(kind="solution_set", equation="x-5", variable="  ", candidates=[])
    )
    assert result.verified is False
    assert "not a legal variable name" in result.detail


def test_legal_variable_absent_from_equation_with_empty_candidates_is_not_verified():
    # "y" is a legal identifier but doesn't appear in "x - 5" at all. Without
    # the free_symbols membership check this also spuriously verifies True
    # for the same reason as the "5x" case (both sides EmptySet).
    result = run_check(
        SympyCheck(kind="solution_set", equation="x - 5", variable="y", candidates=[])
    )
    assert result.verified is False
    assert "does not appear in the equation" in result.detail
