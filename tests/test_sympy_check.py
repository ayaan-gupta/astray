from server.charter.contracts import SympyCheck
from server.verify.sympy_check import run_check


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


def test_equivalence_true_when_simplify_alone_is_inconclusive():
    # simplify() alone leaves `(x**3-1)/(x-1) - (x**2+x+1)` unreduced in some
    # sympy versions; .equals(0) (numeric sampling) is the fallback that catches it.
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
