"""Seed misconceptions so aggregate insights are not cold-start empty and common
errors attach to a curated entry instead of minting near-duplicates."""

import sqlite3

# (slug, canonical_statement, buggy_rule, topic)
SEEDS: list[tuple[str, str, str, str]] = [
    (
        "freshmans-dream",
        "Distributes an exponent across a sum, dropping the cross term.",
        "(a+b)^2 -> a^2 + b^2",
        "algebra.binomial_expansion",
    ),
    (
        "fraction-add-across",
        "Adds numerators and denominators separately.",
        "a/b + c/d -> (a+c)/(b+d)",
        "arithmetic.fractions",
    ),
    (
        "negative-distribute",
        "Distributes a leading minus sign to only the first term.",
        "-(a+b) -> -a + b",
        "algebra.signs",
    ),
    (
        "cancel-across-sum",
        "Cancels a term that is added, not multiplied.",
        "(a+b)/a -> b",
        "algebra.rational_expressions",
    ),
    (
        "log-of-sum",
        "Treats the log of a sum as the sum of logs.",
        "log(a+b) -> log(a) + log(b)",
        "algebra.logarithms",
    ),
    (
        "sqrt-of-sum",
        "Treats the root of a sum as the sum of roots.",
        "sqrt(a+b) -> sqrt(a) + sqrt(b)",
        "algebra.radicals",
    ),
    (
        "power-of-power-add",
        "Adds exponents when raising a power to a power.",
        "(a^m)^n -> a^(m+n)",
        "algebra.exponents",
    ),
    (
        "exponent-mult-multiply",
        "Multiplies exponents when multiplying like bases.",
        "a^m * a^n -> a^(m*n)",
        "algebra.exponents",
    ),
    (
        "negative-exponent-sign",
        "Treats a negative exponent as negating the base.",
        "a^(-n) -> -a^n",
        "algebra.exponents",
    ),
    (
        "square-root-single-sign",
        "Keeps only the positive root when solving by square roots.",
        "x^2 = k -> x = sqrt(k)",
        "algebra.quadratics",
    ),
    (
        "divide-by-variable",
        "Divides both sides by a variable, losing the x=0 solution.",
        "x^2 = x -> x = 1",
        "algebra.quadratics",
    ),
    (
        "cross-multiply-add",
        "Cross-multiplies an equation that is a sum, not a proportion.",
        "a/b + c/d = e -> ad + cb = e",
        "algebra.rational_equations",
    ),
    (
        "inequality-flip",
        "Fails to flip the inequality when multiplying by a negative.",
        "-2x < 6 -> x < -3",
        "algebra.inequalities",
    ),
    (
        "abs-value-single-case",
        "Solves an absolute value equation with only the positive case.",
        "|x| = k -> x = k",
        "algebra.absolute_value",
    ),
    (
        "slope-inverted",
        "Computes slope as run over rise.",
        "m -> (x2-x1)/(y2-y1)",
        "algebra.linear_functions",
    ),
    (
        "function-notation-multiply",
        "Reads f(x) as f times x.",
        "f(a+b) -> f(a) + f(b)",
        "algebra.functions",
    ),
    (
        "pemdas-left-right",
        "Applies operations strictly left to right, ignoring precedence.",
        "2 + 3*4 -> 20",
        "arithmetic.order_of_operations",
    ),
    (
        "percent-of-vs-change",
        "Confuses percent change with percent of the original.",
        "increase by 20% then decrease 20% -> original",
        "arithmetic.percent",
    ),
    (
        "chain-rule-omitted",
        "Differentiates the outer function without the inner derivative.",
        "d/dx f(g(x)) -> f'(g(x))",
        "calculus.derivatives",
    ),
    (
        "product-rule-as-product",
        "Differentiates a product as the product of derivatives.",
        "d/dx (f*g) -> f' * g'",
        "calculus.derivatives",
    ),
    (
        "quotient-rule-as-quotient",
        "Differentiates a quotient as the quotient of derivatives.",
        "d/dx (f/g) -> f'/g'",
        "calculus.derivatives",
    ),
    (
        "power-rule-on-exponential",
        "Applies the power rule to a^x.",
        "d/dx a^x -> x*a^(x-1)",
        "calculus.derivatives",
    ),
    (
        "integral-of-product",
        "Integrates a product as the product of integrals.",
        "int f*g -> (int f)*(int g)",
        "calculus.integrals",
    ),
    (
        "constant-of-integration-omitted",
        "Omits +C on an indefinite integral.",
        "int f -> F",
        "calculus.integrals",
    ),
    (
        "limit-by-substitution-always",
        "Substitutes into an indeterminate form and stops.",
        "lim -> 0/0 as the answer",
        "calculus.limits",
    ),
    (
        "derivative-is-slope-of-secant",
        "Confuses average rate of change with instantaneous.",
        "f'(a) -> (f(b)-f(a))/(b-a)",
        "calculus.derivatives",
    ),
    (
        "trig-distribute",
        "Distributes a trig function across a sum.",
        "sin(a+b) -> sin(a) + sin(b)",
        "trigonometry.identities",
    ),
    (
        "degrees-radians-mixed",
        "Evaluates a trig function in the wrong angle mode.",
        "sin(pi/2) -> 0.0274",
        "trigonometry.evaluation",
    ),
    (
        "pythagorean-nonright",
        "Applies the Pythagorean theorem to a non-right triangle.",
        "a^2+b^2=c^2 for any triangle",
        "geometry.triangles",
    ),
    (
        "area-perimeter-swap",
        "Uses a perimeter formula where area is required.",
        "area of rectangle -> 2(l+w)",
        "geometry.measurement",
    ),
    (
        "scale-factor-area",
        "Scales area by the linear scale factor.",
        "double sides -> double area",
        "geometry.similarity",
    ),
    (
        "probability-add-dependent",
        "Adds probabilities of non-mutually-exclusive events.",
        "P(A or B) -> P(A) + P(B)",
        "probability.basic",
    ),
    (
        "probability-mult-dependent",
        "Multiplies probabilities of dependent events as independent.",
        "P(A and B) -> P(A)*P(B)",
        "probability.basic",
    ),
    (
        "gamblers-fallacy",
        "Believes past independent outcomes change future probability.",
        "after 5 heads, tails is more likely",
        "probability.independence",
    ),
    (
        "mean-median-confusion",
        "Reports the middle value of an unsorted list as the median.",
        "median -> middle of unsorted list",
        "statistics.center",
    ),
    (
        "correlation-causation",
        "Infers causation from correlation.",
        "corr(a,b) -> a causes b",
        "statistics.inference",
    ),
    (
        "distribute-into-denominator",
        "Distributes a factor into only part of a denominator.",
        "c/(a+b) -> c/a + c/b",
        "algebra.rational_expressions",
    ),
    (
        "like-terms-unlike",
        "Combines terms with different powers as like terms.",
        "x + x^2 -> x^3",
        "algebra.simplification",
    ),
    (
        "zero-product-misuse",
        "Sets factors equal to the constant instead of zero.",
        "(x-2)(x-3)=6 -> x-2=6",
        "algebra.quadratics",
    ),
    (
        "sign-error-transposition",
        "Moves a term across the equals sign without changing sign.",
        "x + 5 = 9 -> x = 9 + 5",
        "algebra.linear_equations",
    ),
]


def seed(conn: sqlite3.Connection) -> int:
    """Insert seed rows. Idempotent — returns the number newly inserted."""
    from server.store.taxonomy import canonicalize_rule

    inserted = 0
    for slug, statement, rule, topic in SEEDS:
        existing = conn.execute("SELECT 1 FROM misconceptions WHERE slug = ?", (slug,)).fetchone()
        if existing:
            continue
        conn.execute(
            """INSERT INTO misconceptions
               (slug, canonical_statement, canonical_rule, topic, is_seed)
               VALUES (?, ?, ?, ?, 1)""",
            (slug, statement, canonicalize_rule(rule), topic),
        )
        inserted += 1
    return inserted
