"""Static gate on LLM-authored scene code. Deterministic -- no model involved.

This is s8. It runs on the host, before any container starts, and it is the
second of two independent layers: the container (`--network=none`, read-only,
non-root, memory/pid capped, wall-clock killed) is the first. Neither is trusted
alone, because each fails differently -- a container escape defeats the sandbox
but not the allow-list, and a novel AST shape defeats the allow-list but not the
sandbox.

The code being validated was written by a model whose prompt chain begins with
untrusted student text, so this is an injection boundary, not a lint pass. The
same reasoning that governs `server/verify/sympy_check.py` applies: reject by
allow-list, never by deny-list alone, because a deny-list only stops the attacks
already imagined.

`s8_validate` is also where the grounding contract is enforced. Every beat the
storyboard planned must appear exactly once in a `with beat(self, "bN")` block.
A missing beat is a citation the tutor can make that seeks nowhere; a duplicate
is a citation that is ambiguous between two moments. Both fail the render into
the repair loop rather than degrading grounding silently.
"""

import ast

from server.charter.contracts import SceneCode, Storyboard, ValidationIssue, ValidationReport

# Only these may be imported. `primitives` is the vetted helper package mounted
# read-only into the container; `manim`, `numpy` and `math` are what a scene
# legitimately needs. Everything else -- including stdlib modules that look
# harmless -- is rejected, because "harmless" is exactly the judgement an
# allow-list exists to avoid making case by case.
ALLOWED_IMPORT_ROOTS = frozenset({"manim", "numpy", "np", "math", "primitives"})

# Names that are dangerous even without an import: builtins reachable from any
# scope. `__import__` and `eval`/`exec`/`compile` reconstruct the import system;
# `open` reaches the filesystem; `globals`/`vars`/`locals` walk to everything else.
DENIED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "importlib",
        "builtins",
        "ctypes",
        "pickle",
        "marshal",
    }
)


def _dunder(name: str) -> bool:
    """Any dunder attribute is refused.

    `().__class__.__bases__[0].__subclasses__()` reaches every loaded class from
    a bare tuple literal -- no import, no denied name. Blocking dunder attribute
    access as a category is what closes that whole family, rather than naming
    its members one at a time.
    """
    return name.startswith("__") and name.endswith("__")


def validate(code: str, storyboard: Storyboard, scene_class_name: str) -> ValidationReport:
    """Check `code` against the import allow-list, name deny-list, structure
    requirements and the storyboard's beat coverage."""
    issues: list[ValidationIssue] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # A file that does not parse cannot be inspected further, and running it
        # would just move the same error into the container.
        return ValidationReport(
            ok=False,
            issues=[ValidationIssue(kind="syntax", detail=str(exc.msg), line=exc.lineno)],
        )

    beat_ids: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    issues.append(
                        ValidationIssue(
                            kind="import",
                            detail=f"import of {alias.name!r} is not allowed",
                            line=node.lineno,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            # `from . import x` (level > 0) has no module root to check and would
            # otherwise slip through as an empty string.
            if node.level or root not in ALLOWED_IMPORT_ROOTS:
                issues.append(
                    ValidationIssue(
                        kind="import",
                        detail=f"import from {node.module or '.'!r} is not allowed",
                        line=node.lineno,
                    )
                )
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            issues.append(
                ValidationIssue(
                    kind="name", detail=f"use of {node.id!r} is not allowed", line=node.lineno
                )
            )
        elif isinstance(node, ast.Attribute):
            if _dunder(node.attr):
                issues.append(
                    ValidationIssue(
                        kind="name",
                        detail=f"dunder attribute {node.attr!r} is not allowed",
                        line=node.lineno,
                    )
                )
        elif isinstance(node, ast.Call):
            beat_id = _beat_id_of(node)
            if beat_id is not None:
                beat_ids.append(beat_id)
        # A `with beat(self, "b1"):` statement's call is reached by ast.walk as a
        # plain Call above, so `with` needs no separate handling.

    issues.extend(_structure_issues(tree, scene_class_name))
    issues.extend(_beat_issues(beat_ids, storyboard))

    return ValidationReport(ok=not issues, issues=issues)


def _beat_id_of(node: ast.Call) -> str | None:
    """Return the beat id if `node` is a `beat(scene, "bN")` call, else None."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name != "beat" or len(node.args) < 2:
        return None
    literal = node.args[1]
    if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
        return literal.value
    # A computed beat id (f-string, variable, concatenation) is refused by
    # _beat_issues via the coverage check: it cannot be matched to a planned id
    # statically, which is the whole point of checking before running.
    return None


def _structure_issues(tree: ast.Module, scene_class_name: str) -> list[ValidationIssue]:
    """Exactly one Scene subclass, named as the contract specifies.

    The runner invokes manim with this class name; two Scene subclasses would
    make which one renders depend on manim's discovery order, and zero means the
    render fails inside the container instead of here.
    """
    scenes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(base, ast.Name) and base.id.endswith("Scene"))
            or (isinstance(base, ast.Attribute) and base.attr.endswith("Scene"))
            for base in node.bases
        )
    ]
    if not scenes:
        return [ValidationIssue(kind="structure", detail="no Scene subclass found")]
    if len(scenes) > 1:
        names = ", ".join(scene.name for scene in scenes)
        return [
            ValidationIssue(kind="structure", detail=f"expected exactly one Scene, found: {names}")
        ]
    if scenes[0].name != scene_class_name:
        return [
            ValidationIssue(
                kind="structure",
                detail=f"Scene class must be named {scene_class_name!r}, found {scenes[0].name!r}",
                line=scenes[0].lineno,
            )
        ]
    return []


def _beat_issues(found: list[str], storyboard: Storyboard) -> list[ValidationIssue]:
    """Every planned beat wrapped exactly once; nothing wrapped that was not planned."""
    issues: list[ValidationIssue] = []
    planned = [beat.id for beat in storyboard.beats]

    for beat_id in planned:
        count = found.count(beat_id)
        if count == 0:
            issues.append(
                ValidationIssue(
                    kind="beat", detail=f"planned beat {beat_id!r} is never wrapped in beat(...)"
                )
            )
        elif count > 1:
            issues.append(
                ValidationIssue(
                    kind="beat",
                    detail=f"beat {beat_id!r} is wrapped {count} times; each beat must appear once",
                )
            )

    for beat_id in sorted(set(found) - set(planned)):
        issues.append(
            ValidationIssue(kind="beat", detail=f"beat {beat_id!r} is not in the storyboard")
        )
    return issues


def validate_scene(scene: SceneCode, storyboard: Storyboard) -> ValidationReport:
    """Convenience wrapper: validate a SceneCode against its storyboard."""
    return validate(scene.code, storyboard, scene.scene_class_name)
