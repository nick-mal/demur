"""The library must not depend on the example.

`src/demur/` is domain-blind: it knows about trajectories, constraints, scorers
and statistics, never about SQL or warehouses. `examples/governed_warehouse/`
is one specimen that exercises it.

Architecture rules that aren't executable decay. This one is executable.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY = REPO_ROOT / "src" / "demur"

# Any import whose dotted path starts with one of these is a boundary violation.
FORBIDDEN_ROOTS = ("examples",)


def library_modules() -> list[Path]:
    return sorted(p for p in LIBRARY.rglob("*.py"))


def imported_names(tree: ast.AST, module_path: Path) -> list[tuple[str, int]]:
    """Every dotted name this module imports, with its line number.

    Relative imports (`from . import x`) are resolved against the module's own
    package so that a relative escape out of the library is caught too.
    """
    found: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))

        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.append((node.module, node.lineno))
                continue

            # Relative import: walk up `level` packages from this module.
            package = module_path.parent
            for _ in range(node.level - 1):
                package = package.parent
            try:
                rel = package.relative_to(REPO_ROOT)
            except ValueError:
                # Escaped the repository entirely — definitely a violation.
                found.append(("<escaped-repo-root>", node.lineno))
                continue
            parts = [p for p in rel.parts if p != "src"]
            if node.module:
                parts.append(node.module)
            found.append((".".join(parts), node.lineno))

    return found


@pytest.mark.parametrize(
    "module", library_modules(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_library_does_not_import_example(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))

    violations = [
        (name, lineno)
        for name, lineno in imported_names(tree, module)
        if name.split(".")[0] in FORBIDDEN_ROOTS or name == "<escaped-repo-root>"
    ]

    assert not violations, "\n".join(
        f"{module.relative_to(REPO_ROOT)}:{lineno} imports {name!r} — "
        "the library must not depend on the example. If demur needs to know "
        "this, the abstraction is wrong: lift the concept into the library "
        "instead of importing the specimen."
        for name, lineno in violations
    )


def test_library_has_modules_to_check() -> None:
    """Guard against the suite passing vacuously.

    If the layout moves and `library_modules()` returns nothing, every
    parametrised case silently disappears and the boundary stops being
    enforced without anything going red.
    """
    assert LIBRARY.is_dir(), f"expected library at {LIBRARY}"
    assert library_modules(), "no modules found under src/demur — check the layout"