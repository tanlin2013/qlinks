"""Fast guardrails for repository-layer dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "qlinks"


def _python_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    return tuple(imports)


def _assert_tree_does_not_import(tree: Path, forbidden_prefixes: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in sorted(tree.rglob("*.py")):
        for imported_module in _python_imports(path):
            if any(
                imported_module == prefix or imported_module.startswith(f"{prefix}.")
                for prefix in forbidden_prefixes
            ):
                relative_path = path.relative_to(_REPOSITORY_ROOT)
                violations.append(f"{relative_path}: {imported_module}")
    assert not violations, "Forbidden upward/cross-layer imports:\n" + "\n".join(violations)


def test_caging_does_not_depend_on_open_system() -> None:
    _assert_tree_does_not_import(
        _PACKAGE_ROOT / "caging",
        ("qlinks.open_system",),
    )


def test_local_structure_is_neutral() -> None:
    _assert_tree_does_not_import(
        _PACKAGE_ROOT / "local_structure",
        ("qlinks.caging", "qlinks.open_system"),
    )


def test_current_open_system_does_not_import_caging_implementation() -> None:
    violations: list[str] = []
    open_system_root = _PACKAGE_ROOT / "open_system"
    deprecated_root = open_system_root / "constructions" / "deprecated"

    for path in sorted(open_system_root.rglob("*.py")):
        if deprecated_root in path.parents:
            continue
        for imported_module in _python_imports(path):
            if imported_module == "qlinks.caging" or imported_module.startswith("qlinks.caging."):
                relative_path = path.relative_to(_REPOSITORY_ROOT)
                violations.append(f"{relative_path}: {imported_module}")

    assert (
        not violations
    ), "Current open-system code depends on caging implementation:\n" + "\n".join(violations)
