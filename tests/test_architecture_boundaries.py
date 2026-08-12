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


def test_active_code_does_not_depend_on_temporary_refactor_facades() -> None:
    """Compatibility facades are migration scaffolding, never internal dependencies."""

    forbidden_by_tree = {
        _PACKAGE_ROOT / "caging": ("qlinks.caging.stability",),
        _PACKAGE_ROOT / "open_system": ("qlinks.open_system.manifold_detectors",),
    }
    allowed_paths = {
        _PACKAGE_ROOT / "caging" / "__init__.py",
    }

    violations: list[str] = []
    for tree, forbidden_prefixes in forbidden_by_tree.items():
        for path in sorted(tree.rglob("*.py")):
            if path in allowed_paths:
                continue
            for imported_module in _python_imports(path):
                if any(
                    imported_module == prefix or imported_module.startswith(f"{prefix}.")
                    for prefix in forbidden_prefixes
                ):
                    relative_path = path.relative_to(_REPOSITORY_ROOT)
                    violations.append(f"{relative_path}: {imported_module}")

    assert not violations, (
        "Active code imports a temporary refactor compatibility facade; "
        "import the focused implementation module instead:\n" + "\n".join(violations)
    )


def test_lazy_compatibility_facades_use_pyflakes_safe_all() -> None:
    """Lazy legacy exports must not trigger flake8/pyflakes F822."""

    facade_paths = (
        _PACKAGE_ROOT / "caging" / "stability.py",
        _PACKAGE_ROOT / "open_system" / "manifold_detectors.py",
    )
    violations: list[str] = []

    for path in facade_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        has_lazy_getattr = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__"
            for node in tree.body
        )
        if not has_lazy_getattr:
            continue

        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            is_all_assignment = any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in targets
            )
            if not is_all_assignment:
                continue
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                relative_path = path.relative_to(_REPOSITORY_ROOT)
                violations.append(str(relative_path))

    assert not violations, (
        "Lazy compatibility facades use a literal __all__, which pyflakes interprets as eager "
        "bindings and reports as F822:\n" + "\n".join(violations)
    )
