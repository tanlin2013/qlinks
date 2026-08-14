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


def test_removed_refactor_facades_are_not_reintroduced() -> None:
    """The reviewed API cleanup removes the temporary refactor facades entirely."""

    facade_paths = (
        _PACKAGE_ROOT / "caging" / "local_search.py",
        _PACKAGE_ROOT / "caging" / "stability.py",
        _PACKAGE_ROOT / "open_system" / "manifold_detectors.py",
    )
    violations = [str(path.relative_to(_REPOSITORY_ROOT)) for path in facade_paths if path.exists()]

    assert not violations, (
        "Temporary refactor facades were removed during API cleanup and must not be "
        "reintroduced:\n" + "\n".join(violations)
    )


def test_local_search_modules_follow_one_way_dependency_order() -> None:
    """Focused local-search modules must follow the reviewed dependency DAG."""

    allowed_dependencies = {
        "local_search_types.py": set(),
        "local_search_geometry.py": {"qlinks.caging.local_search_types"},
        "local_search_core.py": {"qlinks.caging.local_search_types"},
        "local_search_qdm.py": {
            "qlinks.caging.local_search_core",
            "qlinks.caging.local_search_geometry",
            "qlinks.caging.local_search_types",
        },
        "local_search_global.py": {
            "qlinks.caging.local_search_qdm",
            "qlinks.caging.local_search_types",
        },
        "local_search_padding.py": {
            "qlinks.caging.local_search_geometry",
            "qlinks.caging.local_search_global",
            "qlinks.caging.local_search_types",
        },
        "local_search_factorized.py": {
            "qlinks.caging.local_search_global",
            "qlinks.caging.local_search_padding",
            "qlinks.caging.local_search_qdm",
            "qlinks.caging.local_search_types",
        },
        "local_search_certification.py": {
            "qlinks.caging.local_search_global",
            "qlinks.caging.local_search_padding",
            "qlinks.caging.local_search_qdm",
            "qlinks.caging.local_search_types",
        },
        "local_search_proposals.py": {
            "qlinks.caging.local_search_core",
            "qlinks.caging.local_search_geometry",
            "qlinks.caging.local_search_types",
        },
        "local_search_scan.py": {
            "qlinks.caging.local_search_core",
            "qlinks.caging.local_search_padding",
            "qlinks.caging.local_search_types",
        },
        "local_search_workflows.py": {
            "qlinks.caging.local_search_certification",
            "qlinks.caging.local_search_proposals",
            "qlinks.caging.local_search_scan",
            "qlinks.caging.local_search_types",
        },
    }

    violations: list[str] = []
    local_prefix = "qlinks.caging.local_search_"
    for filename, allowed in allowed_dependencies.items():
        path = _PACKAGE_ROOT / "caging" / filename
        for imported_module in _python_imports(path):
            if imported_module.startswith(local_prefix) and imported_module not in allowed:
                violations.append(f"{filename}: {imported_module}")

    assert not violations, "Local-search dependency DAG violation:\n" + "\n".join(violations)


def test_repository_does_not_import_removed_refactor_facades() -> None:
    """No code or tests should still import the removed migration-only module paths."""

    forbidden_prefixes = (
        "qlinks.caging.local_search",
        "qlinks.caging.stability",
        "qlinks.open_system.manifold_detectors",
    )

    violations: list[str] = []
    for tree in (_PACKAGE_ROOT, _REPOSITORY_ROOT / "tests"):
        for path in sorted(tree.rglob("*.py")):
            for imported_module in _python_imports(path):
                if any(
                    imported_module == prefix or imported_module.startswith(f"{prefix}.")
                    for prefix in forbidden_prefixes
                ):
                    relative_path = path.relative_to(_REPOSITORY_ROOT)
                    violations.append(f"{relative_path}: {imported_module}")

    assert (
        not violations
    ), "Removed refactor facade imports remain in the repository:\n" + "\n".join(violations)
