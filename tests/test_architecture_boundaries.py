"""Fast guardrails for repository-layer dependency direction."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "qlinks"


@lru_cache(maxsize=None)
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
        "types.py": set(),
        "geometry.py": {"qlinks.caging.local_search.types"},
        "core.py": {"qlinks.caging.local_search.types"},
        "qdm.py": {
            "qlinks.caging.local_search.core",
            "qlinks.caging.local_search.geometry",
            "qlinks.caging.local_search.types",
        },
        "global_ops.py": {
            "qlinks.caging.local_search.qdm",
            "qlinks.caging.local_search.types",
        },
        "padding.py": {
            "qlinks.caging.local_search.geometry",
            "qlinks.caging.local_search.global_ops",
            "qlinks.caging.local_search.types",
        },
        "factorized.py": {
            "qlinks.caging.local_search.global_ops",
            "qlinks.caging.local_search.padding",
            "qlinks.caging.local_search.qdm",
            "qlinks.caging.local_search.types",
        },
        "certification.py": {
            "qlinks.caging.local_search.global_ops",
            "qlinks.caging.local_search.padding",
            "qlinks.caging.local_search.qdm",
            "qlinks.caging.local_search.types",
        },
        "proposals.py": {
            "qlinks.caging.local_search.core",
            "qlinks.caging.local_search.geometry",
            "qlinks.caging.local_search.types",
        },
        "scan.py": {
            "qlinks.caging.local_search.core",
            "qlinks.caging.local_search.padding",
            "qlinks.caging.local_search.types",
        },
        "workflows.py": {
            "qlinks.caging.local_search.certification",
            "qlinks.caging.local_search.proposals",
            "qlinks.caging.local_search.scan",
            "qlinks.caging.local_search.types",
        },
    }

    violations: list[str] = []
    local_prefix = "qlinks.caging.local_search."
    local_root = _PACKAGE_ROOT / "caging" / "local_search"
    for filename, allowed in allowed_dependencies.items():
        path = local_root / filename
        for imported_module in _python_imports(path):
            if imported_module.startswith(local_prefix) and imported_module not in allowed:
                violations.append(f"{filename}: {imported_module}")

    assert not violations, "Local-search dependency DAG violation:\n" + "\n".join(violations)


def test_stability_modules_follow_one_way_dependency_order() -> None:
    """Focused stability modules must follow the reviewed responsibility DAG."""

    allowed_dependencies = {
        "types.py": set(),
        "symmetry.py": set(),
        "core.py": {"qlinks.caging.stability.types"},
        "laurent.py": {"qlinks.caging.stability.types"},
        "topology.py": {
            "qlinks.caging.stability.core",
            "qlinks.caging.stability.symmetry",
            "qlinks.caging.stability.types",
        },
        "boundary.py": {
            "qlinks.caging.stability.core",
            "qlinks.caging.stability.topology",
            "qlinks.caging.stability.types",
        },
        "qdm.py": {
            "qlinks.caging.stability.boundary",
            "qlinks.caging.stability.core",
            "qlinks.caging.stability.symmetry",
            "qlinks.caging.stability.types",
        },
    }

    violations: list[str] = []
    stability_prefix = "qlinks.caging.stability."
    stability_root = _PACKAGE_ROOT / "caging" / "stability"
    for filename, allowed in allowed_dependencies.items():
        path = stability_root / filename
        for imported_module in _python_imports(path):
            if imported_module.startswith(stability_prefix) and imported_module not in allowed:
                violations.append(f"{filename}: {imported_module}")

    assert not violations, "Stability dependency DAG violation:\n" + "\n".join(violations)


def test_caging_analysis_modules_follow_one_way_dependency_order() -> None:
    """Analysis and environment-reduction responsibilities follow reviewed DAGs."""

    allowed_dependencies = {
        "transitions.py": set(),
        "environment/__init__.py": {
            "qlinks.caging.analysis.environment.contracts",
            "qlinks.caging.analysis.environment.diagnosis",
            "qlinks.caging.analysis.environment.monitor",
            "qlinks.caging.analysis.environment.report",
            "qlinks.caging.analysis.environment.support",
            "qlinks.caging.analysis.transitions",
        },
        "environment/contracts.py": {"qlinks.caging.analysis.transitions"},
        "environment/support.py": {"qlinks.caging.analysis.environment.contracts"},
        "environment/monitor.py": {
            "qlinks.caging.analysis.environment.contracts",
            "qlinks.caging.analysis.environment.support",
        },
        "environment/operator.py": {
            "qlinks.caging.analysis.environment.contracts",
            "qlinks.caging.analysis.environment.support",
            "qlinks.caging.analysis.transitions",
        },
        "environment/discovery.py": {
            "qlinks.caging.analysis.environment.contracts",
            "qlinks.caging.analysis.environment.operator",
            "qlinks.caging.analysis.environment.support",
        },
        "environment/mechanisms.py": {
            "qlinks.caging.analysis.environment.contracts",
            "qlinks.caging.analysis.environment.operator",
            "qlinks.caging.analysis.environment.support",
            "qlinks.caging.analysis.transitions",
        },
        "environment/summary.py": {"qlinks.caging.analysis.environment.contracts"},
        "environment/report.py": {
            "qlinks.caging.analysis.environment.contracts",
            "qlinks.caging.analysis.environment.monitor",
        },
        "environment/diagnosis.py": {
            "qlinks.caging.analysis.environment.contracts",
            "qlinks.caging.analysis.environment.discovery",
            "qlinks.caging.analysis.environment.mechanisms",
            "qlinks.caging.analysis.environment.monitor",
            "qlinks.caging.analysis.environment.operator",
            "qlinks.caging.analysis.environment.report",
            "qlinks.caging.analysis.environment.summary",
            "qlinks.caging.analysis.transitions",
        },
        "local_structure.py": {"qlinks.caging.analysis.environment"},
        "support.py": {
            "qlinks.caging.analysis.environment",
            "qlinks.caging.analysis.transitions",
        },
        "support_morphology.py": {"qlinks.caging.analysis.environment"},
        "spectral.py": set(),
        "thermodynamic.py": {
            "qlinks.caging.analysis.environment",
            "qlinks.caging.analysis.support",
        },
        "evidence.py": {"qlinks.caging.analysis.spectral"},
    }

    violations: list[str] = []
    analysis_prefix = "qlinks.caging.analysis."
    analysis_root = _PACKAGE_ROOT / "caging" / "analysis"
    for relative_path, allowed in allowed_dependencies.items():
        path = analysis_root / relative_path
        for imported_module in _python_imports(path):
            if imported_module.startswith(analysis_prefix) and imported_module not in allowed:
                violations.append(f"{relative_path}: {imported_module}")

    assert not violations, "Caging-analysis dependency DAG violation:\n" + "\n".join(violations)


def test_basis_visualizer_modules_follow_reviewed_dependency_dag() -> None:
    """The split basis visualizer keeps rendering roles one-way and facade-free."""

    allowed_dependencies = {
        "styles.py": set(),
        "render_cache.py": {"qlinks.visualizer.basis.styles"},
        "formatting.py": {"qlinks.visualizer.basis.styles"},
        "rendering.py": {
            "qlinks.visualizer.basis.render_cache",
            "qlinks.visualizer.basis.styles",
        },
        "periodic.py": {
            "qlinks.visualizer.basis.render_cache",
            "qlinks.visualizer.basis.styles",
        },
        "plaquette_geometry.py": {
            "qlinks.visualizer.basis.render_cache",
            "qlinks.visualizer.basis.styles",
        },
        "plaquette_symbols.py": {
            "qlinks.visualizer.basis.render_cache",
            "qlinks.visualizer.basis.styles",
        },
        "configuration.py": {
            "qlinks.visualizer.basis.periodic",
            "qlinks.visualizer.basis.plaquette_geometry",
            "qlinks.visualizer.basis.plaquette_symbols",
            "qlinks.visualizer.basis.render_cache",
            "qlinks.visualizer.basis.rendering",
            "qlinks.visualizer.basis.styles",
        },
        "api.py": {
            "qlinks.visualizer.basis.configuration",
            "qlinks.visualizer.basis.styles",
        },
        "grid.py": {
            "qlinks.visualizer.basis.configuration",
            "qlinks.visualizer.basis.formatting",
            "qlinks.visualizer.basis.render_cache",
            "qlinks.visualizer.basis.styles",
        },
        "local_grid.py": {
            "qlinks.visualizer.basis.configuration",
            "qlinks.visualizer.basis.formatting",
            "qlinks.visualizer.basis.render_cache",
            "qlinks.visualizer.basis.styles",
        },
    }

    violations: list[str] = []
    basis_prefix = "qlinks.visualizer.basis."
    basis_root = _PACKAGE_ROOT / "visualizer" / "basis"
    for filename, allowed in allowed_dependencies.items():
        path = basis_root / filename
        for imported_module in _python_imports(path):
            if imported_module.startswith(basis_prefix) and imported_module not in allowed:
                violations.append(f"{filename}: {imported_module}")

    assert not violations, "Basis-visualizer dependency DAG violation:\n" + "\n".join(violations)


def test_open_system_subspace_helpers_do_not_depend_on_diagnostics_or_manifolds() -> None:
    """Shared common-kernel algebra stays below diagnostics and detector workflows."""

    path = _PACKAGE_ROOT / "open_system" / "_subspace.py"
    forbidden = (
        "qlinks.open_system.diagnostics",
        "qlinks.open_system.manifold_dark",
        "qlinks.open_system.manifold_recycling",
        "qlinks.open_system.manifold_residual",
    )
    violations = [
        imported_module
        for imported_module in _python_imports(path)
        if any(
            imported_module == prefix or imported_module.startswith(f"{prefix}.")
            for prefix in forbidden
        )
    ]
    assert not violations, "Shared open-system subspace algebra has upward dependencies: " + str(
        violations
    )


def test_open_system_implementation_avoids_diagnostics_package_facade() -> None:
    """Internal callers import focused diagnostics children rather than the public facade."""

    open_system_root = _PACKAGE_ROOT / "open_system"
    allowed = {open_system_root / "__init__.py"}
    violations: list[str] = []
    for path in sorted(open_system_root.rglob("*.py")):
        if path in allowed or (open_system_root / "diagnostics") in path.parents:
            continue
        for imported_module in _python_imports(path):
            if imported_module == "qlinks.open_system.diagnostics":
                violations.append(str(path.relative_to(_REPOSITORY_ROOT)))
    assert not violations, "Internal imports use the diagnostics package facade: " + str(violations)


def test_open_system_diagnostics_modules_follow_reviewed_dependency_dag() -> None:
    """Diagnostics responsibilities share only the reviewed numerical leaves."""

    allowed_dependencies = {
        "_formatting.py": set(),
        "_linalg.py": set(),
        "target_manifold.py": set(),
        "jumps.py": {"qlinks.open_system.diagnostics.target_manifold"},
        "verification.py": set(),
        "evolution.py": {"qlinks.open_system.diagnostics.verification"},
        "monitor.py": {
            "qlinks.open_system.diagnostics._formatting",
            "qlinks.open_system.diagnostics._linalg",
        },
        "dark.py": {
            "qlinks.open_system.diagnostics._formatting",
            "qlinks.open_system.diagnostics._linalg",
        },
        "absorbing.py": {
            "qlinks.open_system.diagnostics._formatting",
            "qlinks.open_system.diagnostics._linalg",
        },
    }

    violations: list[str] = []
    diagnostics_prefix = "qlinks.open_system.diagnostics."
    diagnostics_root = _PACKAGE_ROOT / "open_system" / "diagnostics"
    for filename, allowed in allowed_dependencies.items():
        path = diagnostics_root / filename
        for imported_module in _python_imports(path):
            if imported_module.startswith(diagnostics_prefix) and imported_module not in allowed:
                violations.append(f"{filename}: {imported_module}")

    assert not violations, "Open-system diagnostics dependency DAG violation:\n" + "\n".join(
        violations
    )


def test_repository_does_not_import_legacy_flat_refactor_modules() -> None:
    """First-party code must use the nested caging subpackage paths."""

    forbidden_modules = {
        "qlinks.caging.local_search_types",
        "qlinks.caging.local_search_geometry",
        "qlinks.caging.local_search_core",
        "qlinks.caging.local_search_qdm",
        "qlinks.caging.local_search_global",
        "qlinks.caging.local_search_padding",
        "qlinks.caging.local_search_factorized",
        "qlinks.caging.local_search_certification",
        "qlinks.caging.local_search_proposals",
        "qlinks.caging.local_search_scan",
        "qlinks.caging.local_search_workflows",
        "qlinks.caging.stability_types",
        "qlinks.caging.stability_core",
        "qlinks.caging.stability_topology",
        "qlinks.caging.stability_boundary",
        "qlinks.caging.stability_qdm",
        "qlinks.caging.stability_laurent",
        "qlinks.caging.stability_symmetry",
        "qlinks.caging.classification",
        "qlinks.caging.diagnostics",
        "qlinks.caging.support",
        "qlinks.caging.spectral",
        "qlinks.caging.thermodynamic",
        "qlinks.caging.evidence",
        "qlinks.open_system.manifold_detectors",
    }

    violations: list[str] = []
    for tree in (_PACKAGE_ROOT, _REPOSITORY_ROOT / "tests"):
        for path in sorted(tree.rglob("*.py")):
            for imported_module in _python_imports(path):
                if imported_module in forbidden_modules:
                    relative_path = path.relative_to(_REPOSITORY_ROOT)
                    violations.append(f"{relative_path}: {imported_module}")

    assert (
        not violations
    ), "Legacy flat refactor-module imports remain in the repository:\n" + "\n".join(violations)
