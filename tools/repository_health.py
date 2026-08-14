#!/usr/bin/env python3
"""Blocking repository-health checks for architecture, API, size, and security drift."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from architecture_report import analyze_repository, discover_imports

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BUDGET = Path(__file__).with_name("repository_health_budget.json")
_TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SENSITIVE_BASENAMES = {
    ".env",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "service-account.json",
}
_SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
_SECRET_PATTERNS = (
    ("private key material", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub personal access token", re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b")),
)
_FLOATING_ACTION_REFS = {"main", "master", "latest", "head", "dev", "develop"}

_IGNORED_WORKSPACE_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "venv",
}
_IGNORED_WORKSPACE_ROOTS = {
    "build",
    "dist",
    "docs/build",
    "docs/_build",
}


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Machine-readable repository-health snapshot."""

    modules: int
    source_lines: int
    package_edges: int
    static_module_cycles: int
    static_package_cycles: int
    import_time_module_cycles: int
    boundary_violations: int
    oversized_modules: int
    tracked_public_apis: int
    sensitive_file_findings: int
    secret_pattern_findings: int
    workflow_permission_findings: int
    floating_action_findings: int
    guardrail_wiring_findings: int


def _load_budget(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"repository health budget must be a JSON object: {path}")
    return data


def _string_collection(
    node: ast.AST,
    bindings: dict[str, set[str]],
) -> set[str] | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: set[str] = set()
        for item in node.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                return None
            values.add(item.value)
        return values
    if isinstance(node, ast.Name):
        value = bindings.get(node.id)
        return None if value is None else set(value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _string_collection(node.left, bindings)
        right = _string_collection(node.right, bindings)
        if left is None or right is None:
            return None
        return left | right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "sorted", "tuple"}
        and len(node.args) == 1
    ):
        return _string_collection(node.args[0], bindings)
    return None


def _static_all_size(path: Path) -> int | None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        value = _string_collection(node.value, bindings)
        for name in names:
            if name != "__all__" and value is not None:
                bindings[name] = value
        if "__all__" in names:
            return None if value is None else len(value)
    return None


def _module_size_violations(analysis: dict[str, object], budget: dict[str, Any]) -> list[str]:
    default_limit = int(budget["max_new_module_lines"])
    exceptions = {
        str(module): int(limit)
        for module, limit in budget.get("oversized_module_line_limits", {}).items()
    }
    violations: list[str] = []
    modules = analysis["modules"]
    assert isinstance(modules, list)
    for record in modules:
        assert isinstance(record, dict)
        module = str(record["module"])
        lines = int(record["lines"])
        limit = exceptions.get(module, default_limit)
        if lines > limit:
            violations.append(f"module size: {module} has {lines} lines; budget is {limit}")
    return violations


def _api_surface_violations(root: Path, budget: dict[str, Any]) -> tuple[list[str], int]:
    limits = {
        str(path): int(limit) for path, limit in budget.get("public_api_export_limits", {}).items()
    }
    violations: list[str] = []
    seen = 0
    for path in sorted((root / "qlinks").rglob("__init__.py")):
        size = _static_all_size(path)
        if size is None:
            continue
        relative = path.relative_to(root).as_posix()
        seen += 1
        if relative not in limits:
            violations.append(
                f"public API: {relative} defines a static __all__ but has no reviewed budget entry"
            )
            continue
        if size > limits[relative]:
            violations.append(
                f"public API: {relative} exports {size} names; budget is {limits[relative]}"
            )
    return violations, seen


def _top_level_dependency_violations(
    analysis: dict[str, object], budget: dict[str, Any]
) -> list[str]:
    allowed = {
        str(source): set(map(str, targets))
        for source, targets in budget.get("allowed_top_level_dependencies", {}).items()
    }
    violations: list[str] = []
    packages = analysis["packages"]
    assert isinstance(packages, list)
    for record in packages:
        assert isinstance(record, dict)
        package = str(record["package"])
        if package not in allowed:
            violations.append(
                f"package topology: new top-level package {package} needs architecture review"
            )
    edges = analysis["package_edges"]
    assert isinstance(edges, list)
    for edge in edges:
        assert isinstance(edge, dict)
        source = str(edge["source"])
        target = str(edge["target"])
        if target not in allowed.get(source, set()):
            violations.append(f"package topology: unreviewed dependency {source} -> {target}")
    return violations


def _ancestor_api_import_violations(root: Path) -> list[str]:
    module_paths, imports = discover_imports(root, "qlinks")
    package_modules = {
        module for module, path in module_paths.items() if path.name == "__init__.py"
    }
    violations: list[str] = []
    for occurrence in imports:
        if occurrence.target not in package_modules:
            continue
        if not occurrence.source.startswith(f"{occurrence.target}."):
            continue
        violations.append(
            "package API back-import: "
            f"{occurrence.path}:{occurrence.line} imports ancestor API {occurrence.target}"
        )
    return violations


def _raw_forbidden_import_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted((root / "qlinks").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if module == "experimental" or module.startswith("experimental."):
                    relative = path.relative_to(root)
                    violations.append(
                        f"promotion boundary: {relative}:{node.lineno} imports {module}"
                    )
    return violations


def _is_ignored_workspace_path(path: Path, root: Path) -> bool:
    """Return whether ``path`` belongs to local/generated workspace state.

    These directories are not repository-owned input and may contain third-party
    certificates, test keys, caches, or generated files. Arbitrary ``.gitignore``
    entries are intentionally *not* trusted here: a sensitive file elsewhere in
    the working tree should still be reported even when Git ignores it.
    """
    relative = path.relative_to(root)
    relative_posix = relative.as_posix()
    if any(part in _IGNORED_WORKSPACE_DIR_NAMES for part in relative.parts):
        return True
    return any(
        relative_posix == prefix or relative_posix.startswith(f"{prefix}/")
        for prefix in _IGNORED_WORKSPACE_ROOTS
    )


def _iter_repository_files(root: Path) -> Iterable[Path]:
    for current_root, directory_names, file_names in os.walk(root):
        current_path = Path(current_root)
        directory_names[:] = [
            name
            for name in directory_names
            if not _is_ignored_workspace_path(current_path / name, root)
        ]
        for name in file_names:
            path = current_path / name
            if not _is_ignored_workspace_path(path, root):
                yield path


def _security_findings(root: Path) -> tuple[list[str], list[str]]:
    file_findings: list[str] = []
    secret_findings: list[str] = []
    for path in _iter_repository_files(root):
        relative = path.relative_to(root).as_posix()
        name = path.name.lower()
        allowed_env_template = name in {".env.example", ".env.sample", ".env.template"}
        if (
            (name in _SENSITIVE_BASENAMES and not allowed_env_template)
            or path.suffix.lower() in _SENSITIVE_SUFFIXES
            or (name.startswith(".env.") and not allowed_env_template)
        ):
            file_findings.append(f"sensitive filename: {relative}")

        if path.suffix.lower() not in _TEXT_SUFFIXES and path.name not in {
            "Dockerfile",
            "Makefile",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                secret_findings.append(f"possible {label}: {relative}")
    return file_findings, secret_findings


def _workflow_findings(root: Path) -> tuple[list[str], list[str]]:
    permission_findings: list[str] = []
    action_findings: list[str] = []
    workflow_root = root / ".github" / "workflows"
    for path in sorted((*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        jobs_index = text.find("\njobs:")
        header = text if jobs_index < 0 else text[:jobs_index]
        permissions_match = re.search(
            r"(?ms)^permissions:\s*\n(?P<body>(?:  [^\n]+\n?)*)",
            header,
        )
        if permissions_match is None:
            permission_findings.append(
                f"workflow permissions: {relative} lacks a top-level permissions baseline"
            )
        else:
            permissions_body = permissions_match.group("body")
            if not re.search(r"(?m)^  contents:\s*read\s*$", permissions_body):
                permission_findings.append(
                    f"workflow permissions: {relative} top-level contents permission is not read"
                )
            if re.search(r"(?m)^  [A-Za-z0-9_-]+:\s*write\s*$", permissions_body):
                permission_findings.append(
                    f"workflow permissions: {relative} grants top-level write permission"
                )
        if re.search(r"(?m)^permissions:\s*write-all\s*$", text):
            permission_findings.append(f"workflow permissions: {relative} uses write-all")

        for match in re.finditer(r"(?m)^\s*-\s+uses:\s+([^\s#]+)", text):
            reference = match.group(1)
            if reference.startswith("./") or "@" not in reference:
                continue
            ref = reference.rsplit("@", 1)[1].lower()
            if ref in _FLOATING_ACTION_REFS:
                action_findings.append(f"floating action reference: {relative}: {reference}")
    return permission_findings, action_findings


def _active_precommit_hook_ids(text: str) -> set[str]:
    """Return active pre-commit hook IDs, excluding commented-out configuration."""
    return set(
        re.findall(
            r"(?m)^\s*-\s+id:\s*([A-Za-z0-9_.-]+)\s*(?:#.*)?$",
            text,
        )
    )


def _guardrail_wiring_findings(root: Path) -> list[str]:
    findings: list[str] = []

    precommit = (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    active_hook_ids = _active_precommit_hook_ids(precommit)
    required_hook_ids = {
        "black",
        "check-added-large-files",
        "commitizen",
        "commitizen-branch",
        "detect-private-key",
        "fast-tests",
        "flake8",
        "isort",
        "nbstripout",
        "repository-health",
        "test-health",
    }
    for hook_id in sorted(required_hook_ids):
        if hook_id not in active_hook_ids:
            findings.append(f"guardrail wiring: pre-commit hook {hook_id!r} is missing")

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    fast_expression = "not integration and not scientific and not manual and not gpu"
    if fast_expression not in pyproject:
        findings.append("guardrail wiring: pytest default fast-lane marker expression is missing")
    if "error::Warning:qlinks" not in pyproject:
        findings.append("guardrail wiring: qlinks-originated warnings are no longer errors")

    blocking_lint = (root / "scripts" / "lint_blocking.sh").read_text(encoding="utf-8")
    if "tools/repository_health.py --check" not in blocking_lint:
        findings.append("guardrail wiring: blocking lint no longer runs repository health")

    test_workflow = (root / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    if "tools/test_health.py" not in test_workflow:
        findings.append("guardrail wiring: test CI no longer runs test-health check")
    if "scripts/test.sh integration" not in test_workflow:
        findings.append("guardrail wiring: pull-request CI no longer runs integration lane")

    scientific_workflow = (root / ".github" / "workflows" / "scientific.yml").read_text(
        encoding="utf-8"
    )
    if "scripts/test.sh scientific" not in scientific_workflow:
        findings.append("guardrail wiring: scientific workflow no longer runs scientific lane")

    return findings


def build_snapshot(root: Path, budget: dict[str, Any]) -> tuple[HealthSnapshot, list[str]]:
    analysis = analyze_repository(root)
    summary = analysis["summary"]
    assert isinstance(summary, dict)

    violations: list[str] = []
    for key, label in (
        ("static_module_cycle_components", "static module cycles"),
        ("static_package_cycle_components", "static package cycles"),
        ("import_time_module_cycle_components", "import-time module cycles"),
        ("import_time_package_cycle_components", "import-time package cycles"),
        ("boundary_violations", "architecture boundary violations"),
    ):
        count = int(summary[key])
        if count:
            violations.append(f"architecture: {label} must remain zero; found {count}")

    violations.extend(_module_size_violations(analysis, budget))
    api_violations, api_count = _api_surface_violations(root, budget)
    violations.extend(api_violations)
    violations.extend(_top_level_dependency_violations(analysis, budget))
    violations.extend(_ancestor_api_import_violations(root))
    violations.extend(_raw_forbidden_import_violations(root))
    file_findings, secret_findings = _security_findings(root)
    workflow_permission_findings, action_findings = _workflow_findings(root)
    wiring_findings = _guardrail_wiring_findings(root)
    violations.extend(file_findings)
    violations.extend(secret_findings)
    violations.extend(workflow_permission_findings)
    violations.extend(action_findings)
    violations.extend(wiring_findings)

    modules = analysis["modules"]
    assert isinstance(modules, list)
    oversized = sum(
        int(record["lines"]) > int(budget["max_new_module_lines"]) for record in modules
    )
    snapshot = HealthSnapshot(
        modules=int(summary["modules"]),
        source_lines=int(summary["lines"]),
        package_edges=int(summary["package_edges"]),
        static_module_cycles=int(summary["static_module_cycle_components"]),
        static_package_cycles=int(summary["static_package_cycle_components"]),
        import_time_module_cycles=int(summary["import_time_module_cycle_components"]),
        boundary_violations=int(summary["boundary_violations"]),
        oversized_modules=oversized,
        tracked_public_apis=api_count,
        sensitive_file_findings=len(file_findings),
        secret_pattern_findings=len(secret_findings),
        workflow_permission_findings=len(workflow_permission_findings),
        floating_action_findings=len(action_findings),
        guardrail_wiring_findings=len(wiring_findings),
    )
    return snapshot, sorted(set(violations))


def snapshot_to_markdown(snapshot: HealthSnapshot, violations: list[str]) -> str:
    rows = [
        ("Python modules", snapshot.modules),
        ("Source lines", snapshot.source_lines),
        ("Top-level package edges", snapshot.package_edges),
        ("Static module SCCs", snapshot.static_module_cycles),
        ("Static package SCCs", snapshot.static_package_cycles),
        ("Import-time module SCCs", snapshot.import_time_module_cycles),
        ("Architecture boundary violations", snapshot.boundary_violations),
        ("Grandfathered oversized modules", snapshot.oversized_modules),
        ("Tracked public APIs", snapshot.tracked_public_apis),
        ("Sensitive filenames", snapshot.sensitive_file_findings),
        ("Secret-pattern findings", snapshot.secret_pattern_findings),
        ("Workflow permission findings", snapshot.workflow_permission_findings),
        ("Floating action findings", snapshot.floating_action_findings),
        ("Guardrail wiring findings", snapshot.guardrail_wiring_findings),
    ]
    lines = [
        "# qlinks repository-health snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *(f"| {label} | {value} |" for label, value in rows),
        "",
        "## Guardrail result",
        "",
    ]
    if violations:
        lines.append("**FAIL**")
        lines.extend(f"- {violation}" for violation in violations)
    else:
        lines.append("**PASS**")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit nonzero when a guardrail fails")
    parser.add_argument("--budget", type=Path, default=_DEFAULT_BUDGET)
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument("--markdown", type=Path, dest="markdown_path")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    budget = _load_budget(args.budget)
    snapshot, violations = build_snapshot(_REPOSITORY_ROOT, budget)
    markdown = snapshot_to_markdown(snapshot, violations)

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps({"snapshot": asdict(snapshot), "violations": violations}, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_path:
        args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_path.write_text(markdown, encoding="utf-8")
    if not args.quiet:
        print(markdown, end="")
    return 1 if args.check and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
