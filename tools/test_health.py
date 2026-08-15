#!/usr/bin/env python3
"""Report and optionally enforce lightweight qlinks test-suite health budgets."""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import io
import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TEST_ROOT = _REPOSITORY_ROOT / "tests"
_DEFAULT_BUDGET = _TEST_ROOT / "test_health_budget.json"
_EXCLUDED_FAST_MARKERS = frozenset({"integration", "scientific", "manual", "gpu"})
_TRACKED_MARKERS = ("integration", "scientific", "manual", "gpu")


@dataclass(frozen=True)
class TestFileMetric:
    path: str
    lines: int
    test_functions: int


@dataclass(frozen=True)
class PrivateImportHotspot:
    path: str
    count: int


@dataclass(frozen=True)
class _TestFileScan:
    path: Path
    metric: TestFileMetric
    private_imports: int
    referenced_names: frozenset[str]
    fixture_definitions: frozenset[str]
    marker_functions: dict[str, int]
    fast_functions: int
    unmarked_manual_visual_functions: int


@dataclass(frozen=True)
class TestHealthSnapshot:
    python_files: int
    test_loc: int
    ast_test_functions: int
    collected_cases: int
    fast_selected_cases: int
    marker_cases: dict[str, int]
    private_symbol_imports: int
    private_import_hotspots: tuple[PrivateImportHotspot, ...]
    largest_files: tuple[TestFileMetric, ...]
    global_fixtures: int
    unused_global_fixtures: tuple[str, ...]
    unmarked_manual_visual_cases: int

    @property
    def max_test_file_lines(self) -> int:
        return self.largest_files[0].lines if self.largest_files else 0


@cache
def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _test_files() -> list[Path]:
    return sorted(_TEST_ROOT.rglob("*.py"))


def _is_test_function(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
        "test_"
    )


def _fixture_decorator_present(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        func = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(func, ast.Attribute) and func.attr == "fixture":
            return True
        if isinstance(func, ast.Name) and func.id == "fixture":
            return True
    return False


def _fixture_definitions(tree: ast.Module) -> frozenset[str]:
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _fixture_decorator_present(node)
    )


def _pytest_marker_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        value = child.value
        if not isinstance(value, ast.Attribute) or value.attr != "mark":
            continue
        if not isinstance(value.value, ast.Name) or value.value.id != "pytest":
            continue
        if child.attr in _TRACKED_MARKERS:
            names.add(child.attr)
    return names


def _module_marker_names(tree: ast.Module) -> set[str]:
    markers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
            markers.update(_pytest_marker_names(node.value))
    return markers


def _module_has_visual_guard(tree: ast.Module, source: str) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        is_pytestmark = any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets
        )
        if not is_pytestmark:
            continue
        value_source = ast.get_source_segment(source, node.value) or ""
        if "QLINKS_SHOW_PLOTS" in value_source:
            return True
    return False


def _static_file_marker_metrics(
    tree: ast.Module,
    source: str,
) -> tuple[dict[str, int], int, int]:
    marker_counts = {marker: 0 for marker in _TRACKED_MARKERS}
    fast_selected = 0
    unmarked_manual_visual = 0
    source_lines = source.splitlines() if "QLINKS_SHOW_PLOTS" in source else []

    def visit_body(body: list[ast.stmt], inherited_markers: set[str]) -> None:
        nonlocal fast_selected, unmarked_manual_visual
        for node in body:
            if isinstance(node, ast.ClassDef):
                class_markers = set(inherited_markers)
                for decorator in node.decorator_list:
                    class_markers.update(_pytest_marker_names(decorator))
                visit_body(node.body, class_markers)
                continue
            if not _is_test_function(node):
                continue

            markers = set(inherited_markers)
            for decorator in node.decorator_list:
                markers.update(_pytest_marker_names(decorator))
            for marker_name in _TRACKED_MARKERS:
                if marker_name in markers:
                    marker_counts[marker_name] += 1
            if not (markers & _EXCLUDED_FAST_MARKERS):
                fast_selected += 1

            if "manual" not in markers and source_lines:
                start = max(node.lineno - 1, 0)
                end = node.end_lineno or node.lineno
                function_source = "\n".join(source_lines[start:end])
                if "QLINKS_SHOW_PLOTS" in function_source:
                    unmarked_manual_visual += 1

    module_markers = _module_marker_names(tree)
    module_has_visual_guard = _module_has_visual_guard(tree, source)
    if module_has_visual_guard and "manual" not in module_markers:
        # The module-level guard applies to every test function in the module.
        source_lines = source.splitlines()

    visit_body(tree.body, module_markers)

    if module_has_visual_guard and "manual" not in module_markers:
        # Functions without their own manual marker were already counted only when they
        # contained the guard. Replace that local criterion with the module-level guard.
        unmarked_manual_visual = 0

        def count_unmarked(body: list[ast.stmt], inherited_markers: set[str]) -> None:
            nonlocal unmarked_manual_visual
            for node in body:
                if isinstance(node, ast.ClassDef):
                    class_markers = set(inherited_markers)
                    for decorator in node.decorator_list:
                        class_markers.update(_pytest_marker_names(decorator))
                    count_unmarked(node.body, class_markers)
                    continue
                if not _is_test_function(node):
                    continue
                markers = set(inherited_markers)
                for decorator in node.decorator_list:
                    markers.update(_pytest_marker_names(decorator))
                if "manual" not in markers:
                    unmarked_manual_visual += 1

        count_unmarked(tree.body, module_markers)

    return marker_counts, fast_selected, unmarked_manual_visual


def _scan_test_file(path: Path) -> _TestFileScan:
    source = _source(path)
    tree = _parse(path)
    nodes = tuple(ast.walk(tree))
    test_functions = sum(1 for node in nodes if _is_test_function(node))
    private_imports = 0
    referenced_names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("qlinks"):
            private_imports += sum(alias.name.startswith("_") for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("qlinks.") and any(
                    part.startswith("_") for part in alias.name.split(".")
                ):
                    private_imports += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            referenced_names.update(
                argument.arg for argument in (*node.args.args, *node.args.kwonlyargs)
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            referenced_names.add(node.value)

    marker_functions, fast_functions, unmarked_visual = _static_file_marker_metrics(tree, source)
    return _TestFileScan(
        path=path,
        metric=TestFileMetric(
            path=str(path.relative_to(_REPOSITORY_ROOT)),
            lines=len(source.splitlines()),
            test_functions=test_functions,
        ),
        private_imports=private_imports,
        referenced_names=frozenset(referenced_names),
        fixture_definitions=_fixture_definitions(tree),
        marker_functions=marker_functions,
        fast_functions=fast_functions,
        unmarked_manual_visual_functions=unmarked_visual,
    )


def _scan_test_files(paths: list[Path]) -> list[_TestFileScan]:
    return [_scan_test_file(path) for path in paths]


def _scan_metrics(
    scans: list[_TestFileScan],
) -> tuple[
    int,
    int,
    tuple[TestFileMetric, ...],
    int,
    tuple[PrivateImportHotspot, ...],
    int,
    tuple[str, ...],
    tuple[int, int, dict[str, int], int],
]:
    total_loc = sum(scan.metric.lines for scan in scans)
    total_tests = sum(scan.metric.test_functions for scan in scans)
    largest = tuple(
        sorted((scan.metric for scan in scans), key=lambda item: (-item.lines, item.path))[:10]
    )

    hotspots = tuple(
        PrivateImportHotspot(path=scan.metric.path, count=scan.private_imports)
        for scan in sorted(scans, key=lambda item: (-item.private_imports, item.metric.path))
        if scan.private_imports
    )
    private_count = sum(scan.private_imports for scan in scans)

    fixture_dir = _TEST_ROOT / "fixtures"
    fixture_names = {
        fixture
        for scan in scans
        if scan.path.parent == fixture_dir
        for fixture in scan.fixture_definitions
    }
    referenced_names = {
        name
        for scan in scans
        if fixture_dir not in scan.path.parents
        for name in scan.referenced_names
    }
    unused = tuple(sorted(fixture_names - referenced_names))

    marker_counts = {marker: 0 for marker in _TRACKED_MARKERS}
    for scan in scans:
        for marker, count in scan.marker_functions.items():
            marker_counts[marker] += count
    static_collection = (
        total_tests,
        sum(scan.fast_functions for scan in scans),
        marker_counts,
        sum(scan.unmarked_manual_visual_functions for scan in scans),
    )

    return (
        total_loc,
        total_tests,
        largest,
        private_count,
        hotspots,
        len(fixture_names),
        unused,
        static_collection,
    )


class _CollectionPlugin:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def pytest_collection_finish(self, session: Any) -> None:
        self.items = list(session.items)


def _collect_pytest_items() -> list[Any]:
    import pytest

    plugin = _CollectionPlugin()
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = pytest.main(
            ["--collect-only", "-q", "-o", "addopts=", str(_TEST_ROOT)],
            plugins=[plugin],
        )
    if exit_code not in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}:
        raise RuntimeError(
            "pytest collection failed while building test-health report:\n"
            f"{stdout.getvalue()}\n{stderr.getvalue()}"
        )
    return plugin.items


def _collection_metrics(items: list[Any]) -> tuple[int, int, dict[str, int], int]:
    marker_counts = {marker: 0 for marker in _TRACKED_MARKERS}
    fast_selected = 0
    unmarked_manual_visual = 0
    source_cache: dict[Path, str] = {}

    for item in items:
        item_markers = {marker.name for marker in item.iter_markers()}
        for marker in _TRACKED_MARKERS:
            if marker in item_markers:
                marker_counts[marker] += 1
        if not (item_markers & _EXCLUDED_FAST_MARKERS):
            fast_selected += 1

        path = Path(str(item.path)).resolve()
        source = source_cache.setdefault(path, _source(path))
        if "manual" not in item_markers:
            function_source = inspect.getsource(item.obj)
            module_level_visual_guard = (
                "pytestmark" in source
                and "QLINKS_SHOW_PLOTS" in source
                and "pytest.mark.manual" not in source
            )
            if "QLINKS_SHOW_PLOTS" in function_source or module_level_visual_guard:
                unmarked_manual_visual += 1

    return len(items), fast_selected, marker_counts, unmarked_manual_visual


def build_snapshot(*, static_collection: bool = False) -> TestHealthSnapshot:
    paths = _test_files()
    scans = _scan_test_files(paths)
    (
        test_loc,
        ast_tests,
        largest,
        private_count,
        hotspots,
        global_fixture_count,
        unused_global_fixtures,
        static_metrics,
    ) = _scan_metrics(scans)
    if static_collection:
        collected, fast_selected, markers, unmarked_visual = static_metrics
    else:
        items = _collect_pytest_items()
        collected, fast_selected, markers, unmarked_visual = _collection_metrics(items)

    return TestHealthSnapshot(
        python_files=len(paths),
        test_loc=test_loc,
        ast_test_functions=ast_tests,
        collected_cases=collected,
        fast_selected_cases=fast_selected,
        marker_cases=markers,
        private_symbol_imports=private_count,
        private_import_hotspots=hotspots,
        largest_files=largest,
        global_fixtures=global_fixture_count,
        unused_global_fixtures=unused_global_fixtures,
        unmarked_manual_visual_cases=unmarked_visual,
    )


def load_budget(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(value, int) for value in data.values()):
        raise ValueError(f"Test-health budget must be a JSON object of integer limits: {path}")
    return data


def budget_violations(
    snapshot: TestHealthSnapshot,
    budget: dict[str, int],
    *,
    static_collection: bool = False,
) -> list[str]:
    actual = {
        "max_private_symbol_imports": snapshot.private_symbol_imports,
        "max_test_file_lines": snapshot.max_test_file_lines,
        "max_unused_global_fixtures": len(snapshot.unused_global_fixtures),
        "max_unmarked_manual_visual_cases": snapshot.unmarked_manual_visual_cases,
    }
    violations: list[str] = []
    for key, value in actual.items():
        limit = budget.get(key)
        if limit is not None and value > limit:
            violations.append(f"{key}: {value} exceeds budget {limit}")

    if static_collection:
        marker_minima = {
            "min_integration_test_functions": snapshot.marker_cases["integration"],
            "min_scientific_test_functions": snapshot.marker_cases["scientific"],
            "min_manual_test_functions": snapshot.marker_cases["manual"],
        }
    else:
        marker_minima = {
            "min_integration_cases": snapshot.marker_cases["integration"],
            "min_scientific_cases": snapshot.marker_cases["scientific"],
            "min_manual_cases": snapshot.marker_cases["manual"],
        }
    for key, value in marker_minima.items():
        minimum = budget.get(key)
        if minimum is not None and value < minimum:
            violations.append(f"{key}: {value} is below budget floor {minimum}")
    return violations


def snapshot_to_dict(snapshot: TestHealthSnapshot) -> dict[str, Any]:
    data = asdict(snapshot)
    data["max_test_file_lines"] = snapshot.max_test_file_lines
    return data


def snapshot_to_markdown(
    snapshot: TestHealthSnapshot,
    *,
    budget: dict[str, int] | None = None,
    violations: list[str] | None = None,
    static_collection: bool = False,
) -> str:
    lines = [
        "# qlinks test-health snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Python test files | {snapshot.python_files} |",
        f"| Test LOC | {snapshot.test_loc} |",
        f"| AST test functions | {snapshot.ast_test_functions} |",
        f"| {'Static test functions' if static_collection else 'Collected pytest cases'} | "
        f"{snapshot.collected_cases} |",
        f"| {'Static fast-lane functions' if static_collection else 'Fast-lane selected cases'} | "
        f"{snapshot.fast_selected_cases} |",
        f"| {'Integration functions' if static_collection else 'Integration cases'} | "
        f"{snapshot.marker_cases['integration']} |",
        f"| {'Scientific functions' if static_collection else 'Scientific cases'} | "
        f"{snapshot.marker_cases['scientific']} |",
        f"| {'Manual functions' if static_collection else 'Manual cases'} | "
        f"{snapshot.marker_cases['manual']} |",
        f"| {'GPU functions' if static_collection else 'GPU cases'} | "
        f"{snapshot.marker_cases['gpu']} |",
        f"| Direct private qlinks imports | {snapshot.private_symbol_imports} |",
        f"| Global fixtures | {snapshot.global_fixtures} |",
        f"| Unused global fixtures | {len(snapshot.unused_global_fixtures)} |",
        f"| Unmarked manual-visual cases | {snapshot.unmarked_manual_visual_cases} |",
        f"| Largest test file (lines) | {snapshot.max_test_file_lines} |",
        "",
        "## Largest test files",
        "",
        "| File | Lines | AST tests |",
        "|---|---:|---:|",
    ]
    for metric in snapshot.largest_files:
        lines.append(f"| `{metric.path}` | {metric.lines} | {metric.test_functions} |")

    lines.extend(["", "## Private-import hotspots", ""])
    if snapshot.private_import_hotspots:
        lines.extend(["| File | Imports |", "|---|---:|"])
        for hotspot in snapshot.private_import_hotspots:
            lines.append(f"| `{hotspot.path}` | {hotspot.count} |")
    else:
        lines.append("None.")

    lines.extend(["", "## Unused global fixtures", ""])
    if snapshot.unused_global_fixtures:
        lines.extend(f"- `{name}`" for name in snapshot.unused_global_fixtures)
    else:
        lines.append("None.")

    if budget is not None:
        lines.extend(["", "## Budget check", ""])
        if violations:
            lines.append("**FAIL**")
            lines.extend(f"- {violation}" for violation in violations)
        else:
            lines.append("**PASS**")

    lines.append("")
    return "\n".join(lines)


def _write(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path, default=_DEFAULT_BUDGET)
    parser.add_argument("--check", action="store_true", help="Fail if the budget is exceeded.")
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument("--markdown", type=Path, dest="markdown_path")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the Markdown snapshot to stdout.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Use static marker/function analysis instead of importing the suite with pytest; "
            "intended for the local pre-push hook. CI should use the full collector."
        ),
    )
    args = parser.parse_args()

    snapshot = build_snapshot(static_collection=args.local)
    budget = load_budget(args.budget) if args.budget.exists() else None
    violations = (
        budget_violations(snapshot, budget, static_collection=args.local)
        if budget is not None
        else []
    )
    markdown = snapshot_to_markdown(
        snapshot,
        budget=budget,
        violations=violations,
        static_collection=args.local,
    )

    if not args.quiet:
        print(markdown, end="")
    _write(args.markdown_path, markdown)
    if args.json_path is not None:
        content = json.dumps(snapshot_to_dict(snapshot), indent=2, sort_keys=True) + "\n"
        _write(args.json_path, content)

    if args.check and budget is None:
        raise FileNotFoundError(f"Test-health budget not found: {args.budget}")
    return 1 if args.check and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
