#!/usr/bin/env python3
"""Report and optionally enforce lightweight qlinks test-suite health budgets."""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _test_files() -> list[Path]:
    return sorted(_TEST_ROOT.rglob("*.py"))


def _is_test_function(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
        "test_"
    )


def _file_metrics(paths: Iterable[Path]) -> tuple[int, int, tuple[TestFileMetric, ...]]:
    metrics: list[TestFileMetric] = []
    total_loc = 0
    total_tests = 0
    for path in paths:
        source = path.read_text(encoding="utf-8")
        lines = len(source.splitlines())
        tree = ast.parse(source, filename=str(path))
        test_functions = sum(1 for node in ast.walk(tree) if _is_test_function(node))
        total_loc += lines
        total_tests += test_functions
        metrics.append(
            TestFileMetric(
                path=str(path.relative_to(_REPOSITORY_ROOT)),
                lines=lines,
                test_functions=test_functions,
            )
        )
    largest = tuple(sorted(metrics, key=lambda item: (-item.lines, item.path))[:10])
    return total_loc, total_tests, largest


def _private_imports(paths: Iterable[Path]) -> tuple[int, tuple[PrivateImportHotspot, ...]]:
    counts: Counter[str] = Counter()
    total = 0
    for path in paths:
        tree = _parse(path)
        relative = str(path.relative_to(_REPOSITORY_ROOT))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("qlinks")
            ):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        counts[relative] += 1
                        total += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("qlinks."):
                        continue
                    if any(part.startswith("_") for part in alias.name.split(".")):
                        counts[relative] += 1
                        total += 1
    hotspots = tuple(
        PrivateImportHotspot(path=path, count=count)
        for path, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return total, hotspots


def _fixture_names_from_module(path: Path) -> set[str]:
    names: set[str] = set()
    tree = _parse(path)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            func = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(func, ast.Attribute) and func.attr == "fixture":
                names.add(node.name)
            elif isinstance(func, ast.Name) and func.id == "fixture":
                names.add(node.name)
    return names


def _fixture_references(paths: Iterable[Path], fixture_names: set[str]) -> Counter[str]:
    references: Counter[str] = Counter()
    for path in paths:
        if _TEST_ROOT / "fixtures" in path.parents:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for argument in (*node.args.args, *node.args.kwonlyargs):
                    if argument.arg in fixture_names:
                        references[argument.arg] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in fixture_names:
                    references[node.value] += 1
    return references


def _global_fixture_health(paths: list[Path]) -> tuple[int, tuple[str, ...]]:
    fixture_dir = _TEST_ROOT / "fixtures"
    fixture_names: set[str] = set()
    for path in fixture_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        fixture_names.update(_fixture_names_from_module(path))
    references = _fixture_references(paths, fixture_names)
    unused = tuple(sorted(name for name in fixture_names if references[name] == 0))
    return len(fixture_names), unused


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
        source = source_cache.setdefault(path, path.read_text(encoding="utf-8"))
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


def build_snapshot() -> TestHealthSnapshot:
    paths = _test_files()
    test_loc, ast_tests, largest = _file_metrics(paths)
    private_count, hotspots = _private_imports(paths)
    global_fixture_count, unused_global_fixtures = _global_fixture_health(paths)
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


def budget_violations(snapshot: TestHealthSnapshot, budget: dict[str, int]) -> list[str]:
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
) -> str:
    lines = [
        "# qlinks test-health snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Python test files | {snapshot.python_files} |",
        f"| Test LOC | {snapshot.test_loc} |",
        f"| AST test functions | {snapshot.ast_test_functions} |",
        f"| Collected pytest cases | {snapshot.collected_cases} |",
        f"| Fast-lane selected cases | {snapshot.fast_selected_cases} |",
        f"| Integration cases | {snapshot.marker_cases['integration']} |",
        f"| Scientific cases | {snapshot.marker_cases['scientific']} |",
        f"| Manual cases | {snapshot.marker_cases['manual']} |",
        f"| GPU cases | {snapshot.marker_cases['gpu']} |",
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
    args = parser.parse_args()

    snapshot = build_snapshot()
    budget = load_budget(args.budget) if args.budget.exists() else None
    violations = budget_violations(snapshot, budget) if budget is not None else []
    markdown = snapshot_to_markdown(snapshot, budget=budget, violations=violations)

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
