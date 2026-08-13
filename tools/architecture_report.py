#!/usr/bin/env python3
"""Build a self-contained HTML architecture diagnosis for qlinks.

The report is intentionally a repository-development tool rather than package
runtime functionality.  It analyzes Python imports statically, aggregates the
file graph into top-level package dependencies, checks the broad dependency
boundaries documented in ``AGENTS.md``, and emits both HTML and JSON artifacts.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import subprocess
import webbrowser
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx


@dataclass(frozen=True, slots=True)
class ModuleRecord:
    """Static metrics for one Python module."""

    module: str
    path: str
    package: str
    lines: int
    fan_in: int
    fan_out: int
    incoming_imports: int
    outgoing_imports: int
    in_import_cycle: bool
    in_static_cycle: bool


@dataclass(frozen=True, slots=True)
class PackageRecord:
    """Aggregated metrics for one top-level qlinks package/module."""

    package: str
    files: int
    lines: int
    fan_in: int
    fan_out: int
    incoming_imports: int
    outgoing_imports: int
    in_cycle: bool


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    """One architecture-boundary import violation."""

    rule: str
    source: str
    target: str
    path: str
    line: int


@dataclass(frozen=True, slots=True)
class ImportOccurrence:
    """One resolved internal import occurrence."""

    source: str
    target: str
    path: str
    line: int
    type_checking: bool
    local_scope: bool


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    """Static import boundary mirrored from the repository architecture guide."""

    name: str
    source_prefix: str
    forbidden_prefixes: tuple[str, ...]
    excluded_paths: tuple[str, ...] = ()
    excluded_exact_paths: tuple[str, ...] = ()


BROAD_BOUNDARY_RULES: tuple[BoundaryRule, ...] = (
    BoundaryRule(
        name="caging must not depend on open_system",
        source_prefix="qlinks.caging",
        forbidden_prefixes=("qlinks.open_system",),
    ),
    BoundaryRule(
        name="local_structure must remain neutral",
        source_prefix="qlinks.local_structure",
        forbidden_prefixes=("qlinks.caging", "qlinks.open_system"),
    ),
    BoundaryRule(
        name="active open_system must not depend on caging implementation",
        source_prefix="qlinks.open_system",
        forbidden_prefixes=("qlinks.caging",),
        excluded_paths=("qlinks/open_system/constructions/deprecated/",),
    ),
    BoundaryRule(
        name="active caging code must not depend on temporary facades",
        source_prefix="qlinks.caging",
        forbidden_prefixes=("qlinks.caging.local_search", "qlinks.caging.stability"),
        excluded_exact_paths=("qlinks/caging/__init__.py",),
    ),
    BoundaryRule(
        name="active open_system code must not depend on temporary detector facade",
        source_prefix="qlinks.open_system",
        forbidden_prefixes=("qlinks.open_system.manifold_detectors",),
    ),
)

LOCAL_SEARCH_ALLOWED_DEPENDENCIES: dict[str, frozenset[str]] = {
    "qlinks.caging.local_search_types": frozenset(),
    "qlinks.caging.local_search_geometry": frozenset({"qlinks.caging.local_search_types"}),
    "qlinks.caging.local_search_core": frozenset({"qlinks.caging.local_search_types"}),
    "qlinks.caging.local_search_qdm": frozenset(
        {
            "qlinks.caging.local_search_core",
            "qlinks.caging.local_search_geometry",
            "qlinks.caging.local_search_types",
        }
    ),
    "qlinks.caging.local_search_global": frozenset(
        {
            "qlinks.caging.local_search_qdm",
            "qlinks.caging.local_search_types",
        }
    ),
    "qlinks.caging.local_search_padding": frozenset(
        {
            "qlinks.caging.local_search_geometry",
            "qlinks.caging.local_search_global",
            "qlinks.caging.local_search_types",
        }
    ),
    "qlinks.caging.local_search_factorized": frozenset(
        {
            "qlinks.caging.local_search_global",
            "qlinks.caging.local_search_padding",
            "qlinks.caging.local_search_qdm",
            "qlinks.caging.local_search_types",
        }
    ),
    "qlinks.caging.local_search_certification": frozenset(
        {
            "qlinks.caging.local_search_global",
            "qlinks.caging.local_search_padding",
            "qlinks.caging.local_search_qdm",
            "qlinks.caging.local_search_types",
        }
    ),
    "qlinks.caging.local_search_proposals": frozenset(
        {
            "qlinks.caging.local_search_core",
            "qlinks.caging.local_search_geometry",
            "qlinks.caging.local_search_types",
        }
    ),
    "qlinks.caging.local_search_scan": frozenset(
        {
            "qlinks.caging.local_search_core",
            "qlinks.caging.local_search_padding",
            "qlinks.caging.local_search_types",
        }
    ),
    "qlinks.caging.local_search_workflows": frozenset(
        {
            "qlinks.caging.local_search_certification",
            "qlinks.caging.local_search_proposals",
            "qlinks.caging.local_search_scan",
            "qlinks.caging.local_search_types",
        }
    ),
}


SURFACE_FACADE_MODULES = {
    "qlinks.caging.local_search",
    "qlinks.caging.stability",
    "qlinks.open_system.manifold_detectors",
}


def _module_name(path: Path, repository_root: Path) -> str:
    relative = path.relative_to(repository_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _top_level_package(module: str, package_name: str) -> str:
    if module == package_name:
        return package_name
    parts = module.split(".")
    return ".".join(parts[:2])


def _resolve_from_base(source: str, path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    source_package = source if path.name == "__init__.py" else source.rpartition(".")[0]
    package_parts = source_package.split(".") if source_package else []
    levels_up = max(node.level - 1, 0)
    if levels_up:
        package_parts = package_parts[:-levels_up] if levels_up <= len(package_parts) else []
    suffix = node.module.split(".") if node.module else []
    return ".".join((*package_parts, *suffix))


def _longest_known_prefix(name: str, known_modules: set[str]) -> str | None:
    candidate = name
    while candidate:
        if candidate in known_modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _is_type_checking_guard(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    if isinstance(node, (ast.BoolOp, ast.UnaryOp)):
        return any(
            isinstance(child, (ast.Name, ast.Attribute))
            and (
                (isinstance(child, ast.Name) and child.id == "TYPE_CHECKING")
                or (isinstance(child, ast.Attribute) and child.attr == "TYPE_CHECKING")
            )
            for child in ast.walk(node)
        )
    return False


class _ImportCollector(ast.NodeVisitor):
    """Collect imports while distinguishing eager, lazy, and type-only edges."""

    def __init__(self) -> None:
        self.records: list[tuple[ast.Import | ast.ImportFrom, bool, bool]] = []
        self._type_checking_depth = 0
        self._function_depth = 0

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.records.append((node, self._type_checking_depth > 0, self._function_depth > 0))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.records.append((node, self._type_checking_depth > 0, self._function_depth > 0))

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if not _is_type_checking_guard(node.test):
            self.generic_visit(node)
            return

        self.visit(node.test)
        self._type_checking_depth += 1
        for statement in node.body:
            self.visit(statement)
        self._type_checking_depth -= 1
        for statement in node.orelse:
            self.visit(statement)

    def _visit_function(self, node: ast.AST) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)


def discover_imports(
    repository_root: Path,
    package_name: str = "qlinks",
) -> tuple[dict[str, Path], tuple[ImportOccurrence, ...]]:
    """Discover Python modules and statically resolve internal imports."""

    package_root = repository_root / package_name
    if not package_root.is_dir():
        raise FileNotFoundError(f"Package directory not found: {package_root}")

    module_paths = {
        _module_name(path, repository_root): path for path in sorted(package_root.rglob("*.py"))
    }
    known_modules = set(module_paths)
    imports: list[ImportOccurrence] = []

    for source, path in sorted(module_paths.items()):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = _ImportCollector()
        collector.visit(tree)
        relative_path = path.relative_to(repository_root).as_posix()

        for node, type_checking, local_scope in collector.records:
            raw_targets: list[str] = []
            if isinstance(node, ast.Import):
                raw_targets.extend(alias.name for alias in node.names)
            else:
                base = _resolve_from_base(source, path, node)
                if base:
                    raw_targets.append(base)
                    for alias in node.names:
                        if alias.name != "*":
                            raw_targets.append(f"{base}.{alias.name}")

            resolved_targets: set[str] = set()
            for raw_target in raw_targets:
                if raw_target != package_name and not raw_target.startswith(f"{package_name}."):
                    continue
                target = _longest_known_prefix(raw_target, known_modules)
                if target is not None and target != source:
                    resolved_targets.add(target)

            for target in sorted(resolved_targets):
                imports.append(
                    ImportOccurrence(
                        source=source,
                        target=target,
                        path=relative_path,
                        line=int(getattr(node, "lineno", 0)),
                        type_checking=type_checking,
                        local_scope=local_scope,
                    )
                )

    return module_paths, tuple(imports)


def _graph_from_imports(
    modules: Iterable[str],
    imports: Iterable[ImportOccurrence],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(modules)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for occurrence in imports:
        edge = counts[(occurrence.source, occurrence.target)]
        edge["weight"] += 1
        if occurrence.type_checking:
            edge["type_checking_weight"] += 1
        elif occurrence.local_scope:
            edge["local_weight"] += 1
        else:
            edge["eager_weight"] += 1

    for (source, target), edge_counts in counts.items():
        graph.add_edge(
            source,
            target,
            weight=int(edge_counts["weight"]),
            eager_weight=int(edge_counts["eager_weight"]),
            local_weight=int(edge_counts["local_weight"]),
            type_checking_weight=int(edge_counts["type_checking_weight"]),
        )
    return graph


def _import_time_graph(graph: nx.DiGraph) -> nx.DiGraph:
    eager = nx.DiGraph()
    eager.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        weight = int(data.get("eager_weight", 0))
        if weight:
            eager.add_edge(source, target, weight=weight)
    return eager


def _package_graph(
    module_graph: nx.DiGraph,
    package_name: str,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    package_nodes = {_top_level_package(str(module), package_name) for module in module_graph.nodes}
    graph.add_nodes_from(package_nodes)

    counts: Counter[tuple[str, str]] = Counter()
    for source, target, data in module_graph.edges(data=True):
        source_package = _top_level_package(str(source), package_name)
        target_package = _top_level_package(str(target), package_name)
        if source_package == target_package:
            continue
        counts[(source_package, target_package)] += int(data.get("weight", 1))

    for (source, target), weight in counts.items():
        graph.add_edge(source, target, weight=weight)
    return graph


def _cycle_components(graph: nx.DiGraph) -> list[list[str]]:
    components: list[list[str]] = []
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            components.append(sorted(str(node) for node in component))
            continue
        node = next(iter(component))
        if graph.has_edge(node, node):
            components.append([str(node)])
    return sorted(components, key=lambda value: (-len(value), value))


def _architecture_violations(
    imports: Sequence[ImportOccurrence],
) -> tuple[BoundaryViolation, ...]:
    violations: list[BoundaryViolation] = []

    for occurrence in imports:
        for rule in BROAD_BOUNDARY_RULES:
            if not (
                occurrence.source == rule.source_prefix
                or occurrence.source.startswith(f"{rule.source_prefix}.")
            ):
                continue
            if occurrence.path in rule.excluded_exact_paths:
                continue
            if any(occurrence.path.startswith(prefix) for prefix in rule.excluded_paths):
                continue
            if any(
                occurrence.target == prefix or occurrence.target.startswith(f"{prefix}.")
                for prefix in rule.forbidden_prefixes
            ):
                violations.append(
                    BoundaryViolation(
                        rule=rule.name,
                        source=occurrence.source,
                        target=occurrence.target,
                        path=occurrence.path,
                        line=occurrence.line,
                    )
                )

        allowed_local_search_dependencies = LOCAL_SEARCH_ALLOWED_DEPENDENCIES.get(occurrence.source)
        if (
            allowed_local_search_dependencies is not None
            and occurrence.target.startswith("qlinks.caging.local_search_")
            and occurrence.target not in allowed_local_search_dependencies
        ):
            violations.append(
                BoundaryViolation(
                    rule="local-search focused modules must follow the reviewed dependency DAG",
                    source=occurrence.source,
                    target=occurrence.target,
                    path=occurrence.path,
                    line=occurrence.line,
                )
            )

    return tuple(
        sorted(
            set(violations),
            key=lambda item: (item.rule, item.path, item.line, item.target),
        )
    )


def _sum_edge_weights(graph: nx.DiGraph, node: str, *, incoming: bool) -> int:
    edges = graph.in_edges(node, data=True) if incoming else graph.out_edges(node, data=True)
    return sum(int(data.get("weight", 1)) for _, _, data in edges)


def analyze_repository(repository_root: Path, package_name: str = "qlinks") -> dict[str, object]:
    """Return a JSON-serializable architecture analysis."""

    module_paths, imports = discover_imports(repository_root, package_name)
    module_graph = _graph_from_imports(module_paths, imports)
    import_time_graph = _import_time_graph(module_graph)
    package_graph = _package_graph(module_graph, package_name)
    import_time_package_graph = _package_graph(import_time_graph, package_name)

    static_module_cycles = _cycle_components(module_graph)
    import_time_module_cycles = _cycle_components(import_time_graph)
    static_package_cycles = _cycle_components(package_graph)
    import_time_package_cycles = _cycle_components(import_time_package_graph)
    surface_import_time_module_cycles = [
        component
        for component in import_time_module_cycles
        if any(
            module_paths[module].name == "__init__.py" or module in SURFACE_FACADE_MODULES
            for module in component
        )
    ]
    implementation_import_time_module_cycles = [
        component
        for component in import_time_module_cycles
        if component not in surface_import_time_module_cycles
    ]
    static_cyclic_modules = {module for component in static_module_cycles for module in component}
    import_time_cyclic_modules = {
        module for component in import_time_module_cycles for module in component
    }
    cyclic_packages = {package for component in static_package_cycles for package in component}

    module_records: list[ModuleRecord] = []
    for module in sorted(module_graph.nodes):
        path = module_paths[module]
        module_records.append(
            ModuleRecord(
                module=module,
                path=path.relative_to(repository_root).as_posix(),
                package=_top_level_package(module, package_name),
                lines=len(path.read_text(encoding="utf-8").splitlines()),
                fan_in=int(module_graph.in_degree(module)),
                fan_out=int(module_graph.out_degree(module)),
                incoming_imports=_sum_edge_weights(module_graph, module, incoming=True),
                outgoing_imports=_sum_edge_weights(module_graph, module, incoming=False),
                in_import_cycle=module in import_time_cyclic_modules,
                in_static_cycle=module in static_cyclic_modules,
            )
        )

    records_by_package: dict[str, list[ModuleRecord]] = defaultdict(list)
    for record in module_records:
        records_by_package[record.package].append(record)

    package_records: list[PackageRecord] = []
    for package in sorted(package_graph.nodes):
        records = records_by_package.get(package, [])
        package_records.append(
            PackageRecord(
                package=package,
                files=len(records),
                lines=sum(record.lines for record in records),
                fan_in=int(package_graph.in_degree(package)),
                fan_out=int(package_graph.out_degree(package)),
                incoming_imports=_sum_edge_weights(package_graph, package, incoming=True),
                outgoing_imports=_sum_edge_weights(package_graph, package, incoming=False),
                in_cycle=package in cyclic_packages,
            )
        )

    violations = _architecture_violations(imports)
    commit = _git_commit(repository_root)

    return {
        "schema_version": 2,
        "package": package_name,
        "git_commit": commit,
        "summary": {
            "modules": len(module_records),
            "packages": len(package_records),
            "lines": sum(record.lines for record in module_records),
            "internal_edges": module_graph.number_of_edges(),
            "package_edges": package_graph.number_of_edges(),
            "import_time_module_cycle_components": len(import_time_module_cycles),
            "surface_import_time_module_cycle_components": len(surface_import_time_module_cycles),
            "implementation_import_time_module_cycle_components": len(
                implementation_import_time_module_cycles
            ),
            "static_module_cycle_components": len(static_module_cycles),
            "import_time_package_cycle_components": len(import_time_package_cycles),
            "static_package_cycle_components": len(static_package_cycles),
            "boundary_violations": len(violations),
        },
        "packages": [asdict(record) for record in package_records],
        "modules": [asdict(record) for record in module_records],
        "package_edges": [
            {
                "source": str(source),
                "target": str(target),
                "weight": int(data.get("weight", 1)),
            }
            for source, target, data in sorted(package_graph.edges(data=True))
        ],
        "module_edges": [
            {
                "source": str(source),
                "target": str(target),
                "weight": int(data.get("weight", 1)),
                "eager_weight": int(data.get("eager_weight", 0)),
                "local_weight": int(data.get("local_weight", 0)),
                "type_checking_weight": int(data.get("type_checking_weight", 0)),
            }
            for source, target, data in sorted(module_graph.edges(data=True))
        ],
        "import_time_module_cycles": import_time_module_cycles,
        "surface_import_time_module_cycles": surface_import_time_module_cycles,
        "implementation_import_time_module_cycles": implementation_import_time_module_cycles,
        "static_module_cycles": static_module_cycles,
        "import_time_package_cycles": import_time_package_cycles,
        "static_package_cycles": static_package_cycles,
        "boundary_violations": [asdict(violation) for violation in violations],
    }


def _git_commit(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _condensation_levels(graph: nx.DiGraph) -> dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}
    condensation = nx.condensation(graph)
    levels: dict[int, int] = {}
    for node in reversed(list(nx.topological_sort(condensation))):
        successors = list(condensation.successors(node))
        levels[node] = 0 if not successors else 1 + max(levels[item] for item in successors)

    module_levels: dict[str, int] = {}
    for component, data in condensation.nodes(data=True):
        for member in data["members"]:
            module_levels[str(member)] = levels[int(component)]
    return module_levels


def _layered_positions(
    graph: nx.DiGraph, width: int = 1180
) -> tuple[dict[str, tuple[float, float]], int]:
    levels = _condensation_levels(graph)
    by_level: dict[int, list[str]] = defaultdict(list)
    for node in graph.nodes:
        by_level[levels.get(str(node), 0)].append(str(node))

    node_width = 150
    x_padding = 90
    y_padding = 70
    row_height = 105
    max_level = max(by_level, default=0)
    positions: dict[str, tuple[float, float]] = {}

    for level in sorted(by_level, reverse=True):
        nodes = sorted(by_level[level])
        usable = max(width - 2 * x_padding, node_width)
        if len(nodes) == 1:
            xs = [width / 2]
        else:
            step = min(usable / (len(nodes) - 1), 190)
            total = step * (len(nodes) - 1)
            start = (width - total) / 2
            xs = [start + index * step for index in range(len(nodes))]
        y = y_padding + (max_level - level) * row_height
        for node, x in zip(nodes, xs):
            positions[node] = (float(x), float(y))

    height = max(340, 2 * y_padding + (max_level + 1) * row_height)
    return positions, int(height)


def _package_svg(analysis: dict[str, object]) -> str:
    package_records = {
        str(record["package"]): record for record in analysis["packages"]  # type: ignore[index]
    }
    graph = nx.DiGraph()
    graph.add_nodes_from(package_records)
    for edge in analysis["package_edges"]:  # type: ignore[index]
        graph.add_edge(str(edge["source"]), str(edge["target"]), weight=int(edge["weight"]))

    positions, height = _layered_positions(graph)
    max_weight = max(
        (int(data.get("weight", 1)) for _, _, data in graph.edges(data=True)),
        default=1,
    )
    parts = [
        f'<svg id="package-graph" class="dependency-graph" viewBox="0 0 1180 {height}" '
        'role="img" aria-label="Top-level qlinks package dependency graph">',
        "<defs>",
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">',
        '<polygon points="0 0, 8 3.5, 0 7" class="arrow-head"></polygon>',
        "</marker>",
        "</defs>",
    ]

    for source, target, data in graph.edges(data=True):
        sx, sy = positions[str(source)]
        tx, ty = positions[str(target)]
        weight = int(data.get("weight", 1))
        stroke_width = 1.2 + 3.2 * math.sqrt(weight / max_weight)
        parts.append(
            f'<path class="pkg-edge" data-source="{html.escape(str(source))}" '
            f'data-target="{html.escape(str(target))}" data-weight="{weight}" '
            f'd="M {sx:.1f} {sy + 24:.1f} Q {(sx + tx) / 2:.1f} {(sy + ty) / 2:.1f} '
            f'{tx:.1f} {ty - 24:.1f}" style="stroke-width:{stroke_width:.2f}" '
            'marker-end="url(#arrow)"></path>'
        )

    for package, record in sorted(package_records.items()):
        x, y = positions[package]
        short_name = package.removeprefix("qlinks.")
        label = short_name if short_name else "qlinks"
        cycle_class = " cyclic" if bool(record["in_cycle"]) else ""
        parts.append(
            f'<g class="pkg-node{cycle_class}" data-node="{html.escape(package)}" '
            f'transform="translate({x - 70:.1f},{y - 24:.1f})">'
            '<rect width="140" height="48" rx="8"></rect>'
            f'<text x="70" y="20" text-anchor="middle">{html.escape(label)}</text>'
            f'<text class="node-subtitle" x="70" y="37" text-anchor="middle">'
            f'{int(record["files"])} files · {int(record["fan_out"])} deps</text>'
            "</g>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows) if body_rows else '<tr><td colspan="99">None detected.</td></tr>'
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _render_html(analysis: dict[str, object]) -> str:
    summary = analysis["summary"]  # type: ignore[assignment]
    packages = list(analysis["packages"])  # type: ignore[arg-type]
    modules = list(analysis["modules"])  # type: ignore[arg-type]
    package_svg = _package_svg(analysis)

    package_rows = [
        (
            record["package"],
            record["files"],
            record["lines"],
            record["fan_in"],
            record["fan_out"],
            record["incoming_imports"],
            record["outgoing_imports"],
        )
        for record in sorted(packages, key=lambda item: (-int(item["lines"]), str(item["package"])))
    ]
    package_table = _html_table(
        ("Package", "Files", "Lines", "Fan-in", "Fan-out", "Imports in", "Imports out"),
        package_rows,
    )

    top_fan_out = sorted(
        modules,
        key=lambda item: (
            -int(item["fan_out"]),
            -int(item["outgoing_imports"]),
            str(item["module"]),
        ),
    )[:15]
    top_fan_in = sorted(
        modules,
        key=lambda item: (
            -int(item["fan_in"]),
            -int(item["incoming_imports"]),
            str(item["module"]),
        ),
    )[:15]
    fan_out_table = _html_table(
        ("Module", "Fan-out", "Import refs", "Lines"),
        [
            (record["module"], record["fan_out"], record["outgoing_imports"], record["lines"])
            for record in top_fan_out
        ],
    )
    fan_in_table = _html_table(
        ("Module", "Fan-in", "Import refs", "Lines"),
        [
            (record["module"], record["fan_in"], record["incoming_imports"], record["lines"])
            for record in top_fan_in
        ],
    )

    surface_import_cycles = list(
        analysis["surface_import_time_module_cycles"]  # type: ignore[arg-type]
    )
    implementation_import_cycles = list(
        analysis["implementation_import_time_module_cycles"]  # type: ignore[arg-type]
    )
    import_cycle_components = surface_import_cycles + implementation_import_cycles
    import_cycle_sets = {tuple(component) for component in import_cycle_components}
    static_only_components = [
        component
        for component in analysis["static_module_cycles"]  # type: ignore[index]
        if tuple(component) not in import_cycle_sets
    ]
    cycle_rows = (
        [
            ("API/re-export import-time", len(component), " → ".join(component))
            for component in surface_import_cycles
        ]
        + [
            ("implementation import-time", len(component), " → ".join(component))
            for component in implementation_import_cycles
        ]
        + [
            ("type/lazy static", len(component), " → ".join(component))
            for component in static_only_components
        ]
    )
    cycle_table = _html_table(("Kind", "Modules", "Strongly connected component"), cycle_rows)

    violation_rows = [
        (
            item["rule"],
            item["source"],
            item["target"],
            f'{item["path"]}:{item["line"]}',
        )
        for item in analysis["boundary_violations"]  # type: ignore[index]
    ]
    violation_table = _html_table(("Rule", "Source", "Target", "Location"), violation_rows)

    package_options = "".join(
        f'<option value="{html.escape(str(record["package"]))}">'
        f'{html.escape(str(record["package"]))}</option>'
        for record in sorted(packages, key=lambda item: str(item["package"]))
        if str(record["package"]) != str(analysis["package"])
    )
    data_json = json.dumps(analysis, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    commit = analysis.get("git_commit") or "working tree / unknown"

    health_class = "good" if not int(summary["boundary_violations"]) else "warn"
    health_text = (
        "Broad architecture guardrails pass."
        if not int(summary["boundary_violations"])
        else f'{int(summary["boundary_violations"])} broad guardrail violation(s) detected.'
    )
    cycle_text = (
        "No import-time package cycles."
        if not int(summary["import_time_package_cycle_components"])
        else (
            f'{int(summary["import_time_package_cycle_components"])} '
            "import-time package cycle component(s)."
        )
    )
    implementation_cycle_text = (
        "No implementation import-time cycles."
        if not int(summary["implementation_import_time_module_cycle_components"])
        else (
            f'{int(summary["implementation_import_time_module_cycle_components"])} '
            "implementation import-time cycle component(s)."
        )
    )

    template_path = Path(__file__).with_name("architecture_report_template.html")
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "%%ARCH_000%%": html.escape(str(commit)),
        "%%ARCH_001%%": int(summary["modules"]),
        "%%ARCH_002%%": int(summary["packages"]),
        "%%ARCH_003%%": format(int(summary["lines"]), ","),
        "%%ARCH_004%%": int(summary["internal_edges"]),
        "%%ARCH_005%%": int(summary["implementation_import_time_module_cycle_components"]),
        "%%ARCH_006%%": int(summary["surface_import_time_module_cycle_components"]),
        "%%ARCH_007%%": int(summary["boundary_violations"]),
        "%%ARCH_008%%": health_class,
        "%%ARCH_009%%": html.escape(health_text),
        "%%ARCH_010%%": (
            "good"
            if not int(summary["implementation_import_time_module_cycle_components"])
            else "warn"
        ),
        "%%ARCH_011%%": html.escape(implementation_cycle_text),
        "%%ARCH_012%%": (
            "good" if not int(summary["import_time_package_cycle_components"]) else "warn"
        ),
        "%%ARCH_013%%": html.escape(cycle_text),
        "%%ARCH_014%%": package_svg,
        "%%ARCH_015%%": package_options,
        "%%ARCH_016%%": package_table,
        "%%ARCH_017%%": fan_out_table,
        "%%ARCH_018%%": fan_in_table,
        "%%ARCH_019%%": cycle_table,
        "%%ARCH_020%%": violation_table,
        "%%ARCH_021%%": data_json,
    }
    for marker, value in replacements.items():
        template = template.replace(marker, str(value))
    return template


def write_report(
    repository_root: Path,
    html_path: Path,
    json_path: Path,
    package_name: str = "qlinks",
) -> dict[str, object]:
    """Analyze the repository and write HTML plus JSON reports."""

    analysis = analyze_repository(repository_root, package_name)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_render_html(analysis), encoding="utf-8")
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return analysis


def _parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    default_dir = repository_root / "docs" / "build" / "html" / "_static" / "architecture"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root, help="Repository root.")
    parser.add_argument("--package", default="qlinks", help="Python package to analyze.")
    parser.add_argument(
        "--html",
        type=Path,
        default=default_dir / "qlinks-architecture.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=default_dir / "qlinks-architecture.json",
        help="Output JSON path.",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in a browser.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.root.resolve()
    html_path = args.html.expanduser().resolve()
    json_path = args.json.expanduser().resolve()
    analysis = write_report(repository_root, html_path, json_path, args.package)
    summary = analysis["summary"]
    print(f"Architecture HTML: {html_path}")
    print(f"Architecture JSON: {json_path}")
    print(
        "Summary: "
        f'{summary["modules"]} modules, {summary["internal_edges"]} edges, '
        f'{summary["implementation_import_time_module_cycle_components"]} '
        "implementation import-time cycles, "
        f'{summary["boundary_violations"]} broad boundary violations.'
    )
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
