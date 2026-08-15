"""Smoke tests for the repository architecture-report generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPORT_PATH = _REPOSITORY_ROOT / "tools" / "architecture_report.py"


@pytest.fixture(scope="module")
def architecture_report_module():
    spec = importlib.util.spec_from_file_location("qlinks_architecture_report", _REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def architecture_analysis(architecture_report_module):
    """Analyze the repository once for all architecture-report tests."""
    return architecture_report_module.analyze_repository(_REPOSITORY_ROOT)


def test_architecture_report_analyzes_current_repository(architecture_analysis) -> None:
    analysis = architecture_analysis

    assert analysis["summary"]["modules"] > 0
    assert analysis["summary"]["packages"] > 0
    assert analysis["summary"]["import_time_package_cycle_components"] == 0
    assert analysis["summary"]["implementation_import_time_module_cycle_components"] == 0
    assert analysis["summary"]["static_module_cycle_components"] == 0
    assert analysis["summary"]["static_package_cycle_components"] == 0
    assert analysis["summary"]["boundary_violations"] == 0
    assert any(record["package"] == "qlinks.caging" for record in analysis["packages"])
    assert not any(
        any(module.startswith("qlinks.caging.local_search.") for module in component)
        for component in analysis["static_module_cycles"]
    )


def test_architecture_report_writes_self_contained_html_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture_report_module,
    architecture_analysis,
) -> None:
    report = architecture_report_module
    html_path = tmp_path / "qlinks-architecture.html"
    json_path = tmp_path / "qlinks-architecture.json"

    # Exercise report serialization/rendering without repeating the repository-wide AST analysis.
    monkeypatch.setattr(
        report,
        "analyze_repository",
        lambda *_args, **_kwargs: architecture_analysis,
    )
    report.write_report(_REPOSITORY_ROOT, html_path, json_path)

    html_text = html_path.read_text(encoding="utf-8")
    assert "qlinks architecture diagnosis" in html_text
    assert 'id="architecture-data"' in html_text
    assert "https://" not in html_text
    assert json_path.is_file()


def test_import_discovery_reuses_raw_targets_without_changing_internal_edges(
    architecture_report_module,
) -> None:
    module_paths, imports = architecture_report_module.discover_imports(_REPOSITORY_ROOT)
    raw_module_paths, raw_imports, raw_targets = (
        architecture_report_module.discover_imports_with_raw(_REPOSITORY_ROOT)
    )

    assert raw_module_paths == module_paths
    assert raw_imports == imports
    assert any(target.target.startswith("numpy") for target in raw_targets)
