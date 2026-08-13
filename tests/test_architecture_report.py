"""Smoke tests for the repository architecture-report generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPORT_PATH = _REPOSITORY_ROOT / "tools" / "architecture_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("qlinks_architecture_report", _REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_architecture_report_analyzes_current_repository() -> None:
    report = _load_report_module()
    analysis = report.analyze_repository(_REPOSITORY_ROOT)

    assert analysis["summary"]["modules"] > 0
    assert analysis["summary"]["packages"] > 0
    assert analysis["summary"]["import_time_package_cycle_components"] == 0
    assert analysis["summary"]["implementation_import_time_module_cycle_components"] == 0
    assert analysis["summary"]["boundary_violations"] == 0
    assert any(record["package"] == "qlinks.caging" for record in analysis["packages"])


def test_architecture_report_writes_self_contained_html_and_json(tmp_path: Path) -> None:
    report = _load_report_module()
    html_path = tmp_path / "qlinks-architecture.html"
    json_path = tmp_path / "qlinks-architecture.json"

    report.write_report(_REPOSITORY_ROOT, html_path, json_path)

    html_text = html_path.read_text(encoding="utf-8")
    assert "qlinks architecture diagnosis" in html_text
    assert 'id="architecture-data"' in html_text
    assert "https://" not in html_text
    assert json_path.is_file()
