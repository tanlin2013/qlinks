"""Regression tests for the repository-health guardrail."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPOSITORY_ROOT / "tools" / "repository_health.py"
_BUDGET_PATH = _REPOSITORY_ROOT / "tools" / "repository_health_budget.json"


@pytest.fixture(scope="module")
def repository_health_module():
    spec = importlib.util.spec_from_file_location("qlinks_repository_health", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    tools_root = str(_TOOL_PATH.parent)
    sys.path.insert(0, tools_root)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(tools_root)
    return module


def test_repository_health_guardrail_passes_current_tree(repository_health_module) -> None:
    budget = repository_health_module._load_budget(_BUDGET_PATH)
    snapshot, violations = repository_health_module.build_snapshot(_REPOSITORY_ROOT, budget)

    assert violations == []
    assert snapshot.static_module_cycles == 0
    assert snapshot.static_package_cycles == 0
    assert snapshot.boundary_violations == 0
    assert snapshot.sensitive_file_findings == 0
    assert snapshot.secret_pattern_findings == 0


def test_repository_health_budget_tracks_curated_api_surfaces(repository_health_module) -> None:
    budget = repository_health_module._load_budget(_BUDGET_PATH)
    limits = budget["public_api_export_limits"]

    assert "qlinks/caging/__init__.py" in limits
    assert "qlinks/caging/analysis/__init__.py" in limits
    assert "qlinks/caging/local_search/__init__.py" in limits
    assert "qlinks/caging/stability/__init__.py" in limits
    assert "qlinks/open_system/__init__.py" in limits
