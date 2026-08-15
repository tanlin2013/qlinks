"""Regression tests for the lightweight local test-health scanner."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPOSITORY_ROOT / "tools" / "test_health.py"


@pytest.fixture(scope="module")
def test_health_module():
    spec = importlib.util.spec_from_file_location("qlinks_test_health", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_static_marker_scan_preserves_test_taxonomy(test_health_module) -> None:
    source = """
import pytest
pytestmark = pytest.mark.integration

@pytest.mark.scientific
def test_scientific():
    pass

@pytest.mark.manual
def test_manual():
    pass

def test_integration_only():
    pass
"""
    tree = ast.parse(source)

    markers, fast_functions, unmarked_visual = test_health_module._static_file_marker_metrics(
        tree, source
    )

    assert markers == {"integration": 3, "scientific": 1, "manual": 1, "gpu": 0}
    assert fast_functions == 0
    assert unmarked_visual == 0
