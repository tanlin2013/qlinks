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
    assert "qlinks/open_system/diagnostics/__init__.py" in limits


def test_security_scan_ignores_local_virtualenv(repository_health_module, tmp_path: Path) -> None:
    virtualenv = tmp_path / ".venv" / "lib" / "python3.14" / "site-packages"
    key_marker = "-----BEGIN " + "PRIVATE KEY-----"
    private_key = virtualenv / "cryptography" / "serialization" / "ssh.py"
    private_key.parent.mkdir(parents=True)
    private_key.write_text(
        f"TEST_KEY = {key_marker!r}\n",
        encoding="utf-8",
    )
    certificate = virtualenv / "certifi" / "cacert.pem"
    certificate.parent.mkdir(parents=True)
    certificate.write_text("third-party certificate bundle\n", encoding="utf-8")

    file_findings, secret_findings = repository_health_module._security_findings(tmp_path)

    assert file_findings == []
    assert secret_findings == []


def test_security_scan_still_flags_repository_owned_secrets(
    repository_health_module, tmp_path: Path
) -> None:
    sensitive_file = tmp_path / "config" / "production.key"
    sensitive_file.parent.mkdir(parents=True)
    sensitive_file.write_text("repository-owned-secret\n", encoding="utf-8")
    key_marker = "-----BEGIN " + "PRIVATE KEY-----"
    secret_source = tmp_path / "config" / "embedded_secret.py"
    secret_source.write_text(f"TEST_KEY = {key_marker!r}\n", encoding="utf-8")

    file_findings, secret_findings = repository_health_module._security_findings(tmp_path)

    assert file_findings == ["sensitive filename: config/production.key"]
    assert secret_findings == ["possible private key material: config/embedded_secret.py"]


def test_precommit_wiring_ignores_commented_hook_ids(repository_health_module) -> None:
    text = """
-   repo: local
    hooks:
#    -   id: repository-health
    -   id: test-health
"""

    assert repository_health_module._active_precommit_hook_ids(text) == {"test-health"}
