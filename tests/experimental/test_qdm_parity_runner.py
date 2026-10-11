"""Fail-fast contracts for preserving evidence identity across SSH sessions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/docker/docker_run_qdm_parity_evidence.sh"
PROTOCOL_NAME = "qdm_checkerboard_fixed_O1_window_recommendation.json"


def _run(stage, data_dir, *, run_id=None):
    env = os.environ.copy()
    env.pop("QLINKS_EVIDENCE_RUN_ID", None)
    env.update(QLINKS_DATA_DIR=str(data_dir), QLINKS_DOCKER_DRY_RUN="1")
    if run_id is not None:
        env["QLINKS_EVIDENCE_RUN_ID"] = run_id
    return subprocess.run(
        ["bash", str(RUNNER), stage], cwd=ROOT, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "stage", ["L12-spectrum", "L12-reconstruct", "L12-canonical", "L12-observables", "gate"]
)
def test_followup_without_run_id_fails_before_creating_a_folder(tmp_path, stage):
    data = tmp_path / "data"
    result = _run(stage, data)
    assert result.returncode == 2
    assert "QLINKS_EVIDENCE_RUN_ID" in result.stderr
    assert "docker run" not in result.stdout
    assert not data.exists()


def test_wrong_run_id_reports_parity_exact_prerequisite_before_launch(tmp_path):
    result = _run("L12-spectrum", tmp_path / "data", run_id="empty_run")
    assert result.returncode == 2
    assert "Missing parity window protocol" in result.stderr
    assert "qdm_parity_evidence.sh exact" in result.stderr
    assert "fixed-O1-pilot" not in result.stderr
    assert "docker run" not in result.stdout
    assert not (tmp_path / "data").exists()


def test_existing_run_id_is_preserved_in_the_docker_command(tmp_path):
    data = tmp_path / "data"
    protocol = data / "evidence_jobs" / "existing_run" / PROTOCOL_NAME
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}")  # Numerical protocol validation remains in the Python stage.
    result = _run("L12-spectrum", data, run_id="existing_run")
    assert result.returncode == 0
    assert "docker run" in result.stdout
    assert "evidence_jobs/existing_run" in result.stdout
    assert "--sy-character target" in result.stdout
    assert "export QLINKS_EVIDENCE_RUN_ID=existing_run" in result.stdout
    assert sorted(p.name for p in (data / "evidence_jobs").iterdir()) == ["existing_run"]


def test_exact_stage_can_still_initialize_a_run(tmp_path):
    data = tmp_path / "data"
    result = _run("exact", data)
    assert result.returncode == 0
    assert "docker run" in result.stdout
    assert "Reuse in every SSH session" in result.stdout
    assert len(list((data / "evidence_jobs").iterdir())) == 1


def test_reconstruction_runner_requests_only_the_saved_4096_budget(tmp_path):
    data = tmp_path / "data"
    protocol = data / "evidence_jobs" / "existing_run" / PROTOCOL_NAME
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}")
    result = _run("L12-reconstruct", data, run_id="existing_run")
    assert result.returncode == 0
    assert "--reconstruct-cached-only --budgets 4096" in result.stdout
    assert "qdm_sec7_fixed_o1_l12_spectrum.py" in result.stdout
    assert "L12-reconstruct.log" not in result.stdout  # Dry run starts no process.
