"""Fail-fast contracts for preserving evidence identity across SSH sessions."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/docker/docker_run_qdm_parity_evidence.sh"
PROTOCOL_NAME = "qdm_checkerboard_fixed_O1_window_recommendation.json"


def _run(stage, data_dir, *, run_id=None, extra_env=None):
    env = os.environ.copy()
    env.pop("QLINKS_EVIDENCE_RUN_ID", None)
    env.pop("QLINKS_DOCKER_DETACH", None)
    env.pop("QLINKS_QDM_PRIMME_WALLTIME_SECONDS", None)
    env.update(QLINKS_DATA_DIR=str(data_dir), QLINKS_DOCKER_DRY_RUN="1")
    if run_id is not None:
        env["QLINKS_EVIDENCE_RUN_ID"] = run_id
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(RUNNER), stage], cwd=ROOT, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "stage",
    [
        "L12-spectrum",
        "L12-reconstruct",
        "L12-refine",
        "L12-canonical",
        "L12-observables",
        "L12-figure9",
        "gate",
    ],
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


def test_figure_runner_uses_the_accepted_data_finalizer_without_a_spectrum_command(tmp_path):
    data = tmp_path / "data"
    protocol = data / "evidence_jobs" / "existing_run" / PROTOCOL_NAME
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}")
    result = _run("L12-figure9", data, run_id="existing_run")
    assert result.returncode == 0
    assert "experimental/jobs/finalize_qdm_parity_figure9.py" in result.stdout
    assert "evidence_jobs/existing_run" in result.stdout
    assert "qdm_sec7_fixed_o1_l12_spectrum.py" not in result.stdout
    assert "--refine-cached-budget" not in result.stdout


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


def test_refinement_runner_tightens_only_the_folded_tolerance(tmp_path):
    data = tmp_path / "data"
    protocol = data / "evidence_jobs" / "existing_run" / PROTOCOL_NAME
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}")
    result = _run("L12-refine", data, run_id="existing_run")
    assert result.returncode == 0
    assert "--refine-cached-budget --budgets 4096 --tolerance 1e-11" in result.stdout
    assert "--residual-tolerance 1e-6" in result.stdout


def test_solver_tuning_overrides_reach_the_container_without_changing_acceptance(tmp_path):
    data = tmp_path / "data"
    protocol = data / "evidence_jobs" / "existing_run" / PROTOCOL_NAME
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}")
    overrides = {
        "QLINKS_QDM_PRIMME_METHOD": "PRIMME_GD_plusK",
        "QLINKS_QDM_PRIMME_MAX_BASIS_SIZE": "512",
        "QLINKS_QDM_PRIMME_MAX_BLOCK_SIZE": "32",
        "QLINKS_QDM_PRIMME_MIN_RESTART_SIZE": "128",
        "QLINKS_QDM_PRIMME_MAX_PREV_RETAIN": "8",
    }
    result = _run("L12-refine", data, run_id="existing_run", extra_env=overrides)
    assert result.returncode == 0
    command = shlex.split(result.stdout.splitlines()[-1])
    environment = [command[i + 1] for i, arg in enumerate(command) if arg == "--env"]
    for name, value in overrides.items():
        assert f"{name}={value}" in environment
    assert "--refine-cached-budget --budgets 4096 --tolerance 1e-11" in result.stdout
    assert "--residual-tolerance 1e-6" in result.stdout


def test_trial_walltime_is_enforced_inside_the_container(tmp_path):
    data = tmp_path / "data"
    protocol = data / "evidence_jobs" / "existing_run" / PROTOCOL_NAME
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}")
    result = _run(
        "L12-refine",
        data,
        run_id="existing_run",
        extra_env={
            "QLINKS_QDM_PRIMME_WALLTIME_SECONDS": "7200",
            "QLINKS_DOCKER_IMAGE": "tested-primme-image",
        },
    )
    assert result.returncode == 0
    command = shlex.split(result.stdout.splitlines()[-1])
    image_index = command.index("tested-primme-image")
    assert command[image_index + 1 : image_index + 6] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "7200s",
        "python",
    ]


@pytest.mark.parametrize("walltime", ["0", "-1", "2h", "1.5"])
def test_invalid_trial_walltime_fails_before_creating_output(tmp_path, walltime):
    data = tmp_path / "data"
    result = _run(
        "L12-refine",
        data,
        run_id="existing_run",
        extra_env={"QLINKS_QDM_PRIMME_WALLTIME_SECONDS": walltime},
    )
    assert result.returncode == 2
    assert "must be a positive integer" in result.stderr
    assert not data.exists()


@pytest.mark.parametrize("mode", ["detached", "launch_failure", "signal_exit"])
def test_container_lifecycle_retains_diagnostics_and_reports_failure(tmp_path, mode):
    data = tmp_path / "data"
    output = data / "evidence_jobs" / "existing_run"
    output.mkdir(parents=True)
    (output / PROTOCOL_NAME).write_text("{}")
    # The launcher is exercised against an executable Docker stand-in; no
    # numerical job or real container is needed to test lifecycle semantics.
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$FAKE_DOCKER_ARGS"\n'
        'printf "fake-container-id\\n"\nexit "$FAKE_DOCKER_EXIT"\n'
    )
    docker.chmod(0o755)
    args_path = tmp_path / "docker-args"
    id_file = output / "L12-refine.container-id"
    id_file.write_text("previous-container-id\n")
    exit_code = {"detached": 0, "launch_failure": 125, "signal_exit": 129}[mode]
    result = _run(
        "L12-refine",
        data,
        run_id="existing_run",
        extra_env={
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "QLINKS_DOCKER_DRY_RUN": "0",
            "QLINKS_DOCKER_DETACH": "0" if mode == "signal_exit" else "1",
            "FAKE_DOCKER_ARGS": str(args_path),
            "FAKE_DOCKER_EXIT": str(exit_code),
        },
    )
    assert result.returncode == exit_code
    args = args_path.read_text().splitlines()
    assert "--refine-cached-budget" in args
    assert args[args.index("--cache-root") + 1].endswith("qdm_target_parity_v1")
    if mode == "signal_exit":
        assert "--rm" in args and "--detach" not in args
        assert "Container exit code: 129" in result.stdout
        assert "Container exit code: 129" in (output / "L12-refine.log").read_text()
    else:
        assert "--detach" in args and "--rm" not in args
        assert "--name" in args
    if mode == "detached":
        assert id_file.read_text() == "fake-container-id\n"
        assert "numerical completion and acceptance are not yet established" in result.stdout
        assert "docker logs -f fake-container-id" in result.stdout
        assert "{{json .State}}" in result.stdout
        assert not (output / "L12-refine.log").exists()  # Read via Docker until explicitly saved.
    else:
        assert id_file.read_text() == "previous-container-id\n"
        assert "Launch succeeded" not in result.stdout


def test_invalid_detach_mode_fails_before_creating_output(tmp_path):
    data = tmp_path / "data"
    result = _run("exact", data, extra_env={"QLINKS_DOCKER_DETACH": "yes"})
    assert result.returncode == 2
    assert "QLINKS_DOCKER_DETACH must be 0 or 1" in result.stderr
    assert not data.exists()
