"""Static contracts for the post-PRIMME Sec. VII P0 lane."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
COMMON = JOBS / "qdm_sec7_fixed_o1.py"
TARGET = JOBS / "qdm_sec7_target_block.py"
PILOT = JOBS / "qdm_sec7_fixed_o1_pilot.py"
RUNNER = ROOT / "scripts" / "docker" / "docker_run_qdm_sec7_p0.sh"


def test_sec7_p0_python_jobs_are_syntactically_valid() -> None:
    for path in (COMMON, TARGET, PILOT):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_target_block_lane_starts_from_persisted_512_checkpoint() -> None:
    source = TARGET.read_text(encoding="utf-8")
    assert "BASELINE_BUDGET = 512" in source
    assert "DEFAULT_BUDGETS = (640, 768)" in source
    assert "DEFAULT_TOLERANCES = (1.0e-9, 1.0e-10)" in source
    assert "validated 512-vector" in source
    assert "target-energy projector refinement only" in source
    assert "window_coverage_complete" not in source
    assert "shift_invert" not in source


def test_fixed_o1_pilot_cannot_start_large_strip_or_primme_solver() -> None:
    source = PILOT.read_text(encoding="utf-8")
    assert "for repeats in (1, 2):" in source
    assert "PILOT_HALF_WIDTHS" in source
    assert "folded_spectrum_partial_spectrum" not in source
    assert "shift_invert_partial_spectrum" not in source
    assert "primme.eigsh" not in source
    assert "repeats=3" not in source
    assert "qdm_checkerboard_fixed_O1_window_systematics.csv" in source
    assert "qdm_checkerboard_fixed_O1_window_recommendation.json" in source


def test_runner_keeps_target_and_thermal_lanes_separate() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "target-block-status)" in script
    assert "target-block-refine)" in script
    assert "fixed-O1-pilot)" in script
    assert "qdm_checkerboard_fullsym_finite_beta_20260810T164206Z" in script
    assert "qdm_checkerboard_primme_staged_20260825T164226Z" in script
    assert "QLINKS_QDM_PRIMME_WARM_START_VECTORS:-512" in script

    pilot = script.split("fixed-O1-pilot)", maxsplit=1)[1].split("status)", maxsplit=1)[0]
    assert "qdm_sec7_fixed_o1_pilot.py" in pilot
    assert "qdm_sec7_target_block.py" not in pilot
    assert "--primme-data-dir" in pilot


def test_common_contract_locks_fixed_width_candidates_and_sector_dimensions() -> None:
    source = COMMON.read_text(encoding="utf-8")
    assert "PILOT_HALF_WIDTHS = (0.10, 0.20, 0.25, 0.50)" in source
    assert "EXPECTED_SECTOR_DIMENSIONS = {4: 15, 8: 1125, 12: 114483}" in source
    assert "window_half_width" in source
    assert "estimated_L12_eigenpair_budget" in source
