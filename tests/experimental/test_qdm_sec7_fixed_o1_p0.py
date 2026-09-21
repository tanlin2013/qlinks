"""Static contracts for the post-PRIMME Sec. VII P0 lane."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
NOTEBOOKS = ROOT / "experimental" / "notebooks"
COMMON = JOBS / "qdm_sec7_fixed_o1.py"
TARGET = JOBS / "qdm_sec7_target_block.py"
PILOT = JOBS / "qdm_sec7_fixed_o1_pilot.py"
SPECTRUM = JOBS / "qdm_sec7_fixed_o1_l12_spectrum.py"
OBSERVABLES = JOBS / "qdm_sec7_fixed_o1_l12_observables.py"
SEQUENCE = JOBS / "qdm_sec7_fixed_o1_sequence.py"
STATUS = JOBS / "qdm_sec7_fixed_o1_status.py"
RUNNER = ROOT / "scripts" / "docker" / "docker_run_qdm_sec7_p0.sh"


def _load(path: Path, name: str):
    for directory in (JOBS, NOTEBOOKS):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sec7_p0_python_jobs_are_syntactically_valid() -> None:
    for path in (COMMON, TARGET, PILOT, SPECTRUM, OBSERVABLES, SEQUENCE, STATUS):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_sec7_p0_python_jobs_import_against_real_public_apis() -> None:
    common = _load(COMMON, "qdm_sec7_fixed_o1")
    target = _load(TARGET, "qdm_sec7_target_block_test")
    pilot = _load(PILOT, "qdm_sec7_fixed_o1_pilot_test")
    spectrum = _load(SPECTRUM, "qdm_sec7_fixed_o1_l12_spectrum_test")
    assert common.PILOT_HALF_WIDTHS == (0.10, 0.20, 0.25, 0.50)
    assert target.BASELINE_BUDGET == 512
    assert pilot.SYSTEMATICS_NAME == "qdm_checkerboard_fixed_O1_window_systematics.csv"
    assert spectrum.CONVERGENCE_NAME == "qdm_checkerboard_L12_fixed_O1_spectral_convergence.csv"


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


def test_l12_spectrum_stage_owns_only_window_coverage_solves() -> None:
    source = SPECTRUM.read_text(encoding="utf-8")
    assert "folded_spectrum_partial_spectrum" in source
    assert "make_resumable_folded_solver" in source
    assert "qdm_checkerboard_L12_fixed_O1_spectral_convergence.csv" in source
    assert "at_least_two_covered_budgets" in source
    assert "projector_deleted_block_covariance" not in source
    assert '"solver_status": "completed"' in source
    assert "transformed_maximum_residual" in source
    assert "residual_acceptance_tolerance" in source
    assert "tau_A_mc_raw" not in source
    assert "w_raw" not in source


def test_spectrum_stage_can_extend_after_failed_observable_budget_gate(tmp_path: Path) -> None:
    spectrum = _load(SPECTRUM, "qdm_sec7_fixed_o1_l12_spectrum_extension_test")
    assert spectrum._observables_request_extension(tmp_path) is False
    (tmp_path / spectrum.OBSERVABLES_ACCEPTANCE_NAME).write_text(
        '{"closed": false}\n',
        encoding="utf-8",
    )
    assert spectrum._observables_request_extension(tmp_path) is True


def test_spectrum_acceptance_requires_two_distinct_covered_budgets() -> None:
    spectrum = _load(SPECTRUM, "qdm_sec7_fixed_o1_l12_spectrum_acceptance_test")
    frame = pd.DataFrame(
        [
            {
                "requested_subspace_size": 2048,
                "window_coverage_complete": True,
                "window_state_count": 100,
                "window_maximum_residual": 1.0e-8,
            },
            {
                "requested_subspace_size": 2048,
                "window_coverage_complete": True,
                "window_state_count": 100,
                "window_maximum_residual": 1.0e-8,
            },
        ]
    )
    acceptance = spectrum._acceptance(
        frame,
        width=0.20,
        residual_tolerance=1.0e-6,
    )
    assert acceptance["closed"] is False


def test_fixed_window_coverage_requires_both_edges() -> None:
    spectrum = _load(SPECTRUM, "qdm_sec7_fixed_o1_l12_spectrum_coverage_test")
    residuals = np.full(4, 1.0e-9)
    covered = spectrum.coverage_metrics(
        np.asarray([11.75, 11.95, 12.05, 12.25]),
        residuals,
        target_energy=12.0,
        half_width=0.20,
        solver_tolerance=1.0e-8,
    )
    one_sided = spectrum.coverage_metrics(
        np.asarray([11.75, 11.95, 12.05, 12.19]),
        residuals,
        target_energy=12.0,
        half_width=0.20,
        solver_tolerance=1.0e-8,
    )
    assert covered["window_coverage_complete"] is True
    assert one_sided["window_coverage_complete"] is False


def test_l12_observables_stage_is_solver_free_and_uses_cached_checkpoints() -> None:
    source = OBSERVABLES.read_text(encoding="utf-8")
    assert "iter_spectral_checkpoints" in source
    assert "load_spectral_checkpoint" in source
    assert "projector_deleted_block_covariance" in source
    assert "qdm_checkerboard_thermal_overlap_fixed_O1.csv" in source
    assert "qdm_checkerboard_concentration_fixed_O1.csv" in source
    assert "folded_spectrum_partial_spectrum" not in source
    assert "make_resumable_folded_solver" not in source
    assert "primme.eigsh" not in source
    assert 'checkpoint.metadata.get("backend", "")' in source
    assert '!= "primme"' in source
    assert "by_budget" in source


def test_three_size_stage_is_descriptive_and_solver_free() -> None:
    source = SEQUENCE.read_text(encoding="utf-8")
    assert "qdm_checkerboard_fixed_O1_three_size_sequence.csv" in source
    assert "qdm_checkerboard_fixed_O1_three_size_fit_diagnostics.csv" in source
    assert '"selected_model": False' in source
    assert "scaling exponent" in source
    assert "folded_spectrum_partial_spectrum" not in source
    assert "primme.eigsh" not in source


def test_p0_status_is_file_only_and_tracks_all_remaining_gates() -> None:
    source = STATUS.read_text(encoding="utf-8")
    assert "qdm_checkerboard_L12_target_block_acceptance.json" in source
    assert "fixed_O1_pilot_recommended" in source
    assert "L12_spectrum_closed" in source
    assert "L12_observables_closed" in source
    assert "three_size_sequence_closed" in source
    assert "p0_thermal_lane_closed" in source
    assert "build_context" not in source
    assert "folded_spectrum_partial_spectrum" not in source


def test_runner_keeps_target_and_thermal_lanes_separate() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "target-block-status)" in script
    assert "target-block-refine)" in script
    assert "fixed-O1-pilot)" in script
    assert "fixed-O1-L12-spectrum)" in script
    assert "fixed-O1-L12-observables)" in script
    assert "fixed-O1-three-size)" in script
    assert "qdm_checkerboard_fullsym_finite_beta_20260810T164206Z" in script
    assert "qdm_checkerboard_primme_staged_20260825T164226Z" in script
    assert "qdm_sec7_fixed_o1_p0_20260831T064812Z" in script
    assert "QLINKS_QDM_PRIMME_WARM_START_VECTORS:-512" in script
    assert '--residual-tolerance "${FIXED_L12_RESIDUAL_TOLERANCE}"' in script

    pilot = script.split("fixed-O1-pilot)", maxsplit=1)[1].split(
        "fixed-O1-L12-spectrum)", maxsplit=1
    )[0]
    assert "qdm_sec7_fixed_o1_pilot.py" in pilot
    assert "qdm_sec7_target_block.py" not in pilot
    assert "--primme-data-dir" in pilot

    observables = script.split("fixed-O1-L12-observables)", maxsplit=1)[1].split(
        "fixed-O1-three-size)", maxsplit=1
    )[0]
    assert "qdm_sec7_fixed_o1_l12_observables.py" in observables
    assert "qdm_sec7_target_block.py" not in observables


def test_runner_recommends_only_remaining_thermal_lane() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    recommended = script.split("Recommended remaining P0 sequence", maxsplit=1)[1]
    assert "--stage fixed-O1-pilot" in recommended
    assert "--stage fixed-O1-L12-spectrum" in recommended
    assert "--stage fixed-O1-L12-observables" in recommended
    assert "--stage fixed-O1-three-size" in recommended
    assert "--stage status" in recommended
    assert "--stage target-block-refine" not in recommended
    assert "target-block lane is already closed" in recommended


def test_common_contract_locks_fixed_width_candidates_and_sector_dimensions() -> None:
    source = COMMON.read_text(encoding="utf-8")
    assert "PILOT_HALF_WIDTHS = (0.10, 0.20, 0.25, 0.50)" in source
    assert "EXPECTED_SECTOR_DIMENSIONS = {4: 15, 8: 1125, 12: 114483}" in source
    assert "window_half_width" in source
    assert "estimated_L12_eigenpair_budget" in source
