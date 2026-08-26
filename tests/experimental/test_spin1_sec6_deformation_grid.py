from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

import spin1_sec6_deformation_grid as grid  # noqa: E402


def _write_representative_products(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    concentration_rows = []
    sequence_rows = []
    worst_rows = []
    for length in (8, 10, 12, 14):
        concentration_rows.append(
            {
                "L": length,
                "M": -2,
                "J3_over_J": 0.1,
                "kappa_over_J": 0.1,
                "variant": "raw",
                "window_protocol": grid.WINDOW_PROTOCOL,
                "window_half_width": length**0.25,
                "window_state_count": 20 + length,
                "retained_state_count": 19 + length,
                "joint_dark_rank": 1,
                "removed_fraction": 1.0 / (20 + length),
                "energy_block_count": 10,
                "covered_spectral_half_width": 10.0,
                "window_max_eigenpair_residual": 1.0e-12,
                "window_median_eigenpair_residual": 5.0e-13,
                "tower_residual": 1.0e-13,
                "w_L": 1.0 / length,
                "median_nonidentity_width": 0.5 / length,
                "spectrum_method": "scipy.linalg.eigh",
                "resolved_sector_dimension": 100 + length,
                "requested_eigenpairs": 100 + length,
                "checkpoint_path": f"representative-L{length}",
            }
        )
        if length <= 12:
            for index, witness in enumerate(("A", "Z", "Y"), start=1):
                sequence_rows.append(
                    {
                        "L": length,
                        "witness": witness,
                        "tau_mc_raw": index / 10.0,
                    }
                )
            for index in range(grid.OPERATOR_BASIS_DIMENSION):
                worst_rows.append(
                    {
                        "L": length,
                        "kappa_over_J": 0.1,
                        "variant": "raw",
                        "window_protocol": grid.WINDOW_PROTOCOL,
                        "window_half_width": length**0.25,
                        "basis_operator": f"B{index:02d}",
                        "coefficient": 1.0 if index == 1 else 0.0,
                    }
                )
    pd.DataFrame(concentration_rows).to_csv(root / grid.COMMON_NAME, index=False)
    pd.DataFrame(sequence_rows).to_csv(root / grid.PANEL_B_NAME, index=False)
    pd.DataFrame(worst_rows).to_csv(root / grid.REPRESENTATIVE_WORST_NAME, index=False)


def _fake_point(length: int, kappa_over_j: float):
    half_width = length**0.25
    row = {
        "schema_version": grid.SCHEMA_VERSION,
        "L": length,
        "M": -2,
        "J3_over_J": 0.1,
        "kappa_over_J": kappa_over_j,
        "window_protocol": grid.WINDOW_PROTOCOL,
        "window_exponent": grid.WINDOW_EXPONENT,
        "window_prefactor": grid.WINDOW_PREFACTOR,
        "window_half_width": half_width,
        "window_state_count": 30 + length,
        "retained_state_count": 29 + length,
        "joint_dark_rank": 1,
        "joint_dark_fraction": 1.0 / (30 + length),
        "energy_block_count": 12,
        "covered_spectral_half_width": 10.0,
        "window_max_eigenpair_residual": 1.0e-12,
        "window_median_eigenpair_residual": 5.0e-13,
        "tower_residual": 1.0e-13,
        "w_L_raw": (1.0 + kappa_over_j) / length,
        "median_nonidentity_width_raw": 0.5 / length,
        "tau_A_mc_raw": 0.1 + kappa_over_j,
        "tau_Z_mc_raw": 0.2 + kappa_over_j,
        "tau_Y_mc_raw": 0.3 + kappa_over_j,
        "worst_basis_operator": "B01",
        "worst_basis_coefficient_abs": 1.0,
        "operator_basis_dimension": grid.OPERATOR_BASIS_DIMENSION,
        "operator_basis_version": grid.OPERATOR_BASIS_VERSION,
        "operator_basis_sha256": grid._sha256_strings(
            [f"B{index:02d}" for index in range(grid.OPERATOR_BASIS_DIMENSION)]
        ),
        "spectrum_method": "scipy.linalg.eigh",
        "full_spectrum": True,
        "sector_dimension": 100 + length,
        "returned_eigenpairs": 100 + length,
        "solve_seconds": 0.01,
        "source_role": "test_dense_point",
        "checkpoint_reused": False,
        "checkpoint_path": "",
    }
    worst = [
        {
            "L": length,
            "kappa_over_J": kappa_over_j,
            "window_protocol": grid.WINDOW_PROTOCOL,
            "window_half_width": half_width,
            "basis_operator": f"B{index:02d}",
            "coefficient_real": 1.0 if index == 1 else 0.0,
            "coefficient_imag": 0.0,
            "coefficient_abs": 1.0 if index == 1 else 0.0,
            "source_role": "test_dense_point",
        }
        for index in range(grid.OPERATOR_BASIS_DIMENSION)
    ]
    return row, worst


def test_status_lane_reuses_representative_points_without_solving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration = tmp_path / "integration"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    _write_representative_products(integration)

    def forbidden_solve(**_kwargs):
        raise AssertionError("status lane must not diagonalize")

    monkeypatch.setattr(grid, "_compute_dense_point", forbidden_solve)
    frame = grid.run_grid(
        integration_data_dir=integration,
        cache_root=cache,
        output_dir=output,
        compute_missing=False,
    )

    assert set(frame["kappa_over_J"]) == {0.1}
    assert set(frame["L"]) == {8, 10, 12}
    progress = json.loads((output / grid.PROGRESS_NAME).read_text(encoding="utf-8"))
    assert progress["completed_count"] == 3
    assert progress["target_count"] == 12
    assert len(progress["pending_points"]) == 9
    assert not progress["p0_grid_complete"]


def test_complete_grid_is_checkpointed_and_second_run_performs_no_solves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration = tmp_path / "integration"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    _write_representative_products(integration)
    calls: list[tuple[int, float]] = []

    def fake_solve(*, length: int, kappa_over_j: float):
        calls.append((length, kappa_over_j))
        return _fake_point(length, kappa_over_j)

    monkeypatch.setattr(grid, "_compute_dense_point", fake_solve)
    first = grid.run_grid(
        integration_data_dir=integration,
        cache_root=cache,
        output_dir=output,
        compute_missing=True,
    )

    assert len(first) == 12
    assert len(calls) == 9
    progress = json.loads((output / grid.PROGRESS_NAME).read_text(encoding="utf-8"))
    assert progress["p0_grid_complete"]
    panel_c = pd.read_csv(output / grid.PANEL_C_NAME)
    panel_d = pd.read_csv(output / grid.PANEL_D_NAME)
    assert len(panel_c) == 12
    assert set(panel_d["L"]) == {8, 10, 12}
    assert panel_d["complete"].all()

    def forbidden_solve(**_kwargs):
        raise AssertionError("validated point checkpoints must be reused")

    monkeypatch.setattr(grid, "_compute_dense_point", forbidden_solve)
    second = grid.run_grid(
        integration_data_dir=integration,
        cache_root=cache,
        output_dir=output,
        compute_missing=True,
    )
    assert len(second) == 12
    assert second["checkpoint_reused"].all()
