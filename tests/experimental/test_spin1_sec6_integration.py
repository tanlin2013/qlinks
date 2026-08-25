from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

import spin1_sec6_integration as integration  # noqa: E402


def _write_reference_evidence(root: Path, *, deformation_protocol: bool = True) -> None:
    (root / "spin1_xy_sec6_provisioning_summary.json").write_text(
        json.dumps({"representative_sparse_budget_certified": True}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "L": 14,
                "kappa_over_J": 0.1,
                "variant": "raw",
                "w_L": integration.REFERENCE_L14_RAW_WIDTH,
                "window_state_count": integration.REFERENCE_L14_RAW_STATES,
                "joint_dark_rank": integration.REFERENCE_L14_DARK_RANK,
                "removed_fraction": integration.REFERENCE_L14_REMOVED_FRACTION,
                "sparse_convergence_passed": True,
            },
            {
                "L": 14,
                "kappa_over_J": 0.1,
                "variant": "clean",
                "w_L": integration.REFERENCE_L14_CLEAN_WIDTH,
                "window_state_count": integration.REFERENCE_L14_RAW_STATES,
                "joint_dark_rank": integration.REFERENCE_L14_DARK_RANK,
                "removed_fraction": integration.REFERENCE_L14_REMOVED_FRACTION,
                "sparse_convergence_passed": True,
            },
        ]
    ).to_csv(root / "spin1_xy_kappa0p1_concentration_L14.csv", index=False)

    tolerance_rows = []
    for variant, width in (
        ("raw", integration.REFERENCE_L14_RAW_WIDTH),
        ("clean", integration.REFERENCE_L14_CLEAN_WIDTH),
    ):
        for tolerance in (1.0e-10, 3.0e-10, 1.0e-9):
            tolerance_rows.append(
                {
                    "variant": variant,
                    "energy_block_tolerance": tolerance,
                    "energy_block_count": 4007,
                    "largest_covariance_width": width + tolerance * 1.0e-3,
                }
            )
    pd.DataFrame(tolerance_rows).to_csv(
        root / "spin1_xy_kappa0p1_concentration_L14_tolerance_audit.csv", index=False
    )

    bridge_rows = []
    for length in (8, 10, 12, 14):
        for bridge, distance in (
            ("mc_to_beta0_resolved", 0.02 / (length / 8.0)),
            ("beta0_resolved_to_fixedM", 2.78e-5 if length == 14 else 1.0e-3 / length),
        ):
            bridge_rows.append(
                {
                    "L": length,
                    "kappa_over_J": 0.1,
                    "window_exponent": 0.25,
                    "window_prefactor": 1.0,
                    "window_role": "alpha_0.25_c_1",
                    "window_half_width": length**0.25,
                    "bridge": bridge,
                    "trace_distance": distance,
                    "abs_delta_tau_A": distance / 4.0,
                    "abs_delta_tau_Z": distance / 3.0,
                    "abs_delta_tau_Y": distance / 2.0,
                    "raw_window_state_count": 100 * length,
                }
            )
    pd.DataFrame(bridge_rows).to_csv(
        root / "spin1_xy_kappa0p1_two_bridge_rdm_distance.csv", index=False
    )

    micro_rows = []
    for length in (8, 10, 12, 14):
        micro_rows.append(
            {
                "L": length,
                "kappa_over_J": 0.1,
                "window_exponent": 0.25,
                "window_prefactor": 1.0,
                "window_role": "alpha_0.25_c_1",
                "window_half_width": length**0.25,
                "window_state_count": 100 * length,
                "tau_A_mc_raw": 0.10 + 0.001 * length,
                "tau_Z_mc_raw": 0.20 + 0.001 * length,
                "tau_Y_mc_raw": 0.30 + 0.001 * length,
            }
        )
    pd.DataFrame(micro_rows).to_csv(
        root / "spin1_xy_kappa0p1_microcanonical_windows_sec6.csv", index=False
    )

    pd.DataFrame(
        {
            "L": [12, 12, 12],
            "energy_density": [-0.1, 0.0, 0.1],
            "Q_A": [0.1, 0.0, 0.12],
            "Q_Z": [0.2, 0.0, 0.22],
            "Q_Y": [0.3, 0.0, 0.32],
            "is_tower_state": [False, True, False],
        }
    ).to_csv(root / "spin1_xy_kappa0p1_eth_scatter_Lmax.csv", index=False)

    deformation_rows = []
    for kappa in (0.05, 0.10, 0.15, 0.20):
        row = {
            "L": 12,
            "kappa_over_J": kappa,
            "tau_A_mc_raw": 0.11 + kappa,
            "tau_Z_mc_raw": 0.21 + kappa,
            "tau_Y_mc_raw": 0.31 + kappa,
        }
        if deformation_protocol:
            row.update({"window_exponent": 0.25, "window_prefactor": 1.0})
        deformation_rows.append(row)
    pd.DataFrame(deformation_rows).to_csv(
        root / "spin1_xy_kappa_matching_grid.csv", index=False
    )

    concentration_rows = []
    for length in (8, 10, 12):
        for kappa in (0.05, 0.10, 0.15, 0.20):
            row = {
                "L": length,
                "kappa_over_J": kappa,
                "largest_covariance_width": 0.08 / length + 0.01 * kappa,
            }
            if deformation_protocol:
                row.update({"window_exponent": 0.25, "window_prefactor": 1.0})
            concentration_rows.append(row)
    pd.DataFrame(concentration_rows).to_csv(
        root / "spin1_xy_kappa_concentration_grid.csv", index=False
    )

    pd.DataFrame(
        {
            "real_t2_over_J": [-0.01, 0.0, 0.01],
            "imag_t2_over_J": [0.1, 0.1, 0.1],
            "normalized_tower_residual": [1.0e-2, 1.0e-12, 1.0e-2],
        }
    ).to_csv(root / "spin1_xy_complex_t2_obstruction_grid.csv", index=False)


def test_validate_established_sec6_evidence_without_recomputation(tmp_path: Path) -> None:
    _write_reference_evidence(tmp_path)
    audit = integration.validate_established_evidence(tmp_path)
    assert audit["representative_l14_validated"] is True
    assert audit["sparse_budget_certified"] is True
    assert audit["exact_energy_tolerance_stable"] is True
    assert audit["beta0_second_bridge_trace_distance"] == 2.78e-5


def test_build_figure_data_uses_primary_window_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_reference_evidence(source)

    result = integration.build_figure_data(source, output)

    assert "spin1_xy_figure6_panel_b_witness_sequence.csv" in result["written"]
    assert "spin1_xy_figure6_panel_c_deformation.csv" in result["written"]
    assert "spin1_xy_figure6_panel_d_family_band.csv" in result["written"]
    panel_b = pd.read_csv(output / "spin1_xy_figure6_panel_b_witness_sequence.csv")
    assert set(panel_b["L"]) == {8, 10, 12, 14}
    assert set(panel_b["witness"]) == {"A", "Z", "Y"}


def test_legacy_deformation_window_is_reported_pending(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_reference_evidence(source, deformation_protocol=False)

    result = integration.build_figure_data(source, output)

    panel_c = "spin1_xy_figure6_panel_c_deformation.csv"
    assert panel_c not in result["written"]
    assert panel_c in result["pending"]
    assert "do not reuse" in result["pending"][panel_c]
    panel_d = "spin1_xy_figure6_panel_d_family_band.csv"
    assert panel_d not in result["written"]
    assert "do not reuse" in result["pending"][panel_d]


def test_primary_window_mask_rejects_untagged_legacy_table() -> None:
    frame = pd.DataFrame({"L": [8, 10], "value": [1.0, 2.0]})
    try:
        integration._primary_window_mask(frame)
    except integration.EvidenceValidationError as exc:
        assert "cannot identify" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("untagged window table should be rejected")
