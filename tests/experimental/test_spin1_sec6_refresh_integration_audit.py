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

import spin1_exchange_convention as convention  # noqa: E402
import spin1_sec6_refresh_integration_audit as refresh  # noqa: E402


def _stamp(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[convention.EXCHANGE_CONVENTION_METADATA_KEY] = convention.CURRENT_EXCHANGE_CONVENTION
    return result


def _write_integration_products(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for length in refresh.TARGET_LENGTHS:
        for protocol, half_width in (
            (
                refresh.PRIMARY_WINDOW_PROTOCOL,
                convention.PRIMARY_WINDOW_PREFACTOR * length**convention.PRIMARY_WINDOW_EXPONENT,
            ),
            (refresh.FIXED_WINDOW_PROTOCOL, convention.FIXED_CONTROL_HALF_WIDTH),
        ):
            for variant in ("raw", "clean"):
                rows.append(
                    {
                        "L": length,
                        "kappa_over_J": refresh.REPRESENTATIVE_KAPPA_OVER_J,
                        "variant": variant,
                        "window_protocol": protocol,
                        "window_half_width": half_width,
                        "w_L": 1.0 / length,
                        "window_state_count": 20 + length,
                        "covered_spectral_half_width": 5.0,
                        "window_max_eigenpair_residual": 1.0e-6,
                    }
                )
    _stamp(pd.DataFrame(rows)).to_csv(
        root / "spin1_xy_kappa0p1_concentration_common_windows.csv",
        index=False,
    )
    (root / "spin1_xy_kappa0p1_common_window_summary.json").write_text(
        json.dumps(
            {
                "lengths": list(refresh.TARGET_LENGTHS),
                "window_protocols": [
                    refresh.PRIMARY_WINDOW_PROTOCOL,
                    refresh.FIXED_WINDOW_PROTOCOL,
                ],
                "power_law_fit_computed": False,
                convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                    convention.CURRENT_EXCHANGE_CONVENTION
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "spin1_xy_kappa0p1_common_window_checkpoint_audit.csv",
        "spin1_xy_kappa0p1_common_window_tolerance_audit.csv",
        "spin1_xy_kappa0p1_common_window_worst_eigenoperator.csv",
        "spin1_xy_figure6_panel_a_scatter.csv",
        "spin1_xy_figure6_panel_b_witness_sequence.csv",
        "spin1_xy_appendix_beta0_bridges_data.csv",
        "spin1_xy_appendix_complex_t2_obstruction_data.csv",
    ):
        _stamp(pd.DataFrame([{"value": 1.0}])).to_csv(root / name, index=False)


def _validation() -> dict[str, object]:
    return {
        "representative_l14_validated": True,
        "sparse_budget_certified": True,
        "exact_energy_tolerance_stable": True,
        "beta0_second_bridge_trace_distance": 2.78e-5,
        "source_files": {"source.json": "abc"},
    }


def test_refresh_closes_representative_common_window_without_residual_veto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    integration = tmp_path / "integration"
    source.mkdir()
    _write_integration_products(integration)
    monkeypatch.setattr(
        refresh.integration,
        "validate_established_evidence",
        lambda _source: _validation(),
    )

    report = refresh.refresh_audit(
        source_data_dir=source,
        integration_data_dir=integration,
    )

    assert report["common_window_status"] == "READY"
    assert report["representative_common_window_closed"]
    assert report["missing_primary_concentration_sizes"] == []
    assert report["primary_window_available_sizes"] == [8, 10, 12, 14]
    assert not report["deformation_grid"]["p0_grid_complete"]
    assert report["next_numerical_action"].startswith("compute or reuse only missing")
    written = report["figure_data_products"]["written"]
    assert "spin1_xy_figure6_panel_a_scatter.csv" in written
    assert "spin1_xy_figure6_panel_c_deformation.csv" not in written


def test_refresh_recognizes_completed_deformation_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    integration = tmp_path / "integration"
    source.mkdir()
    _write_integration_products(integration)
    (integration / refresh.GRID_PROGRESS_NAME).write_text(
        json.dumps(
            {
                "panel_c_complete": True,
                "panel_d_complete": True,
                "p0_grid_complete": True,
                "pending_points": [],
                "completed_count": 12,
                "target_count": 12,
                convention.EXCHANGE_CONVENTION_METADATA_KEY: (
                    convention.CURRENT_EXCHANGE_CONVENTION
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        refresh.integration,
        "validate_established_evidence",
        lambda _source: _validation(),
    )

    report = refresh.refresh_audit(
        source_data_dir=source,
        integration_data_dir=integration,
    )

    assert report["deformation_grid"]["p0_grid_complete"]
    assert report["next_numerical_action"] == (
        "none; render final figures from convention-mapped CSVs"
    )
    assert "spin1_xy_figure6_panel_c_deformation.csv" in report["figure_data_products"]["written"]
    assert "spin1_xy_figure6_panel_d_family_band.csv" in report["figure_data_products"]["written"]
