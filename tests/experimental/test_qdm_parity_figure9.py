"""Render-only data contracts; synthetic certificates are not production evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.linalg as la
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experimental/jobs"))

from qdm_figure9_l12_evidence import (  # noqa: E402
    CANONICAL_NAME,
    GATES,
    PRIMARY_NAME,
    SYSTEMATICS_NAME,
    WIDTHS,
    load_l12_evidence,
)
from qdm_parity_protocol import PROTOCOL  # noqa: E402

PHASES = (0.025, 0.05, 0.075, 0.10)


@pytest.fixture
def renderer():
    pytest.importorskip("IPython", reason="The figure runtime uses the notebook plotting helpers")
    import render_qdm_parity_figure9

    return render_qdm_parity_figure9


@pytest.fixture
def figure_data(tmp_path):
    """Simulate the producer schemas without solving a Hamiltonian."""
    data = tmp_path / "data"
    data.mkdir()
    identity = {
        "symmetry_protocol": PROTOCOL,
        "parity_policy": "measured_target",
        "fully_symmetry_resolved": True,
        "Ly": 4,
    }
    scatter, exact, blocks = [], [], []
    energy = np.linspace(6, 10, 875)
    for phase in PHASES:
        a = 0.05 + phase + 0.001 * (energy - 8) ** 2
        z = 2 * a
        a[437] = z[437] = 0
        for index, e in enumerate(energy):
            scatter.append(
                {
                    **identity,
                    "Lx": 8,
                    "Sy": 1,
                    "phase": phase,
                    "energy": e,
                    "energy_density": e / 32,
                    "Q_A": a[index],
                    "Q_Z": z[index],
                    "is_tower_state": index == 437,
                }
            )
        for width in WIDTHS:
            selected = np.abs(energy - 8) <= width + 1e-9
            raw_a, raw_z = a[selected].mean(), z[selected].mean()
            exact.append(
                {
                    **identity,
                    "Lx": 8,
                    "Sy": 1,
                    "phase": phase,
                    "sector_dimension": 875,
                    "raw_population_includes_target": True,
                    "operator_normalization": "local_HS_quotient_in_target_parity",
                    "maximum_eigenpair_residual": 1e-14,
                    "gram_frobenius_residual": 1e-13,
                    "window_half_width": width,
                    "target_energy": 8,
                    "raw_window_state_count": int(selected.sum()),
                    "matched_beta_raw": 0,
                    "tau_A_mc_raw": raw_a,
                    "tau_Z_mc_raw": raw_z,
                    "tau_A_can_raw": a.mean(),
                    "tau_Z_can_raw": z.mean(),
                    "matching_distance_raw": max(abs(raw_a - a.mean()), abs(raw_z - z.mean())),
                    "w_raw": 0.08 - phase / 10,
                    "joint_dark_removed_rank": 1,
                    "removed_fraction": 1 / int(selected.sum()),
                }
            )
        blocks.append({"Lx": 8, "phase": phase, "dimension": 875, "Sy": 1})
    four = []
    for row in exact:
        copy = {
            **row,
            "Lx": 4,
            "Sy": -1,
            "sector_dimension": 1,
            "target_energy": 4,
            "raw_window_state_count": 1,
            "removed_fraction": 1,
            "w_raw": 0,
            "matching_distance_raw": 0,
        }
        for key in ("tau_A_mc_raw", "tau_Z_mc_raw", "tau_A_can_raw", "tau_Z_can_raw"):
            copy[key] = 0
        four.append(copy)
    exact_frame = pd.DataFrame([*exact, *four])
    pd.DataFrame(scatter).to_csv(data / "qdm_parity_all_eigenstates.csv", index=False)
    exact_frame.to_csv(data / "qdm_parity_phase_window_systematics.csv", index=False)
    exact_frame[np.isclose(exact_frame.phase, 0.05)].to_csv(
        data / "qdm_checkerboard_fixed_O1_window_systematics.csv", index=False
    )
    (data / "qdm_parity_exact_audit.json").write_text(json.dumps({"blocks": blocks}))
    l12_identity = {**identity, "Lx": 12, "Sy": -1, "sector_dimension": 39667, "phase": 0.05}
    controls = []
    for budget in (4096, 5120):
        for width, count in zip(WIDTHS, (10, 20, 25, 50), strict=True):
            controls.append(
                {
                    **l12_identity,
                    "requested_subspace_size": budget,
                    "returned_eigenpairs": budget,
                    "window_protocol": "fixed_O1_total_energy",
                    "window_half_width": width,
                    "window_energy_density_half_width": width / 48,
                    "window_lower_energy": 12 - width,
                    "window_upper_energy": 12 + width,
                    "window_state_count": count,
                    "maximum_eigenpair_residual": 1e-9,
                    "gram_frobenius_residual": 1e-13,
                    "cage_window_projector_weight": 1,
                    "window_coverage_complete": True,
                    "raw_window_state_count": count,
                    "joint_dark_removed_rank": 3,
                    "removed_fraction": 3 / count,
                    "tau_A_mc_raw": 0.04,
                    "tau_Z_mc_raw": 0.08,
                    "tau_A_can": 0.041,
                    "tau_Z_can": 0.082,
                    "tau_A_can_stderr": 1e-5,
                    "tau_Z_can_stderr": 2e-5,
                    "matching_distance_raw": 0.002,
                    "matched_beta": 0.046,
                    "matched_beta_stderr": np.nan,
                    "w_raw": 0.025,
                }
            )
    frame = pd.DataFrame(controls)
    frame.to_csv(data / SYSTEMATICS_NAME, index=False)
    frame[np.isclose(frame.window_half_width, 0.25)].to_csv(data / PRIMARY_NAME, index=False)
    pd.DataFrame(
        [
            {
                **l12_identity,
                "beta_star": 0.046,
                "beta_stderr": np.nan,
                "tau_A_target": 0.041,
                "tau_Z_target": 0.082,
                "tau_A_stderr": 1e-5,
                "tau_Z_stderr": 2e-5,
            }
        ]
    ).to_csv(data / CANONICAL_NAME, index=False)
    for name, filename in GATES.items():
        payload = {**identity, "Sy": -1, "closed": True, "checks": {"fixture_passed": True}}
        if name != "canonical":
            payload.update(primary_window_half_width=0.25)
        if name == "spectral":
            payload["last_two_covered_budgets"] = [4096, 5120]
        if name == "observables":
            payload["last_two_budgets"] = [4096, 5120]
        (data / filename).write_text(json.dumps(payload))
    (data / "qdm_checkerboard_fixed_O1_window_recommendation.json").write_text(
        json.dumps(
            {
                **identity,
                "status": "recommended",
                "recommended_half_width": 0.25,
                "neighbor_controls": [0.10, 0.20, 0.50],
            }
        )
    )
    (data / "qdm_parity_character_preflight.json").write_text(
        json.dumps(
            {
                **identity,
                "blocks": [
                    {"Sy": sy, "target_Sy_residual": 0, "target_sector_dimension": dimension}
                    for sy, dimension in ((-1, 1), (1, 875), (-1, 39667))
                ],
            }
        )
    )
    return data


def test_l12_loader_reconciles_two_budgets_and_allows_unestimated_beta_error(figure_data):
    row, controls, hashes, report = load_l12_evidence(figure_data)
    assert row.requested_subspace_size == 5120
    assert np.isnan(row.matched_beta_stderr)
    assert len(controls) == 8
    assert len(hashes) == 6
    assert report["budgets"] == [4096, 5120]
    assert report["phase_range_drawn"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "open_gate",
        "wrong_parity",
        "missing_width",
        "count_changed",
        "primary_changed",
        "canonical_changed",
        "bad_residual",
    ],
)
def test_l12_loader_rejects_unaccepted_or_inconsistent_inputs(figure_data, fault):
    if fault == "open_gate":
        path = figure_data / GATES["observables"]
        payload = json.loads(path.read_text())
        payload["closed"] = False
        path.write_text(json.dumps(payload))
    else:
        path = figure_data / (PRIMARY_NAME if fault == "primary_changed" else SYSTEMATICS_NAME)
        if fault == "canonical_changed":
            path = figure_data / CANONICAL_NAME
        frame = pd.read_csv(path)
        if fault == "wrong_parity":
            frame.loc[0, "Sy"] = 1
        elif fault == "missing_width":
            frame = frame.iloc[:-1]
        elif fault == "count_changed":
            frame.loc[4, ["raw_window_state_count", "window_state_count"]] = 11
            frame.loc[4, "removed_fraction"] = 3 / 11
        elif fault == "primary_changed":
            frame.loc[0, "tau_A_mc_raw"] += 1e-6
        elif fault == "canonical_changed":
            frame.loc[0, "tau_A_target"] += 0.01
        elif fault == "bad_residual":
            frame.loc[0, "maximum_eigenpair_residual"] = 1e-3
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_l12_evidence(figure_data)


@pytest.mark.parametrize("include_l12", [False, True])
@pytest.mark.integration
def test_render_preserves_layout_and_marks_only_measured_sizes(
    figure_data, tmp_path, monkeypatch, include_l12, renderer
):
    def forbidden(*args, **kwargs):
        pytest.fail("A render must not call an eigensolver")

    for module, name in ((np.linalg, "eigh"), (la, "eigh"), (spla, "eigsh")):
        monkeypatch.setattr(module, name, forbidden)
    preserved = {p.name: p.read_bytes() for p in figure_data.iterdir()}
    output = tmp_path / "figure"
    with plt.rc_context():
        report = renderer.render(figure_data, output, formats=("png",), include_l12=include_l12)
    expected = {8, 12} if include_l12 else {8}
    for panel in ("b", "d"):
        frame = pd.read_csv(output / f"qdm_checkerboard_figure9_panel_{panel}_plot.csv")
        assert set(frame.Lx) == expected
        if include_l12:
            columns = ["value_min", "value_max"] if panel == "b" else ["w_min", "w_max"]
            assert frame.loc[frame.Lx.eq(12), columns].isna().all().all()
    phase = pd.read_csv(output / "qdm_checkerboard_figure9_panel_c_plot.csv")
    assert set(phase.Lx) == {8}
    assert len(phase) == 16
    assert report["canvas_inches"] == [7.05, 5.75]
    assert report["four_by_four_thermal_point_plotted"] is False
    assert report["twelve_by_four_plotted"] == include_l12
    assert report["size_fit_performed"] is False
    assert all(p.read_bytes() == preserved[p.name] for p in figure_data.iterdir())
    if include_l12:
        assert report["primary_values_by_size"]["12"]["matched_beta_stderr"] is None
        assert (output / "figure9_parity_provenance.json").is_file()
    else:
        assert (output / "figure9_8x4_provenance.json").is_file()


@pytest.mark.integration
def test_requested_l12_cannot_silently_fall_back_to_an_eight_only_figure(
    figure_data, tmp_path, renderer
):
    (figure_data / GATES["observables"]).unlink()
    output = tmp_path / "figure"
    with pytest.raises(ValueError, match="--include-l12 requires"):
        renderer.render(figure_data, output, formats=("png",), include_l12=True)
    assert not output.exists()


@pytest.mark.integration
@pytest.mark.parametrize("gate_valid", [False, True])
def test_finalize_assembles_gated_two_size_figure_without_a_solver(
    figure_data, monkeypatch, renderer, gate_valid
):
    from finalize_qdm_parity_figure9 import finalize

    def forbidden(*args, **kwargs):
        pytest.fail("Finalization must use the persisted data")

    for module, name in ((np.linalg, "eigh"), (la, "eigh"), (spla, "eigsh")):
        monkeypatch.setattr(module, name, forbidden)
    if not gate_valid:
        (figure_data / "qdm_parity_character_preflight.json").unlink()
        with pytest.raises(ValueError, match="Parity submission gate is open"):
            finalize(figure_data, use_tex=False, formats=("png",))
        assert not (figure_data / "figures").exists()
        return
    with plt.rc_context():
        report = finalize(figure_data, use_tex=False, formats=("png",))
    gate = json.loads((figure_data / "qdm_parity_submission_gate.json").read_text())
    status = json.loads(
        (figure_data / "qdm_checkerboard_fixed_O1_three_size_status.json").read_text()
    )
    assert gate["closed"] is True
    assert status["nontrivial_sizes"] == [8, 12]
    assert status["fit_status"] == "insufficient_nontrivial_sizes"
    assert report["thermal_sizes"] == [8, 12]
    assert (figure_data / "figures/qdm_checkerboard_figure9_prx.png").is_file()
