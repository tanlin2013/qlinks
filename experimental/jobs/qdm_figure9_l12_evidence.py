"""Validate accepted 12x4 figure inputs without loading large eigenvector caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qdm_parity_protocol import PROTOCOL, require_parity_protocol, window_controls_stable

WIDTHS = (0.10, 0.20, 0.25, 0.50)
PRIMARY_NAME = "qdm_checkerboard_thermal_overlap_fixed_O1.csv"
SYSTEMATICS_NAME = "qdm_checkerboard_window_systematics_fixed_O1.csv"
CANONICAL_NAME = "qdm_checkerboard_finite_beta_transfer_target.csv"
GATES = {
    "canonical": "qdm_parity_canonical_acceptance.json",
    "spectral": "qdm_checkerboard_L12_fixed_O1_spectral_acceptance.json",
    "observables": "qdm_checkerboard_L12_fixed_O1_observables_acceptance.json",
}


def _require_identity(frame: pd.DataFrame, description: str) -> None:
    expected = {
        "Lx": 12,
        "Ly": 4,
        "Sy": -1,
        "sector_dimension": 39667,
        "symmetry_protocol": PROTOCOL,
        "parity_policy": "measured_target",
        "fully_symmetry_resolved": True,
    }
    if frame.empty or any(
        key not in frame or not frame[key].eq(value).all() for key, value in expected.items()
    ):
        raise ValueError(f"{description} does not identify the 12x4 measured-target irrep")
    if "phase" not in frame or not np.allclose(frame.phase, 0.05, atol=1e-12, rtol=0):
        raise ValueError(f"{description} must use the representative phase 0.05")


def load_l12_evidence(data: Path):
    """Reconcile numerical gates, two-budget controls and the canonical source.

    The numerical jobs own Hamiltonian actions and full-Gram validation. This
    render-only check validates their persisted certificates and internal
    agreement; it does not repeat a 39667-dimensional calculation.
    """
    data = Path(data)
    paths = [data / name for name in (PRIMARY_NAME, SYSTEMATICS_NAME, CANONICAL_NAME)]
    paths.extend(data / name for name in GATES.values())
    for path in paths:
        if not path.is_file():
            raise ValueError(f"--include-l12 requires the completed numerical source: {path}")
    gates = {}
    for name, filename in GATES.items():
        payload = json.loads((data / filename).read_text())
        require_parity_protocol(payload, description=f"12x4 {name} gate")
        checks = payload.get("checks", {})
        if (
            payload.get("closed") is not True
            or payload.get("Sy") != -1
            or not checks
            or any(value is not True for value in checks.values())
        ):
            raise ValueError(f"12x4 {name} acceptance is not closed")
        gates[name] = payload

    primary, systematics, canonical = (pd.read_csv(path) for path in paths[:3])
    for frame, description in ((primary, "primary rows"), (systematics, "window controls")):
        _require_identity(frame, f"12x4 {description}")
        required = {
            "requested_subspace_size",
            "returned_eigenpairs",
            "window_half_width",
            "window_energy_density_half_width",
            "maximum_eigenpair_residual",
            "gram_frobenius_residual",
            "cage_window_projector_weight",
            "window_coverage_complete",
            "raw_window_state_count",
            "joint_dark_removed_rank",
            "removed_fraction",
            "tau_A_mc_raw",
            "tau_Z_mc_raw",
            "tau_A_can",
            "tau_Z_can",
            "tau_A_can_stderr",
            "tau_Z_can_stderr",
            "matching_distance_raw",
            "matched_beta",
            "w_raw",
            "window_lower_energy",
            "window_upper_energy",
            "window_state_count",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"12x4 {description} is missing required columns")
        numeric = required - {"window_coverage_complete"}
        if not np.isfinite(frame[sorted(numeric)]).all().all():
            raise ValueError(f"12x4 {description} contains nonfinite figure inputs")
        for key in ("maximum_eigenpair_residual", "gram_frobenius_residual"):
            if not frame[key].between(0, 1e-6).all():
                raise ValueError(f"12x4 {description} fails its {key} certificate")
        if (
            "window_protocol" not in frame
            or not frame.window_protocol.eq("fixed_O1_total_energy").all()
            or not frame.returned_eigenpairs.eq(frame.requested_subspace_size).all()
            or not frame.window_coverage_complete.eq(True).all()  # noqa: E712
            or not np.allclose(frame.cage_window_projector_weight, 1, atol=1e-6, rtol=0)
            or not frame.raw_window_state_count.between(2, frame.returned_eigenpairs).all()
            or not frame.joint_dark_removed_rank.between(0, frame.raw_window_state_count).all()
            or not frame[["tau_A_can_stderr", "tau_Z_can_stderr", "w_raw"]].ge(0).all().all()
            or frame.duplicated(["requested_subspace_size", "window_half_width"]).any()
        ):
            raise ValueError(
                f"12x4 {description} has invalid population or uncertainty certificates"
            )
        counts = frame[
            [
                "requested_subspace_size",
                "returned_eigenpairs",
                "raw_window_state_count",
                "joint_dark_removed_rank",
            ]
        ]
        if (
            not np.equal(counts, np.round(counts)).all().all()
            or not frame.window_state_count.eq(frame.raw_window_state_count).all()
        ):
            raise ValueError(f"12x4 {description} has invalid integer state counts")
        if not np.allclose(
            frame.window_lower_energy, 12 - frame.window_half_width, atol=1e-12, rtol=0
        ) or not np.allclose(
            frame.window_upper_energy, 12 + frame.window_half_width, atol=1e-12, rtol=0
        ):
            raise ValueError(f"12x4 {description} has a different target-energy window")
        if not np.allclose(
            frame.window_energy_density_half_width, frame.window_half_width / 48, atol=1e-12, rtol=0
        ) or not np.allclose(
            frame.removed_fraction,
            frame.joint_dark_removed_rank / frame.raw_window_state_count,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"12x4 {description} has inconsistent window or deletion conventions")
        mismatch = np.maximum(
            np.abs(frame.tau_A_mc_raw - frame.tau_A_can),
            np.abs(frame.tau_Z_mc_raw - frame.tau_Z_can),
        )
        if not np.allclose(mismatch, frame.matching_distance_raw, atol=1e-10, rtol=0):
            raise ValueError(f"12x4 {description} has inconsistent witness mismatches")

    if not np.allclose(primary.window_half_width, 0.25, atol=1e-12, rtol=0):
        raise ValueError("12x4 primary half-width must remain 0.25")
    for name in ("spectral", "observables"):
        if not np.isclose(gates[name].get("primary_window_half_width", np.nan), 0.25):
            raise ValueError(f"12x4 {name} gate has a different primary window")
    if not window_controls_stable(systematics, widths=WIDTHS, tolerance=5e-5):
        raise ValueError("12x4 prescribed window controls are not two-budget stable")
    budgets = sorted(primary.requested_subspace_size.unique())[-2:]
    if (
        len(budgets) != 2
        or sorted(systematics.requested_subspace_size.unique())[-2:] != budgets
        or list(gates["spectral"].get("last_two_covered_budgets", [])) != budgets
        or list(gates["observables"].get("last_two_budgets", [])) != budgets
    ):
        raise ValueError("12x4 figure rows disagree with the accepted budgets")
    latest = primary[primary.requested_subspace_size.isin(budgets)].sort_values(
        "requested_subspace_size"
    )
    if len(latest) != 2:
        raise ValueError("12x4 figure requires two distinct accepted budget rows")
    compare = ["tau_A_mc_raw", "tau_Z_mc_raw", "matching_distance_raw", "w_raw"]
    if float(np.max(np.abs(latest.iloc[1][compare] - latest.iloc[0][compare]))) > 5e-5:
        raise ValueError("12x4 primary observables are not budget-stable")
    selected_controls = systematics[
        systematics.requested_subspace_size.isin(budgets)
        & np.isclose(systematics.window_half_width, 0.25)
    ].sort_values("requested_subspace_size")
    reconcile = sorted(numeric)
    if not np.allclose(latest[reconcile], selected_controls[reconcile], atol=1e-12, rtol=0):
        raise ValueError("12x4 primary table disagrees with its window-control source")

    _require_identity(canonical, "12x4 canonical source")
    if len(canonical) != 1:
        raise ValueError("12x4 figure requires one accepted canonical source row")
    source = canonical.iloc[0]
    for name in ("A", "Z"):
        for column, canonical_column in (
            (f"tau_{name}_can", f"tau_{name}_target"),
            (f"tau_{name}_can_stderr", f"tau_{name}_stderr"),
            ("matched_beta", "beta_star"),
        ):
            if canonical_column not in canonical or not np.allclose(
                latest[column], source[canonical_column], atol=1e-12, rtol=0
            ):
                raise ValueError("12x4 plotted means/errors disagree with the canonical source")
    selected = latest.iloc[-1]
    report = {
        "budgets": list(map(int, budgets)),
        "selected_budget": int(selected.requested_subspace_size),
        "primary_half_width": 0.25,
        "phase": 0.05,
        "phase_scan_available": False,
        "phase_range_drawn": False,
        "raw_window_state_count": int(selected.raw_window_state_count),
        "canonical_uncertainty": "conditional ratio jackknife; separate grid/sample acceptance",
        "eigenvector_validation_repeated_by_renderer": False,
    }
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return selected, systematics, hashes, report
