#!/usr/bin/env python
"""Postprocess covered Lx=12 fixed-O(1) checkpoints into Sec. VII observables.

This stage is solver-free. It requires the exact-ED pilot recommendation and at
least two valid cached PRIMME budgets that fully cover the selected fixed total-
energy window. It computes the defining raw A/Z microcanonical traces and raw
stripe covariance, plus basis-independent joint-dark-cleaned companions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la
from evidence_cache import (
    CacheValidationStatus,
    iter_spectral_checkpoints,
    load_spectral_checkpoint,
)
from helpers import projector_deleted_block_covariance, projector_deleted_observable_moments
from qdm_resumable_spectrum import folded_problem_description
from qdm_sec7_fixed_o1 import (
    DARK_TOL,
    ENERGY_BLOCK_TOL,
    REPRESENTATIVE_PHASE,
    atomic_write_csv,
    atomic_write_json,
    build_context,
    orthonormalize,
    recover_reference_geometry,
    stripe_algebra,
)
from qdm_sec7_fixed_o1_l12_spectrum import RECOMMENDATION_NAME, coverage_metrics

from qlinks.caging.analysis.spectral import select_microcanonical_window_by_width

THERMAL_NAME = "qdm_checkerboard_thermal_overlap_fixed_O1.csv"
SYSTEMATICS_NAME = "qdm_checkerboard_window_systematics_fixed_O1.csv"
CONCENTRATION_NAME = "qdm_checkerboard_concentration_fixed_O1.csv"
CONCENTRATION_DETAIL_NAME = "qdm_checkerboard_concentration_L12_raw_clean.csv"
WORST_NAME = "qdm_checkerboard_worst_eigenoperator_fixed_O1.csv"
BLOCK_AUDIT_NAME = "qdm_checkerboard_L12_fixed_O1_joint_dark_block_audit.csv"
ACCEPTANCE_NAME = "qdm_checkerboard_L12_fixed_O1_observables_acceptance.json"
CANONICAL_NAME = "qdm_checkerboard_finite_beta_transfer_target.csv"


def _configure_cache(cache_root: Path) -> None:
    os.environ["QLINKS_EVIDENCE_CACHE_ROOT"] = str(Path(cache_root).resolve(strict=False))
    os.environ["QLINKS_EVIDENCE_CACHE_RESUME"] = "1"
    os.environ["QLINKS_EVIDENCE_CACHE_WRITE"] = "1"
    os.environ["QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE"] = "0"


def _load_recommendation(output_dir: Path) -> tuple[dict[str, Any], float, tuple[float, ...]]:
    path = Path(output_dir) / RECOMMENDATION_NAME
    if not path.is_file():
        raise FileNotFoundError(f"fixed-O(1) pilot recommendation is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "recommended" or payload.get("recommended_half_width") is None:
        raise RuntimeError("fixed-O(1) pilot has not selected a production half-width")
    primary = float(payload["recommended_half_width"])
    controls = []
    for raw in payload.get("neighbor_controls", []):
        value = float(raw)
        if value > 0 and not math.isclose(value, primary, abs_tol=1e-12):
            controls.append(value)
    widths = tuple(sorted({primary, *controls}))
    return payload, primary, widths


def _canonical_target(primme_data_dir: Path) -> dict[str, float]:
    path = Path(primme_data_dir) / CANONICAL_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"persisted Lx=12 canonical target is missing: {path}; refusing to recompute it"
        )
    frame = pd.read_csv(path)
    if "Lx" not in frame.columns:
        raise RuntimeError(f"{path} has no Lx column")
    rows = frame[frame["Lx"].astype(int) == 12].copy()
    if "phase" in rows.columns:
        phase = pd.to_numeric(rows["phase"], errors="coerce")
        selected = rows[np.isclose(phase, REPRESENTATIVE_PHASE, atol=1e-12)]
        if not selected.empty:
            rows = selected
    if rows.empty:
        raise RuntimeError(f"{path} has no Lx=12 canonical row")
    row = rows.iloc[-1]
    required = ("beta_star", "tau_A_target", "tau_Z_target")
    missing = [name for name in required if name not in rows.columns or pd.isna(row[name])]
    if missing:
        raise RuntimeError(f"Lx=12 canonical row is missing columns/values: {missing}")
    return {
        "beta_star": float(row["beta_star"]),
        "beta_stderr": float(row.get("beta_stderr", math.nan)),
        "tau_A_target": float(row["tau_A_target"]),
        "tau_Z_target": float(row["tau_Z_target"]),
        "tau_A_stderr": float(row.get("tau_A_stderr", math.nan)),
        "tau_Z_stderr": float(row.get("tau_Z_stderr", math.nan)),
    }


def _joint_dark_subspace(
    *,
    energies: np.ndarray,
    vectors: np.ndarray,
    indices: np.ndarray,
    q_all,
    energy_tolerance: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected = np.asarray(indices, dtype=int)
    if selected.size == 0:
        raise ValueError("joint-dark inventory requires a nonempty window")
    selected = selected[np.argsort(np.asarray(energies, dtype=float)[selected])]
    groups: list[list[int]] = [[int(selected[0])]]
    spectrum = np.asarray(energies, dtype=float)
    for raw_index in selected[1:]:
        index = int(raw_index)
        if abs(spectrum[index] - spectrum[groups[-1][-1]]) <= float(energy_tolerance):
            groups[-1].append(index)
        else:
            groups.append([index])

    columns: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for block_id, group in enumerate(groups):
        basis = np.asarray(vectors[:, group], dtype=np.complex128)
        compressed = basis.conj().T @ (q_all @ basis)
        compressed = 0.5 * (compressed + compressed.conj().T)
        q_values, rotation = la.eigh(compressed, check_finite=False)
        scale = max(1.0, float(np.max(np.abs(q_values), initial=0.0)))
        keep = np.flatnonzero(q_values <= DARK_TOL * scale)
        if keep.size:
            dark = basis @ rotation[:, keep]
            columns.extend(dark[:, column] for column in range(dark.shape[1]))
        rows.append(
            {
                "energy_block_id": block_id,
                "energy": float(np.mean(spectrum[group])),
                "block_dimension": len(group),
                "joint_dark_rank": int(keep.size),
                "q_all_min_eigenvalue": (
                    float(np.min(q_values)) if q_values.size else math.nan
                ),
                "q_all_max_eigenvalue": (
                    float(np.max(q_values)) if q_values.size else math.nan
                ),
                "energy_block_tolerance": float(energy_tolerance),
            }
        )
    exceptional = (
        orthonormalize(np.column_stack(columns))
        if columns
        else np.zeros((vectors.shape[0], 0), dtype=np.complex128)
    )
    return exceptional, rows


def _worst_coefficients(
    coefficients: np.ndarray,
    *,
    quotient_coefficients: np.ndarray,
    ambient_names: tuple[str, ...],
) -> str:
    ambient = quotient_coefficients @ np.asarray(coefficients)
    return json.dumps(
        {
            ambient_names[index]: [float(complex(value).real), float(complex(value).imag)]
            for index, value in enumerate(ambient)
            if abs(value) > 1.0e-10
        },
        sort_keys=True,
    )


def _load_covered_checkpoints(
    *,
    context,
    cache_root: Path,
    half_width: float,
    residual_tolerance: float,
) -> list[tuple[int, Any, dict[str, Any]]]:
    problem = folded_problem_description(context.h_sector, target_energy=context.tower_energy)
    records: list[tuple[int, Any, dict[str, Any]]] = []
    for directory in iter_spectral_checkpoints(
        namespace="qdm/checkerboard_large_strip",
        problem=problem,
        cache_root=cache_root,
    ):
        checkpoint = load_spectral_checkpoint(
            directory,
            expected_problem=problem,
            hamiltonian=context.h_sector,
            requested_solver_tolerance=None,
            residual_tolerance=float(residual_tolerance),
            sample_vectors=8,
        )
        if checkpoint is None or checkpoint.status is not CacheValidationStatus.VALID_FINAL:
            continue
        tolerance = float(checkpoint.metadata.get("solver_tolerance", 1.0e-8))
        coverage = coverage_metrics(
            checkpoint.energies,
            checkpoint.residuals,
            target_energy=context.tower_energy,
            half_width=half_width,
            solver_tolerance=tolerance,
        )
        if not coverage["window_coverage_complete"]:
            continue
        budget = int(checkpoint.metadata.get("requested_budget", checkpoint.energies.size))
        records.append((budget, checkpoint, coverage))
    records.sort(key=lambda item: item[0])
    return records


def _evaluate_width(
    *,
    context,
    checkpoint,
    budget: int,
    width: float,
    canonical: dict[str, float],
    stripe_ops,
    stripe_meta: dict[str, Any],
    ambient_names: tuple[str, ...],
    quotient_coefficients: np.ndarray,
    residual_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_energies = np.asarray(checkpoint.energies)
    solver_tolerance = float(checkpoint.metadata.get("solver_tolerance", 1.0e-8))
    coverage = coverage_metrics(
        all_energies,
        checkpoint.residuals,
        target_energy=context.tower_energy,
        half_width=width,
        solver_tolerance=solver_tolerance,
    )
    if not coverage["window_coverage_complete"]:
        raise RuntimeError(f"budget {budget} does not fully cover DeltaE={width}")

    window = select_microcanonical_window_by_width(
        all_energies,
        target_energy=context.tower_energy,
        half_width=float(width),
        degeneracy_tolerance=max(ENERGY_BLOCK_TOL, 10.0 * solver_tolerance),
    )
    checkpoint_indices = np.asarray(window.indices, dtype=int)
    energies = np.asarray(all_energies[checkpoint_indices], dtype=float)
    residuals = np.asarray(checkpoint.residuals)[checkpoint_indices]
    window_residual = float(np.max(residuals, initial=0.0))
    if window_residual > float(residual_tolerance):
        raise RuntimeError(
            f"budget {budget} window residual {window_residual:.3e} exceeds "
            f"{residual_tolerance:.3e}"
        )
    vectors = np.asarray(checkpoint.eigenvectors[:, checkpoint_indices], dtype=np.complex128)
    indices = np.arange(energies.size, dtype=int)
    block_tolerance = max(ENERGY_BLOCK_TOL, 20.0 * window_residual, 10.0 * solver_tolerance)
    exceptional, block_rows = _joint_dark_subspace(
        energies=energies,
        vectors=vectors,
        indices=indices,
        q_all=context.q_all,
        energy_tolerance=block_tolerance,
    )
    window_vectors = vectors
    empty_exceptional = np.zeros((context.sector.sector_dimension, 0), dtype=np.complex128)
    witness: dict[str, dict[str, float]] = {}
    for name, operator in context.projected_q.items():
        raw = projector_deleted_observable_moments(
            window_vectors,
            empty_exceptional,
            operator,
            tolerance=1.0e-9,
        )
        clean = projector_deleted_observable_moments(
            window_vectors,
            exceptional,
            operator,
            tolerance=1.0e-9,
        )
        witness[name] = {"raw": float(raw["mean"]), "clean": float(clean["mean"])}

    raw_covariance = projector_deleted_block_covariance(
        energies,
        vectors,
        empty_exceptional,
        stripe_ops,
        indices,
        energy_tolerance=block_tolerance,
        vector_tolerance=1.0e-9,
    )
    clean_covariance = projector_deleted_block_covariance(
        energies,
        vectors,
        exceptional,
        stripe_ops,
        indices,
        energy_tolerance=block_tolerance,
        vector_tolerance=1.0e-9,
    )

    row = {
        "Lx": context.lx,
        "Ly": 4,
        "phase": context.phase,
        "sector_dimension": context.sector.sector_dimension,
        "window_protocol": "fixed_O1_total_energy",
        "window_half_width": float(width),
        "window_energy_density_half_width": float(width) / context.lx,
        "requested_subspace_size": int(budget),
        "returned_eigenpairs": int(np.asarray(checkpoint.energies).size),
        "raw_window_state_count": int(indices.size),
        "joint_dark_removed_rank": int(exceptional.shape[1]),
        "removed_fraction": float(exceptional.shape[1] / max(1, indices.size)),
        "matched_beta": canonical["beta_star"],
        "matched_beta_stderr": canonical["beta_stderr"],
        "tau_A_mc_raw": witness["A"]["raw"],
        "tau_Z_mc_raw": witness["Z"]["raw"],
        "tau_A_mc_clean": witness["A"]["clean"],
        "tau_Z_mc_clean": witness["Z"]["clean"],
        "tau_A_can": canonical["tau_A_target"],
        "tau_Z_can": canonical["tau_Z_target"],
        "tau_A_can_stderr": canonical["tau_A_stderr"],
        "tau_Z_can_stderr": canonical["tau_Z_stderr"],
        "matching_distance_raw": max(
            abs(witness["A"]["raw"] - canonical["tau_A_target"]),
            abs(witness["Z"]["raw"] - canonical["tau_Z_target"]),
        ),
        "matching_distance_clean": max(
            abs(witness["A"]["clean"] - canonical["tau_A_target"]),
            abs(witness["Z"]["clean"] - canonical["tau_Z_target"]),
        ),
        "w_raw": float(raw_covariance["largest_width"]),
        "w_clean": float(clean_covariance["largest_width"]),
        "formal_local_dimension": int(stripe_meta["formal_operator_dimension"]),
        "formal_nonidentity_dimension": int(stripe_meta["ambient_nonidentity_dimension"]),
        "projected_quotient_dimension": int(stripe_meta["projected_operator_dimension"]),
        "projected_quotient_nonidentity_dimension": max(
            0, int(stripe_meta["projected_operator_dimension"]) - 1
        ),
        "energy_block_tolerance": float(block_tolerance),
        "window_maximum_residual": window_residual,
        **coverage,
    }
    tagged_blocks = [
        {
            "Lx": context.lx,
            "phase": context.phase,
            "requested_subspace_size": int(budget),
            "window_half_width": float(width),
            **block,
        }
        for block in block_rows
    ]
    worst = [
        {
            "Lx": context.lx,
            "phase": context.phase,
            "requested_subspace_size": int(budget),
            "window_half_width": float(width),
            "background": "raw",
            "width": float(raw_covariance["largest_width"]),
            "coefficients": _worst_coefficients(
                raw_covariance["worst_coefficients"],
                quotient_coefficients=quotient_coefficients,
                ambient_names=ambient_names,
            ),
        },
        {
            "Lx": context.lx,
            "phase": context.phase,
            "requested_subspace_size": int(budget),
            "window_half_width": float(width),
            "background": "clean",
            "width": float(clean_covariance["largest_width"]),
            "coefficients": _worst_coefficients(
                clean_covariance["worst_coefficients"],
                quotient_coefficients=quotient_coefficients,
                ambient_names=ambient_names,
            ),
        },
    ]
    return row, tagged_blocks, worst


def _acceptance(primary: pd.DataFrame, *, tolerance: float) -> dict[str, Any]:
    ordered = primary.sort_values("requested_subspace_size")
    last = ordered.tail(2)
    if len(last) == 2:
        a = last.iloc[0]
        b = last.iloc[1]
        observable_change = max(
            abs(float(a["tau_A_mc_raw"]) - float(b["tau_A_mc_raw"])),
            abs(float(a["tau_Z_mc_raw"]) - float(b["tau_Z_mc_raw"])),
            abs(float(a["matching_distance_raw"]) - float(b["matching_distance_raw"])),
            abs(float(a["w_raw"]) - float(b["w_raw"])),
        )
        counts_stable = int(a["raw_window_state_count"]) == int(b["raw_window_state_count"])
    else:
        observable_change = math.inf
        counts_stable = False
    checks = {
        "at_least_two_covered_budgets": len(ordered) >= 2,
        "last_two_window_counts_stable": counts_stable,
        "last_two_raw_observables_stable": observable_change <= float(tolerance),
    }
    return {
        "schema_version": 1,
        "closed": all(checks.values()),
        "checks": checks,
        "last_two_budgets": last["requested_subspace_size"].astype(int).tolist(),
        "last_two_maximum_raw_change": float(observable_change),
        "observable_budget_tolerance": float(tolerance),
        "claim_boundary": (
            "This closes the Lx=12 fixed-window numerical row and budget convergence only; "
            "thermodynamic extrapolation remains a separate task."
        ),
    }


def run(
    *,
    primme_data_dir: Path,
    output_dir: Path,
    cache_root: Path,
    residual_tolerance: float,
    observable_budget_tolerance: float,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, primary_width, widths = _load_recommendation(output)
    canonical = _canonical_target(primme_data_dir)
    _configure_cache(cache_root)

    reference = recover_reference_geometry()
    context = build_context(reference=reference, repeats=3, phase=REPRESENTATIVE_PHASE)
    covered = _load_covered_checkpoints(
        context=context,
        cache_root=cache_root,
        half_width=primary_width,
        residual_tolerance=residual_tolerance,
    )
    if len(covered) < 2:
        raise RuntimeError(
            "need at least two valid cached PRIMME budgets covering the primary fixed window; "
            "run --stage fixed-O1-L12-spectrum first"
        )

    stripe_ops, _, stripe_meta, ambient_names, quotient_coefficients = stripe_algebra(
        context,
        z_placement=reference.z_placement,
    )
    systematics_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []

    for budget, checkpoint, _primary_coverage in covered:
        for width in widths:
            solver_tolerance = float(checkpoint.metadata.get("solver_tolerance", 1.0e-8))
            coverage = coverage_metrics(
                checkpoint.energies,
                checkpoint.residuals,
                target_energy=context.tower_energy,
                half_width=width,
                solver_tolerance=solver_tolerance,
            )
            if not coverage["window_coverage_complete"]:
                continue
            row, blocks, worst = _evaluate_width(
                context=context,
                checkpoint=checkpoint,
                budget=budget,
                width=width,
                canonical=canonical,
                stripe_ops=stripe_ops,
                stripe_meta=stripe_meta,
                ambient_names=ambient_names,
                quotient_coefficients=quotient_coefficients,
                residual_tolerance=residual_tolerance,
            )
            row["is_primary_window"] = bool(math.isclose(width, primary_width, abs_tol=1e-12))
            systematics_rows.append(row)
            block_rows.extend(blocks)
            worst_rows.extend(worst)
            if row["is_primary_window"]:
                primary_rows.append(row)

            atomic_write_csv(output / SYSTEMATICS_NAME, pd.DataFrame(systematics_rows))
            atomic_write_csv(output / BLOCK_AUDIT_NAME, pd.DataFrame(block_rows))
            atomic_write_csv(output / WORST_NAME, pd.DataFrame(worst_rows))

    primary = (
        pd.DataFrame(primary_rows)
        .sort_values("requested_subspace_size")
        .reset_index(drop=True)
    )
    if len(primary) < 2:
        raise RuntimeError("postprocessing did not produce two primary-window budget rows")
    atomic_write_csv(output / THERMAL_NAME, primary)

    concentration_columns = [
        "Lx",
        "Ly",
        "phase",
        "window_protocol",
        "window_half_width",
        "requested_subspace_size",
        "raw_window_state_count",
        "joint_dark_removed_rank",
        "removed_fraction",
        "formal_local_dimension",
        "formal_nonidentity_dimension",
        "projected_quotient_dimension",
        "projected_quotient_nonidentity_dimension",
        "w_raw",
        "w_clean",
        "energy_block_tolerance",
        "window_maximum_residual",
    ]
    concentration = primary[concentration_columns].copy()
    atomic_write_csv(output / CONCENTRATION_NAME, concentration)
    atomic_write_csv(output / CONCENTRATION_DETAIL_NAME, concentration)

    acceptance = _acceptance(primary, tolerance=observable_budget_tolerance)
    acceptance.update(
        {
            "primary_window_half_width": primary_width,
            "canonical_source": str(Path(primme_data_dir) / CANONICAL_NAME),
            "covered_budgets_postprocessed": (
                primary["requested_subspace_size"].astype(int).tolist()
            ),
        }
    )
    atomic_write_json(output / ACCEPTANCE_NAME, acceptance)
    if not acceptance["closed"]:
        raise RuntimeError(
            "Lx=12 fixed-O(1) observables are not budget-stable; "
            f"see {output / ACCEPTANCE_NAME}"
        )
    return primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primme-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--observable-budget-tolerance", type=float, default=5.0e-5)
    args = parser.parse_args()
    frame = run(
        primme_data_dir=args.primme_data_dir,
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        residual_tolerance=args.residual_tolerance,
        observable_budget_tolerance=args.observable_budget_tolerance,
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
