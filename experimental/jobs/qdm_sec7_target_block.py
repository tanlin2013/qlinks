#!/usr/bin/env python
"""Converge/classify the Lx=12 checkerboard target-energy block only."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evidence_cache import (
    CacheValidationStatus,
    iter_spectral_checkpoints,
    load_spectral_checkpoint,
)
from qdm_checkerboard_large_strip import folded_spectrum_partial_spectrum
from qdm_resumable_spectrum import (
    folded_problem_description,
    make_resumable_folded_solver,
)
from qdm_sec7_fixed_o1 import (
    REPRESENTATIVE_PHASE,
    atomic_write_csv,
    atomic_write_json,
    build_context,
    compact_type1_orbit,
    compare_subspaces,
    process_memory_gib,
    recover_reference_geometry,
    target_dark_kernel,
)

CONVERGENCE_NAME = "qdm_checkerboard_L12_target_block_convergence.csv"
DARK_NAME = "qdm_checkerboard_joint_dark_kernel.csv"
COMPARE_NAME = "qdm_checkerboard_joint_dark_vs_type1.csv"
COMPACT_NAME = "qdm_checkerboard_compact_dark_manifold.csv"
STATUS_NAME = "qdm_checkerboard_L12_target_block_status.json"
ACCEPTANCE_NAME = "qdm_checkerboard_L12_target_block_acceptance.json"
DEFAULT_BUDGETS = (640, 768)
DEFAULT_TOLERANCES = (1.0e-9, 1.0e-10)
BASELINE_BUDGET = 512


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("at least one budget is required")
    if any(value <= BASELINE_BUDGET for value in values):
        raise ValueError(f"refinement budgets must exceed cached baseline {BASELINE_BUDGET}")
    return values


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("all solver tolerances must be positive")
    return values


def _configure_cache(cache_root: Path) -> None:
    os.environ["QLINKS_EVIDENCE_CACHE_ROOT"] = str(Path(cache_root).resolve(strict=False))
    os.environ["QLINKS_EVIDENCE_CACHE_RESUME"] = "1"
    os.environ["QLINKS_EVIDENCE_CACHE_WRITE"] = "1"
    os.environ["QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE"] = "0"
    os.environ.setdefault("QLINKS_QDM_FOLDED_BACKEND", "primme")
    os.environ.setdefault("QLINKS_QDM_PRIMME_WARM_START_VECTORS", "512")


def _checkpoint_inventory(context, *, cache_root: Path) -> tuple[dict[str, Any], list[Any]]:
    problem = folded_problem_description(
        context.h_sector,
        target_energy=context.tower_energy,
    )
    records: list[dict[str, Any]] = []
    checkpoints: list[Any] = []
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
            sample_vectors=8,
        )
        if checkpoint is None:
            continue
        checkpoints.append(checkpoint)
        records.append(
            {
                "path": str(directory),
                "budget": int(checkpoint.metadata.get("requested_budget", -1)),
                "returned_eigenpairs": int(checkpoint.energies.size),
                "backend": checkpoint.metadata.get("backend"),
                "solver_tolerance": checkpoint.metadata.get("solver_tolerance"),
                "status": checkpoint.status.value,
                "minimum_energy": float(np.min(checkpoint.energies)),
                "maximum_energy": float(np.max(checkpoint.energies)),
                "maximum_residual": float(np.max(checkpoint.residuals, initial=0.0)),
                "producer_run_id": checkpoint.metadata.get("producer_run_id"),
            }
        )
    baseline = [
        checkpoint
        for checkpoint in checkpoints
        if int(checkpoint.metadata.get("requested_budget", -1)) == BASELINE_BUDGET
        and checkpoint.status is not CacheValidationStatus.INCOMPATIBLE
    ]
    if not baseline:
        raise RuntimeError(
            "validated 512-vector Lx=12 PRIMME checkpoint is missing; refusing to start "
            "target-block refinement without the required staged prerequisite"
        )
    return {
        "schema_version": 1,
        "Lx": context.lx,
        "phase": context.phase,
        "sector_dimension": context.sector.sector_dimension,
        "target_energy": context.tower_energy,
        "tower_residual": context.tower_residual,
        "cage_projection_norm": context.cage_projection_norm,
        "cage_QA": context.cage_q["A"],
        "cage_QZ": context.cage_q["Z"],
        "baseline_budget": BASELINE_BUDGET,
        "baseline_validated": True,
        "checkpoints": sorted(records, key=lambda row: (row["budget"], str(row["path"]))),
        "solve_policy": "status is cache-only; refinement is target-block-only PRIMME",
    }, checkpoints


def status(*, cache_root: Path, output_dir: Path) -> dict[str, Any]:
    _configure_cache(cache_root)
    reference = recover_reference_geometry()
    context = build_context(reference=reference, repeats=3, phase=REPRESENTATIVE_PHASE)
    payload, _ = _checkpoint_inventory(context, cache_root=cache_root)
    atomic_write_json(Path(output_dir) / STATUS_NAME, payload)
    return payload


def _stage_rows(
    *,
    context,
    partial,
    budget: int,
    tolerance: float,
    elapsed: float,
    candidate: np.ndarray,
    compact_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    target = target_dark_kernel(
        context,
        partial.energies,
        partial.eigenvectors,
        solver_tolerance=tolerance,
    )
    comparison = compare_subspaces(candidate, target["dark"])
    convergence = {
        "Lx": context.lx,
        "phase": context.phase,
        "requested_subspace_size": int(budget),
        "returned_eigenpairs": int(partial.energies.size),
        "solver_tolerance": float(tolerance),
        "spectrum_method": partial.method,
        "runtime_seconds": float(elapsed),
        "peak_rss_gib": float(partial.peak_rss_gib or process_memory_gib()),
        "partial_min_energy": partial.min_energy,
        "partial_max_energy": partial.max_energy,
        "partial_maximum_residual": partial.maximum_residual,
        "target_block_dimension": target["block_dimension"],
        "target_joint_dark_rank": target["joint_dark_rank"],
        "target_energy_min": target["target_energy_min"],
        "target_energy_max": target["target_energy_max"],
        "target_maximum_residual": target["target_maximum_residual"],
        "target_median_residual": target["target_median_residual"],
        "target_orthogonality_residual": target["target_orthogonality_residual"],
        "cage_target_projector_weight": target["cage_projector_weight"],
        "cage_joint_dark_weight": target["cage_dark_weight"],
        "type1_projected_rank": comparison["type1_projected_rank"],
        "unexplained_joint_dark_norm": comparison["unexplained_joint_dark_norm"],
        "candidate_outside_joint_dark_norm": comparison["candidate_outside_joint_dark_norm"],
        "maximum_principal_angle_rad": comparison["maximum_principal_angle_rad"],
        "extra_dark_rank_beyond_type1": max(
            0,
            int(target["joint_dark_rank"]) - int(comparison["type1_projected_rank"]),
        ),
    }
    dark_rows = [
        {
            "Lx": context.lx,
            "phase": context.phase,
            "requested_subspace_size": int(budget),
            "solver_tolerance": float(tolerance),
            "target_block_dimension": target["block_dimension"],
            "joint_dark_rank": target["joint_dark_rank"],
            "q_all_eigenvalue_index": int(index),
            "q_all_eigenvalue": float(value),
            "is_joint_dark": bool(value <= 1.0e-9 * max(1.0, float(np.max(np.abs(target["q_values"]))))),
        }
        for index, value in enumerate(target["q_values"])
    ]
    compare_row = {
        "Lx": context.lx,
        "phase": context.phase,
        "requested_subspace_size": int(budget),
        "solver_tolerance": float(tolerance),
        **comparison,
        "principal_angles_rad": json.dumps(comparison["principal_angles_rad"]),
        "target_energy_block_dimension": target["block_dimension"],
        "cage_target_projector_weight": target["cage_projector_weight"],
        "cage_joint_dark_weight": target["cage_dark_weight"],
        "classification_scope": "target_energy_block_only",
    }
    tagged_compact = [
        {
            **row,
            "requested_subspace_size": int(budget),
            "solver_tolerance": float(tolerance),
        }
        for row in compact_rows
    ]
    return convergence, dark_rows, compare_row, tagged_compact


def _acceptance(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 2:
        return {"closed": False, "reason": "need at least two refinement stages"}
    ordered = frame.sort_values(["requested_subspace_size", "solver_tolerance"])
    last = ordered.iloc[-2:]
    dimensions = set(last["target_block_dimension"].astype(int))
    dark_ranks = set(last["target_joint_dark_rank"].astype(int))
    extra_ranks = set(last["extra_dark_rank_beyond_type1"].astype(int))
    cage_weights = last["cage_target_projector_weight"].astype(float).to_numpy()
    residuals = last["target_maximum_residual"].astype(float).to_numpy()
    tolerances = last["solver_tolerance"].astype(float).to_numpy()
    residual_ok = bool(
        np.all(residuals <= np.maximum(1.0e-7, 100.0 * tolerances))
    )
    checks = {
        "target_block_dimension_stable": len(dimensions) == 1,
        "joint_dark_rank_stable": len(dark_ranks) == 1,
        "extra_dark_rank_stable": len(extra_ranks) == 1,
        "cage_projector_weight_commensurate": bool(np.all(cage_weights >= 1.0 - 1.0e-7)),
        "target_residuals_commensurate": residual_ok,
    }
    return {
        "closed": all(checks.values()),
        "checks": checks,
        "last_two_budgets": last["requested_subspace_size"].astype(int).tolist(),
        "target_block_dimension": (next(iter(dimensions)) if len(dimensions) == 1 else None),
        "joint_dark_rank": (next(iter(dark_ranks)) if len(dark_ranks) == 1 else None),
        "extra_dark_rank_beyond_type1": (next(iter(extra_ranks)) if len(extra_ranks) == 1 else None),
        "claim_boundary": (
            "P0-A closes only the converged E=12 target-block classification; it does not "
            "establish a thermal-window result."
        ),
    }


def refine(
    *,
    cache_root: Path,
    output_dir: Path,
    budgets: tuple[int, ...],
    tolerances: tuple[float, ...],
) -> pd.DataFrame:
    if len(budgets) != len(tolerances):
        raise ValueError("budgets and tolerances must have equal length")
    if len(budgets) < 2:
        raise ValueError("the target-block audit requires at least two refinement stages")
    _configure_cache(cache_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    reference = recover_reference_geometry()
    context = build_context(reference=reference, repeats=3, phase=REPRESENTATIVE_PHASE)
    inventory, _ = _checkpoint_inventory(context, cache_root=cache_root)
    atomic_write_json(output / STATUS_NAME, inventory)
    candidate, compact_rows = compact_type1_orbit(context)

    original = folded_spectrum_partial_spectrum
    solver = (
        original
        if getattr(original, "__name__", "") == "resumable_folded_spectrum_partial_spectrum"
        else make_resumable_folded_solver(original)
    )

    convergence_rows: list[dict[str, Any]] = []
    dark_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    compact_all: list[dict[str, Any]] = []
    for stage_index, (budget, tolerance) in enumerate(
        zip(budgets, tolerances, strict=True), start=1
    ):
        started = time.perf_counter()
        partial = solver(
            context.h_sector,
            target_energy=context.tower_energy,
            subspace_size=int(budget),
            tolerance=float(tolerance),
            maxiter=None,
            ncv_factor=2.05,
            random_seed=20260831 + int(budget),
        )
        elapsed = time.perf_counter() - started
        convergence, dark, comparison, compact = _stage_rows(
            context=context,
            partial=partial,
            budget=budget,
            tolerance=tolerance,
            elapsed=elapsed,
            candidate=candidate,
            compact_rows=compact_rows,
        )
        convergence["refinement_stage"] = stage_index
        convergence_rows.append(convergence)
        dark_rows.extend(dark)
        compare_rows.append(comparison)
        compact_all.extend(compact)

        convergence_frame = pd.DataFrame(convergence_rows)
        atomic_write_csv(output / CONVERGENCE_NAME, convergence_frame)
        atomic_write_csv(output / DARK_NAME, pd.DataFrame(dark_rows))
        atomic_write_csv(output / COMPARE_NAME, pd.DataFrame(compare_rows))
        atomic_write_csv(output / COMPACT_NAME, pd.DataFrame(compact_all))
        stage_metadata = {
            "schema_version": 1,
            "refinement_stage": stage_index,
            "requested_subspace_size": int(budget),
            "solver_tolerance": float(tolerance),
            "target_block_dimension": int(convergence["target_block_dimension"]),
            "target_joint_dark_rank": int(convergence["target_joint_dark_rank"]),
            "cage_target_projector_weight": float(convergence["cage_target_projector_weight"]),
            "target_maximum_residual": float(convergence["target_maximum_residual"]),
            "runtime_seconds": float(elapsed),
            "peak_rss_gib": float(convergence["peak_rss_gib"]),
            "solve_policy": "target-energy projector refinement only; broad-window coverage ignored",
        }
        atomic_write_json(
            output / f"qdm_checkerboard_L12_target_block_stage_{stage_index:02d}.json",
            stage_metadata,
        )

    frame = pd.DataFrame(convergence_rows)
    atomic_write_json(output / ACCEPTANCE_NAME, _acceptance(frame))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("status", "refine"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument(
        "--tolerances",
        default=",".join(f"{value:.12g}" for value in DEFAULT_TOLERANCES),
    )
    args = parser.parse_args()
    if args.mode == "status":
        print(json.dumps(status(cache_root=args.cache_root, output_dir=args.output_dir), indent=2))
        return
    frame = refine(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        budgets=_parse_int_tuple(args.budgets),
        tolerances=_parse_float_tuple(args.tolerances),
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
