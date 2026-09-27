#!/usr/bin/env python
"""Blind coupling-chart obstruction hierarchy for the spin-1 tower.

This job deliberately does not encode the staggered tower phase rule in the
input chart.  It computes the state-readjusting first-order cage tangent and
the exact fixed-state affine subspace from the generic stability API, and only
then compares them with the analytic tower-compatible exchange subspace.

The bounded-local operator/state co-continuation layer requested by the PRX P1
handoff is not yet implemented in the public stability API.  The output records
that gap explicitly as T_joint=not_implemented rather than identifying it with
T_fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from spin1_exchange_convention import (
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
)

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.caging.stability import (
    cage_compatibility_hierarchy_from_hamiltonians,
    combine_perturbations_from_coefficients,
    scan_support_eigenstate_branch,
    subspace_principal_overlaps,
)
from qlinks.lattice import BoundaryCondition
from qlinks.models import (
    SpinOneXYChainModel,
    spin_one_xy_hxy_h3_imaginary_j2_model,
    spin_one_xy_periodic_range_couplings,
    spin_one_xy_scar_tower_states,
)

COORDINATES = ("Re_t1", "Im_t1", "Re_t2", "Im_t2", "Re_t3", "Im_t3")
TOLERANCE = 1.0e-10
BASE_J3_OVER_J = 0.10
BASE_KAPPA_OVER_J = 0.10


def coordinate_model(length: int, total_sz: int, coordinate: str) -> SpinOneXYChainModel:
    """Return one unit coordinate of the blind translation-invariant chart."""
    if coordinate not in COORDINATES:
        raise ValueError(f"unknown coordinate {coordinate!r}")
    distance = int(coordinate[-1])
    coefficient = 1.0j if coordinate.startswith("Im") else 1.0
    extra = spin_one_xy_periodic_range_couplings(
        length=length,
        distance=distance,
        coefficient=coefficient,
    )
    return SpinOneXYChainModel(
        length=length,
        boundary_condition=BoundaryCondition.PERIODIC,
        j_xy=0.0,
        total_sz=total_sz,
        extra_xy_couplings=extra,
    )


def orthonormal_columns(array: np.ndarray, *, tolerance: float = TOLERANCE) -> np.ndarray:
    """Return an orthonormal real basis for a column span."""
    matrix = np.asarray(array, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("basis must be a matrix")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    u, singular_values, _vh = np.linalg.svd(matrix, full_matrices=False)
    keep = singular_values > tolerance
    return np.asarray(u[:, keep], dtype=float)


def tower_basis() -> np.ndarray:
    """Analytic odd-real/even-imaginary exchange subspace, post hoc only."""
    basis = np.zeros((len(COORDINATES), 3), dtype=float)
    basis[COORDINATES.index("Re_t1"), 0] = 1.0
    basis[COORDINATES.index("Im_t2"), 1] = 1.0
    basis[COORDINATES.index("Re_t3"), 2] = 1.0
    return basis


def _smallest_positive(values: np.ndarray, tolerance: float) -> float | None:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > tolerance]
    return None if positive.size == 0 else float(np.min(positive))


def _subspace_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = orthonormal_columns(candidate)
    reference = orthonormal_columns(reference)
    overlaps = subspace_principal_overlaps(candidate, reference)
    angles = np.degrees(np.arccos(np.clip(overlaps, 0.0, 1.0))) if overlaps.size else np.array([])
    candidate_projector = candidate @ candidate.T
    reference_projector = reference @ reference.T
    ref_in_candidate = (
        float(np.linalg.norm((np.eye(reference.shape[0]) - candidate_projector) @ reference))
        if reference.shape[1]
        else 0.0
    )
    candidate_in_ref = (
        float(np.linalg.norm((np.eye(candidate.shape[0]) - reference_projector) @ candidate))
        if candidate.shape[1]
        else 0.0
    )
    projection_overlap = float(np.linalg.norm(candidate.T @ reference, ord="fro") ** 2)
    return {
        "principal_overlaps": [float(value) for value in overlaps],
        "principal_angles_deg": [float(value) for value in angles],
        "tower_in_layer_residual": ref_in_candidate,
        "layer_in_tower_residual": candidate_in_ref,
        "projection_overlap_frobenius_sq": projection_overlap,
    }


def _spot_check(
    base_hamiltonian: object,
    perturbations: list[object],
    support: np.ndarray,
    state: np.ndarray,
    coefficients: np.ndarray,
    *,
    label: str,
    classification: str,
) -> list[dict[str, Any]]:
    perturbation = combine_perturbations_from_coefficients(
        perturbations,
        np.asarray(coefficients, dtype=float)[:, None],
    )[0]
    parameters = (0.0, 5.0e-5, 1.0e-4, 2.0e-4)
    branch = scan_support_eigenstate_branch(
        base_hamiltonian,
        perturbation,
        support,
        parameters,
        reference_state=state,
        tolerance=TOLERANCE,
    )
    rows: list[dict[str, Any]] = []
    for point in branch.points:
        rows.append(
            {
                "check": label,
                "classification": classification,
                "lambda": float(point.parameter),
                "boundary_residual": float(point.boundary_residual),
                "internal_eigen_residual": float(point.internal_eigen_residual),
                "full_residual": float(point.full_residual),
                **{
                    coordinate: float(coefficients[index])
                    for index, coordinate in enumerate(COORDINATES)
                },
            }
        )
    return rows


def analyze(length: int, total_sz: int) -> dict[str, Any]:
    """Analyze one even size in the blind six-coordinate chart."""
    base_model = spin_one_xy_hxy_h3_imaginary_j2_model(
        length=length,
        j=1.0,
        j3=BASE_J3_OVER_J,
        kappa=BASE_KAPPA_OVER_J,
        total_sz=total_sz,
    )
    base = base_model.build(builder="optimized", basis_solver="dfs", sort_basis=True)
    configs = basis_configs_from_build_result(base)
    towers, labels = spin_one_xy_scar_tower_states(
        basis_configs=configs,
        length=length,
        normalize=True,
    )
    if towers.shape[1] != 1:
        raise RuntimeError(f"expected one fixed-M tower state, found {labels}")
    state = np.asarray(towers[:, 0], dtype=np.complex128)
    support = np.flatnonzero(np.abs(state) > TOLERANCE)

    perturbations: list[object] = []
    for coordinate in COORDINATES:
        built = coordinate_model(length, total_sz, coordinate).build(
            builder="optimized",
            basis_solver="dfs",
            sort_basis=True,
        )
        perturbations.append(built.hamiltonian)

    hierarchy = cage_compatibility_hierarchy_from_hamiltonians(
        base.hamiltonian,
        perturbations,
        support,
        state,
        coefficient_field="real",
        tolerance=TOLERANCE,
    )
    spaces = {
        "T_cage": orthonormal_columns(hierarchy.first_order.compatible_coefficient_basis),
        "T_fixed": orthonormal_columns(hierarchy.fixed_state.compatible_coefficient_basis),
        "T_tower": tower_basis(),
    }

    layer_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    reference = spaces["T_tower"]

    layer_specs = (
        (
            "T_cage",
            hierarchy.first_order.rank,
            hierarchy.first_order.compatible_dimension,
            hierarchy.first_order.constraint_matrix.shape[0],
            hierarchy.first_order.singular_values,
            "computed",
        ),
        (
            "T_fixed",
            hierarchy.fixed_state.rank,
            hierarchy.fixed_state.compatible_dimension,
            hierarchy.fixed_state.constraint_matrix.shape[0],
            hierarchy.fixed_state.singular_values,
            "computed",
        ),
        (
            "T_tower",
            len(COORDINATES) - reference.shape[1],
            reference.shape[1],
            len(COORDINATES) - reference.shape[1],
            np.array([], dtype=float),
            "analytic_post_hoc_reference",
        ),
    )
    for layer, rank, dimension, obstruction_dimension, singular_values, status in layer_specs:
        comparison = _subspace_comparison(spaces[layer], reference)
        layer_rows.append(
            {
                "L": length,
                "M": total_sz,
                EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
                "layer": layer,
                "status": status,
                "n_parameters": len(COORDINATES),
                "rank": int(rank),
                "nullity": int(dimension),
                "obstruction_dimension": int(obstruction_dimension),
                "rank_tolerance": TOLERANCE,
                "smallest_nonzero_transverse_obstruction": _smallest_positive(
                    singular_values, TOLERANCE
                ),
                "principal_overlaps_with_T_tower": json.dumps(comparison["principal_overlaps"]),
                "principal_angles_deg_with_T_tower": json.dumps(comparison["principal_angles_deg"]),
                "tower_in_layer_residual": comparison["tower_in_layer_residual"],
                "layer_in_tower_residual": comparison["layer_in_tower_residual"],
                "projection_overlap_frobenius_sq": comparison["projection_overlap_frobenius_sq"],
                "bounded_local_support": (
                    "translation-invariant two-site exchange perturbations at separations 1,2,3"
                ),
                "implementation_gap": "",
            }
        )
        for basis_index in range(spaces[layer].shape[1]):
            row = {
                "L": length,
                "M": total_sz,
                EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
                "layer": layer,
                "basis_vector": basis_index,
            }
            row.update(
                {
                    coordinate: float(spaces[layer][coordinate_index, basis_index])
                    for coordinate_index, coordinate in enumerate(COORDINATES)
                }
            )
            basis_rows.append(row)
        for singular_index, singular_value in enumerate(np.asarray(singular_values, dtype=float)):
            singular_rows.append(
                {
                    "L": length,
                    "M": total_sz,
                    EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
                    "layer": layer,
                    "singular_index": singular_index,
                    "singular_value": float(singular_value),
                    "rank_tolerance": TOLERANCE,
                    "is_nonzero": bool(singular_value > TOLERANCE),
                }
            )

    layer_rows.append(
        {
            "L": length,
            "M": total_sz,
            EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
            "layer": "T_joint",
            "status": "not_implemented",
            "n_parameters": len(COORDINATES),
            "rank": pd.NA,
            "nullity": pd.NA,
            "obstruction_dimension": pd.NA,
            "rank_tolerance": TOLERANCE,
            "smallest_nonzero_transverse_obstruction": np.nan,
            "principal_overlaps_with_T_tower": "[]",
            "principal_angles_deg_with_T_tower": "[]",
            "tower_in_layer_residual": np.nan,
            "layer_in_tower_residual": np.nan,
            "projection_overlap_frobenius_sq": np.nan,
            "bounded_local_support": (
                "not declared because the joint-local continuation map is absent"
            ),
            "implementation_gap": (
                "The public stability API does not yet expose simultaneous continuation "
                "of the cage state and a bounded local caging operator with a finite "
                "local coefficient basis. T_joint is therefore not identified with T_fixed."
            ),
        }
    )

    spot_rows: list[dict[str, Any]] = []
    if spaces["T_fixed"].shape[1]:
        spot_rows.extend(
            _spot_check(
                base.hamiltonian,
                perturbations,
                support,
                state,
                spaces["T_fixed"][:, 0],
                label="fixed_preserving",
                classification="T_fixed",
            )
        )
    tangent_only = orthonormal_columns(hierarchy.tangent_only_coefficient_basis)
    if tangent_only.shape[1]:
        spot_rows.extend(
            _spot_check(
                base.hamiltonian,
                perturbations,
                support,
                state,
                tangent_only[:, 0],
                label="tangent_only",
                classification="T_cage_minus_T_fixed",
            )
        )
    elif spaces["T_cage"].shape[1]:
        spot_rows.extend(
            _spot_check(
                base.hamiltonian,
                perturbations,
                support,
                state,
                spaces["T_cage"][:, 0],
                label="first_order_preserving",
                classification="T_cage",
            )
        )
    incompatible = np.zeros(len(COORDINATES), dtype=float)
    incompatible[COORDINATES.index("Im_t1")] = 1.0
    spot_rows.extend(
        _spot_check(
            base.hamiltonian,
            perturbations,
            support,
            state,
            incompatible,
            label="incompatible_Im_t1",
            classification="deliberate_control",
        )
    )

    summary = {
        "L": length,
        "M": total_sz,
        EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION,
        "base_point": {
            "J_over_J": 1.0,
            "J3_over_J": BASE_J3_OVER_J,
            "kappa_over_J": BASE_KAPPA_OVER_J,
        },
        "coordinates": list(COORDINATES),
        "rank_tolerance": TOLERANCE,
        "perturbation_support": (
            "translation-invariant two-site Hermitian exchanges at distances 1,2,3"
        ),
        "fixed_subspace_inclusion_residual": float(hierarchy.fixed_subspace_inclusion_residual),
        "tangent_only_dimension": int(hierarchy.tangent_only_dimension),
        "T_joint_status": "not_implemented",
        "T_joint_gap": layer_rows[-1]["implementation_gap"],
    }
    return {
        "layers": layer_rows,
        "basis": basis_rows,
        "singular_values": singular_rows,
        "spot_checks": spot_rows,
        "summary": summary,
    }


def run(
    output_dir: Path,
    *,
    lengths: tuple[int, ...] = (8, 10),
    magnetization: int = -2,
) -> dict[str, Any]:
    """Run the blind hierarchy and write tidy machine-readable evidence tables."""
    output = output_dir.resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    if not lengths or any(length < 4 or length % 2 for length in lengths):
        raise ValueError("lengths must contain even integers >=4")

    all_layers: list[dict[str, Any]] = []
    all_basis: list[dict[str, Any]] = []
    all_singular: list[dict[str, Any]] = []
    all_spot: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for length in lengths:
        result = analyze(length, magnetization)
        all_layers.extend(result["layers"])
        all_basis.extend(result["basis"])
        all_singular.extend(result["singular_values"])
        all_spot.extend(result["spot_checks"])
        summaries.append(result["summary"])

    pd.DataFrame(all_layers).to_csv(output / "spin1_obstruction_hierarchy.csv", index=False)
    pd.DataFrame(all_basis).to_csv(output / "spin1_obstruction_basis.csv", index=False)
    pd.DataFrame(all_singular).to_csv(output / "spin1_obstruction_singular_values.csv", index=False)
    pd.DataFrame(all_spot).to_csv(output / "spin1_obstruction_spot_checks.csv", index=False)
    payload = {
        "schema_version": 1,
        "sizes": summaries,
        "joint_local_layer": {
            "status": "not_implemented",
            "scientific_interpretation": (
                "Report T_cage and T_fixed without forcing equality to the "
                "analytic tower-compatible subspace; the missing T_joint layer "
                "remains an implementation gap."
            ),
        },
    }
    (output / "spin1_obstruction_hierarchy.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        default="8,10",
        help="Comma-separated even sizes; 8,10 is the default size-compatibility check.",
    )
    parser.add_argument("--magnetization", type=int, default=-2)
    args = parser.parse_args()
    lengths = tuple(int(value) for value in args.lengths.split(",") if value.strip())
    run(args.output_dir, lengths=lengths, magnetization=args.magnetization)


if __name__ == "__main__":
    main()
