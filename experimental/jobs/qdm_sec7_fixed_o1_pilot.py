#!/usr/bin/env python
"""Validate candidate fixed-O(1) microcanonical windows at exact-ED sizes Lx=4,8."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la
from helpers import (
    projector_deleted_block_covariance,
    projector_resolved_energy_basis,
)
from qdm_sec7_fixed_o1 import (
    ENERGY_BLOCK_TOL,
    PILOT_HALF_WIDTHS,
    REPRESENTATIVE_PHASE,
    atomic_write_csv,
    atomic_write_json,
    build_context,
    canonical_weights,
    estimate_l12_window_budget,
    process_memory_gib,
    recommend_fixed_width,
    recover_reference_geometry,
    stripe_algebra,
    target_dark_kernel,
)

from qlinks.caging.analysis.spectral import select_microcanonical_window_by_width

SYSTEMATICS_NAME = "qdm_checkerboard_fixed_O1_window_systematics.csv"
RECOMMENDATION_NAME = "qdm_checkerboard_fixed_O1_window_recommendation.json"
AUDIT_NAME = "qdm_checkerboard_fixed_O1_exact_energy_block_audit.csv"


def _parse_widths(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("all fixed-window half-widths must be positive")
    return tuple(sorted(set(values)))


def _orthonormalize(columns: np.ndarray, *, tolerance: float = 1.0e-9) -> np.ndarray:
    matrix = np.asarray(columns, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("columns must be a matrix")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > tolerance * max(1.0, float(singular[0]))
    return np.asarray(u[:, keep], dtype=np.complex128)


def _joint_dark_all(
    energies: np.ndarray,
    vectors: np.ndarray,
    q_all,
    tower: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    order = np.argsort(np.asarray(energies, dtype=float))
    values = np.asarray(energies, dtype=float)[order]
    basis_all = np.asarray(vectors, dtype=np.complex128)[:, order]
    groups: list[list[int]] = []
    for index, energy in enumerate(values):
        if not groups or abs(energy - values[groups[-1][-1]]) > ENERGY_BLOCK_TOL:
            groups.append([index])
        else:
            groups[-1].append(index)
    columns: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for block_id, group in enumerate(groups):
        basis = basis_all[:, group]
        compressed = basis.conj().T @ (q_all @ basis)
        compressed = 0.5 * (compressed + compressed.conj().T)
        q_values, rotation = la.eigh(compressed, check_finite=False)
        scale = max(1.0, float(np.max(np.abs(q_values), initial=0.0)))
        keep = np.flatnonzero(q_values <= 1.0e-9 * scale)
        dark = (
            basis @ rotation[:, keep]
            if keep.size
            else np.zeros((basis.shape[0], 0), dtype=np.complex128)
        )
        target_weight = float(np.linalg.norm(dark.conj().T @ tower) ** 2) if keep.size else 0.0
        columns.extend(dark[:, column] for column in range(dark.shape[1]))
        rows.append(
            {
                "energy_block_id": block_id,
                "energy": float(np.mean(values[group])),
                "block_dimension": len(group),
                "joint_dark_rank": int(keep.size),
                "target_weight": target_weight,
            }
        )
    exceptional = (
        _orthonormalize(np.column_stack(columns))
        if columns
        else np.zeros((vectors.shape[0], 0), dtype=np.complex128)
    )
    return exceptional, rows


def _validate_authoritative_base(base_data_dir: Path) -> dict[str, Any]:
    base = Path(base_data_dir)
    required = {
        "common_sector": base / "qdm_checkerboard_common_symmetry_sector.csv",
        "thermal": base / "qdm_checkerboard_thermal_overlap.csv",
        "concentration": base / "qdm_checkerboard_concentration_grid.csv",
        "canonical": base / "qdm_checkerboard_finite_beta_transfer_target.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing authoritative Lx=4,8 prerequisite files: " + ", ".join(missing))
    common = pd.read_csv(required["common_sector"])
    result: dict[str, Any] = {"files": {name: str(path) for name, path in required.items()}}
    for lx, expected in ((4, 15), (8, 1125)):
        rows = common[
            (common["Lx"].astype(int) == lx)
            & np.isclose(common["phase"].astype(float), REPRESENTATIVE_PHASE)
        ]
        if rows.empty:
            raise RuntimeError(f"authoritative common-sector row missing at Lx={lx}")
        dimension = int(rows["sector_dimension"].dropna().iloc[0])
        if dimension != expected:
            raise RuntimeError(
                f"authoritative sector dimension mismatch at Lx={lx}: {dimension} != {expected}"
            )
        result[f"Lx{lx}_sector_dimension"] = dimension
    return result


def run(
    *,
    base_data_dir: Path,
    primme_data_dir: Path,
    output_dir: Path,
    widths: tuple[float, ...],
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prerequisite = _validate_authoritative_base(base_data_dir)
    atomic_write_json(
        output / "qdm_checkerboard_fixed_O1_prerequisite_audit.json",
        prerequisite,
    )

    reference = recover_reference_geometry()
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for repeats in (1, 2):
        started = time.perf_counter()
        context = build_context(
            reference=reference,
            repeats=repeats,
            phase=REPRESENTATIVE_PHASE,
        )
        h_dense = np.asarray(context.h_sector.toarray(), dtype=np.complex128)
        energies, vectors = la.eigh(h_dense, check_finite=False)
        eigen_residuals = np.linalg.norm(
            context.h_sector @ vectors - vectors * energies[None, :],
            axis=0,
        )
        if float(np.max(eigen_residuals, initial=0.0)) > 1.0e-8:
            raise RuntimeError(
                f"Lx={context.lx} exact ED residual exceeded tolerance: "
                f"{np.max(eigen_residuals):.3e}"
            )

        raw_q = {
            name: np.real(np.einsum("ij,ij->j", vectors.conj(), operator @ vectors))
            for name, operator in context.projected_q.items()
        }
        exceptional, dark_blocks = _joint_dark_all(
            energies,
            vectors,
            context.q_all,
            context.tower,
        )
        resolved = projector_resolved_energy_basis(
            energies,
            vectors,
            exceptional,
            energy_tolerance=ENERGY_BLOCK_TOL,
            vector_tolerance=1.0e-9,
        )
        exceptional_mask = resolved["is_exceptional"].astype(bool)
        clean_mask = ~exceptional_mask
        clean_energies = np.asarray(resolved["energies"])[clean_mask]
        clean_q = {
            name: np.real(
                np.einsum(
                    "ij,ij->j",
                    resolved["basis"][:, clean_mask].conj(),
                    operator @ resolved["basis"][:, clean_mask],
                )
            )
            for name, operator in context.projected_q.items()
        }
        beta_raw, canonical_raw_weights = canonical_weights(energies, context.tower_energy)
        beta_clean, canonical_clean_weights = canonical_weights(
            clean_energies,
            context.tower_energy,
        )
        canonical_raw = {
            name: float(np.dot(canonical_raw_weights, values)) for name, values in raw_q.items()
        }
        canonical_clean = {
            name: float(np.dot(canonical_clean_weights, values)) for name, values in clean_q.items()
        }
        stripe_ops, _, stripe_meta, ambient_names, quotient_coefficients = stripe_algebra(
            context,
            z_placement=reference.z_placement,
        )
        target = target_dark_kernel(
            context,
            energies,
            vectors,
            solver_tolerance=1.0e-12,
        )
        block_rows.append(
            {
                "Lx": context.lx,
                "phase": context.phase,
                "sector_dimension": context.sector.sector_dimension,
                "target_energy": context.tower_energy,
                "target_block_dimension": target["block_dimension"],
                "target_joint_dark_rank": target["joint_dark_rank"],
                "cage_target_projector_weight": target["cage_projector_weight"],
                "cage_joint_dark_weight": target["cage_dark_weight"],
                "target_maximum_residual": target["target_maximum_residual"],
                "full_joint_dark_rank": exceptional.shape[1],
                "exact_energy_block_count": len(dark_blocks),
            }
        )
        atomic_write_csv(output / AUDIT_NAME, pd.DataFrame(block_rows))

        empty_exceptional = np.zeros((context.sector.sector_dimension, 0), dtype=np.complex128)
        for half_width in widths:
            raw_window = select_microcanonical_window_by_width(
                energies,
                target_energy=context.tower_energy,
                half_width=float(half_width),
                degeneracy_tolerance=ENERGY_BLOCK_TOL,
            )
            raw_indices = np.asarray(raw_window.indices, dtype=int)
            clean_window = select_microcanonical_window_by_width(
                clean_energies,
                target_energy=context.tower_energy,
                half_width=float(half_width),
                degeneracy_tolerance=ENERGY_BLOCK_TOL,
            )
            clean_indices = np.asarray(clean_window.indices, dtype=int)
            raw_mc = {name: float(np.mean(values[raw_indices])) for name, values in raw_q.items()}
            clean_mc = {
                name: float(np.mean(values[clean_indices])) for name, values in clean_q.items()
            }
            raw_covariance = projector_deleted_block_covariance(
                energies,
                vectors,
                empty_exceptional,
                stripe_ops,
                raw_indices,
                energy_tolerance=ENERGY_BLOCK_TOL,
                vector_tolerance=1.0e-9,
            )
            clean_covariance = projector_deleted_block_covariance(
                energies,
                vectors,
                exceptional,
                stripe_ops,
                raw_indices,
                energy_tolerance=ENERGY_BLOCK_TOL,
                vector_tolerance=1.0e-9,
            )
            removed_rank = int(np.count_nonzero(exceptional_mask[raw_indices]))
            row = {
                "Lx": context.lx,
                "Ly": 4,
                "phase": context.phase,
                "window_protocol": "fixed_O1_total_energy",
                "window_half_width": float(half_width),
                "window_energy_density_half_width": float(half_width) / context.lx,
                "sector_dimension": context.sector.sector_dimension,
                "raw_window_state_count": int(raw_window.n_states),
                "clean_window_state_count": int(clean_window.n_states),
                "joint_dark_removed_rank": removed_rank,
                "removed_fraction": float(removed_rank / max(1, raw_window.n_states)),
                "matched_beta_raw": beta_raw,
                "matched_beta_clean": beta_clean,
                "tau_A_mc_raw": raw_mc["A"],
                "tau_Z_mc_raw": raw_mc["Z"],
                "tau_A_mc_clean": clean_mc["A"],
                "tau_Z_mc_clean": clean_mc["Z"],
                "tau_A_can_raw": canonical_raw["A"],
                "tau_Z_can_raw": canonical_raw["Z"],
                "tau_A_can_clean": canonical_clean["A"],
                "tau_Z_can_clean": canonical_clean["Z"],
                "matching_distance_raw": max(
                    abs(raw_mc["A"] - canonical_raw["A"]),
                    abs(raw_mc["Z"] - canonical_raw["Z"]),
                ),
                "matching_distance_clean": max(
                    abs(clean_mc["A"] - canonical_clean["A"]),
                    abs(clean_mc["Z"] - canonical_clean["Z"]),
                ),
                "w_raw": float(raw_covariance["largest_width"]),
                "w_clean": float(clean_covariance["largest_width"]),
                "raw_clean_width_difference": float(
                    raw_covariance["largest_width"] - clean_covariance["largest_width"]
                ),
                "formal_local_dimension": int(stripe_meta["formal_operator_dimension"]),
                "formal_nonidentity_dimension": int(stripe_meta["ambient_nonidentity_dimension"]),
                "projected_quotient_dimension": int(stripe_meta["projected_operator_dimension"]),
                "projected_quotient_nonidentity_dimension": max(
                    0,
                    int(stripe_meta["projected_operator_dimension"]) - 1,
                ),
                "energy_block_tolerance": ENERGY_BLOCK_TOL,
                "target_block_dimension": target["block_dimension"],
                "target_joint_dark_rank": target["joint_dark_rank"],
                "cage_target_projector_weight": target["cage_projector_weight"],
                "maximum_eigenpair_residual": float(np.max(eigen_residuals, initial=0.0)),
                "exact_ED_runtime_seconds": float(time.perf_counter() - started),
                "peak_rss_gib": process_memory_gib(),
                "worst_raw_coefficients": json.dumps(
                    {
                        ambient_names[index]: [
                            float(complex(value).real),
                            float(complex(value).imag),
                        ]
                        for index, value in enumerate(
                            quotient_coefficients @ np.asarray(raw_covariance["worst_coefficients"])
                        )
                        if abs(value) > 1.0e-10
                    },
                    sort_keys=True,
                ),
            }
            rows.append(row)
            atomic_write_json(
                output / "fixed_O1_checkpoints" / f"Lx{context.lx}_dE{half_width:.3f}.json",
                row,
            )
            atomic_write_csv(output / SYSTEMATICS_NAME, pd.DataFrame(rows))

    frame = pd.DataFrame(rows).sort_values(["Lx", "window_half_width"]).reset_index(drop=True)
    estimated_budgets = {
        float(width): estimate_l12_window_budget(
            primme_data_dir,
            half_width=float(width),
        )
        for width in widths
    }
    recommendation = recommend_fixed_width(
        frame,
        estimated_budgets=estimated_budgets,
    )
    recommendation.update(
        {
            "authoritative_base": str(Path(base_data_dir)),
            "primme_staged_source": str(Path(primme_data_dir)),
            "estimated_L12_budgets": {
                f"{width:.3f}": value for width, value in estimated_budgets.items()
            },
            "selection_note": (
                "Delta E=0.20 is preferred only if it passes the declared finite-size and "
                "coverability heuristics; otherwise the JSON requests scientific review."
            ),
        }
    )
    atomic_write_csv(output / SYSTEMATICS_NAME, frame)
    atomic_write_json(output / RECOMMENDATION_NAME, recommendation)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--primme-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--widths",
        default=",".join(f"{value:.2f}" for value in PILOT_HALF_WIDTHS),
    )
    args = parser.parse_args()
    frame = run(
        base_data_dir=args.base_data_dir,
        primme_data_dir=args.primme_data_dir,
        output_dir=args.output_dir,
        widths=_parse_widths(args.widths),
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
