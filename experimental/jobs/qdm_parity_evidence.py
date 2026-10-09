#!/usr/bin/env python
"""Regenerate target-parity QDM evidence and track the 12x4 submission gate.

Positive phases only: phi=0 has enhanced symmetry and needs a separate analysis.
The primary fixed total-energy half-width is inherited, not fitted: DeltaE=0.25.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experimental/jobs", ROOT / "experimental/notebooks"):
    sys.path.insert(0, str(path))

import helpers  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as la  # noqa: E402
from evidence_job_utils import collect_file_manifest, git_metadata, write_json  # noqa: E402
from qdm_character_dimensions import checkerboard_character_dimensions  # noqa: E402
from qdm_checkerboard_large_strip import (  # noqa: E402
    canonical_typicality_scan,
    energy_matched_canonical_estimate,
)
from qdm_followup_provenance import validate_eigensystem  # noqa: E402
from qdm_parity_protocol import (  # noqa: E402
    PROTOCOL,
    require_parity_protocol,
    window_controls_stable,
)
from qdm_resumable_spectrum import folded_problem_description  # noqa: E402
from qdm_sec7_fixed_o1 import (  # noqa: E402
    ENERGY_BLOCK_TOL,
    EXPECTED_SECTOR_DIMENSIONS,
    build_context,
    canonical_weights,
    checkerboard_instance,
    recover_reference_geometry,
    stripe_algebra,
)
from qdm_sec7_fixed_o1_l12_observables import (  # noqa: E402
    ACCEPTANCE_NAME as OBSERVABLES_ACCEPTANCE,
)
from qdm_sec7_fixed_o1_l12_spectrum import (  # noqa: E402
    ACCEPTANCE_NAME as SPECTRAL_ACCEPTANCE,
)
from qdm_sec7_fixed_o1_pilot import SYSTEMATICS_NAME, joint_dark_energy_subspace  # noqa: E402
from qdm_sec7_fixed_o1_sequence import STATUS_NAME as SEQUENCE_STATUS  # noqa: E402
from qdm_sy_followup import _table  # noqa: E402
from qdm_target_parity import compact_target_parity  # noqa: E402

PHASES = (0.025, 0.05, 0.075, 0.10)
WIDTHS = (0.10, 0.20, 0.25, 0.50)
IDENTITY = {
    "parity_policy": "measured_target",
    "symmetry_protocol": PROTOCOL,
    "fully_symmetry_resolved": True,
}


def sector_identity(context):
    return {
        **IDENTITY,
        "Sy": context.sector.labels["Sy_character"],
        "target_Sy_residual": context.sector.labels["target_Sy_residual"],
    }


def evaluate_exact(context, energies, vectors, operators, *, widths=WIDTHS):
    """Raw traces include the target; joint-dark deletion is a companion only."""
    validation = validate_eigensystem(context.h_sector, energies, vectors)
    beta, weights = canonical_weights(energies, context.tower_energy)
    q = {
        name: np.einsum("ij,ij->j", vectors.conj(), op @ vectors).real
        for name, op in context.projected_q.items()
    }
    canonical = {name: float(weights @ values) for name, values in q.items()}
    exceptional, _ = joint_dark_energy_subspace(energies, vectors, context.q_all, context.tower)
    empty = np.zeros((vectors.shape[0], 0), complex)
    rows = []
    for width in widths:
        indices = np.flatnonzero(
            np.abs(energies - context.tower_energy) <= width + ENERGY_BLOCK_TOL
        )
        raw = {name: float(values[indices].mean()) for name, values in q.items()}
        deleted = helpers.projector_deleted_basis(vectors[:, indices], exceptional, tolerance=1e-9)
        n_clean = deleted["retained_rank"]
        clean = {
            name: helpers.projector_deleted_observable_moments(
                vectors[:, indices],
                exceptional,
                context.projected_q[name],
                tolerance=1e-9,
            )
            if n_clean
            else {"mean": None}
            for name in q
        }
        covariance = helpers.projector_deleted_block_covariance(
            energies,
            vectors,
            empty,
            operators,
            indices,
            energy_tolerance=ENERGY_BLOCK_TOL,
            vector_tolerance=1e-9,
        )
        clean_covariance = (
            helpers.projector_deleted_block_covariance(
                energies,
                vectors,
                exceptional,
                operators,
                indices,
                energy_tolerance=ENERGY_BLOCK_TOL,
                vector_tolerance=1e-9,
            )
            if n_clean
            else None
        )
        removed = len(indices) - n_clean
        rows.append(
            {
                **sector_identity(context),
                "Lx": context.lx,
                "Ly": 4,
                "phase": context.phase,
                "sector_dimension": len(energies),
                "target_energy": context.tower_energy,
                "window_protocol": "fixed_O1_total_energy",
                "window_half_width": width,
                "raw_window_state_count": len(indices),
                "clean_window_state_count": n_clean,
                "joint_dark_removed_rank": removed,
                "removed_fraction": removed / len(indices),
                "matched_beta_raw": beta,
                **{f"tau_{name}_can_raw": value for name, value in canonical.items()},
                **{f"tau_{name}_mc_raw": value for name, value in raw.items()},
                **{f"tau_{name}_mc_clean": clean[name]["mean"] for name in q},
                "matching_distance_raw": max(abs(raw[name] - canonical[name]) for name in q),
                "w_raw": covariance["largest_width"],
                "w_clean": clean_covariance["largest_width"] if n_clean else None,
                "operator_normalization": "local_HS_quotient_in_target_parity",
                "raw_population_includes_target": True,
                "nontrivial_thermal_sector": len(energies) > 1,
                "scaling_eligible": len(energies) > 1 and len(indices) > 1,
                "window_energy_density_half_width": width / (4 * context.lx),
                **validation,
            }
        )
    return pd.DataFrame(rows)


def _plot_exact(output, scatter, rows):
    frame = scatter[(scatter.Lx == 8) & np.isclose(scatter.phase, 0.05)]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
    for axis, name in zip(axes, ("A", "Z"), strict=True):
        background = frame[~frame.is_tower_state]
        target = frame[frame.is_tower_state]
        axis.scatter(background.energy_density, background[f"Q_{name}"], s=4, alpha=0.6)
        axis.scatter(
            target.energy_density,
            target[f"Q_{name}"],
            s=55,
            marker="*",
            color="#b52130",
            label="Compact target",
        )
        axis.set_ylabel(rf"$\langle Q_{name}\rangle$")
    axes[0].set_title(r"$8\times4$, $\phi=0.05$, $S_y=+1$", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel(r"Energy density $E/32$")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"qdm_fig9a_target_parity.{suffix}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4))
    selected = rows[np.isclose(rows.window_half_width, 0.25) & rows.scaling_eligible]
    labels = (r"$\tau_A^{\mathrm{mc}}$", r"$\tau_Z^{\mathrm{mc}}$", r"$w_{\mathrm{raw}}$")
    for axis, key, label in zip(
        axes, ("tau_A_mc_raw", "tau_Z_mc_raw", "w_raw"), labels, strict=True
    ):
        for lx, group in selected.groupby("Lx"):
            axis.plot(group.phase, group[key], "o-", markersize=3, label=f"{lx} x 4")
        axis.set_xlabel(r"$\phi$")
        axis.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "qdm_parity_phase_evidence.pdf")
    fig.savefig(output / "qdm_parity_phase_evidence.png", dpi=180)
    plt.close(fig)


def regenerate_exact(output, *, phases=PHASES):
    reference = recover_reference_geometry()
    frames, rows, audits = [], [], []
    for repeats in (1, 2):
        for phase in phases:
            print(f"[qdm-parity] exact {4 * repeats}x4 phi={phase}", flush=True)
            context = build_context(
                reference=reference, repeats=repeats, phase=phase, sy_character="target"
            )
            directory = output / "exact_eigensystems" / f"Lx{context.lx}_phi{phase:.6f}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "energies.npy").is_file() and (directory / "vectors.npy").is_file():
                energies = np.load(directory / "energies.npy", allow_pickle=False)
                vectors = np.load(directory / "vectors.npy", allow_pickle=False)
                validate_eigensystem(context.h_sector, energies, vectors)
            else:
                energies, vectors = la.eigh(context.h_sector.toarray())
                validate_eigensystem(context.h_sector, energies, vectors)
                np.save(directory / "energies.npy", energies)
                np.save(directory / "vectors.npy", vectors)
            operators, _, metadata, _, _ = stripe_algebra(
                context, z_placement=reference.z_placement
            )
            frame = _table(
                context,
                energies,
                vectors,
                parity=context.sector.labels["Sy_character"],
                target=context.tower,
            )
            if frame.is_tower_state.sum() != 1:
                raise RuntimeError("target-exclusion rank changed")
            frames.append(frame.assign(**sector_identity(context)))
            rows.append(evaluate_exact(context, energies, vectors, operators))
            audit = {
                **sector_identity(context),
                "Lx": context.lx,
                "phase": phase,
                "tower_residual": context.tower_residual,
                "cage_Q": context.cage_q,
                "dimension": len(energies),
                "stripe_algebra": metadata,
                "spectral_problem": folded_problem_description(
                    context.h_sector, target_energy=context.tower_energy
                ),
            }
            audits.append(audit)
            write_json(directory / "metadata.json", audit)
            pd.concat(rows).to_csv(output / "qdm_parity_phase_window_systematics.csv", index=False)
    scatter = pd.concat(frames, ignore_index=True)
    table = pd.concat(rows, ignore_index=True)
    scatter.to_csv(output / "qdm_parity_all_eigenstates.csv", index=False)
    scatter[~scatter.is_tower_state].to_csv(output / "qdm_parity_eth_scatter.csv", index=False)
    table[np.isclose(table.phase, 0.05)].to_csv(output / SYSTEMATICS_NAME, index=False)
    write_json(output / "qdm_parity_exact_audit.json", {"blocks": audits, "phases": list(phases)})
    write_json(
        output / "qdm_checkerboard_fixed_O1_window_recommendation.json",
        {
            **IDENTITY,
            "status": "recommended",
            "recommended_half_width": 0.25,
            "neighbor_controls": [0.10, 0.20, 0.50],
            "estimated_L12_budgets": {},
            "selection_policy": (
                "prespecified continuation of PR151 DeltaE=0.25; no observable fitting"
            ),
            "claim_boundary": "Finite-size evidence; no selected thermodynamic extrapolation.",
        },
    )
    _plot_exact(output, scatter, table)
    return table


def matrix_preflight(output, *, budget=1024):
    reference = recover_reference_geometry()
    context = build_context(reference=reference, repeats=3, sy_character="target")
    h = context.h_sector
    n = h.shape[0]
    payload = {
        **sector_identity(context),
        "Lx": 12,
        "Ly": 4,
        "phase": context.phase,
        "sector_dimension": n,
        "legacy_dimension": EXPECTED_SECTOR_DIMENSIONS[12],
        "dimension_ratio": n / EXPECTED_SECTOR_DIMENSIONS[12],
        "nnz": h.nnz,
        "sparse_storage_gib": sum(x.nbytes for x in (h.data, h.indices, h.indptr)) / 2**30,
        "one_complex_vector_gib": n * 16 / 2**30,
        "requested_budget": budget,
        "eigenvector_storage_gib": n * budget * 16 / 2**30,
        "krylov_2p05_budget_storage_gib": n * int(np.ceil(2.05 * budget)) * 16 / 2**30,
        "tower_residual": context.tower_residual,
        "cage_Q": context.cage_q,
        "problem": folded_problem_description(h, target_energy=context.tower_energy),
        "runtime_prediction": (
            "Measured dimensions/storage only; convergence speed requires a benchmark."
        ),
        "solve_launched": False,
    }
    write_json(output / "qdm_parity_L12_preflight.json", payload)
    return payload


def preflight(output, *, budget=1024):
    reference = recover_reference_geometry()
    rows = []
    for repeats in (1, 2, 3):
        measured = compact_target_parity(
            checkerboard_instance(reference, repeats, 0.05), repeats=repeats
        )
        dimensions = checkerboard_character_dimensions(4 * repeats)
        n = dimensions["Sy_dimensions"][str(measured["Sy"])]
        rows.append(
            {
                **IDENTITY,
                **measured,
                **dimensions,
                "target_sector_dimension": n,
                "dimension_ratio": n / dimensions["translation_dimension"],
                "one_complex_vector_gib": n * 16 / 2**30,
                "eigenvector_storage_gib": n * budget * 16 / 2**30,
                "krylov_2p05_budget_storage_gib": n * int(np.ceil(2.05 * budget)) * 16 / 2**30,
                "requested_budget": budget,
                "solve_launched": False,
                "matrix_constructed": False,
            }
        )
    write_json(output / "qdm_parity_character_preflight.json", {"blocks": rows, **IDENTITY})
    return rows


def regenerate_canonical(output, *, samples=8, stderr_tolerance=5e-4, grid_tolerance=5e-5):
    reference = recover_reference_geometry()
    context = build_context(reference=reference, repeats=3, sy_character="target")
    rows = []
    for n_samples, points, seed in (
        (samples, 41, 20261009),
        (2 * samples, 41, 20261010),
        (2 * samples, 81, 20261010),
    ):
        print(f"[qdm-parity] canonical samples={n_samples} points={points}", flush=True)
        scan = canonical_typicality_scan(
            context.h_sector,
            context.projected_q,
            beta_max=0.25,
            beta_points=points,
            n_samples=n_samples,
            random_seed=seed,
        )
        matched = energy_matched_canonical_estimate(scan, target_energy=context.tower_energy)
        rows.append(
            {
                **sector_identity(context),
                "Lx": 12,
                "Ly": 4,
                "phase": context.phase,
                "sector_dimension": context.h_sector.shape[0],
                "beta_star": matched.beta,
                "beta_stderr": None,
                "target_energy": context.tower_energy,
                "stochastic_samples": n_samples,
                "beta_points": points,
                "random_seed": seed,
                **{f"tau_{name}_target": value for name, value in matched.observables.items()},
                **{
                    f"tau_{name}_stderr": value for name, value in matched.observable_stderr.items()
                },
                "uncertainty_scope": (
                    "conditional ratio jackknife; beta-match tested by grid/sample refinement"
                ),
            }
        )
        pd.DataFrame(rows).to_csv(output / "qdm_parity_canonical_refinement.csv", index=False)
    a, coarse, fine = rows
    checks = {
        "conditional_stderr_below_tolerance": all(
            fine[f"tau_{name}_stderr"] <= stderr_tolerance for name in ("A", "Z")
        ),
        "grid_refinement_stable": max(
            abs(coarse[f"tau_{name}_target"] - fine[f"tau_{name}_target"]) for name in ("A", "Z")
        )
        <= grid_tolerance,
        "independent_sample_refinement_stable": all(
            abs(a[f"tau_{name}_target"] - fine[f"tau_{name}_target"])
            <= 3 * np.hypot(a[f"tau_{name}_stderr"], fine[f"tau_{name}_stderr"]) + grid_tolerance
            for name in ("A", "Z")
        ),
    }
    pd.DataFrame([fine]).to_csv(
        output / "qdm_checkerboard_finite_beta_transfer_target.csv", index=False
    )
    checks = {key: bool(value) for key, value in checks.items()}
    acceptance = {
        **sector_identity(context),
        "closed": all(checks.values()),
        "checks": checks,
        "stderr_tolerance": stderr_tolerance,
        "grid_tolerance": grid_tolerance,
    }
    write_json(output / "qdm_parity_canonical_acceptance.json", acceptance)
    if not acceptance["closed"]:
        raise RuntimeError(
            "canonical refinement gate remains open; increase samples or refine grid"
        )
    return acceptance


def submission_gate(output):
    checks = {}
    for name in (
        "qdm_parity_canonical_acceptance.json",
        SPECTRAL_ACCEPTANCE,
        OBSERVABLES_ACCEPTANCE,
        SEQUENCE_STATUS,
    ):
        path = output / name
        payload = json.loads(path.read_text()) if path.is_file() else {}
        if payload:
            require_parity_protocol(payload, description=name)
        checks[name] = payload.get("closed") is True
    exact = output / "qdm_parity_phase_window_systematics.csv"
    if exact.is_file():
        frame = pd.read_csv(exact)
        pairs = set(zip(frame.Lx, frame.phase, strict=True))
        checks["positive_phase_exact_scan"] = (
            all((lx, phase) in pairs for lx in (4, 8) for phase in PHASES)
            and frame.Sy.isin([-1, 1]).all()
            and frame.symmetry_protocol.eq(PROTOCOL).all()
        )
    else:
        checks["positive_phase_exact_scan"] = False
    preflight_path = output / "qdm_parity_character_preflight.json"
    if preflight_path.is_file():
        counted = json.loads(preflight_path.read_text())
        require_parity_protocol(counted, description="character preflight")
        blocks = counted.get("blocks", [])
        checks["target_irreps_measured_all_sizes"] = (
            len(blocks) == 3
            and [b["Sy"] for b in blocks] == [-1, 1, -1]
            and all(b["target_Sy_residual"] < 1e-8 for b in blocks)
            and [b["target_sector_dimension"] for b in blocks] == [1, 875, 39667]
        )
    else:
        checks["target_irreps_measured_all_sizes"] = False
    systematics_path = output / "qdm_checkerboard_window_systematics_fixed_O1.csv"
    checks["L12_primary_and_neighbor_controls_stable"] = (
        window_controls_stable(pd.read_csv(systematics_path))
        if systematics_path.is_file()
        else False
    )
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        **IDENTITY,
        "closed": all(checks.values()),
        "checks": checks,
        "manuscript_submission_ready": all(checks.values()),
        "claim_boundary": (
            "Numerical evidence gate only; manuscript wording/figures require human review."
        ),
    }
    write_json(output / "qdm_parity_submission_gate.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "exact",
            "target-parity",
            "L12-preflight",
            "L12-matrix-preflight",
            "L12-canonical",
            "gate",
        ),
        default="exact",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    helpers.set_revtex_matplotlib_style(base_font_size=8, prefer_tex=False)
    if args.stage == "target-parity":
        reference = recover_reference_geometry()
        measured = [
            compact_target_parity(checkerboard_instance(reference, n, 0.05), repeats=n)
            for n in (1, 2, 3)
        ]
        write_json(output / "qdm_compact_target_parities.json", {"blocks": measured, **IDENTITY})
        print(json.dumps(measured, indent=2), flush=True)
    elif args.stage == "exact":
        regenerate_exact(output)
    elif args.stage == "L12-preflight":
        preflight(output)
    elif args.stage == "L12-matrix-preflight":
        matrix_preflight(output)
    elif args.stage == "L12-canonical":
        regenerate_canonical(output, samples=args.samples)
    gate = submission_gate(output)
    write_json(
        output / "qdm_parity_run_metadata.json",
        {
            "git": git_metadata(ROOT),
            "host_source_commit": os.environ.get("QLINKS_EVIDENCE_SOURCE_COMMIT"),
            "stage": args.stage,
            "runtime_seconds": time.perf_counter() - started,
            **IDENTITY,
        },
    )
    write_json(output / "file_manifest.json", collect_file_manifest(output))
    print(json.dumps(gate, indent=2), flush=True)
    if args.stage == "gate" and not gate["closed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
