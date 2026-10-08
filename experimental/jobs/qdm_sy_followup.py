#!/usr/bin/env python
"""Bounded 8x4 S_y resolution and Fig. 9(a) provenance follow-up to PR #150.

No 12x4 eigensolve, manuscript replacement, or production cache mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experimental/jobs", ROOT / "experimental/notebooks"):
    sys.path.insert(0, str(path))

import helpers  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import prx_referee_sanity_qdm as sanity  # noqa: E402
import qdm_checkerboard_symmetry as symmetry  # noqa: E402
import qdm_sec7_fixed_o1 as fixed  # noqa: E402
import scipy.linalg as la  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from evidence_job_utils import (  # noqa: E402
    collect_file_manifest,
    git_metadata,
    utc_run_id,
    write_json,
)
from qdm_followup_provenance import compare_populations, validate_eigensystem  # noqa: E402

from qlinks.caging.analysis.spectral import permutation_matrix  # noqa: E402

TOL = 1.0e-8


def _table(context, energies, vectors, *, parity=None, target=None):
    exceptional = np.zeros((vectors.shape[0], 0), complex) if target is None else target[:, None]
    resolved = helpers.projector_resolved_energy_basis(
        energies,
        vectors,
        exceptional,
        energy_tolerance=fixed.ENERGY_BLOCK_TOL,
        vector_tolerance=1.0e-9,
    )
    v = resolved["basis"]
    e = resolved["energies"]
    potential = sp.diags(context.build.hamiltonian.diagonal())
    b = context.sector.basis
    potential = sp.csr_array(b.conj().T @ (potential @ b))
    frame = pd.DataFrame(
        {
            "energy": e,
            "energy_density": e / 32.0,
            "Q_A": sanity._expectation(context.projected_q["A"], v),
            "Q_Z": sanity._expectation(context.projected_q["Z"], v),
            "potential_energy": sanity._expectation(potential, v),
            "is_tower_state": resolved["is_exceptional"].astype(bool),
        }
    )
    frame["kinetic_energy"] = frame.energy - frame.potential_energy
    frame["Sy"] = parity
    frame["Lx"], frame["Ly"], frame["phase"] = 8, 4, context.phase
    frame["background_variant"] = "raw_target_excluded"
    return frame


def resolve_parity_blocks(context, *, stripe_operators=()):
    """Build sparse character projectors, then solve only the two bounded blocks."""
    b = context.sector.basis
    blocks = []
    frames = []
    concentration = []
    reconstructed_h = sp.csr_array(context.h_sector.shape, dtype=complex)
    projector_sum = sp.csr_array(context.h_sector.shape, dtype=complex)
    for parity in (1, -1):
        resolved, _ = symmetry.checkerboard_fully_resolved_sector(
            context.model,
            context.basis,
            packed_index=context.packed_index,
            repeats=2,
            sy_character=parity,
            chunk_size=16384,
        )
        c = sp.csr_array(b.conj().T @ resolved.sector.basis)
        h = sp.csr_array(c.conj().T @ context.h_sector @ c)
        e, v = la.eigh(h.toarray(), check_finite=True)
        checks = validate_eigensystem(h, e, v)
        tower = np.asarray(c.conj().T @ context.tower).reshape(-1)
        weight = float(np.vdot(tower, tower).real)
        target = tower / np.sqrt(weight) if weight > 1.0e-16 else None
        # Evaluate the original local witnesses in legacy coordinates; never
        # replace Q by a product of separately projected L operators.
        lifted = np.asarray(c @ v)
        target_lifted = np.asarray(c @ target).reshape(-1) if target is not None else None
        frames.append(_table(context, e, lifted, parity=parity, target=target_lifted))
        if stripe_operators:
            concentration.append(
                _concentration_row(
                    e,
                    v,
                    tuple(sp.csr_array(c.conj().T @ op @ c) for op in stripe_operators),
                    parity=parity,
                    target_energy=context.tower_energy,
                )
            )
        blocks.append({"Sy": parity, "dimension": len(e), "target_weight": weight, **checks})
        projector_sum += c @ c.conj().T
        reconstructed_h += c @ h @ c.conj().T
    resolution = {
        "blocks": blocks,
        "projector_completeness_residual": float(
            sp.linalg.norm(projector_sum - sp.eye(b.shape[1]))
        ),
        "hamiltonian_reconstruction_residual": float(
            sp.linalg.norm(reconstructed_h - context.h_sector)
        ),
    }
    if (
        max(
            resolution[k]
            for k in ("projector_completeness_residual", "hamiltonian_reconstruction_residual")
        )
        > TOL
    ):
        raise RuntimeError(f"incomplete parity decomposition: {resolution}")
    return pd.concat(frames, ignore_index=True), resolution, concentration


def _concentration_row(energies, vectors, operators, *, parity, target_energy):
    indices = np.flatnonzero(np.abs(energies - target_energy) <= 0.25)
    result = helpers.projector_deleted_block_covariance(
        energies,
        vectors,
        np.zeros((vectors.shape[0], 0), complex),
        operators,
        indices,
        energy_tolerance=fixed.ENERGY_BLOCK_TOL,
        vector_tolerance=1.0e-9,
    )
    return {
        "Sy": parity,
        "window_half_width": 0.25,
        "window_count": len(indices),
        "largest_covariance_eigenvalue": result["largest_eigenvalue"],
        "w": result["largest_width"],
        "operator_count": len(operators),
        "normalization": "same_local_HS_frame_as_translation_union",
    }


def _antiunitary_audit(context):
    """U K in reduced coordinates is B^dagger U B*, not B^dagger U B."""
    b = context.sector.basis
    permutations = symmetry.checkerboard_positive_phase_permutations(
        context.model,
        context.basis,
        packed_index=context.packed_index,
    )
    h = context.h_sector.toarray()
    rows = []
    for name in ("Rx", "Ry", "Sx", "Sy", "C2"):
        p = sp.csr_array(permutation_matrix(permutations[name]))
        a = (b.conj().T @ p @ b.conjugate()).toarray()
        rows.append(
            {
                "operation": name + " K",
                "preservation_residual": float(
                    np.linalg.norm(a.conj().T @ a - np.eye(len(h))) / np.sqrt(len(h))
                ),
                "commuting_residual": float(
                    np.linalg.norm(a @ h.conjugate() - h @ a) / np.linalg.norm(h)
                ),
                "square_plus_one_residual": float(
                    np.linalg.norm(a @ a.conjugate() - np.eye(len(h))) / np.sqrt(len(h))
                ),
            }
        )
    for row in rows:
        row["exact_commuting_antiunitary_inside_sector"] = bool(
            max(
                row[key]
                for key in (
                    "preservation_residual",
                    "commuting_residual",
                    "square_plus_one_residual",
                )
            )
            <= TOL
        )
    return rows


def _plot(output, frame):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
    for axis, witness in zip(axes, ("Q_A", "Q_Z"), strict=True):
        for parity, color, marker in ((1, "#246a9b", "o"), (-1, "#b85d24", "^")):
            f = frame[(frame.Sy == parity) & ~frame.is_tower_state]
            axis.scatter(
                f.energy_density,
                f[witness],
                s=8,
                alpha=0.6,
                color=color,
                marker=marker,
                label=f"$S_y={parity:+d}$ ({len(f)} states)",
            )
        axis.set_ylabel(witness)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Energy density E/32")
    axes[0].set_title("8 x 4 checkerboard QDM: exact shifted-reflection sectors")
    fig.tight_layout()
    fig.savefig(output / "qdm_fig9a_sy_diagnostic.pdf")
    fig.savefig(output / "qdm_fig9a_sy_diagnostic.png", dpi=180)
    plt.close(fig)


def _means(frame, target_energy):
    rows = []
    for parity in (None, 1, -1):
        full = frame if parity is None else frame[frame.Sy == parity]
        # Canonical and microcanonical primary comparisons include the target.
        e = full.energy.to_numpy()
        beta, weights = fixed.canonical_weights(e, target_energy)
        window = full[np.abs(full.energy - target_energy) <= 0.25]
        rows.append(
            {
                "Sy": "translation_union" if parity is None else parity,
                "dimension": len(full),
                "target_weight": int(full.is_tower_state.sum()),
                "beta_matched": beta,
                "window_half_width": 0.25,
                "window_count": len(window),
                **{
                    f"canonical_{key}": float(weights @ full[key].to_numpy())
                    for key in ("Q_A", "Q_Z")
                },
                **{f"window_{key}": float(window[key].mean()) for key in ("Q_A", "Q_Z")},
            }
        )
    return pd.DataFrame(rows)


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    helpers.set_revtex_matplotlib_style(base_font_size=8, prefer_tex=False)
    print("[qdm-sy] recovering reference motif", flush=True)
    reference = fixed.recover_reference_geometry()
    context = fixed.build_context(reference=reference, repeats=2)
    print("[qdm-sy] validating legacy 1125-dimensional eigensystem", flush=True)
    e, v, metadata, source = sanity._spectrum(
        context, roots=(args.data_dir, args.cache_root), allow_small_dense=args.allow_small_dense
    )
    validate_eigensystem(context.h_sector, e, v)
    cache = output / "legacy_complete_eigensystem"
    cache.mkdir(exist_ok=True)
    np.save(cache / "energies.npy", e)
    np.save(cache / "vectors.npy", v)
    write_json(cache / "metadata.json", {**metadata, "fully_symmetry_resolved": False})
    legacy = _table(context, e, v, target=context.tower)
    legacy_bg = legacy[~legacy.is_tower_state].copy()
    legacy_bg.to_csv(output / "qdm_fig9a_legacy_reconstructed.csv", index=False)
    source_frame = sanity._source_population(args.data_dir)
    print("[qdm-sy] building exact S_y +/- character blocks", flush=True)
    stripe_operators, _, stripe_metadata, _, _ = fixed.stripe_algebra(
        context,
        z_placement=reference.z_placement,
    )
    frame, resolution, concentration = resolve_parity_blocks(
        context, stripe_operators=stripe_operators
    )
    concentration.insert(
        0,
        _concentration_row(
            e, v, stripe_operators, parity="translation_union", target_energy=context.tower_energy
        ),
    )
    pd.DataFrame(concentration).to_csv(output / "qdm_sy_stripe_concentration.csv", index=False)
    write_json(output / "qdm_sy_stripe_algebra.json", stripe_metadata)
    if len(frame) != context.sector.sector_dimension or frame.is_tower_state.sum() != 1:
        raise RuntimeError("parity resolution changed the target-exclusion rank or population")
    if frame.loc[frame.is_tower_state, ["Q_A", "Q_Z"]].to_numpy().max() > TOL:
        raise RuntimeError("target darkness failed after parity resolution")
    frame.to_csv(output / "qdm_sy_all_eigenstates.csv", index=False)
    background = frame[~frame.is_tower_state].copy()
    classifier = sanity._classify(background)
    background["branch_label"] = classifier["labels"]
    background["branch_score"] = classifier["score"]
    background.to_csv(output / "qdm_fig9a_sy_scatter.csv", index=False)
    contingency = pd.crosstab(background.Sy, background.branch_label)
    contingency.to_csv(output / "qdm_sy_branch_contingency.csv")
    agreement = float(
        max(contingency.to_numpy().trace(), np.fliplr(contingency.to_numpy()).trace())
        / len(background)
    )
    provenance = {
        "source_vs_legacy": compare_populations(source_frame, legacy_bg),
        "source_vs_sy_resolved": compare_populations(source_frame, background),
        "legacy_vs_sy_resolved": compare_populations(legacy_bg, background),
        "source_path": str(args.data_dir / "qdm_checkerboard_eth_scatter.csv"),
    }
    write_json(output / "qdm_sy_provenance.json", provenance)
    audit = sanity._symmetry_audit(context)
    write_json(
        output / "qdm_sy_symmetry_audit.json",
        {
            "unitary_candidates": audit["unitary_candidate_rows"],
            "antiunitary_candidates": _antiunitary_audit(context),
            **resolution,
        },
    )
    means = _means(frame, context.tower_energy)
    means.to_csv(output / "qdm_sy_thermal_means.csv", index=False)
    verdict = {
        "parity_resolution": resolution,
        "branch_parity_best_label_agreement": agreement,
        "branch_classifier_is_quantum_number": False,
        "heavy_new_run_required": False,
        "source_provenance_resolved": provenance["source_vs_legacy"].get("population_matches")
        is True,
        "production_update_authorized_by_this_diagnostic": False,
        "stripe_concentration_status": "bounded_8x4_fixed_window_compared",
        "source_provenance_status": provenance["source_vs_legacy"]["status"],
    }
    write_json(output / "verdict.json", verdict)
    _plot(output, frame)
    lines = [
        "# QDM S_y follow-up",
        "",
        "Exact 8x4, electric winding (0,0), phi=0.05, lambda=1, Tdiag=i, Ty2=+1.",
        "Local A/Z placements and operator-norm normalization are recovered by the",
        "existing motif search.",
        "Validation tolerance 1e-8; energy-block tolerance 1e-9; deterministic dense",
        "solves, no random sampling.",
        "",
        "## Symmetry resolution",
        "",
        json.dumps(resolution, indent=2),
        "",
        f"PCA label / parity best agreement: {agreement:.6f}. PCA labels are a",
        "heuristic, not a quantum number.",
        "",
        "## Provenance",
        "",
        json.dumps(provenance, indent=2),
        "",
        "A row mismatch with matching energy-block witness traces is compatible with a different",
        "basis inside degeneracies; it does not establish changed physics. Failed",
        "block traces require",
        "checking the local motif, witness placement/normalization, Hamiltonian, and",
        "exclusion policy.",
        "The candidate population is never trimmed to fit source rows.",
        "",
        "## Next provisioning decision",
        "",
        "The legacy translation union is not a single spatial symmetry irrep. Use",
        "the target parity",
        "for an ETH comparison, and keep the union only as a clearly labelled companion.",
        "Full-union canonical traces survive basis rotations, but parity-conditioned",
        "means can differ.",
        "Parity-resolved stripe concentration is compared in the fixed |E-8|<=0.25 window only.",
        "The scan over phases, lengths and all production windows remains to be",
        "provisioned; no 12x4 solve",
        "or manuscript asset replacement is launched by this job.",
        "",
        "## Reproducibility",
        "",
        f"Spectrum source: {source}",
        f"Git: {git_metadata(ROOT)}",
        f"Command: {sys.argv}",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    write_json(output / "file_manifest.json", {"files": collect_file_manifest(output)})
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"[qdm-sy] completed: {output}", flush=True)
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experimental/data/evidence_jobs" / utc_run_id("qdm_sy_followup"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT
        / "experimental/data/evidence_jobs/qdm_checkerboard_primme_staged_20260825T164226Z",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "experimental/data/evidence_cache"
    )
    parser.add_argument("--allow-small-dense", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
