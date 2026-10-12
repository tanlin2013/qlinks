"""Isolated small-system parity preflight and folded-backend comparison worker."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdm_checkerboard_large_strip import folded_spectrum_partial_spectrum
from qdm_parity_evidence import regenerate_exact
from qdm_parity_protocol import PROTOCOL
from qdm_resumable_spectrum import make_resumable_folded_solver
from qdm_sec7_fixed_o1 import REPRESENTATIVE_PHASE, build_context, recover_reference_geometry
from qdm_sec7_fixed_o1_l12_spectrum import coverage_metrics


def compare_backend(
    backend: str,
    *,
    output: Path,
    cache_root: Path,
    repeats: list[int],
    budgets: list[int],
    tolerance: float,
) -> pd.DataFrame:
    """Compare timed solves in measured target irreps, without importing legacy notebooks.

    The 4x4 target irrep has dimension one and is a dense symmetry check.
    Only the nontrivial 8x4 irrep exercises the iterative backends. The caller
    must first run the complete exact parity preflight for both sizes.
    """

    if backend not in {"arpack", "primme"} or any(r not in (1, 2) for r in repeats):
        raise ValueError("backend comparison is restricted to arpack/primme and 4x4/8x4")
    if not repeats or not budgets or any(b <= 0 for b in budgets):
        raise ValueError("backend comparison requires sizes and positive budgets")
    os.environ.update(
        QLINKS_QDM_FOLDED_BACKEND=backend,
        QLINKS_EVIDENCE_CACHE_ROOT=str(cache_root),
        QLINKS_EVIDENCE_CACHE_RESUME="0",
        QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE="1",
        QLINKS_EVIDENCE_CACHE_WRITE="1",
    )
    solver = make_resumable_folded_solver(folded_spectrum_partial_spectrum)
    reference = recover_reference_geometry()
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for repeat in repeats:
        context = build_context(
            reference=reference,
            repeats=repeat,
            phase=REPRESENTATIVE_PHASE,
            sy_character="target",
        )
        for budget in budgets:
            started = time.perf_counter()
            partial = solver(
                context.h_sector,
                target_energy=context.tower_energy,
                subspace_size=budget,
                tolerance=tolerance,
                random_seed=20261012 + budget,
            )
            vectors = partial.eigenvectors
            gram_error = float(
                np.linalg.norm(
                    vectors.conj().T @ vectors - np.eye(vectors.shape[1]),
                    ord=2,
                )
            )
            if partial.maximum_residual > 1e-6 or gram_error > 1e-6:
                raise RuntimeError(f"{backend} physical residual/full-Gram check failed")
            rows.append(
                {
                    "repeats": repeat,
                    "Lx": context.lx,
                    "window_half_width": 0.5,
                    "requested_subspace_size": budget,
                    "sector_dimension": context.sector.sector_dimension,
                    "Sy": context.sector.labels["Sy_character"],
                    "symmetry_protocol": PROTOCOL,
                    "parity_policy": "measured_target",
                    "comparison_role": "dense_symmetry_check"
                    if context.lx == 4
                    else "iterative_backend",
                    "returned_eigenpairs": len(partial.energies),
                    "spectral_method": partial.method,
                    "solver_status": "completed",
                    "partial_min_energy": partial.min_energy,
                    "partial_max_energy": partial.max_energy,
                    "partial_maximum_residual": partial.maximum_residual,
                    "full_Gram_residual": gram_error,
                    "transformed_maximum_residual": float(np.max(partial.transformed_residuals)),
                    "runtime_seconds": time.perf_counter() - started,
                    "peak_rss_gib": partial.peak_rss_gib,
                    **coverage_metrics(
                        partial.energies,
                        partial.residuals,
                        target_energy=context.tower_energy,
                        half_width=0.5,
                        solver_tolerance=tolerance,
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(
                output / "qdm_checkerboard_L12_spectral_convergence.csv", index=False
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("preflight", "arpack", "primme"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--repeats", default="1,2")
    parser.add_argument("--budgets", default="16,32")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == "preflight":
        # Full eigensystem/Gram, measured parity, target darkness, thermal
        # witnesses and joint-dark covariance checks remain mandatory.
        regenerate_exact(args.output, phases=(REPRESENTATIVE_PHASE,))
    else:
        if args.cache_root is None:
            parser.error("backend solves require --cache-root")
        compare_backend(
            args.backend,
            output=args.output,
            cache_root=args.cache_root,
            repeats=list(map(int, args.repeats.split(","))),
            budgets=list(map(int, args.budgets.split(","))),
            tolerance=args.tolerance,
        )


if __name__ == "__main__":
    main()
