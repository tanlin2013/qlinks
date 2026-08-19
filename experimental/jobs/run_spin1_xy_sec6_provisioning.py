#!/usr/bin/env python
"""Run the restartable Spin-1 XY Sec. VI P0 provisioning workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from evidence_job_utils import (
    build_parser,
    find_repo_root,
    parse_float_tuple,
    parse_int_tuple,
    run_evidence_notebook,
    run_evidence_renderer,
)


DEFAULT_BASELINE = "experimental/data/evidence_jobs/spin1_production_20260806T074051Z"
DEFAULT_SPARSE_ADDENDUM = "experimental/data/evidence_jobs/spin1_production_20260810T082123Z"


def _repo_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = find_repo_root() / path
    return path.resolve(strict=False)


def main() -> None:
    parser = build_parser(description=__doc__ or "Run Sec. VI provisioning.")
    parser.add_argument(
        "--baseline-data-dir",
        default=DEFAULT_BASELINE,
        help="Authoritative dense production evidence directory (20260806 by default).",
    )
    parser.add_argument(
        "--sparse-convergence-data-dir",
        default=DEFAULT_SPARSE_ADDENDUM,
        help="Sparse-convergence addendum containing the completed 8192->10000 audit.",
    )
    parser.add_argument(
        "--checkpoint-source-dir",
        default=None,
        help="Optional existing spectral-checkpoint root to reuse before solving.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint root for newly solved spectra. Defaults to <data-dir>/checkpoints.",
    )
    parser.add_argument(
        "--dense-sizes",
        default="8,10,12",
        help="Even full-spectrum sizes used for raw microcanonical/two-bridge diagnostics.",
    )
    parser.add_argument("--large-size", type=int, default=14)
    parser.add_argument("--representative-eigenpairs", type=int, default=8192)
    parser.add_argument("--family-eigenpairs", type=int, default=8192)
    parser.add_argument(
        "--safe-fixed-widths",
        default="0.75,1.0",
        help="Contained fixed half-widths included in the bridge/window systematic.",
    )
    parser.add_argument(
        "--concentration-half-width",
        type=float,
        default=1.0,
        help="Contained half-width for the L=14 complete 19-operator covariance.",
    )
    parser.add_argument("--representative-kappa", type=float, default=0.10)
    parser.add_argument("--family-kappa", type=float, default=0.20)
    parser.add_argument("--shift", type=float, default=1.0e-7)
    parser.add_argument("--arpack-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--residual-chunk-size", type=int, default=64)
    parser.add_argument(
        "--run-family-l14",
        action="store_true",
        help=(
            "After the representative P0.1 solve, run the nonrepresentative L=14 kappa/J=0.20 "
            "family point. This is intentionally opt-in."
        ),
    )
    parser.add_argument(
        "--skip-large-representative",
        action="store_true",
        help=(
            "Run only dense postprocessing/bridges. Useful when the L=14 representative products "
            "have already been generated in another run."
        ),
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching spectral checkpoints before starting an eigensolve.",
    )
    parser.add_argument(
        "--write-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist energies/eigenvectors immediately after each expensive sparse solve.",
    )
    parser.add_argument(
        "--quarter-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the prefactor-1 L^(1/4) window as a systematic companion.",
    )
    args = parser.parse_args()

    if args.stage == "render":
        run_evidence_renderer(
            job_name="spin1_xy_sec6_provisioning",
            renderer_filename="render_spin1_xy_draft_figures.py",
            args=args,
        )
        return

    dense_sizes = parse_int_tuple(args.dense_sizes)
    safe_widths = parse_float_tuple(args.safe_fixed_widths)
    if not dense_sizes:
        raise ValueError("--dense-sizes must contain at least one size")
    if any(length < 8 or length % 2 for length in dense_sizes):
        raise ValueError("--dense-sizes must contain even L>=8")
    if args.large_size < 8 or args.large_size % 2:
        raise ValueError("--large-size must be an even integer >=8")
    if args.large_size > 14:
        raise ValueError(
            "Sec. VI P0 explicitly stops at L=14 until the two-bridge decomposition is examined."
        )
    if args.representative_eigenpairs < 2 or args.family_eigenpairs < 2:
        raise ValueError("sparse eigenpair budgets must be >=2")
    if not safe_widths or any(width <= 0.0 for width in safe_widths):
        raise ValueError("--safe-fixed-widths must contain positive values")
    if args.concentration_half_width <= 0.0:
        raise ValueError("--concentration-half-width must be positive")
    if args.shift == 0.0:
        raise ValueError("--shift must be nonzero")
    if args.residual_chunk_size < 1:
        raise ValueError("--residual-chunk-size must be >=1")

    baseline = _repo_path(args.baseline_data_dir)
    convergence = _repo_path(args.sparse_convergence_data_dir)
    checkpoint_source = _repo_path(args.checkpoint_source_dir)
    checkpoint_dir = _repo_path(args.checkpoint_dir)

    overrides = {
        "BASELINE_DATA_DIR": baseline,
        "SPARSE_CONVERGENCE_DATA_DIR": convergence,
        "CHECKPOINT_SOURCE_DIR": checkpoint_source,
        "CHECKPOINT_DIR": checkpoint_dir,
        "DENSE_SIZES": tuple(dense_sizes),
        "LARGE_SIZE": int(args.large_size),
        "REPRESENTATIVE_KAPPA_OVER_J": float(args.representative_kappa),
        "FAMILY_KAPPA_OVER_J": float(args.family_kappa),
        "REPRESENTATIVE_EIGENPAIRS": int(args.representative_eigenpairs),
        "FAMILY_EIGENPAIRS": int(args.family_eigenpairs),
        "SAFE_FIXED_HALF_WIDTHS": tuple(safe_widths),
        "INCLUDE_QUARTER_WINDOW": bool(args.quarter_window),
        "CONCENTRATION_HALF_WIDTH": float(args.concentration_half_width),
        "SHIFT": float(args.shift),
        "ARPACK_TOLERANCE": float(args.arpack_tolerance),
        "RESIDUAL_CHUNK_SIZE": int(args.residual_chunk_size),
        "REUSE_CHECKPOINTS": bool(args.reuse_checkpoints),
        "WRITE_CHECKPOINTS": bool(args.write_checkpoints),
        "RUN_LARGE_REPRESENTATIVE": (
            args.profile == "production" and not args.skip_large_representative
        ),
        "RUN_FAMILY_LARGE_SIZE": bool(args.run_family_l14),
    }
    data_dir = run_evidence_notebook(
        job_name="spin1_xy_sec6_provisioning",
        notebook_filename="spin1_xy_sec6_provisioning.ipynb",
        assignment_overrides=overrides,
        args=args,
    )

    if args.stage == "all":
        args.source_data_dir = data_dir
        run_evidence_renderer(
            job_name="spin1_xy_sec6_provisioning",
            renderer_filename="render_spin1_xy_draft_figures.py",
            args=args,
        )


if __name__ == "__main__":
    main()
