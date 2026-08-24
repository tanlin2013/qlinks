#!/usr/bin/env python
"""Run or render the square-QDM Sec. VII checkerboard evidence workflow."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from evidence_job_utils import (
    build_parser,
    find_repo_root,
    parse_float_tuple,
    parse_int_tuple,
    run_evidence_notebook,
    run_evidence_renderer,
)


def _repo_path(raw: str | Path | None, *, default: Path) -> Path:
    path = default if raw is None else Path(raw).expanduser()
    if not path.is_absolute():
        path = find_repo_root() / path
    return path.resolve(strict=False)


def _configure_resumable_spectrum(args: argparse.Namespace) -> Path:
    """Configure the opt-in stable cache and folded-spectrum backend."""

    repo_root = find_repo_root()
    cache_root = _repo_path(
        args.evidence_cache_root,
        default=repo_root / "experimental" / "data" / "evidence_cache",
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["QLINKS_EVIDENCE_CACHE_ROOT"] = str(cache_root)
    os.environ["QLINKS_EVIDENCE_CACHE_RESUME"] = "1" if args.resume_cache else "0"
    os.environ["QLINKS_EVIDENCE_CACHE_WRITE"] = "1" if args.write_cache else "0"
    os.environ["QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE"] = "1" if args.force_recompute_cache else "0"
    os.environ["QLINKS_QDM_FOLDED_BACKEND"] = str(args.large_strip_folded_backend)
    os.environ["QLINKS_QDM_RESUMABLE_SPECTRUM"] = "1"
    os.environ["QLINKS_QDM_PRIMME_WARM_START_VECTORS"] = str(int(args.primme_warm_start_vectors))
    os.environ["QLINKS_QDM_PRIMME_METHOD"] = str(args.primme_method)
    os.environ["QLINKS_QDM_PRIMME_MAX_BLOCK_SIZE"] = str(int(args.primme_max_block_size))

    # ``sitecustomize`` must be importable when the notebook kernel starts, not
    # only after the notebook later inserts experimental/jobs into sys.path.
    jobs_dir = repo_root / "experimental" / "jobs"
    resume_site = jobs_dir / "qdm_resume_site"
    inherited = os.environ.get("PYTHONPATH")
    parts = [str(resume_site), str(jobs_dir)]
    if inherited:
        parts.append(inherited)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    if args.large_strip_folded_backend == "primme" and importlib.util.find_spec("primme") is None:
        raise RuntimeError(
            "--large-strip-folded-backend primme requires the PRIMME evidence image. "
            "Build it with scripts/docker/build_primme_evidence_image.sh and set "
            "QLINKS_DOCKER_IMAGE=tanlin2013/qlinks:notebook-primme."
        )
    return cache_root


def main() -> None:
    parser = build_parser(description=__doc__ or "Run square-QDM checkerboard evidence job.")
    parser.add_argument(
        "--transport-repeats", default=None, help="Local checkerboard-family repeats, e.g. 1,2,3."
    )
    parser.add_argument(
        "--microcanonical-repeats",
        "--ed-repeats",
        dest="ed_repeats",
        default=None,
        help=(
            "Full energy-resolved repeats. Production-safe default: 1,2. "
            "Repeat 3 is disabled unless explicitly acknowledged."
        ),
    )
    parser.add_argument("--transfer-max-length", type=int, default=None)
    parser.add_argument(
        "--phase-values", default=None, help="Checkerboard pilot grid including endpoint control."
    )
    parser.add_argument(
        "--positive-phase-values", default=None, help="Principal positive phase grid."
    )
    parser.add_argument("--representative-phase", type=float, default=None)
    parser.add_argument(
        "--thermal-protocol", choices=("auto", "beta0", "finite-beta"), default="auto"
    )
    parser.add_argument("--energy-match-tolerance", type=float, default=None)
    parser.add_argument("--window-prefactors", default=None)
    parser.add_argument("--primary-window-prefactor", type=float, default=None)
    parser.add_argument("--energy-block-tolerance", type=float, default=None)
    parser.add_argument(
        "--symmetry-chunk-size",
        type=int,
        default=None,
        help="Chunk size for 12x4 checkerboard symmetry permutations.",
    )
    parser.add_argument("--large-strip-repeats", default=None)
    parser.add_argument("--run-large-strip", action="store_true")
    parser.add_argument(
        "--large-strip-spectral-method",
        choices=("folded", "shift-invert"),
        default=None,
        help=(
            "Interior-spectrum method. Production default is factorization-free folded spectrum; "
            "the folded eigensolver backend is selected separately. "
            "shift-invert is diagnostic-only and requires --allow-direct-lu-shift-invert."
        ),
    )
    parser.add_argument(
        "--large-strip-folded-backend",
        choices=("auto", "arpack", "primme"),
        default="auto",
        help=(
            "Eigensolver used for the folded operator. auto selects PRIMME when installed and "
            "otherwise SciPy/ARPACK. Every completed budget is cached independently."
        ),
    )
    parser.add_argument(
        "--evidence-cache-root",
        type=Path,
        default=None,
        help=(
            "Stable reusable cache root. Defaults to experimental/data/evidence_cache, separate "
            "from timestamped evidence-job attempt directories."
        ),
    )
    parser.add_argument(
        "--resume-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate and reuse compatible completed spectral budgets before solving.",
    )
    parser.add_argument(
        "--write-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Persist every completed folded-spectrum budget immediately after validation data "
            "exist."
        ),
    )
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        help="Ignore compatible final checkpoints for this run but keep writing new checkpoints.",
    )
    parser.add_argument(
        "--primme-warm-start-vectors",
        type=int,
        default=256,
        help=(
            "Maximum vectors reused from the largest compatible lower-budget checkpoint by PRIMME."
        ),
    )
    parser.add_argument(
        "--primme-method",
        default="PRIMME_DYNAMIC",
        help="PRIMME eigensolver method name used by the optional folded-spectrum backend.",
    )
    parser.add_argument(
        "--primme-max-block-size",
        type=int,
        default=0,
        help="Optional PRIMME maxBlockSize override. Zero keeps PRIMME's own default.",
    )
    parser.add_argument(
        "--large-strip-subspace-budgets",
        default=None,
        help=(
            "Comma-separated folded-spectrum requested eigenpair budgets. The workflow escalates "
            "until all requested windows are covered and then performs one extra budget when "
            "requested."
        ),
    )
    parser.add_argument("--large-strip-folded-tolerance", type=float, default=None)
    parser.add_argument("--large-strip-folded-ncv-factor", type=float, default=None)
    parser.add_argument("--large-strip-folded-random-seed", type=int, default=None)
    parser.add_argument("--large-strip-convergence-tolerance", type=float, default=None)
    parser.add_argument(
        "--allow-direct-lu-shift-invert",
        action="store_true",
        help="Explicitly allow the memory-heavy SuperLU shift-invert diagnostic backend.",
    )
    parser.add_argument(
        "--large-strip-canonical-only",
        action="store_true",
        help="Checkpoint the large-strip canonical target and stop before any spectral solve.",
    )
    parser.add_argument("--large-strip-eigenpairs", type=int, default=None)
    parser.add_argument(
        "--large-strip-eigenpair-budgets",
        default=None,
        help=(
            "Legacy comma-separated budget ladder. With the folded method these values are "
            "treated as subspace sizes; with shift-invert they remain requested eigenpair counts."
        ),
    )
    parser.add_argument(
        "--large-strip-extra-convergence-step",
        action="store_true",
        help=(
            "After first full window coverage, run one additional budget from the ladder as "
            "a solver cross-check."
        ),
    )
    parser.add_argument(
        "--allow-extreme-large-strip",
        action="store_true",
        help=(
            "Allow repeats >=4 (16x4 and larger). This is not production-feasible by default: "
            "the 16x4 zero-winding basis already contains about 4.59e8 states."
        ),
    )
    parser.add_argument("--large-strip-sigma-offset", type=float, default=None)
    parser.add_argument("--large-strip-eig-tolerance", type=float, default=None)
    parser.add_argument("--large-strip-maxiter", type=int, default=None)
    parser.add_argument("--finite-beta-samples", type=int, default=None)
    parser.add_argument("--finite-beta-beta-max", type=float, default=None)
    parser.add_argument("--finite-beta-beta-points", type=int, default=None)
    parser.add_argument("--finite-beta-random-seed", type=int, default=None)
    parser.add_argument("--dark-classification-repeats", default=None)
    parser.add_argument("--skip-dark-manifold-classification", action="store_true")
    parser.add_argument("--large-strip-phase-check-values", default=None)
    parser.add_argument("--skip-checkerboard-thermal-scan", action="store_true")
    parser.add_argument("--skip-checkerboard-concentration", action="store_true")
    parser.add_argument(
        "--allow-large-dense-ed",
        action="store_true",
        help="Acknowledge the repeat>=3 dense-ED memory risk.",
    )
    args = parser.parse_args()
    if args.primme_warm_start_vectors < 0:
        raise ValueError("--primme-warm-start-vectors must be >=0")
    if args.primme_max_block_size < 0:
        raise ValueError("--primme-max-block-size must be >=0")

    if args.stage == "render":
        run_evidence_renderer(
            job_name="square_qdm_draft_evidence",
            renderer_filename="render_square_qdm_draft_figures.py",
            args=args,
        )
        return

    cache_root = _configure_resumable_spectrum(args)
    print(
        {
            "stable_evidence_cache": str(cache_root),
            "resume_cache": bool(args.resume_cache),
            "write_cache": bool(args.write_cache),
            "force_recompute_cache": bool(args.force_recompute_cache),
            "folded_backend": args.large_strip_folded_backend,
        },
        flush=True,
    )

    ed = parse_int_tuple(args.ed_repeats)
    if ed is not None and max(ed) >= 3 and not args.allow_large_dense_ed:
        raise ValueError(
            "QDM full ED repeat >=3 is disabled: the current algorithm can exceed 400 GiB. "
            "Use repeats 1,2, or implement a controlled partial-spectrum/typicality method."
        )
    transport = parse_int_tuple(args.transport_repeats)
    phases = parse_float_tuple(args.phase_values)
    positive = parse_float_tuple(args.positive_phase_values)
    prefactors = parse_float_tuple(args.window_prefactors)
    large = parse_int_tuple(args.large_strip_repeats)
    large_budgets = parse_int_tuple(args.large_strip_eigenpair_budgets)
    large_subspace_budgets = parse_int_tuple(args.large_strip_subspace_budgets)
    dark_classification = parse_int_tuple(args.dark_classification_repeats)
    large_phase_check = parse_float_tuple(args.large_strip_phase_check_values)
    overrides = {
        "SAVE_FIGURES": bool(args.figure_formats.strip()) and args.stage != "compute",
        "SAVE_PDF": "pdf" in {p.strip().lower() for p in args.figure_formats.split(",")},
        "RUN_CHECKERBOARD_THERMAL_SCAN": not args.skip_checkerboard_thermal_scan,
        "RUN_CHECKERBOARD_CONCENTRATION": not args.skip_checkerboard_concentration,
        "RUN_LARGE_STRIP": bool(args.run_large_strip),
        "LARGE_STRIP_CANONICAL_ONLY": bool(args.large_strip_canonical_only),
        "RUN_DARK_MANIFOLD_CLASSIFICATION": not args.skip_dark_manifold_classification,
        "CHECKERBOARD_THERMAL_PROTOCOL": args.thermal_protocol,
        "STRICT_CLAIMS": bool(args.strict_claims),
    }
    if ed is not None:
        overrides["CHECKERBOARD_ED_REPEATS"] = ed
    if transport is not None:
        overrides["CHECKERBOARD_TRANSPORT_REPEATS"] = transport
    if phases is not None:
        overrides["CHECKERBOARD_PHASE_VALUES"] = phases
    if positive is not None:
        overrides["CHECKERBOARD_POSITIVE_PHASE_VALUES"] = positive
    if prefactors is not None:
        overrides["MICROCANONICAL_PREFACTORS"] = prefactors
    if large is not None:
        if max(large) >= 4 and not args.allow_extreme_large_strip:
            raise ValueError(
                "QDM large-strip production is capped at repeats=3 (12x4). "
                "The 16x4 zero-winding space is about 4.59e8 states; use "
                "--allow-extreme-large-strip only for an explicit feasibility experiment."
            )
        overrides["LARGE_STRIP_REPEATS"] = large
    if large_budgets is not None:
        if any(value < 4 for value in large_budgets):
            raise ValueError("--large-strip-eigenpair-budgets entries must be at least four")
        legacy_budgets = tuple(sorted(set(large_budgets)))
        overrides["LARGE_STRIP_EIGENPAIR_BUDGETS"] = legacy_budgets
        if large_subspace_budgets is None:
            overrides["LARGE_STRIP_SUBSPACE_BUDGETS"] = legacy_budgets
    if large_subspace_budgets is not None:
        if any(value < 4 for value in large_subspace_budgets):
            raise ValueError("--large-strip-subspace-budgets entries must be at least four")
        overrides["LARGE_STRIP_SUBSPACE_BUDGETS"] = tuple(sorted(set(large_subspace_budgets)))
    if args.large_strip_spectral_method is not None:
        if (
            args.large_strip_spectral_method == "shift-invert"
            and not args.allow_direct_lu_shift_invert
        ):
            raise ValueError(
                "direct-LU shift-invert is disabled after the 12x4 SuperLU MemoryError; "
                "use --allow-direct-lu-shift-invert only for an explicit diagnostic retry"
            )
        overrides["LARGE_STRIP_SPECTRAL_METHOD"] = str(args.large_strip_spectral_method)
    if args.large_strip_folded_tolerance is not None:
        if args.large_strip_folded_tolerance <= 0:
            raise ValueError("--large-strip-folded-tolerance must be positive")
        overrides["LARGE_STRIP_FOLDED_TOL"] = float(args.large_strip_folded_tolerance)
    if args.large_strip_folded_ncv_factor is not None:
        if args.large_strip_folded_ncv_factor <= 1.0:
            raise ValueError("--large-strip-folded-ncv-factor must exceed one")
        overrides["LARGE_STRIP_FOLDED_NCV_FACTOR"] = float(args.large_strip_folded_ncv_factor)
    if args.large_strip_folded_random_seed is not None:
        overrides["LARGE_STRIP_FOLDED_RANDOM_SEED"] = int(args.large_strip_folded_random_seed)
    if args.large_strip_convergence_tolerance is not None:
        if args.large_strip_convergence_tolerance <= 0:
            raise ValueError("--large-strip-convergence-tolerance must be positive")
        overrides["LARGE_STRIP_METHOD_CONVERGENCE_TOL"] = float(
            args.large_strip_convergence_tolerance
        )
    overrides["LARGE_STRIP_EXTRA_CONVERGENCE_STEP"] = bool(args.large_strip_extra_convergence_step)
    if dark_classification is not None:
        overrides["DARK_CLASSIFICATION_REPEATS"] = dark_classification
    if large_phase_check is not None:
        overrides["LARGE_STRIP_PHASE_CHECK_VALUES"] = large_phase_check
    if args.large_strip_eigenpairs is not None:
        if args.large_strip_eigenpairs < 4:
            raise ValueError("--large-strip-eigenpairs must be at least four")
        overrides["LARGE_STRIP_EIGENPAIRS"] = int(args.large_strip_eigenpairs)
    if args.large_strip_sigma_offset is not None:
        overrides["LARGE_STRIP_SIGMA_OFFSET"] = float(args.large_strip_sigma_offset)
    if args.large_strip_eig_tolerance is not None:
        overrides["LARGE_STRIP_EIG_TOL"] = float(args.large_strip_eig_tolerance)
    if args.large_strip_maxiter is not None:
        overrides["LARGE_STRIP_MAXITER"] = int(args.large_strip_maxiter)
    if args.finite_beta_samples is not None:
        if args.finite_beta_samples < 2:
            raise ValueError("--finite-beta-samples must be at least two")
        overrides["FINITE_BETA_TYPICALITY_SAMPLES"] = int(args.finite_beta_samples)
    if args.finite_beta_beta_max is not None:
        if args.finite_beta_beta_max <= 0:
            raise ValueError("--finite-beta-beta-max must be positive")
        overrides["FINITE_BETA_BETA_MAX"] = float(args.finite_beta_beta_max)
    if args.finite_beta_beta_points is not None:
        if args.finite_beta_beta_points < 3:
            raise ValueError("--finite-beta-beta-points must be at least three")
        overrides["FINITE_BETA_BETA_POINTS"] = int(args.finite_beta_beta_points)
    if args.finite_beta_random_seed is not None:
        overrides["FINITE_BETA_RANDOM_SEED"] = int(args.finite_beta_random_seed)
    if args.transfer_max_length is not None:
        if args.transfer_max_length < 4 or args.transfer_max_length % 4:
            raise ValueError("--transfer-max-length must be a multiple of four >=4")
        overrides["CHECKERBOARD_TRANSFER_MAX_LENGTH"] = int(args.transfer_max_length)
    if args.representative_phase is not None:
        overrides["CHECKERBOARD_REPRESENTATIVE_PHASE"] = float(args.representative_phase)
    if args.energy_match_tolerance is not None:
        overrides["CHECKERBOARD_ENERGY_MATCH_TOL"] = float(args.energy_match_tolerance)
    if args.primary_window_prefactor is not None:
        overrides["PRIMARY_WINDOW_PREFACTOR"] = float(args.primary_window_prefactor)
    if args.energy_block_tolerance is not None:
        overrides["ENERGY_BLOCK_TOL"] = float(args.energy_block_tolerance)
    if args.symmetry_chunk_size is not None:
        if args.symmetry_chunk_size <= 0:
            raise ValueError("--symmetry-chunk-size must be positive")
        overrides["CHECKERBOARD_SYMMETRY_CHUNK_SIZE"] = int(args.symmetry_chunk_size)
    run_evidence_notebook(
        job_name="square_qdm_draft_evidence",
        notebook_filename="square_qdm_draft_evidence.ipynb",
        assignment_overrides=overrides,
        args=args,
    )


if __name__ == "__main__":
    main()
