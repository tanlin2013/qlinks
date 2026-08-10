#!/usr/bin/env python
"""Run or render the square-QDM Sec. VII checkerboard evidence workflow."""

from __future__ import annotations

from evidence_job_utils import (
    build_parser,
    parse_float_tuple,
    parse_int_tuple,
    run_evidence_notebook,
    run_evidence_renderer,
)


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
    parser.add_argument("--large-strip-repeats", default=None)
    parser.add_argument("--run-large-strip", action="store_true")
    parser.add_argument("--large-strip-eigenpairs", type=int, default=None)
    parser.add_argument(
        "--large-strip-eigenpair-budgets",
        default=None,
        help=(
            "Comma-separated shift-invert budget ladder for the 12x4 strip. "
            "The notebook checkpoints each attempt and stops after the first budget that "
            "fully covers every requested window unless an extra convergence step is requested."
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
    if args.stage == "render":
        run_evidence_renderer(
            job_name="square_qdm_draft_evidence",
            renderer_filename="render_square_qdm_draft_figures.py",
            args=args,
        )
        return

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
    dark_classification = parse_int_tuple(args.dark_classification_repeats)
    large_phase_check = parse_float_tuple(args.large_strip_phase_check_values)
    overrides = {
        "SAVE_FIGURES": bool(args.figure_formats.strip()) and args.stage != "compute",
        "SAVE_PDF": "pdf" in {p.strip().lower() for p in args.figure_formats.split(",")},
        "RUN_CHECKERBOARD_THERMAL_SCAN": not args.skip_checkerboard_thermal_scan,
        "RUN_CHECKERBOARD_CONCENTRATION": not args.skip_checkerboard_concentration,
        "RUN_LARGE_STRIP": bool(args.run_large_strip),
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
        overrides["LARGE_STRIP_EIGENPAIR_BUDGETS"] = tuple(sorted(set(large_budgets)))
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
    run_evidence_notebook(
        job_name="square_qdm_draft_evidence",
        notebook_filename="square_qdm_draft_evidence.ipynb",
        assignment_overrides=overrides,
        args=args,
    )


if __name__ == "__main__":
    main()
