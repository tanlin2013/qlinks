#!/usr/bin/env python
"""Run or render the Spin-1 XY Sec. 6 evidence workflow."""

from __future__ import annotations

import argparse

from evidence_job_utils import (
    build_parser,
    parse_float_tuple,
    parse_int_tuple,
    run_evidence_notebook,
    run_evidence_renderer,
)


def main() -> None:
    parser = build_parser(description=__doc__ or "Run Spin-1 XY evidence job.")
    parser.add_argument(
        "--microcanonical-sizes",
        "--sizes",
        dest="microcanonical_sizes",
        default=None,
        help="Comma-separated full-spectrum dense-ED sizes. Keep L=14 in "
        "--large-size-sizes so it uses the partial-spectrum path.",
    )
    parser.add_argument(
        "--deformation-sizes",
        default=None,
        help="Comma-separated sizes used for preserving-neighborhood scans; "
        "may be smaller than ED sizes.",
    )
    parser.add_argument(
        "--large-size-sizes",
        default=None,
        help=(
            "Comma-separated partial-spectrum sizes. Production defaults to 14; "
            "these sizes are not added to the full deformation grid."
        ),
    )
    parser.add_argument(
        "--large-size-eigenpairs",
        type=int,
        default=None,
        help="Legacy single shift-invert eigenpair budget near E=0.",
    )
    parser.add_argument(
        "--large-size-eigenpair-budgets",
        default=None,
        help=(
            "Comma-separated independent shift-invert budgets for the L=14 "
            "solver-convergence study, e.g. 10000,12000. Each budget is run "
            "from scratch and checkpointed before the next budget."
        ),
    )
    parser.add_argument(
        "--large-size-safe-fixed-widths",
        default=None,
        help=(
            "Comma-separated deliberately safe fixed half-widths used for the "
            "large-size convergence table, e.g. 0.75,1.0."
        ),
    )
    parser.add_argument(
        "--large-size-quarter-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include the prefactor-1 L^(1/4) safe window in the L=14 convergence audit.",
    )
    parser.add_argument(
        "--large-size-concentration-half-width",
        type=float,
        default=None,
        help="Fully covered fixed half-width used for the L=14 19-operator covariance (default 1).",
    )
    parser.add_argument(
        "--large-size-baseline-data-dir",
        default=None,
        help=(
            "Optional repository-relative evidence directory containing the authoritative "
            "lower-budget L=14 run. When present, its fully covered safe-window rows are "
            "prepended to the sparse convergence table."
        ),
    )
    parser.add_argument(
        "--large-size-family-kappa-values",
        default=None,
        help=(
            "Additional L=14 compatible-family couplings for the larger-size envelope, "
            "normally 0.2 after the representative convergence run."
        ),
    )
    parser.add_argument(
        "--large-size-family-eigenpairs",
        type=int,
        default=None,
        help="Shift-invert eigenpair budget for each additional L=14 family point.",
    )
    parser.add_argument(
        "--allow-extreme-large-size",
        action="store_true",
        help=(
            "Allow partial-spectrum sizes above L=14. This is experimental: L=16 has "
            "about 2.7e5 states in the target momentum sector and is not a production default."
        ),
    )
    parser.add_argument(
        "--large-size-shift",
        type=float,
        default=None,
        help="Nonzero ARPACK shift used to avoid singular factorization at the exact tower energy.",
    )
    parser.add_argument(
        "--large-size-concentration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compute the complete 19-operator covariance diagnostic at the "
            "large-size representative point. Production defaults to enabled; "
            "use --no-large-size-concentration only for a preflight run."
        ),
    )
    parser.add_argument(
        "--window-exponents",
        default=None,
        help="Comma-separated microcanonical width exponents, e.g. 0.5,0.25,0.",
    )
    parser.add_argument(
        "--window-prefactors",
        default=None,
        help="Comma-separated window prefactors shared by every width exponent.",
    )
    parser.add_argument(
        "--fit-bootstrap-repeats",
        type=int,
        default=None,
        help="Number of window-systematic bootstrap replicates for revised matching fits.",
    )
    parser.add_argument(
        "--counting-max-length",
        type=int,
        default=None,
        help="Largest even length for the exact fixed-M beta-zero counting sequence.",
    )
    parser.add_argument(
        "--finite-d", type=float, default=None, help="Generic T1 single-ion anisotropy."
    )
    parser.add_argument(
        "--j3-values",
        default=None,
        help="Deprecated supplementary real-J3 scan values.",
    )
    parser.add_argument(
        "--kappa-values",
        default=None,
        help=(
            "Comma-separated imaginary second-neighbor kappa/J values along "
            "the compatible family anchored at J3/J=0.1."
        ),
    )
    parser.add_argument(
        "--representative-kappa",
        type=float,
        default=None,
        help=(
            "Interior representative coupling kappa_star/J used for the detailed "
            "ETH scatter, finite-size matching, and L=14 point."
        ),
    )
    parser.add_argument(
        "--principal-kappa-values",
        default=None,
        help=(
            "Comma-separated positive compatible couplings defining the main "
            "family-wide matching and concentration interval."
        ),
    )
    parser.add_argument(
        "--deformed-type1-kappa-values",
        default=None,
        help="Representative kappa/J points where reference Type-1 states are continuation-tested.",
    )
    parser.add_argument(
        "--obstruction-grid-points",
        type=int,
        default=None,
        help="Number of points per axis in the complex-t2 residual heatmap.",
    )
    parser.add_argument(
        "--obstruction-t2-bound",
        type=float,
        default=None,
        help="Symmetric |Re t2/J|, |Im t2/J| bound for the residual heatmap.",
    )
    parser.add_argument(
        "--exceptional-projector-mode",
        choices=("type1", "target-only"),
        default="type1",
        help=(
            "Legacy provenance option. The primary exceptional projector is the "
            "translated joint-dark kernel; Type-1 search is used only at kappa=0."
        ),
    )
    parser.add_argument(
        "--skip-protocol-m",
        action="store_true",
        help="Deprecated: the primary evidence workflow requires undeformed cage excision.",
    )
    parser.add_argument("--skip-background-concentration", action="store_true")
    parser.add_argument("--skip-deformation-concentration", action="store_true")
    parser.add_argument("--skip-complex-hermitian-path", action="store_true")
    parser.add_argument("--skip-joint-continuation", action="store_true")
    parser.add_argument("--skip-deformed-type1-inventory", action="store_true")
    args = parser.parse_args()
    if args.skip_protocol_m:
        raise ValueError(
            "--skip-protocol-m is no longer supported by the manuscript evidence job: "
            "T1 requires the representative-point joint-dark-cleaned ensemble. Use "
            "--exceptional-projector-mode target-only only for an explicitly diagnostic run."
        )

    if args.stage == "render":
        run_evidence_renderer(
            job_name="spin1_xy_draft_evidence",
            renderer_filename="render_spin1_xy_draft_figures.py",
            args=args,
        )
        return

    micro_sizes = parse_int_tuple(args.microcanonical_sizes)
    deformation_sizes = parse_int_tuple(args.deformation_sizes)
    large_sizes = parse_int_tuple(args.large_size_sizes)
    large_budgets = parse_int_tuple(args.large_size_eigenpair_budgets)
    large_safe_widths = parse_float_tuple(args.large_size_safe_fixed_widths)
    large_family_kappas = parse_float_tuple(args.large_size_family_kappa_values)
    window_exponents = parse_float_tuple(args.window_exponents)
    window_prefactors = parse_float_tuple(args.window_prefactors)
    j3_values = parse_float_tuple(args.j3_values)
    kappa_values = parse_float_tuple(args.kappa_values)
    principal_kappa_values = parse_float_tuple(args.principal_kappa_values)
    type1_kappa_values = parse_float_tuple(args.deformed_type1_kappa_values)
    overrides = {
        "RUN_BACKGROUND_CONCENTRATION": not args.skip_background_concentration,
        "RUN_DEFORMATION_CONCENTRATION": not args.skip_deformation_concentration,
        "RUN_COMPLEX_HERMITIAN_PATH": not args.skip_complex_hermitian_path,
        "RUN_JOINT_CONTINUATION_CROSSCHECK": not args.skip_joint_continuation,
        "RUN_DEFORMED_TYPE1_INVENTORY": not args.skip_deformed_type1_inventory,
        "EXCEPTIONAL_PROJECTOR_MODE": args.exceptional_projector_mode,
    }
    if args.large_size_concentration is not None:
        overrides["RUN_LARGE_SIZE_CONCENTRATION"] = bool(args.large_size_concentration)
    if micro_sizes is not None:
        overrides["MICROCANONICAL_SIZES"] = micro_sizes
        overrides["SIZES"] = micro_sizes
    if deformation_sizes is not None:
        overrides["DEFORMATION_SIZES"] = deformation_sizes
    if large_sizes is not None:
        if max(large_sizes) > 14 and not args.allow_extreme_large_size:
            raise ValueError(
                "Spin-1 partial-spectrum production is capped at L=14. "
                "Use --allow-extreme-large-size only for an explicitly experimental L>=16 run."
            )
        overrides["LARGE_SIZE_SIZES"] = large_sizes
    if large_budgets is not None:
        if any(value < 2 for value in large_budgets):
            raise ValueError("--large-size-eigenpair-budgets entries must be at least 2")
        overrides["LARGE_SIZE_EIGENPAIR_BUDGETS"] = tuple(sorted(set(large_budgets)))
    if large_safe_widths is not None:
        if any(value <= 0.0 for value in large_safe_widths):
            raise ValueError("--large-size-safe-fixed-widths entries must be positive")
        overrides["LARGE_SIZE_SAFE_FIXED_HALF_WIDTHS"] = large_safe_widths
    if args.large_size_quarter_window is not None:
        overrides["LARGE_SIZE_INCLUDE_QUARTER_WINDOW"] = bool(args.large_size_quarter_window)
    if args.large_size_concentration_half_width is not None:
        if args.large_size_concentration_half_width <= 0.0:
            raise ValueError("--large-size-concentration-half-width must be positive")
        overrides["LARGE_SIZE_CONCENTRATION_HALF_WIDTH"] = float(
            args.large_size_concentration_half_width
        )
    if args.large_size_baseline_data_dir is not None:
        overrides["LARGE_SIZE_BASELINE_DATA_DIR"] = args.large_size_baseline_data_dir
    if large_family_kappas is not None:
        if any(value <= 0.0 for value in large_family_kappas):
            raise ValueError("--large-size-family-kappa-values must be positive")
        overrides["LARGE_SIZE_FAMILY_KAPPA_VALUES"] = large_family_kappas
    if args.large_size_family_eigenpairs is not None:
        if args.large_size_family_eigenpairs < 2:
            raise ValueError("--large-size-family-eigenpairs must be at least 2")
        overrides["LARGE_SIZE_FAMILY_EIGENPAIRS"] = int(args.large_size_family_eigenpairs)
    if args.large_size_eigenpairs is not None:
        if args.large_size_eigenpairs < 2:
            raise ValueError("--large-size-eigenpairs must be at least 2")
        overrides["LARGE_SIZE_EIGENPAIRS"] = int(args.large_size_eigenpairs)
    if args.large_size_shift is not None:
        if args.large_size_shift == 0.0:
            raise ValueError("--large-size-shift must be nonzero")
        overrides["LARGE_SIZE_SHIFT"] = float(args.large_size_shift)
    if window_exponents is not None:
        if any(value < 0.0 or value >= 1.0 for value in window_exponents):
            raise ValueError("--window-exponents must satisfy 0 <= alpha < 1")
        overrides["WINDOW_SCALING_EXPONENTS"] = window_exponents
    if window_prefactors is not None:
        if any(value <= 0.0 for value in window_prefactors):
            raise ValueError("--window-prefactors must be positive")
        overrides["WINDOW_PREFACTORS"] = window_prefactors
    if args.fit_bootstrap_repeats is not None:
        if args.fit_bootstrap_repeats < 0:
            raise ValueError("--fit-bootstrap-repeats must be nonnegative")
        overrides["FIT_BOOTSTRAP_REPEATS"] = int(args.fit_bootstrap_repeats)
    if args.counting_max_length is not None:
        if args.counting_max_length < 4 or args.counting_max_length % 2:
            raise ValueError("--counting-max-length must be an even integer >= 4")
        overrides["COUNTING_LENGTHS"] = tuple(range(4, args.counting_max_length + 1, 2))
    if args.finite_d is not None:
        overrides["D_THERMAL"] = float(args.finite_d)
    if j3_values is not None:
        overrides["PRESERVING_J3_PATH"] = j3_values
    if kappa_values is not None:
        overrides["KAPPA_OVER_J_PATH"] = kappa_values
    if args.representative_kappa is not None:
        if not __import__("math").isfinite(args.representative_kappa):
            raise ValueError("--representative-kappa must be finite")
        overrides["REPRESENTATIVE_KAPPA_OVER_J"] = float(args.representative_kappa)
    if principal_kappa_values is not None:
        if any(value <= 0.0 for value in principal_kappa_values):
            raise ValueError("--principal-kappa-values must be strictly positive")
        overrides["PRINCIPAL_KAPPA_OVER_J_PATH"] = principal_kappa_values
    if type1_kappa_values is not None:
        overrides["DEFORMED_TYPE1_KAPPA_VALUES"] = type1_kappa_values
    if args.obstruction_grid_points is not None:
        if args.obstruction_grid_points < 3:
            raise ValueError("--obstruction-grid-points must be at least 3")
        overrides["OBSTRUCTION_GRID_POINTS"] = int(args.obstruction_grid_points)
    if args.obstruction_t2_bound is not None:
        if args.obstruction_t2_bound <= 0:
            raise ValueError("--obstruction-t2-bound must be positive")
        overrides["OBSTRUCTION_T2_BOUND"] = float(args.obstruction_t2_bound)

    run_evidence_notebook(
        job_name="spin1_xy_draft_evidence",
        notebook_filename="spin1_xy_draft_evidence.ipynb",
        assignment_overrides=overrides,
        args=args,
    )


if __name__ == "__main__":
    main()
