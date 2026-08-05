#!/usr/bin/env python
"""Run or render the Spin-1 XY Sec. 6 evidence workflow."""

from __future__ import annotations

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
        help="Comma-separated dense-ED sizes. Known profile: 8,10,12; "
        "add 14 only on a large-memory host.",
    )
    parser.add_argument(
        "--deformation-sizes",
        default=None,
        help="Comma-separated sizes used for preserving-neighborhood scans; "
        "may be smaller than ED sizes.",
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
            "T1 requires the undeformed cage-excised ensemble. Use "
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
    j3_values = parse_float_tuple(args.j3_values)
    kappa_values = parse_float_tuple(args.kappa_values)
    type1_kappa_values = parse_float_tuple(args.deformed_type1_kappa_values)
    overrides = {
        "RUN_BACKGROUND_CONCENTRATION": not args.skip_background_concentration,
        "RUN_DEFORMATION_CONCENTRATION": not args.skip_deformation_concentration,
        "RUN_COMPLEX_HERMITIAN_PATH": not args.skip_complex_hermitian_path,
        "RUN_JOINT_CONTINUATION_CROSSCHECK": not args.skip_joint_continuation,
        "RUN_DEFORMED_TYPE1_INVENTORY": not args.skip_deformed_type1_inventory,
        "EXCEPTIONAL_PROJECTOR_MODE": args.exceptional_projector_mode,
    }
    if micro_sizes is not None:
        overrides["MICROCANONICAL_SIZES"] = micro_sizes
        overrides["SIZES"] = micro_sizes
    if deformation_sizes is not None:
        overrides["DEFORMATION_SIZES"] = deformation_sizes
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
