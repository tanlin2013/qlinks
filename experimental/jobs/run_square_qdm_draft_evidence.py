#!/usr/bin/env python
"""Run or render the square-QDM Sec. 7 evidence workflow."""

from __future__ import annotations

from evidence_job_utils import (
    build_parser,
    parse_int_tuple,
    run_evidence_notebook,
    run_evidence_renderer,
)


def main() -> None:
    parser = build_parser(description=__doc__ or "Run square-QDM evidence job.")
    parser.add_argument(
        "--microcanonical-repeats",
        "--ed-repeats",
        dest="microcanonical_repeats",
        default=None,
        help=(
            "Dense energy-resolved repeat counts. Production-safe default: "
            "1,2. Repeat 3 requires --allow-large-dense-ed and a memory-scalable algorithm "
            "or an effectively exclusive very-large-memory host."
        ),
    )
    parser.add_argument(
        "--sequence-repeats",
        default=None,
        help="Repeat counts for exact compact-family and revised-Y transport certificates.",
    )
    parser.add_argument(
        "--transfer-max-length",
        type=int,
        default=None,
        help="Largest multiple of four used by the beta-zero strip transfer calculation.",
    )
    parser.add_argument("--product-max-support-size", type=int, default=None)
    parser.add_argument(
        "--allow-large-dense-ed",
        action="store_true",
        help=(
            "Acknowledge the O(d^2) memory risk and allow dense microcanonical repeat >=3. "
            "Without this flag the job fails before notebook execution."
        ),
    )
    parser.add_argument(
        "--exceptional-projector-mode",
        choices=("identified-compact", "target-only"),
        default="identified-compact",
        help="Exceptional projector used by the undeformed fixed-width Protocol M calculation.",
    )
    parser.add_argument(
        "--deformation-reference",
        "--generic-reference",
        dest="deformation_reference",
        choices=("peierls", "nonuniform-potential"),
        default="nonuniform-potential",
        help=(
            "Path used for the provisional T3 stable export. The primary T1 "
            "ensemble is always the undeformed cage-excised Hamiltonian. "
            "--generic-reference is retained as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--deformation-reference-parameter",
        "--generic-reference-parameter",
        dest="deformation_reference_parameter",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--skip-protocol-m",
        action="store_true",
        help="Deprecated: the primary evidence workflow requires undeformed cage excision.",
    )
    parser.add_argument("--skip-background-concentration", action="store_true")
    parser.add_argument("--skip-nonuniform-potential-path", action="store_true")
    parser.add_argument("--skip-non-gauge-kinetic-path", action="store_true")
    parser.add_argument("--skip-collective-cluster-scan", action="store_true")
    args = parser.parse_args()
    if args.skip_protocol_m:
        raise ValueError(
            "--skip-protocol-m is no longer supported by the manuscript evidence job: "
            "T1 requires the undeformed cage-excised ensemble. Use "
            "--exceptional-projector-mode target-only only for an explicitly diagnostic run."
        )

    if args.stage == "render":
        run_evidence_renderer(
            job_name="square_qdm_draft_evidence",
            renderer_filename="render_square_qdm_draft_figures.py",
            args=args,
        )
        return

    micro_repeats = parse_int_tuple(args.microcanonical_repeats)
    if micro_repeats is not None and max(micro_repeats) >= 3 and not args.allow_large_dense_ed:
        raise ValueError(
            "dense QDM microcanonical repeat >=3 is disabled by default because the complete "
            "eigenvector matrix and LAPACK work arrays can exceed hundreds of GiB. Re-run with "
            "--microcanonical-repeats 1,2, or explicitly add --allow-large-dense-ed after "
            "confirming a suitable algorithm and memory budget."
        )
    sequence_repeats = parse_int_tuple(args.sequence_repeats)
    overrides = {
        "SAVE_FIGURES": bool(args.figure_formats.strip()) and args.stage != "compute",
        "SAVE_PDF": "pdf" in {part.strip().lower() for part in args.figure_formats.split(",")},
        "RUN_BACKGROUND_CONCENTRATION": not args.skip_background_concentration,
        "RUN_NONUNIFORM_POTENTIAL_PATH": not args.skip_nonuniform_potential_path,
        "RUN_NON_GAUGE_KINETIC_PATH": not args.skip_non_gauge_kinetic_path,
        "RUN_COLLECTIVE_CLUSTER_SCAN": not args.skip_collective_cluster_scan,
        "RUN_REVISED_Y_VALIDATION": True,
        "STRICT_CLAIMS": bool(args.strict_claims),
        "EXCEPTIONAL_PROJECTOR_MODE": args.exceptional_projector_mode,
        "DEFORMATION_REFERENCE": args.deformation_reference,
    }
    if micro_repeats is not None:
        overrides["MICROCANONICAL_REPEAT_COUNTS"] = micro_repeats
    if sequence_repeats is not None:
        overrides["PRODUCT_SCALING_REPEAT_COUNTS"] = sequence_repeats
    if args.product_max_support_size is not None:
        if args.product_max_support_size <= 0:
            raise ValueError("--product-max-support-size must be positive")
        overrides["PRODUCT_SCALING_MAX_SUPPORT_SIZE"] = int(args.product_max_support_size)
    if args.transfer_max_length is not None:
        if args.transfer_max_length < 4 or args.transfer_max_length % 4:
            raise ValueError("--transfer-max-length must be a multiple of four >= 4")
        overrides["strip_lengths"] = tuple(range(4, args.transfer_max_length + 1, 4))
    if args.deformation_reference_parameter is not None:
        if args.deformation_reference == "peierls":
            overrides["PEIERLS_REFERENCE_PHASE"] = float(args.deformation_reference_parameter)
        else:
            value = float(args.deformation_reference_parameter)
            overrides["NONUNIFORM_POTENTIAL_REFERENCE"] = value

    run_evidence_notebook(
        job_name="square_qdm_draft_evidence",
        notebook_filename="square_qdm_draft_evidence.ipynb",
        assignment_overrides=overrides,
        args=args,
    )


if __name__ == "__main__":
    main()
