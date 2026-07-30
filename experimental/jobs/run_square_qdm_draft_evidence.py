#!/usr/bin/env python
"""Run the square-QDM draft-evidence notebook as a batch job.

This script executes ``experimental/notebooks/square_qdm_draft_evidence.ipynb``
with explicit job parameters and writes all CSV, figure, notebook, log, and
manifest outputs to a timestamped data directory by default.
"""

from __future__ import annotations

from evidence_job_utils import build_parser, parse_int_tuple, run_evidence_notebook


def main() -> None:
    parser = build_parser(description=__doc__ or "Run square-QDM evidence job.")
    parser.add_argument(
        "--ed-repeats",
        default=None,
        help=(
            "Comma-separated repeat counts for dense microcanonical ED. Defaults "
            "are profile dependent; production intentionally uses 1,2 only. Use "
            "--ed-repeats 1,2,3 only when enough memory is exclusively available."
        ),
    )
    parser.add_argument(
        "--sequence-repeats",
        default=None,
        help=(
            "Comma-separated repeat counts for non-ED product/transfer-style "
            "sequence diagnostics. This can be larger than --ed-repeats."
        ),
    )
    parser.add_argument(
        "--product-max-support-size",
        type=int,
        default=None,
        help=(
            "Override the formal-support guard for product-sequence diagnostics. "
            "For the compact 4-state unit, repeats 4 needs 256 and repeats 5 needs 1024."
        ),
    )
    parser.add_argument(
        "--skip-protocol-m",
        action="store_true",
        help="Skip the projector-deletion control table.",
    )
    parser.add_argument(
        "--skip-background-concentration",
        action="store_true",
        help="Skip background concentration diagnostics.",
    )
    parser.add_argument(
        "--skip-nonuniform-potential-path",
        action="store_true",
        help="Skip the nonuniform flippability-potential path.",
    )
    parser.add_argument(
        "--skip-non-gauge-kinetic-path",
        action="store_true",
        help="Skip the non-gauge kinetic deformation path.",
    )
    parser.add_argument(
        "--skip-collective-cluster-scan",
        action="store_true",
        help="Skip the collective multi-row locality scan.",
    )
    args = parser.parse_args()
    ed_repeats = parse_int_tuple(args.ed_repeats)
    sequence_repeats = parse_int_tuple(args.sequence_repeats)
    overrides = {
        "SAVE_FIGURES": bool(args.figure_formats.strip()),
        "SAVE_PDF": "pdf" in {part.strip().lower() for part in args.figure_formats.split(",")},
        "RUN_PROTOCOL_M": not args.skip_protocol_m,
        "RUN_BACKGROUND_CONCENTRATION": not args.skip_background_concentration,
        "RUN_NONUNIFORM_POTENTIAL_PATH": not args.skip_nonuniform_potential_path,
        "RUN_NON_GAUGE_KINETIC_PATH": not args.skip_non_gauge_kinetic_path,
        "RUN_COLLECTIVE_CLUSTER_SCAN": not args.skip_collective_cluster_scan,
    }
    if ed_repeats is not None:
        overrides["MICROCANONICAL_REPEAT_COUNTS"] = ed_repeats
    if sequence_repeats is not None:
        overrides["PRODUCT_SCALING_REPEAT_COUNTS"] = sequence_repeats
    if args.product_max_support_size is not None:
        if args.product_max_support_size <= 0:
            raise ValueError("--product-max-support-size must be positive")
        overrides["PRODUCT_SCALING_MAX_SUPPORT_SIZE"] = int(args.product_max_support_size)

    run_evidence_notebook(
        job_name="square_qdm_draft_evidence",
        notebook_filename="square_qdm_draft_evidence.ipynb",
        assignment_overrides=overrides,
        args=args,
    )


if __name__ == "__main__":
    main()
