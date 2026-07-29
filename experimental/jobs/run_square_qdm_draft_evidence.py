#!/usr/bin/env python
"""Run the square-QDM draft-evidence notebook as a batch job.

This script executes ``experimental/notebooks/square_qdm_draft_evidence.ipynb``
with explicit job parameters and writes all CSV, figure, notebook, log, and
manifest outputs to a timestamped data directory by default.
"""

from __future__ import annotations

from evidence_job_utils import build_parser, run_evidence_notebook


def main() -> None:
    parser = build_parser(description=__doc__ or "Run square-QDM evidence job.")
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

    run_evidence_notebook(
        job_name="square_qdm_draft_evidence",
        notebook_filename="square_qdm_draft_evidence.ipynb",
        assignment_overrides={
            "SAVE_FIGURES": bool(args.figure_formats.strip()),
            "SAVE_PDF": "pdf" in {part.strip().lower() for part in args.figure_formats.split(",")},
            "RUN_PROTOCOL_M": not args.skip_protocol_m,
            "RUN_BACKGROUND_CONCENTRATION": not args.skip_background_concentration,
            "RUN_NONUNIFORM_POTENTIAL_PATH": not args.skip_nonuniform_potential_path,
            "RUN_NON_GAUGE_KINETIC_PATH": not args.skip_non_gauge_kinetic_path,
            "RUN_COLLECTIVE_CLUSTER_SCAN": not args.skip_collective_cluster_scan,
        },
        args=args,
    )


if __name__ == "__main__":
    main()
