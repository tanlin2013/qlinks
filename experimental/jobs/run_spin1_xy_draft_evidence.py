#!/usr/bin/env python
"""Run the Spin-1 XY draft-evidence notebook as a batch job.

This script executes ``experimental/notebooks/spin1_xy_draft_evidence.ipynb``
with explicit job parameters and writes all CSV, figure, notebook, log, and
manifest outputs to a timestamped data directory by default.
"""

from __future__ import annotations

from evidence_job_utils import build_parser, run_evidence_notebook


def main() -> None:
    parser = build_parser(description=__doc__ or "Run Spin-1 XY evidence job.")
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
        "--skip-complex-hermitian-path",
        action="store_true",
        help="Skip the finite complex-Hermitian preserving path.",
    )
    parser.add_argument(
        "--skip-joint-continuation",
        action="store_true",
        help="Skip the joint cage/local-channel continuation cross-check.",
    )
    args = parser.parse_args()

    run_evidence_notebook(
        job_name="spin1_xy_draft_evidence",
        notebook_filename="spin1_xy_draft_evidence.ipynb",
        assignment_overrides={
            "RUN_PROTOCOL_M": not args.skip_protocol_m,
            "RUN_BACKGROUND_CONCENTRATION": not args.skip_background_concentration,
            "RUN_COMPLEX_HERMITIAN_PATH": not args.skip_complex_hermitian_path,
            "RUN_JOINT_CONTINUATION_CROSSCHECK": not args.skip_joint_continuation,
        },
        args=args,
    )


if __name__ == "__main__":
    main()
