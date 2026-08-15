#!/usr/bin/env python
"""CLI for the independent jump-bridge near-zero Liouvillian job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from jump_bridge_liouvillian import run_liouvillian_benchmark
from jump_bridge_p0 import benchmark_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an independent partial Liouvillian spectrum check for the four "
            "claim-critical 4x4 QDM jump-bridge cases."
        )
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-dir", "--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=("compute", "all"),
        default="compute",
        help="Accepted for compatibility with scripts/docker_run_evidence_job.sh.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in benchmark_cases()],
        help="Repeat to select cases; default runs all four.",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=("A_retargeted_single", "ML", "final"),
        help="Repeat to select jump families; default checks all three.",
    )
    parser.add_argument(
        "--method",
        choices=("largest-real", "smallest-magnitude", "shift-invert"),
        default="largest-real",
        help=(
            "ARPACK mode. largest-real avoids sparse-LU and is the recommended first "
            "server run; shift-invert is an explicit fallback."
        ),
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--ncv", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--eig-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--peripheral-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--sigma", type=float, default=1.0e-10)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    selected = set(args.case or [])
    cases = tuple(case for case in benchmark_cases() if not selected or case.name in selected)
    families = tuple(args.family or ("A_retargeted_single", "ML", "final"))

    if args.output_dir is None:
        run_id = args.run_id
        if run_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"jump_bridge_liouvillian_{stamp}"
        output_dir = Path("experimental/data/evidence_jobs") / run_id
    else:
        output_dir = args.output_dir

    run_liouvillian_benchmark(
        cases=cases,
        output_dir=output_dir,
        family_names=families,
        method=args.method,
        k=args.k,
        eig_tolerance=args.eig_tolerance,
        zero_tolerance=args.zero_tolerance,
        peripheral_tolerance=args.peripheral_tolerance,
        maxiter=args.maxiter,
        ncv=args.ncv,
        sigma=args.sigma,
        strict=args.strict,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
