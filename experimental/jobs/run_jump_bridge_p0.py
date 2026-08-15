#!/usr/bin/env python
"""Run the claim-critical ICQMBS-to-Lindblad jump-bridge benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from jump_bridge_p0 import benchmark_cases, run_jump_bridge_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark directed caging rows against current Lindblad jump designs."
    )
    parser.add_argument(
        "--output-dir",
        "--data-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Evidence directory. Defaults to a timestamped folder under experimental/data.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id used when --output-dir/--data-dir is omitted.",
    )
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
        help="Run only one named case. Repeat to select several cases.",
    )
    parser.add_argument(
        "--no-legacy-single",
        action="store_true",
        help="Skip the deprecated single-cage reduced-IZ/block-reset baseline.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir
    if output_dir is None:
        run_id = args.run_id or f"jump_bridge_p0_{timestamp}"
        output_dir = Path("experimental/data/evidence_jobs") / run_id

    run_jump_bridge_benchmark(
        output_dir=output_dir,
        selected_cases=args.case,
        include_legacy_single=not args.no_legacy_single,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
