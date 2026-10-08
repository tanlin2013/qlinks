#!/usr/bin/env python
"""Run the bounded, cache-first PRX referee sanity checks from the 2026-10-08 handoff.

The job never launches an L=14 Spin-1 or 12x4 checkerboard-QDM production solve.
Spin-1 adjacent-gap statistics are accepted only from complete validated spectra.
For the 8x4 QDM check, a bounded 1125x1125 dense fallback is allowed by default
when no compatible complete cache is found.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir() and (candidate / "experimental").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate qlinks repository root")

JOBS = ROOT / "experimental" / "jobs"
NOTEBOOKS = ROOT / "experimental" / "notebooks"
for path in (JOBS, NOTEBOOKS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import spin1_exchange_convention as convention  # noqa: E402
from evidence_cache import default_cache_root  # noqa: E402
from evidence_job_utils import (  # noqa: E402
    collect_file_manifest,
    git_metadata,
    utc_run_id,
    write_json,
)
from helpers import set_revtex_matplotlib_style  # noqa: E402
from prx_referee_sanity_qdm import analyze_qdm  # noqa: E402
from prx_referee_sanity_spin1 import MANDATORY_KAPPA, analyze_spin1  # noqa: E402

DEFAULT_LENGTHS = (8, 10, 12)
DEFAULT_KAPPAS = (0.05, 0.10, 0.15, 0.20)


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"expected positive comma-separated integers, got {raw!r}")
    return values


def _parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError(f"expected positive comma-separated floats, got {raw!r}")
    return values


def _default_output_dir() -> Path:
    run_id = utc_run_id("prx_referee_sanity_checks_20261008")
    return ROOT / "experimental" / "data" / "evidence_jobs" / run_id


def _default_spin1_roots() -> tuple[Path, ...]:
    return (
        ROOT / "experimental" / "data" / "evidence_cache" / "spin1",
        ROOT
        / "experimental"
        / "data"
        / "evidence_jobs"
        / "spin1_convention_migration_20260930T035701Z",
        ROOT
        / "experimental"
        / "data"
        / "evidence_jobs"
        / "spin1_sec6_integration_20260825T073925Z",
    )


def _default_qdm_data_dir() -> Path:
    run_id = os.environ.get(
        "QLINKS_QDM_THERMAL_RUN_ID", "qdm_checkerboard_primme_staged_20260825T164226Z"
    )
    return ROOT / "experimental" / "data" / "evidence_jobs" / run_id


def _write_spin1_followup(output: Path, missing: list[int]) -> Path | None:
    if not missing:
        return None
    path = output / "followup_spin1_dense_cache.sh"
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

# Generated because one or more mandatory kappa/J=0.10 complete small-size
# spectra were absent. This existing seed lane is bounded to L=8,10,12 and
# explicitly never launches L=14.
python experimental/jobs/spin1_sec6_seed_dense_cache.py \\
  --cache-root experimental/data/evidence_cache/spin1 \\
  --output-dir experimental/data/evidence_jobs/spin1_sec6_dense_cache_followup_20261008
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_hidden_qdm_followup(output: Path) -> Path:
    path = output / "followup_qdm_hidden_block.md"
    path.write_text(
        """# QDM hidden-block follow-up required

The sanity check found an exact additional block inside the nominal Fig. 9(a)
sector. Do not silently replace manuscript-facing assets. Resolve the operation
identified in `qdm_fig9a_symmetry_audit.md`, then recompute the 8x4 raw scatter,
raw A/Z means, and stripe concentration before updating Fig. 9. No 12x4 raw
spectrum is required for this follow-up.
""",
        encoding="utf-8",
    )
    return path


def _write_readme(
    output: Path,
    *,
    spin1: dict,
    qdm: dict,
    verdict: dict,
    spin1_followup: Path | None,
) -> None:
    git = git_metadata(ROOT)
    lines = [
        "# PRX referee sanity checks",
        "",
        "Bounded validation run for the 2026-10-08 referee handoff. It does not edit "
        "manuscript prose.",
        "",
        "## Exact command",
        "",
        "```bash",
        " ".join(shlex.quote(value) for value in sys.argv),
        "```",
        "",
        "## Provenance",
        "",
        f"- git commit: `{git.get('commit')}`",
        f"- git branch: `{git.get('branch')}`",
        f"- spin-1 exchange convention: `{convention.CURRENT_EXCHANGE_CONVENTION}`",
        f"- QDM eigensystem source: `{qdm['source']}` ({qdm['source_kind']})",
        "- L=14 spin-1 production solve launched: no",
        "- 12x4 QDM raw-spectrum solve launched: no",
        "",
        "## Spin-1 cache inventory",
        "",
        "| L | kappa/J | status | source |",
        "|---:|---:|---|---|",
    ]
    for row in spin1["inventory"]:
        lines.append(
            f"| {row['L']} | {row['kappa_over_J']:.2f} | {row['status']} | "
            f"{row.get('source', '')} |"
        )
    lines += [
        "",
        "## Scientific verdict",
        "",
        f"- spin-1: `{verdict['spin1_nonintegrability_check']}`",
        f"- spin-1 RMT class: `{verdict['spin1_rmt_class']}`",
        f"- QDM: `{qdm['verdict']}`",
        f"- QDM hidden symmetry found: `{verdict['qdm_hidden_symmetry_found']}`",
        f"- QDM Fig. 9 recompute required: `{verdict['qdm_fig9_recompute_required']}`",
        f"- heavy new run required: `{verdict['heavy_new_run_required']}`",
        "",
        "The spin-1 diagnostic is supporting evidence for a numerically nonintegrable background;",
        "it is not a proof of strong ETH. The QDM result is restricted to the finite 8x4 "
        "raw sector.",
    ]
    if spin1_followup is not None:
        lines += [
            "",
            f"Mandatory complete spin-1 caches were missing; run `{spin1_followup.name}` and rerun",
            "this job. No incomplete sparse window was substituted into level statistics.",
        ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    set_revtex_matplotlib_style(base_font_size=8.0, prefer_tex=False)

    lengths = _parse_ints(args.spin1_lengths)
    kappas = _parse_floats(args.spin1_kappas)
    if not any(math.isclose(value, MANDATORY_KAPPA, abs_tol=1.0e-12) for value in kappas):
        raise ValueError("--spin1-kappas must include mandatory kappa/J=0.10")
    roots = tuple(Path(path).resolve(strict=False) for path in args.spin1_cache_root)

    spin1 = analyze_spin1(
        output,
        roots=roots,
        lengths=lengths,
        kappas=kappas,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    spin1_followup = _write_spin1_followup(output, spin1["missing_mandatory_lengths"])

    qdm = analyze_qdm(
        output,
        data_dir=Path(args.qdm_data_dir).resolve(strict=False),
        cache_root=Path(args.evidence_cache_root).resolve(strict=False),
        allow_small_dense=bool(args.allow_small_qdm_dense),
    )
    if qdm["recompute_required"]:
        _write_hidden_qdm_followup(output)

    verdict = {
        "spin1_nonintegrability_check": spin1["verdict"],
        "spin1_rmt_class": spin1["rmt_class"],
        "qdm_hidden_symmetry_found": qdm["hidden_symmetry_found"],
        "qdm_disconnected_exact_block_structure_found": qdm["disconnected_block_found"],
        "qdm_fig9_recompute_required": qdm["recompute_required"],
        "heavy_new_run_required": False,
        "small_qdm_dense_fallback_used": qdm["small_dense_fallback_used"],
        "spin1_mandatory_cache_missing_lengths": spin1["missing_mandatory_lengths"],
    }
    write_json(output / "verdict.json", verdict)
    _write_readme(
        output,
        spin1=spin1,
        qdm=qdm,
        verdict=verdict,
        spin1_followup=spin1_followup,
    )
    write_json(output / "file_manifest.json", {"files": collect_file_manifest(output)})
    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    print(f"[prx-referee-sanity] output: {output}", flush=True)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--spin1-cache-root",
        type=Path,
        action="append",
        default=None,
        help="Repeatable root searched recursively for complete spin-1 spectra.",
    )
    parser.add_argument("--spin1-lengths", default="8,10,12")
    parser.add_argument("--spin1-kappas", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--qdm-data-dir", type=Path, default=_default_qdm_data_dir())
    parser.add_argument("--evidence-cache-root", type=Path, default=default_cache_root())
    parser.add_argument(
        "--allow-small-qdm-dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow a bounded 1125x1125 Lx=8 dense fallback if no complete cache exists.",
    )
    args = parser.parse_args()
    if args.spin1_cache_root is None:
        args.spin1_cache_root = list(_default_spin1_roots())
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    run(args)


if __name__ == "__main__":
    main()
