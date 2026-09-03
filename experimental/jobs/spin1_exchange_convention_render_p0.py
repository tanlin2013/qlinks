#!/usr/bin/env python
"""Prepare and render convention-mapped Spin-1 Sec. VI P0 figures.

The convention migration copies/rescales historical evidence products, but the
standardized ``spin1_xy_figure6_panel_*.csv`` tables are integration products rather
than primary evidence. This post-processing entry point first verifies or repairs the
migration provenance stamp from the immutable historical source, then runs the current
convention-aware Sec. VI integration formatter in place and invokes the renderer. It
contains no eigensolver path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import render_spin1_xy_sec6_integration_figures as renderer

import spin1_exchange_convention_repair_manifest as manifest_repair
import spin1_sec6_integration as integration

_REQUIRED_RENDER_INPUTS = (
    "spin1_xy_figure6_panel_a_scatter.csv",
    "spin1_xy_figure6_panel_b_witness_sequence.csv",
)


def prepare_and_render(
    data_dir: Path,
    *,
    use_tex: bool,
    allow_incomplete: bool,
    historical_source_dir: Path | None = None,
    source_run_id: str | None = None,
) -> dict[str, object]:
    """Verify mapped provenance, build standardized figure tables, then render them."""

    data = Path(data_dir).resolve(strict=False)
    manifest_path = data / "spin1_exchange_convention_migration_manifest.json"
    repaired_manifest = False
    if historical_source_dir is not None:
        had_manifest = manifest_path.is_file()
        manifest_repair.repair_missing_manifest(
            source_dir=historical_source_dir,
            output_dir=data,
            source_run_id=source_run_id,
        )
        repaired_manifest = not had_manifest

    integration.run_integration(data, data)

    missing = [name for name in _REQUIRED_RENDER_INPUTS if not (data / name).is_file()]
    if missing:
        raise RuntimeError(
            "convention-aware integration did not produce required Fig. 6 inputs: "
            + ", ".join(missing)
        )

    written = renderer.render(
        data,
        use_tex=use_tex,
        allow_incomplete=allow_incomplete,
    )
    return {
        "data_dir": str(data),
        "manifest_repaired": repaired_manifest,
        "prepared_figure_data": list(_REQUIRED_RENDER_INPUTS),
        "rendered": list(written),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--historical-source-dir", type=Path, default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = prepare_and_render(
        args.data_dir,
        use_tex=args.use_tex,
        allow_incomplete=args.allow_incomplete,
        historical_source_dir=args.historical_source_dir,
        source_run_id=args.source_run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
