#!/usr/bin/env python
"""Prepare and render convention-mapped Spin-1 Sec. VI P0 figures.

The convention migration copies/rescales historical evidence products, but the
standardized ``spin1_xy_figure6_panel_*.csv`` tables are integration products rather
than primary evidence.  This post-processing entry point first runs the current
convention-aware Sec. VI integration formatter in place, then invokes the renderer.
It contains no eigensolver path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import render_spin1_xy_sec6_integration_figures as renderer

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
) -> dict[str, object]:
    """Build standardized figure tables from mapped evidence, then render them."""

    data = Path(data_dir).resolve(strict=False)
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
        "prepared_figure_data": list(_REQUIRED_RENDER_INPUTS),
        "rendered": list(written),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = prepare_and_render(
        args.data_dir,
        use_tex=args.use_tex,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
