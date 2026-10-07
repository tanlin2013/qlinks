#!/usr/bin/env python
"""Render the PRX checkerboard-QDM Fig. 9 from frozen evidence only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate qlinks repository")
sys.path[:0] = [
    str(ROOT / "experimental" / "notebooks"),
    str(ROOT / "experimental" / "jobs"),
    str(ROOT),
]

from helpers import set_revtex_matplotlib_style, write_figure_manifest  # noqa: E402
from prx_main_thermal_figure_polish import render_qdm_figure9  # noqa: E402

MANUSCRIPT_STEMS = (
    "qdm_checkerboard_figure7_combined",
    "qdm_checkerboard_figure9_prx",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--figure-formats", default="pdf,svg")
    parser.add_argument("--use-tex", action="store_true")
    args = parser.parse_args()

    data = args.data_dir.resolve()
    formats = tuple(value.strip() for value in args.figure_formats.split(",") if value.strip())
    set_revtex_matplotlib_style(base_font_size=9.0, prefer_tex=args.use_tex)
    if args.use_tex and not bool(plt.rcParams.get("text.usetex", False)):
        raise RuntimeError(
            "--use-tex requires a working LaTeX executable; refusing mathtext fallback"
        )

    render_qdm_figure9(data=data, formats=formats)
    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
