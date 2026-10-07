#!/usr/bin/env python
"""Render the PRX checkerboard-QDM Fig. 9 from frozen evidence only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

import prx_main_thermal_figure_polish as _polish  # noqa: E402
from helpers import set_revtex_matplotlib_style, write_figure_manifest  # noqa: E402
from prx_main_thermal_figure_polish import render_qdm_figure9  # noqa: E402

MANUSCRIPT_STEMS = (
    "qdm_checkerboard_figure7_combined",
    "qdm_checkerboard_figure9_prx",
)


def _qdm_range_legend_handles() -> list[Rectangle]:
    """Show only the ensemble fill/edge grammar used by Fig. 9(c)."""

    neutral = "0.35"
    return [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=neutral,
            edgecolor=neutral,
            linewidth=_polish.RANGE_BOX_EDGE_WIDTH,
            linestyle="-",
            alpha=_polish.RANGE_BOX_FACE_ALPHA,
            label="raw MC",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor=neutral,
            linewidth=_polish.RANGE_BOX_EDGE_WIDTH,
            linestyle="--",
            label="canonical",
        ),
    ]


def _install_qdm_panel_c_legend() -> None:
    """Add a compact ensemble key to panel 9(c) without changing range geometry."""

    original = _polish._draw_qdm_b_c

    def _draw_with_panel_c_legend(
        *,
        axes_b: list,
        axes_c: list,
        panel_b,
        panel_c,
    ) -> None:
        original(
            axes_b=axes_b,
            axes_c=axes_c,
            panel_b=panel_b,
            panel_c=panel_c,
        )
        axes_c[0].legend(
            handles=_qdm_range_legend_handles(),
            loc="upper right",
            frameon=False,
            fontsize=7.0,
            handletextpad=0.5,
            borderaxespad=0.35,
        )

    _polish._draw_qdm_b_c = _draw_with_panel_c_legend


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

    # Keep the sampled-phase range geometry literal and consistent with Fig. 6.
    # The verified QDM phase spans are physically present but can be much smaller
    # than a printed line width, so the caption should explain their near-invisibility
    # instead of introducing a figure-specific visual exaggeration.
    _install_qdm_panel_c_legend()
    render_qdm_figure9(data=data, formats=formats)
    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
