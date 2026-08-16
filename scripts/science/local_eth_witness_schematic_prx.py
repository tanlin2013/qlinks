#!/usr/bin/env python3
"""PRX-style schematic local-ETH witness scatter plot with fake data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# PRX/REVTeX two-column papers use a single-column width of about 3.375 in.
# Keep the aspect compact enough to sit naturally inside one column.
SINGLE_COLUMN_WIDTH_IN = 3.375 * 1.2
FIGURE_HEIGHT_IN = 2.50


def local_point_density(x: np.ndarray, y: np.ndarray, bins: int = 60) -> np.ndarray:
    """Estimate local scatter density without requiring SciPy."""
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    return hist[ix, iy]


def main() -> None:
    rng = np.random.default_rng(12)

    # Place the caged eigenstate at an arbitrary energy density.
    e_cage = 0.37
    energy_window_half_width = 0.10

    # Fake thermal eigenstates: deliberately denser near e_cage and sparser
    # toward the spectral edges.
    n_points = 2600
    core = rng.normal(loc=e_cage, scale=0.23, size=int(0.82 * n_points))
    background = rng.uniform(e_cage - 0.70, e_cage + 0.70, size=n_points - core.size)
    energy_density = np.concatenate([core, background])
    energy_density = np.clip(energy_density, e_cage - 0.75, e_cage + 0.75)

    # Finite nonzero ETH band with weak smooth energy dependence.
    scaled = (energy_density - e_cage) / 0.75
    q_center = 0.265 - 0.030 * scaled**2
    q_r = q_center + rng.normal(0.0, 0.027, size=energy_density.size)
    q_r = np.clip(q_r, 0.15, None)

    # Draw low-density points first so dense regions remain visible.
    density = local_point_density(energy_density, q_r)
    order = np.argsort(density)
    energy_density = energy_density[order]
    q_r = q_r[order]
    density = density[order]

    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH_IN, FIGURE_HEIGHT_IN),
        constrained_layout=True,
    )

    # Microcanonical energy window centered on the caged eigenstate.
    ax.axvspan(
        e_cage - energy_window_half_width,
        e_cage + energy_window_half_width,
        color="0.82",
        alpha=0.45,
        linewidth=0,
        zorder=0,
    )

    scatter = ax.scatter(
        energy_density,
        q_r,
        c=density,
        cmap="inferno",
        norm=Normalize(vmin=1, vmax=np.percentile(density, 98)),
        s=7.0,
        alpha=0.84,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    # One caged/scar eigenstate at Q_R = 0.
    ax.scatter(
        [e_cage],
        [0.0],
        marker="*",
        s=180,
        facecolor="deepskyblue",
        edgecolor="navy",
        linewidth=0.9,
        zorder=5,
        clip_on=False,
    )

    ax.axhline(
        0.0,
        color="0.35",
        linewidth=0.75,
        linestyle="--",
        zorder=1,
    )

    ax.set_xlabel(r"Energy density $e=E/L$")
    ax.set_ylabel(r"Local ETH witness $Q_R$")

    ax.set_xlim(e_cage - 0.78, e_cage + 0.78)
    ax.set_ylim(-0.025, 0.355)

    # Omit numerical x ticks: the absolute value of e_cage is schematic.
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.025,
        fraction=0.052,
    )
    cbar.set_label("Local point density")
    cbar.set_ticks([])

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        width=0.75,
        length=3.0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    fig.savefig("local_eth_witness_schematic_prx.pdf", transparent=True)
    fig.savefig("local_eth_witness_schematic_prx.svg", transparent=True)
    # fig.savefig("local_eth_witness_schematic_prx.png", dpi=400, transparent=True)


if __name__ == "__main__":
    # REVTeX-compatible visual style:
    # - Times-like serif text
    # - STIX math glyphs
    # - 8 pt body text, appropriate for final single-column figure size
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    main()
