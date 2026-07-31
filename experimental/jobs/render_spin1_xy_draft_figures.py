#!/usr/bin/env python
"""Render final-size REVTeX figures from completed Spin-1 XY evidence tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate qlinks repository")
NOTEBOOK_DIR = ROOT / "experimental" / "notebooks"
sys.path.insert(0, str(NOTEBOOK_DIR))
sys.path.insert(0, str(ROOT))
from helpers import (  # noqa: E402
    PRX_FOUR_PANEL_FIGSIZE,
    add_panel_label,
    save_prx_figure,
    set_revtex_matplotlib_style,
    write_figure_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--figure-formats", default="pdf,svg")
    parser.add_argument("--use-tex", action="store_true")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    formats = tuple(x.strip() for x in args.figure_formats.split(",") if x.strip())
    set_revtex_matplotlib_style(base_font_size=8, prefer_tex=args.use_tex)

    scatter = pd.read_csv(data / "finiteD_eth_scatter_Lmax.csv")
    finite = pd.read_csv(data / "finiteD_microcanonical_window_sensitivity.csv")
    beta0 = pd.read_csv(data / "spin1_xy_beta0_ensemble_overlap.csv")
    deform = pd.read_csv(data / "spin1_xy_preserving_j3_scan.csv")
    summary = pd.read_csv(data / "symmetry_resolved_spectral_evidence.csv")
    primary_pref = float(
        finite.loc[(finite["window_prefactor"] - 1.0).abs().idxmin(), "window_prefactor"]
    )
    central = finite[np.isclose(finite["window_prefactor"], primary_pref)].sort_values("L")
    largest = int(central["L"].max())
    row = summary[summary["L"] == largest].iloc[0]
    center = float(row["finiteD_scar_energy"]) / largest
    half = float(row["finiteD_window_actual_half_width"]) / largest

    fig = plt.figure(figsize=PRX_FOUR_PANEL_FIGSIZE)
    gs = fig.add_gridspec(
        2, 2, left=0.085, right=0.985, bottom=0.09, top=0.90, wspace=0.30, hspace=0.34
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    ax = axes[0]
    ax.axvspan(center - half, center + half, color="0.5", alpha=0.10, zorder=0)
    ax.axvline(center, color="0.45", ls="--", lw=0.8)
    for col, label, marker in [
        ("QY", r"$Q_r^Y$", "o"),
        ("QA_normalized", r"$Q_{r,r+1}^A/(8J^2)$", "s"),
        ("QZ_normalized", r"$Q_{r,r+1}^Z/(8J^2)$", "^"),
    ]:
        ax.scatter(
            scatter["energy_density"], scatter[col], s=9, alpha=0.55, marker=marker, label=label
        )
    ax.scatter(
        [center],
        [0],
        marker="*",
        s=70,
        edgecolors="black",
        linewidths=0.4,
        label="exact tower",
        zorder=5,
    )
    ax.set_xlabel(r"Energy density $e=E/L$")
    ax.set_ylabel("Normalized local activity")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(a)")

    ax = axes[1]
    group = finite.groupby("L")
    for col, label, marker in [
        ("tau_Y", r"$Q_r^Y$", "o"),
        ("tau_A_normalized", r"$Q_{r,r+1}^A/(8J^2)$", "s"),
        ("tau_Z_normalized", r"$Q_{r,r+1}^Z/(8J^2)$", "^"),
    ]:
        vals = central[col].to_numpy()
        lows = group[col].min().reindex(central["L"]).to_numpy()
        highs = group[col].max().reindex(central["L"]).to_numpy()
        ax.errorbar(
            central["L"],
            vals,
            yerr=np.vstack([vals - lows, highs - vals]),
            marker=marker,
            capsize=2.5,
            label=label,
        )
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel("Microcanonical activity")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(b)")

    ax = axes[2]
    ax.plot(
        beta0["L"], beta0["energy_density_mismatch"], marker="o", label=r"$|e_\psi-e_{\beta=0}|$"
    )
    for col, label, marker in [
        ("delta_A", r"$\delta_A$", "s"),
        ("delta_Z", r"$\delta_Z$", "^"),
        ("delta_Y", r"$\delta_Y$", "v"),
    ]:
        ax.plot(beta0["L"], beta0[col], marker=marker, label=label)
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel("Matching difference")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(c)")

    ax = axes[3]
    max_length = int(deform["L"].max())
    frame = deform[deform["L"] == max_length]
    for col, label, marker in [
        ("tau_Y", r"$Q_r^Y$", "o"),
        ("tau_A_normalized", r"$Q_{r,r+1}^A/(8J^2)$", "s"),
        ("tau_Z_normalized", r"$Q_{r,r+1}^Z/(8J^2)$", "^"),
    ]:
        ax.plot(frame["J3_over_J"], frame[col], marker=marker, label=label)
    ax.set_xlabel(r"Preserving exchange $J_3/J$")
    ax.set_ylabel("Microcanonical activity")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(d)")

    # Shared witness legend above the grid; matching-specific labels remain local.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False
    )
    axes[2].legend(loc="upper right", frameon=False, fontsize=6)
    for ax in (axes[0], axes[1], axes[3]):
        leg = ax.get_legend()
        if leg:
            leg.remove()
    save_prx_figure(fig, "spin1_xy_evidence_main", directory=figures, formats=formats)
    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
