#!/usr/bin/env python
"""Render final-size REVTeX figures from completed square-QDM evidence tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--figure-formats", default="pdf,svg")
    p.add_argument("--use-tex", action="store_true")
    a = p.parse_args()
    data = a.data_dir.resolve()
    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    formats = tuple(x.strip() for x in a.figure_formats.split(",") if x.strip())
    set_revtex_matplotlib_style(base_font_size=8, prefer_tex=a.use_tex)
    fixed = pd.read_csv(data / "qdm_fixed_width_microcanonical_primary.csv")
    overlap = pd.read_csv(data / "qdm_beta0_ensemble_overlap.csv")
    concentration = pd.read_csv(data / "qdm_background_concentration.csv")
    path = pd.read_csv(data / "qdm_nonuniform_potential_path.csv")
    yvalid = pd.read_csv(data / "qdm_revised_Y_size_validation.csv")
    y_enabled = bool(yvalid.get("resolved_shell_valid", pd.Series(False, index=yvalid.index)).all())

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
    fixed_specs = [("thermal_A_activity", r"$Q_R^A$", "o"), ("thermal_Z_activity", r"$Q_R^Z$", "s")]
    if y_enabled:
        fixed_specs.append(("thermal_Y_activity", r"$Q_R^Y$", "^"))
    for col, label, marker in fixed_specs:
        ax.plot(fixed["Lx"], fixed[col], marker=marker, label=label)
    ax.axhline(0, ls="--", lw=0.8, color="0.45")
    ax.set_xlabel(r"Strip length $L_x$")
    ax.set_ylabel("Microcanonical activity")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(a)")
    ax = axes[1]
    ax.plot(
        overlap["Lx"],
        overlap["energy_density_mismatch"],
        marker="o",
        label=r"$|e_\psi-e_{\beta=0}|$",
    )
    overlap_specs = [("delta_A", r"$\delta_A$", "s"), ("delta_Z", r"$\delta_Z$", "^")]
    if y_enabled:
        overlap_specs.append(("delta_Y", r"$\delta_Y$", "v"))
    for col, label, marker in overlap_specs:
        ax.plot(overlap["Lx"], overlap[col], marker=marker, label=label)
    ax.set_xlabel(r"Strip length $L_x$")
    ax.set_ylabel("Matching difference")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(b)")
    ax = axes[2]
    if concentration.empty:
        ax.text(
            0.5,
            0.5,
            "concentration data unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        env = (
            concentration.groupby("Lx")
            .agg(
                median=("basis_independent_std", "median"), maximum=("basis_independent_std", "max")
            )
            .reset_index()
        )
        ax.plot(env["Lx"], env["median"], marker="o", label="median local spread")
        ax.plot(env["Lx"], env["maximum"], marker="s", ls="--", label="maximum local spread")
    ax.set_xlabel(r"Strip length $L_x$")
    ax.set_ylabel("Window EEV spread")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", frameon=False)
    add_panel_label(ax, "(c)")
    ax = axes[3]
    path_specs = [
        ("thermal_A_activity", r"$Q_R^A$", "o"),
        ("thermal_Z_activity", r"$Q_R^Z$", "s"),
        ("thermal_Y_activity", r"$Q_R^Y$", "^"),
    ]
    for col, label, marker in path_specs:
        ax.plot(path["g"], path[col], marker=marker, label=label)
    ax.set_xlabel(r"Nonuniform potential deformation $g$")
    ax.set_ylabel("Microcanonical activity")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", frameon=False, fontsize=6)
    add_panel_label(ax, "(d)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False
    )
    for ax in (axes[0],):
        leg = ax.get_legend()
        if leg:
            leg.remove()
    axes[1].legend(loc="upper right", frameon=False, fontsize=6)
    save_prx_figure(fig, "qdm_evidence_main", directory=figures, formats=formats)
    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
