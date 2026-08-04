#!/usr/bin/env python
"""Render final-size REVTeX figures from completed spin-1 XY evidence tables."""

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
    PRX_TWO_PANEL_FIGSIZE,
    add_panel_label,
    save_prx_figure,
    set_revtex_matplotlib_style,
    write_figure_manifest,
)

WITNESS_SPECS = [
    ("A", r"$Q_R^A$", "o"),
    ("Z", r"$Q_R^Z$", "s"),
    ("Y", r"$Q_R^Y$", "^"),
]


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


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

    sequence = pd.read_csv(data / "spin1_xy_cage_excised_sequence.csv")
    scatter = pd.read_csv(data / "spin1_xy_cage_excised_eth_scatter.csv")
    concentration = pd.read_csv(data / "spin1_xy_cage_excised_concentration.csv")
    overlap = pd.read_csv(data / "spin1_xy_beta0_cage_excised_overlap.csv")
    exact = _read_optional(data / "exact_fixed_M_activities.csv")
    deform = _read_optional(data / "spin1_xy_deformed_cage_excised_grid.csv")

    prefactors = np.sort(sequence["window_prefactor"].unique())
    primary_prefactor = float(prefactors[np.argmin(np.abs(prefactors - 1.0))])
    central = sequence[np.isclose(sequence["window_prefactor"], primary_prefactor)].sort_values("L")
    largest = int(central["L"].max())
    scatter_largest = scatter[scatter["L"] == largest]
    row = central[central["L"] == largest].iloc[0]

    fig = plt.figure(figsize=PRX_FOUR_PANEL_FIGSIZE)
    gs = fig.add_gridspec(
        2, 2, left=0.085, right=0.985, bottom=0.09, top=0.90, wspace=0.30, hspace=0.34
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    ax = axes[0]
    half = float(row["window_energy_density_half_width"])
    ax.axvspan(-half, half, color="0.5", alpha=0.10, zorder=0)
    ax.axvline(0.0, color="0.45", ls="--", lw=0.8)
    retained = scatter_largest[~scatter_largest["is_exceptional"].astype(bool)]
    removed = scatter_largest[scatter_largest["is_exceptional"].astype(bool)]
    for key, label, marker in WITNESS_SPECS:
        col = f"Q_{key}"
        ax.scatter(
            retained["energy_density"], retained[col], s=9, alpha=0.50, marker=marker, label=label
        )
        if not removed.empty:
            ax.scatter(
                removed["energy_density"],
                removed[col],
                s=20,
                marker=marker,
                facecolors="none",
                edgecolors="0.25",
                linewidths=0.6,
            )
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=70,
        edgecolors="black",
        linewidths=0.4,
        label="selected tower",
        zorder=5,
    )
    ax.set_xlabel(r"Energy density $e=E/L$")
    ax.set_ylabel("Local witness activity")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(a)")

    ax = axes[1]
    grouped = sequence.groupby("L")
    for key, label, marker in WITNESS_SPECS:
        col = f"tau_{key}"
        values = central[col].to_numpy()
        low = grouped[col].min().reindex(central["L"]).to_numpy()
        high = grouped[col].max().reindex(central["L"]).to_numpy()
        ax.errorbar(
            central["L"],
            values,
            yerr=np.vstack([values - low, high - values]),
            marker=marker,
            capsize=2.5,
            label=label,
        )
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel(r"$\tau_Q^{\mathrm{mc,th}}$")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(b)")
    inset = ax.inset_axes([0.56, 0.55, 0.40, 0.38])
    inset.plot(central["L"], central["removed_fraction"], marker="o", lw=0.8)
    inset.set_xlabel(r"$L$", fontsize=6)
    inset.set_ylabel(r"$f_{\rm cage}$", fontsize=6)
    inset.tick_params(labelsize=6)

    ax = axes[2]
    for key, _label, marker in WITNESS_SPECS:
        ax.plot(overlap["L"], overlap[f"delta_{key}"], marker=marker, label=rf"$\delta_{key}$")
    ax.plot(
        overlap["L"],
        overlap["energy_density_mismatch"],
        marker="v",
        ls="--",
        label=r"$|e_\psi-e_{\beta=0}|$",
    )
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel("Matching difference")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=6)
    add_panel_label(ax, "(c)")

    ax = axes[3]
    c0 = concentration[np.isclose(concentration["window_prefactor"], primary_prefactor)]
    env = (
        c0.groupby("L")
        .agg(
            median=("basis_independent_std", "median"),
            maximum=("basis_independent_std", "max"),
        )
        .reset_index()
    )
    ax.plot(env["L"], env["median"], marker="o", label="median")
    ax.plot(env["L"], env["maximum"], marker="s", ls="--", label="maximum")
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel("Retained local spread")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    add_panel_label(ax, "(d)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4)
    for ax in axes[:2]:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    save_prx_figure(fig, "spin1_xy_undeformed_evidence_main", directory=figures, formats=formats)

    if not exact.empty:
        fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=PRX_TWO_PANEL_FIGSIZE)
        ax0.plot(exact["length"], exact["A2_direct_normalized"], marker="o", label=r"$Q_R^A$")
        ax0.plot(exact["length"], exact["Z2_direct_normalized"], marker="s", label=r"$Q_R^Z$")
        ax0.plot(exact["length"], exact["y2_activity"], marker="^", label=r"$Q_R^Y$")
        ax0.set_xlabel(r"System size $L$")
        ax0.set_ylabel(r"$\mathrm{Tr}(\rho_{\beta=0}Q_R)$")
        ax0.grid(alpha=0.22)
        ax0.legend(loc="upper right")
        add_panel_label(ax0, "(a)")
        if not deform.empty:
            max_l = int(deform["L"].max())
            frame = deform[deform["L"] == max_l]
            for key, label, marker in WITNESS_SPECS:
                column = {"A": "tau_A_normalized", "Z": "tau_Z_normalized", "Y": "tau_Y"}[key]
                ax1.plot(frame["J3_over_J"], frame[column], marker=marker, label=label)
            ax1.legend(loc="upper right")
            if "cage_excision_applied" in frame and not bool(frame["cage_excision_applied"].all()):
                ax1.text(
                    0.03,
                    0.04,
                    "ordinary window; cage excision pending",
                    transform=ax1.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
                )
        ax1.set_xlabel(r"Preserving exchange $J_3/J$")
        ax1.set_ylabel("Deformed local activity")
        ax1.grid(alpha=0.22)
        add_panel_label(ax1, "(b)")
        fig2.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.96, wspace=0.30)
        save_prx_figure(fig2, "spin1_xy_beta0_and_deformation", directory=figures, formats=formats)

    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
