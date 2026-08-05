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
    use_integer_ticks,
    write_figure_manifest,
)

WITNESS_SPECS = [
    ("A", r"$Q_R^A$", "o"),
    ("Z", r"$Q_R^Z$", "s"),
    ("Y", r"$Q_R^Y$", "^"),
]


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _resolved_beta0_column(frame: pd.DataFrame, key: str) -> str:
    for candidate in (f"tau_{key}_resolved_beta0", f"tau_{key}_beta0"):
        if candidate in frame:
            return candidate
    raise KeyError(f"No resolved beta=0 column for witness {key}")


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
    set_revtex_matplotlib_style(base_font_size=9.0, prefer_tex=args.use_tex)

    sequence = pd.read_csv(data / "spin1_xy_cage_excised_sequence.csv")
    scatter = pd.read_csv(data / "spin1_xy_cage_excised_eth_scatter.csv")
    overlap = pd.read_csv(data / "spin1_xy_beta0_cage_excised_overlap.csv")
    obstruction = pd.read_csv(data / "spin1_xy_complex_t2_obstruction_grid.csv")
    deformation = pd.read_csv(data / "spin1_xy_kappa_matching_grid.csv")
    concentration = _read_optional(data / "spin1_xy_kappa_concentration_grid.csv")
    exact = _read_optional(data / "exact_fixed_M_activities.csv")

    prefactors = np.sort(sequence["window_prefactor"].unique())
    primary_prefactor = float(prefactors[np.argmin(np.abs(prefactors - 1.0))])
    central = sequence[np.isclose(sequence["window_prefactor"], primary_prefactor)].sort_values("L")
    largest = int(central["L"].max())
    scatter_largest = scatter[scatter["L"] == largest]
    row = central[central["L"] == largest].iloc[0]

    fig = plt.figure(figsize=(PRX_FOUR_PANEL_FIGSIZE[0], 6.15))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.09,
        top=0.91,
        wspace=0.38,
        hspace=0.38,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # (a) Reference-point ETH scatter.
    ax = axes[0]
    half = float(row["window_energy_density_half_width"])
    ax.axvspan(-half, half, color="0.5", alpha=0.10, zorder=0)
    ax.axvline(0.0, color="0.45", ls="--", lw=0.8)
    retained = scatter_largest[~scatter_largest["is_exceptional"].astype(bool)]
    removed = scatter_largest[scatter_largest["is_exceptional"].astype(bool)]
    for key, label, marker in WITNESS_SPECS:
        column = f"Q_{key}"
        ax.scatter(
            retained["energy_density"],
            retained[column],
            s=10,
            alpha=0.50,
            marker=marker,
            label=label,
        )
        if not removed.empty:
            ax.scatter(
                removed["energy_density"],
                removed[column],
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
        s=75,
        edgecolors="black",
        linewidths=0.4,
        label="selected tower",
        zorder=5,
    )
    ax.set_xlabel(r"Energy density $e=E/L$")
    ax.set_ylabel("Local witness activity")
    ax.grid(alpha=0.22)
    add_panel_label(ax, "(a)")

    # (b) Reference-point microcanonical and resolved beta=0 values.
    ax = axes[1]
    grouped = sequence.groupby("L")
    for key, label, marker in WITNESS_SPECS:
        mc_column = f"tau_{key}_mc_th"
        beta_column = _resolved_beta0_column(overlap, key)
        line = ax.plot(
            overlap["L"],
            overlap[mc_column],
            marker=marker,
            label=label,
        )[0]
        ax.plot(
            overlap["L"],
            overlap[beta_column],
            linestyle="--",
            color=line.get_color(),
        )
        sequence_column = f"tau_{key}"
        if sequence_column in sequence:
            values = central[sequence_column].to_numpy()
            low = grouped[sequence_column].min().reindex(central["L"]).to_numpy()
            high = grouped[sequence_column].max().reindex(central["L"]).to_numpy()
            ax.errorbar(
                central["L"],
                values,
                yerr=np.vstack([values - low, high - values]),
                fmt="none",
                ecolor=line.get_color(),
                capsize=2.5,
                linewidth=0.8,
            )
    ax.set_xlabel(r"System size $L$")
    ax.set_ylabel("Local activity")
    use_integer_ticks(ax, axis="x")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8)
    add_panel_label(ax, "(b)")
    inset = ax.inset_axes([0.56, 0.55, 0.40, 0.36])
    if "delta_max" in overlap:
        inset.plot(overlap["L"], overlap["delta_max"], marker="o")
    else:
        delta_columns = [column for column in overlap if column.startswith("delta_")]
        inset.plot(overlap["L"], overlap[delta_columns].max(axis=1), marker="o")
    inset.set_xlabel(r"$L$", fontsize=8)
    inset.set_ylabel(r"$\delta_L(0)$", fontsize=8)
    inset.tick_params(labelsize=8)
    use_integer_ticks(inset, axis="x")

    # (c) Ambient complex-t2 residual plane, matching along compatible line in inset.
    ax = axes[2]
    pivot = obstruction.pivot(
        index="imag_t2_over_J",
        columns="real_t2_over_J",
        values="normalized_tower_residual",
    )
    mesh = ax.pcolormesh(
        pivot.columns.to_numpy(),
        pivot.index.to_numpy(),
        pivot.to_numpy(),
        shading="auto",
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="white")
    ax.set_xlabel(r"$\operatorname{Re}t_2/J$")
    ax.set_ylabel(r"$\operatorname{Im}t_2/J$")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Normalized tower residual", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    add_panel_label(ax, "(c)")
    inset = ax.inset_axes([0.49, 0.55, 0.46, 0.38])
    for length, frame in deformation.groupby("L"):
        frame = frame.sort_values("kappa_over_J")
        inset.plot(
            frame["kappa_over_J"],
            frame["delta_max"],
            marker="o",
            label=rf"$L={int(length)}$",
        )
    inset.set_xlabel(r"$\kappa/J$", fontsize=8)
    inset.set_ylabel(r"$\delta_L(\kappa)$", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.legend(fontsize=7, loc="upper right")

    # (d) Complete local-algebra concentration.
    ax = axes[3]
    if not concentration.empty:
        cpivot = concentration.pivot(
            index="L",
            columns="kappa_over_J",
            values="largest_covariance_width",
        )
        cmesh = ax.pcolormesh(
            cpivot.columns.to_numpy(),
            cpivot.index.to_numpy(),
            cpivot.to_numpy(),
            shading="nearest",
        )
        colorbar = fig.colorbar(cmesh, ax=ax, pad=0.02)
        colorbar.set_label(r"$\sqrt{\lambda_{\max}(\Gamma)}$", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "concentration data not computed", ha="center", va="center")
    ax.set_xlabel(r"$\kappa/J$")
    ax.set_ylabel(r"System size $L$")
    use_integer_ticks(ax, axis="y")
    add_panel_label(ax, "(d)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=4,
        frameon=False,
    )
    save_prx_figure(
        fig,
        "spin1_xy_figure6_combined",
        directory=figures,
        formats=formats,
    )

    # Supporting two-panel output for exact fixed-M targets and deformation matching.
    if not exact.empty:
        fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=PRX_TWO_PANEL_FIGSIZE)
        ax0.plot(exact["length"], exact["A2_direct_normalized"], marker="o", label=r"$Q_R^A$")
        ax0.plot(exact["length"], exact["Z2_direct_normalized"], marker="s", label=r"$Q_R^Z$")
        ax0.plot(exact["length"], exact["y2_activity"], marker="^", label=r"$Q_R^Y$")
        ax0.set_xlabel(r"System size $L$")
        ax0.set_ylabel(r"$\mathrm{Tr}(\rho_{\beta=0,M}Q_R)$")
        use_integer_ticks(ax0, axis="x")
        ax0.grid(alpha=0.22)
        ax0.legend(loc="upper right")
        add_panel_label(ax0, "(a)")

        for length, frame in deformation.groupby("L"):
            frame = frame.sort_values("kappa_over_J")
            ax1.plot(
                frame["kappa_over_J"],
                frame["delta_max"],
                marker="o",
                label=rf"$L={int(length)}$",
            )
        ax1.set_xlabel(r"Compatible deformation $\kappa/J$")
        ax1.set_ylabel(r"$\delta_L(\kappa)$")
        ax1.grid(alpha=0.22)
        ax1.legend(loc="upper right")
        add_panel_label(ax1, "(b)")
        fig2.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.96, wspace=0.30)
        save_prx_figure(
            fig2,
            "spin1_xy_beta0_and_deformation",
            directory=figures,
            formats=formats,
        )

    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
