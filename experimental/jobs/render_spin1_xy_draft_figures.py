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
    for candidate in (
        f"tau_{key}_resolved_beta0_clean",
        f"tau_{key}_resolved_beta0",
        f"tau_{key}_beta0",
    ):
        if candidate in frame:
            return candidate
    raise KeyError(f"No resolved beta=0 column for witness {key}")


def _centered_edges(values: np.ndarray, *, fallback_half_width: float) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.size == 1:
        return np.asarray(
            [coordinates[0] - fallback_half_width, coordinates[0] + fallback_half_width]
        )
    midpoint = 0.5 * (coordinates[:-1] + coordinates[1:])
    return np.concatenate(
        (
            [coordinates[0] - (midpoint[0] - coordinates[0])],
            midpoint,
            [coordinates[-1] + (coordinates[-1] - midpoint[-1])],
        )
    )


def _primary_sequence(sequence: pd.DataFrame) -> pd.DataFrame:
    mask = np.isclose(sequence["window_prefactor"], 1.0)
    if "window_exponent" in sequence:
        mask &= np.isclose(sequence["window_exponent"], 0.5)
    if "window_coverage_complete" in sequence:
        mask &= sequence["window_coverage_complete"].fillna(False).astype(bool)
    return sequence[mask].sort_values("L")


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

    central = _primary_sequence(sequence)
    largest_scatter = int(scatter["L"].max())
    scatter_largest = scatter[scatter["L"] == largest_scatter]
    row = central[central["L"] == largest_scatter].iloc[0]

    # Nested lower strips replace overlaying insets in panels (b) and (c).
    fig = plt.figure(figsize=(PRX_FOUR_PANEL_FIGSIZE[0], 6.45))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.94,
        bottom=0.085,
        top=0.92,
        wspace=0.48,
        hspace=0.40,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gs[0, 1].subgridspec(2, 1, height_ratios=(2.25, 1.0), hspace=0.08)
    ax_b = fig.add_subplot(gs_b[0])
    ax_b_delta = fig.add_subplot(gs_b[1], sharex=ax_b)
    gs_c = gs[1, 0].subgridspec(2, 1, height_ratios=(2.35, 1.0), hspace=0.48)
    ax_c = fig.add_subplot(gs_c[0])
    ax_c_delta = fig.add_subplot(gs_c[1])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) Reference-point ETH scatter.
    half = float(row["window_energy_density_half_width"])
    ax_a.axvspan(-half, half, color="0.5", alpha=0.10, zorder=0)
    ax_a.axvline(0.0, color="0.45", ls="--", lw=0.8)
    retained = scatter_largest[~scatter_largest["is_exceptional"].astype(bool)]
    removed = scatter_largest[scatter_largest["is_exceptional"].astype(bool)]
    for key, label, marker in WITNESS_SPECS:
        column = f"Q_{key}"
        ax_a.scatter(
            retained["energy_density"],
            retained[column],
            s=10,
            alpha=0.50,
            marker=marker,
            label=label,
        )
        if not removed.empty:
            ax_a.scatter(
                removed["energy_density"],
                removed[column],
                s=20,
                marker=marker,
                facecolors="none",
                edgecolors="0.25",
                linewidths=0.6,
            )
    ax_a.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=75,
        edgecolors="black",
        linewidths=0.4,
        label="selected tower",
        zorder=5,
    )
    ax_a.set_xlabel(r"Energy density $e=E/L$")
    ax_a.set_ylabel("Local witness activity")
    ax_a.grid(alpha=0.22)
    add_panel_label(ax_a, "(a)")

    # (b) Reference-point clean microcanonical and clean resolved beta=0 values.
    for key, label, marker in WITNESS_SPECS:
        mc_column = (
            f"tau_{key}_mc_clean" if f"tau_{key}_mc_clean" in overlap else f"tau_{key}_mc_th"
        )
        beta_column = _resolved_beta0_column(overlap, key)
        line = ax_b.plot(overlap["L"], overlap[mc_column], marker=marker, label=label)[0]
        ax_b.plot(
            overlap["L"],
            overlap[beta_column],
            linestyle="--",
            color=line.get_color(),
        )
        delta_column = (
            f"delta_{key}_clean_clean" if f"delta_{key}_clean_clean" in overlap else f"delta_{key}"
        )
        regular = overlap[overlap["L"] >= 8]
        pre = overlap[overlap["L"] < 8]
        ax_b_delta.plot(regular["L"], regular[delta_column], marker=marker, label=label)
        if not pre.empty:
            ax_b_delta.scatter(pre["L"], pre[delta_column], marker=marker, color="0.55", zorder=4)
    ax_b.set_ylabel("Local activity")
    ax_b.grid(alpha=0.22)
    ax_b.legend(loc="lower right", fontsize=9)
    ax_b.tick_params(labelbottom=False)
    use_integer_ticks(ax_b, axis="x")
    ax_b.set_xticks(np.sort(overlap["L"].unique()).astype(int))
    add_panel_label(ax_b, "(b)")
    ax_b_delta.set_xlabel(r"System size $L$")
    ax_b_delta.set_ylabel(r"$\delta_{\alpha,L}$")
    ax_b_delta.grid(alpha=0.20)
    use_integer_ticks(ax_b_delta, axis="x")
    ax_b_delta.set_xticks(np.sort(overlap["L"].unique()).astype(int))

    # (c) Ambient complex-t2 residual plane with a non-overlapping matching strip.
    pivot = obstruction.pivot(
        index="imag_t2_over_J",
        columns="real_t2_over_J",
        values="normalized_tower_residual",
    )
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    mesh = ax_c.pcolormesh(
        _centered_edges(x, fallback_half_width=0.01),
        _centered_edges(y, fallback_half_width=0.01),
        pivot.to_numpy(),
        shading="flat",
    )
    ax_c.axvline(0.0, linestyle="--", linewidth=1.0, color="white")
    ax_c.set_xlabel(r"$\operatorname{Re}t_2/J$")
    ax_c.set_ylabel(r"$\operatorname{Im}t_2/J$")
    colorbar = fig.colorbar(mesh, ax=ax_c, pad=0.02)
    colorbar.set_label("Normalized tower residual", fontsize=9)
    colorbar.ax.tick_params(labelsize=9)
    add_panel_label(ax_c, "(c)")
    for length, frame in deformation.groupby("L"):
        frame = frame.sort_values("kappa_over_J")
        ax_c_delta.plot(
            frame["kappa_over_J"],
            frame["delta_max"],
            marker="o",
            label=rf"$L={int(length)}$",
        )
    ax_c_delta.set_xlabel(r"$\kappa/J$")
    ax_c_delta.set_ylabel(r"$\Delta_L(\kappa)$")
    ax_c_delta.grid(alpha=0.20)
    ax_c_delta.legend(fontsize=8.5, loc="upper right", ncol=2)

    # (d) Complete local-algebra concentration.
    if not concentration.empty:
        cpivot = concentration.pivot(
            index="L",
            columns="kappa_over_J",
            values="largest_covariance_width",
        )
        cx = cpivot.columns.to_numpy(dtype=float)
        cy = cpivot.index.to_numpy(dtype=float)
        cmesh = ax_d.pcolormesh(
            _centered_edges(cx, fallback_half_width=0.025),
            _centered_edges(cy, fallback_half_width=0.5),
            cpivot.to_numpy(),
            shading="flat",
        )
        ax_d.set_yticks(cy.astype(int))
        colorbar = fig.colorbar(cmesh, ax=ax_d, pad=0.02)
        colorbar.set_label(r"$\sqrt{\lambda_{\max}(\Gamma)}$", fontsize=9)
        colorbar.ax.tick_params(labelsize=9)
    else:
        ax_d.text(0.5, 0.5, "concentration data not computed", ha="center", va="center")
    ax_d.set_xlabel(r"$\kappa/J$")
    ax_d.set_ylabel(r"System size $L$")
    use_integer_ticks(ax_d, axis="y")
    if not concentration.empty:
        ax_d.set_yticks(np.sort(concentration["L"].unique()).astype(int))
    add_panel_label(ax_d, "(d)")

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    save_prx_figure(fig, "spin1_xy_figure6_combined", directory=figures, formats=formats)

    # Separate matching-distance figures for reading and extrapolation checks.
    fig_delta, (ax0, ax1) = plt.subplots(1, 2, figsize=PRX_TWO_PANEL_FIGSIZE)
    for key, label, marker in WITNESS_SPECS:
        column = (
            f"delta_{key}_clean_clean" if f"delta_{key}_clean_clean" in overlap else f"delta_{key}"
        )
        regular = overlap[overlap["L"] >= 8]
        pre = overlap[overlap["L"] < 8]
        ax0.plot(regular["L"], regular[column], marker=marker, label=label)
        if not pre.empty:
            ax0.scatter(pre["L"], pre[column], marker=marker, color="0.55")
    ax0.set_xlabel(r"System size $L$")
    ax0.set_ylabel(r"$\delta_{\alpha,L}$")
    use_integer_ticks(ax0, axis="x")
    ax0.grid(alpha=0.22)
    ax0.legend(loc="upper right")
    add_panel_label(ax0, "(a)")
    for length, frame in deformation.groupby("L"):
        frame = frame.sort_values("kappa_over_J")
        ax1.plot(frame["kappa_over_J"], frame["delta_max"], marker="o", label=rf"$L={int(length)}$")
    ax1.set_xlabel(r"Compatible deformation $\kappa/J$")
    ax1.set_ylabel(r"$\Delta_L(\kappa)$")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="upper right", ncol=2)
    add_panel_label(ax1, "(b)")
    fig_delta.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.96, wspace=0.30)
    save_prx_figure(
        fig_delta,
        "spin1_xy_matching_distances_separate",
        directory=figures,
        formats=formats,
    )

    # Supporting two-panel output for exact fixed-M targets and deformation matching.
    if not exact.empty:
        fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=PRX_TWO_PANEL_FIGSIZE)
        a_column = (
            "A_activity_normalized" if "A_activity_normalized" in exact else "A2_direct_normalized"
        )
        z_column = (
            "Z_activity_normalized" if "Z_activity_normalized" in exact else "Z2_direct_normalized"
        )
        ax0.plot(exact["length"], exact[a_column], marker="o", label=r"$Q_R^A$")
        ax0.plot(exact["length"], exact[z_column], marker="s", label=r"$Q_R^Z$")
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
        ax1.set_ylabel(r"$\Delta_L(\kappa)$")
        ax1.grid(alpha=0.22)
        ax1.legend(loc="upper right", ncol=2)
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
