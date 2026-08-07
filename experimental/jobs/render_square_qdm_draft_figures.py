#!/usr/bin/env python
"""Render final-size REVTeX figures from square-QDM checkerboard evidence tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate qlinks repository")
sys.path[:0] = [str(ROOT / "experimental" / "notebooks"), str(ROOT)]

from helpers import (  # noqa: E402
    PRX_FOUR_PANEL_FIGSIZE,
    add_panel_label,
    save_prx_figure,
    set_revtex_matplotlib_style,
    use_integer_ticks,
    write_figure_manifest,
)


def read(path):
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def edges(values, default_half):
    values = np.asarray(values, float)
    if len(values) == 1:
        return np.array([values[0] - default_half, values[0] + default_half])
    return np.r_[
        values[0] - (values[1] - values[0]) / 2,
        (values[:-1] + values[1:]) / 2,
        values[-1] + (values[-1] - values[-2]) / 2,
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--figure-formats", default="pdf,svg")
    p.add_argument("--use-tex", action="store_true")
    a = p.parse_args()
    data = a.data_dir.resolve()
    figs = data / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    formats = tuple(x.strip() for x in a.figure_formats.split(",") if x.strip())
    set_revtex_matplotlib_style(base_font_size=9.0, prefer_tex=a.use_tex)
    thermal_path = data / "qdm_checkerboard_thermal_overlap.csv"
    thermal = read(
        thermal_path if thermal_path.exists() else data / "qdm_checkerboard_beta0_overlap.csv"
    )
    scatter = read(data / "qdm_checkerboard_eth_scatter.csv")
    concentration = read(data / "qdm_checkerboard_concentration_grid.csv")
    rep = read(data / "qdm_checkerboard_representative_phase.csv")
    # gates = read(data / "qdm_checkerboard_scientific_gates.csv")
    if thermal.empty or scatter.empty:
        raise RuntimeError("Checkerboard thermal products are unavailable; run compute first.")
    primary_pref = float(
        thermal.window_prefactor.iloc[(thermal.window_prefactor - 0.75).abs().argmin()]
    )
    primary = thermal[np.isclose(thermal.window_prefactor, primary_pref)].copy()
    phi = (
        float(rep.phi_star.iloc[0])
        if not rep.empty
        else float(sorted(primary.phase.unique())[len(primary.phase.unique()) // 2])
    )
    protocol = str(primary.thermal_protocol.iloc[0])
    reference_label = r"$\beta=0$ trace" if protocol == "beta0" else r"matched canonical"
    use_physical_target = "Delta_physical_target" in primary.columns

    fig = plt.figure(figsize=PRX_FOUR_PANEL_FIGSIZE)
    outer = fig.add_gridspec(
        2, 2, left=0.085, right=0.975, bottom=0.10, top=0.91, wspace=0.32, hspace=0.36
    )
    axa = fig.add_subplot(outer[0, 0])
    # Background file contains only joint-dark-cleaned states. The cage is drawn once as a star.
    for col, label, marker in [("Q_A", r"$Q_R^A$", "o"), ("Q_Z", r"$Q_R^Z$", "s")]:
        axa.scatter(
            scatter.energy_density, scatter[col], s=10, alpha=0.52, marker=marker, label=label
        )
    largest = int(scatter.Lx.max())
    row = primary[(primary.Lx == largest) & np.isclose(primary.phase, phi)].iloc[0]
    axa.axvspan(
        row.cage_energy_density - row.window_energy_density_half_width,
        row.cage_energy_density + row.window_energy_density_half_width,
        color="0.5",
        alpha=0.10,
        zorder=0,
    )
    axa.scatter(
        [row.cage_energy_density],
        [0],
        marker="*",
        s=76,
        edgecolors="black",
        linewidths=0.45,
        zorder=7,
        label="compact cage",
    )
    axa.set_xlabel(r"Energy density $e=E/(4L_x)$")
    axa.set_ylabel("Witness activity")
    axa.grid(alpha=0.22)
    add_panel_label(axa, "(a)")

    gsb = outer[0, 1].subgridspec(2, 1, height_ratios=(3.0, 1.35), hspace=0.08)
    axb = fig.add_subplot(gsb[0])
    axb2 = fig.add_subplot(gsb[1], sharex=axb)
    r = primary[np.isclose(primary.phase, phi)].sort_values("Lx")
    for key, label, marker in [("A", r"$Q_R^A$", "o"), ("Z", r"$Q_R^Z$", "s")]:
        reference_column = (
            f"tau_{key}_reference_physical" if use_physical_target else f"tau_{key}_reference"
        )
        delta_column = f"delta_{key}_physical_target" if use_physical_target else f"delta_{key}"
        axb.plot(r.Lx, r[f"tau_{key}_mc"], marker=marker, label=label)
        axb.plot(r.Lx, r[reference_column], marker=marker, fillstyle="none", ls="--")
        axb2.plot(r.Lx, r[delta_column], marker=marker, label=rf"$\delta_{key}$")
    axb.set_ylabel(r"Local activity $\tau$")
    axb.grid(alpha=0.22)
    axb.tick_params(labelbottom=False)
    add_panel_label(axb, "(b)")
    axb2.set_xlabel(r"Strip length $L_x$")
    axb2.set_ylabel(r"$\delta$")
    axb2.grid(alpha=0.22)
    use_integer_ticks(axb2, axis="x")
    axb2.set_xticks(r.Lx.astype(int))
    if r.Lx.nunique() == 1:
        axb2.set_xlim(float(r.Lx.iloc[0]) - 0.5, float(r.Lx.iloc[0]) + 0.5)
    style_handles = [
        Line2D([0], [0], marker="o", color="0.25", lw=1, label="microcanonical"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            color="0.25",
            ls="--",
            lw=1,
            label=reference_label,
        ),
    ]
    axb.legend(handles=style_handles, fontsize=8.2, loc="best")

    gsc = outer[1, 0].subgridspec(2, 1, height_ratios=(3.0, 1.35), hspace=0.08)
    axc = fig.add_subplot(gsc[0])
    axc2 = fig.add_subplot(gsc[1], sharex=axc)
    fam = primary[primary.phase > 0].sort_values(["Lx", "phase"])
    matching_column = "Delta_physical_target" if use_physical_target else "Delta"
    for lx, g in fam.groupby("Lx"):
        g = g.dropna(subset=[matching_column])
        if g.empty:
            continue
        axc.plot(g.phase, g[matching_column], marker="o", label=rf"$L_x={int(lx)}$")
        axc2.plot(g.phase, int(lx) * g[matching_column], marker="o")
    axc.axvline(phi, color=".45", ls="--", lw=0.8)
    axc.set_ylabel(r"$\Delta_{L_x}(\varphi)$")
    axc.grid(alpha=0.22)
    axc.tick_params(labelbottom=False)
    add_panel_label(axc, "(c)")
    axc.legend(fontsize=8.5)
    axc2.set_xlabel(r"Checkerboard phase $\varphi$")
    axc2.set_ylabel(r"$L_x\Delta_{L_x}$")
    axc2.grid(alpha=0.22)

    axd = fig.add_subplot(outer[1, 1])
    c = concentration[concentration.phase > 0].copy()
    if c.empty:
        axd.text(
            0.5, 0.5, "concentration unavailable", ha="center", va="center", transform=axd.transAxes
        )
    else:
        piv = c.pivot(index="Lx", columns="phase", values="w").sort_index()
        x = np.asarray(piv.columns, float)
        y = np.asarray(piv.index, int)
        mesh = axd.pcolormesh(edges(x, 0.0125), edges(y, 1.0), piv.to_numpy(), shading="flat")
        cb = fig.colorbar(mesh, ax=axd, pad=0.03)
        cb.set_label(r"$w_{L_x}(\varphi)$")
        cb.ax.tick_params(labelsize=8.5)
        use_integer_ticks(axd, axis="y")
        axd.set_yticks(y)
        if len(y) == 1:
            axd.set_ylim(y[0] - 1, y[0] + 1)
    axd.axvline(phi, color="w", ls="--", lw=0.9, alpha=0.8)
    axd.set_xlabel(r"Checkerboard phase $\varphi$")
    axd.set_ylabel(r"Strip length $L_x$")
    add_panel_label(axd, "(d)")
    h, l = axa.get_legend_handles_labels()  # noqa: E741
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=8.8)
    save_prx_figure(fig, "qdm_checkerboard_figure7_combined", directory=figs, formats=formats)
    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
