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


def prefer_partial_rows(frame: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    """Deduplicate mixed exact/partial smoke products, preferring partial rows.

    Production normally has distinct sizes for the two methods, but smoke tests
    may intentionally reuse a small size to exercise the large-strip lane.
    """
    if frame.empty:
        return frame
    result = frame.copy()
    if "spectrum_method" in result.columns:
        result["_method_priority"] = (
            result["spectrum_method"].eq("shift_invert_partial").astype(int)
        )
    else:
        result["_method_priority"] = 0
    result = (
        result.sort_values([*keys, "_method_priority"])
        .drop_duplicates(keys, keep="last")
        .drop(columns="_method_priority")
    )
    return result


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
    if a.use_tex and not bool(plt.rcParams.get("text.usetex", False)):
        raise RuntimeError(
            "--use-tex requires a working LaTeX executable; refusing mathtext fallback"
        )
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
    if "window_coverage_complete" in primary.columns:
        large = primary["Lx"].astype(int) >= 12
        primary = primary[~large | primary["window_coverage_complete"].fillna(False).astype(bool)]
    if "converged_vs_previous_budget" in primary.columns:
        large = primary["Lx"].astype(int) >= 12
        converged = primary["converged_vs_previous_budget"].fillna(False).astype(bool)
        primary = primary[~large | converged]
    primary = prefer_partial_rows(primary, keys=["Lx", "phase", "window_prefactor"])
    phi = (
        float(rep.phi_star.iloc[0])
        if not rep.empty
        else float(sorted(primary.phase.unique())[len(primary.phase.unique()) // 2])
    )
    protocol = str(primary.thermal_protocol.iloc[0])
    reference_label = r"$\beta=0$ trace" if protocol == "beta0" else r"matched canonical"
    use_physical_target = "Delta_physical_target" in primary.columns

    # Match the spin-1 Fig. 6 physical canvas and nested-strip grammar.
    fig = plt.figure(figsize=(7.05, 6.85))
    outer = fig.add_gridspec(
        2, 2, left=0.08, right=0.955, bottom=0.08, top=0.955, wspace=0.38, hspace=0.34
    )
    representative = primary[np.isclose(primary.phase, phi)].sort_values("Lx")
    verified_lengths = set(representative["Lx"].astype(int))
    scatter_lengths = set(scatter["Lx"].astype(int))
    common_lengths = sorted(verified_lengths.intersection(scatter_lengths))
    if not common_lengths:
        raise RuntimeError("No verified thermal size has matching ETH-scatter data")
    largest = int(common_lengths[-1])
    scatter_largest = scatter[scatter.Lx.astype(int) == largest].copy()
    row = representative[representative.Lx.astype(int) == largest].iloc[-1]

    # (a) Witness-resolved ETH strips.  The raw comparison window and cage star
    # use exactly the same visual semantics as spin-1 Fig. 6.
    gsa = outer[0, 0].subgridspec(2, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gsa[i]) for i in range(2)]
    for index, ((col, label, _marker), axa) in enumerate(
        zip([("Q_A", r"$Q_R^A$", "o"), ("Q_Z", r"$Q_R^Z$", "s")], axes_a, strict=True)
    ):
        axa.axvspan(
            row.cage_energy_density - row.window_energy_density_half_width,
            row.cage_energy_density + row.window_energy_density_half_width,
            color="0.5",
            alpha=0.10,
            zorder=0,
        )
        axa.axvline(row.cage_energy_density, color="0.45", ls="--", lw=0.8)
        axa.scatter(
            scatter_largest.energy_density,
            scatter_largest[col],
            s=10,
            alpha=0.52,
            marker="o",
            linewidths=0,
        )
        axa.scatter(
            [row.cage_energy_density],
            [0],
            marker="*",
            s=78,
            edgecolors="black",
            linewidths=0.45,
            zorder=8,
        )
        axa.set_ylabel(label)
        axa.grid(alpha=0.18)
        if index == 0:
            axa.tick_params(labelbottom=False)
        else:
            axa.set_xlabel(r"Energy density $e=E/(4L_x)$")
    add_panel_label(axes_a[0], "(a)")

    gsb = outer[0, 1].subgridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.08)
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
    axb.grid(alpha=0.20)
    axb.tick_params(labelbottom=False)
    add_panel_label(axb, "(b)")
    axb2.set_xlabel(r"Strip length $L_x$")
    axb2.set_ylabel(r"$\delta_{\alpha,L_x}$")
    axb2.set_ylim(bottom=0.0)
    axb2.grid(alpha=0.20)
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

    gsc = outer[1, 0].subgridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.08)
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
    axc.axvline(phi, color=".45", ls=":", lw=0.9)
    axc2.axvline(phi, color=".45", ls=":", lw=0.9)
    axc.set_ylabel(r"$\Delta_{L_x}(\varphi)$")
    axc.grid(alpha=0.20)
    axc.tick_params(labelbottom=False)
    add_panel_label(axc, "(c)")
    axc.legend(fontsize=8.5)
    axc2.set_xlabel(r"Checkerboard phase $\varphi$")
    axc2.set_ylabel(r"$L_x\Delta_{L_x}$")
    axc2.grid(alpha=0.20)

    axd = fig.add_subplot(outer[1, 1])
    c = concentration[concentration.phase > 0].copy()
    if "window_coverage_complete" in c.columns:
        large = c["Lx"].astype(int) >= 12
        c = c[~large | c["window_coverage_complete"].fillna(False).astype(bool)]
    if "converged_vs_previous_budget" in c.columns:
        large = c["Lx"].astype(int) >= 12
        converged = c["converged_vs_previous_budget"].fillna(False).astype(bool)
        c = c[~large | converged]
    c = prefer_partial_rows(c, keys=["Lx", "phase"])
    if c.empty:
        axd.text(
            0.5, 0.5, "concentration unavailable", ha="center", va="center", transform=axd.transAxes
        )
    else:
        value_column = "w_raw" if "w_raw" in c.columns else "w"
        piv = c.pivot(index="Lx", columns="phase", values=value_column).sort_index()
        x = np.asarray(piv.columns, float)
        y = np.asarray(piv.index, int)
        mesh = axd.pcolormesh(edges(x, 0.0125), edges(y, 1.0), piv.to_numpy(), shading="flat")
        cb = fig.colorbar(mesh, ax=axd, pad=0.03)
        cb.ax.set_title(r"$w_{L_x}(\varphi)$", fontsize=9, pad=4)
        cb.ax.tick_params(labelsize=9)
        use_integer_ticks(axd, axis="y")
        axd.set_yticks(y)
        if len(y) == 1:
            axd.set_ylim(y[0] - 1, y[0] + 1)
    axd.axvline(phi, color="w", ls=":", lw=1.0, alpha=0.9)
    axd.set_xlabel(r"Checkerboard phase $\varphi$")
    axd.set_ylabel(r"Strip length $L_x$")
    add_panel_label(axd, "(d)")
    # Keep the historical stem and emit the manuscript-facing Fig. 9 alias.
    save_prx_figure(fig, "qdm_checkerboard_figure7_combined", directory=figs, formats=formats)
    save_prx_figure(fig, "qdm_checkerboard_figure9_prx", directory=figs, formats=formats)
    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
