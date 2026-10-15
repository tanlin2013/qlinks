"""Render the frozen 8x4 target-parity comparison without eigensolves or 12x4 data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from qdm_parity_protocol import PROTOCOL

PHASES = (0.025, 0.05, 0.075, 0.10)
WIDTHS = (0.10, 0.20, 0.25, 0.50)
COLORS = {"A": "#0072B2", "Z": "#009E73"}


def load_evidence(data: Path):
    """Check frozen certificates and independently reconcile all raw/canonical means."""
    paths = [
        data / "qdm_parity_all_eigenstates.csv",
        data / "qdm_parity_phase_window_systematics.csv",
        data / "qdm_parity_exact_audit.json",
    ]
    scatter, table = (pd.read_csv(path) for path in paths[:2])
    audit = json.loads(paths[2].read_text())
    for frame in (scatter, table):
        if (
            not frame.symmetry_protocol.eq(PROTOCOL).all()
            or not frame.parity_policy.eq("measured_target").all()
            or not frame.fully_symmetry_resolved.eq(True).all()  # noqa: E712
            or not set(frame.Lx).issubset({4, 8})
            or not frame.Ly.eq(4).all()
            or not frame.loc[frame.Lx.eq(8), "Sy"].eq(1).all()
            or not frame.loc[frame.Lx.eq(4), "Sy"].eq(-1).all()
        ):
            raise ValueError("Fig. 9 requires the frozen 4x4/8x4 measured-target parity protocol")
    table = table.loc[table.Lx.eq(8)].copy()
    scatter = scatter.loc[scatter.Lx.eq(8)].copy()
    if len(table) != 16 or table.duplicated(["phase", "window_half_width"]).any():
        raise ValueError("Expected one 8x4 row for every prescribed phase/window pair")
    if not table.sector_dimension.eq(875).all() or not table.raw_population_includes_target.all():
        raise ValueError("The 8x4 raw population must include the target in the 875-state block")
    if not table.operator_normalization.eq("local_HS_quotient_in_target_parity").all():
        raise ValueError("Unexpected stripe-operator normalization")
    for key in ("maximum_eigenpair_residual", "gram_frobenius_residual"):
        if not np.isfinite(table[key]).all() or not table[key].between(0, 1e-8).all():
            raise ValueError(f"Frozen eigensystem certificate fails: {key}")
    for phase in PHASES:
        block = scatter.loc[np.isclose(scatter.phase, phase)]
        rows = table.loc[np.isclose(table.phase, phase)]
        if len(block) != 875 or block.is_tower_state.sum() != 1:
            raise ValueError("Incomplete eigensystem population or target multiplicity")
        if not np.allclose(sorted(rows.window_half_width), WIDTHS, atol=1e-12, rtol=0):
            raise ValueError("Prescribed window controls are incomplete")
        if not np.isfinite(block[["energy", "energy_density", "Q_A", "Q_Z"]]).all().all():
            raise ValueError("Nonfinite scatter data")
        if not np.allclose(block.energy_density, block.energy / 32, atol=1e-12):
            raise ValueError("Energy-density convention changed")
        if not np.allclose(block.loc[block.is_tower_state, ["Q_A", "Q_Z"]], 0, atol=1e-8):
            raise ValueError("The target is not locally dark")
        blocks = [b for b in audit["blocks"] if b["Lx"] == 8 and np.isclose(b["phase"], phase)]
        if len(blocks) != 1 or blocks[0]["dimension"] != 875 or blocks[0]["Sy"] != 1:
            raise ValueError("The exact audit does not identify the selected parity block")
        for row in rows.itertuples():
            indices = (
                np.abs(block.energy.to_numpy() - row.target_energy) <= row.window_half_width + 1e-9
            )
            if indices.sum() != row.raw_window_state_count:
                raise ValueError("Raw-window state count differs from the eigensystem")
            weights = np.exp(-row.matched_beta_raw * (block.energy.to_numpy() - row.target_energy))
            weights /= weights.sum()
            if not np.isclose(weights @ block.energy, row.target_energy, atol=1e-8):
                raise ValueError("Canonical energy matching fails")
            mismatches = []
            for name in COLORS:
                raw = block.loc[indices, f"Q_{name}"].mean()
                canonical = weights @ block[f"Q_{name}"]
                if not np.allclose(
                    [raw, canonical],
                    [getattr(row, f"tau_{name}_mc_raw"), getattr(row, f"tau_{name}_can_raw")],
                    atol=1e-10,
                    rtol=0,
                ):
                    raise ValueError("Raw/canonical means differ from the frozen eigensystem")
                mismatches.append(abs(raw - canonical))
            if not np.isclose(max(mismatches), row.matching_distance_raw, atol=1e-10):
                raise ValueError("Ensemble mismatch differs from the frozen means")
            if not np.isfinite(row.w_raw) or row.w_raw < 0:
                raise ValueError("Invalid stripe width")
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return scatter, table, sources


def render(data: Path, output: Path):
    scatter, table, sources = load_evidence(data)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )
    fig = plt.figure(figsize=(7.1, 4.9))
    grid = fig.add_gridspec(
        2, 2, left=0.095, right=0.985, bottom=0.105, top=0.94, hspace=0.47, wspace=0.42
    )
    agrid, bgrid = (
        grid[0, 0].subgridspec(2, 1, hspace=0.10),
        grid[0, 1].subgridspec(2, 1, hspace=0.14),
    )
    axes_a, axes_b = [], []
    representative = table.loc[np.isclose(table.phase, 0.05)].sort_values("window_half_width")
    primary = table.loc[np.isclose(table.window_half_width, 0.25)].sort_values("phase")
    background = scatter.loc[np.isclose(scatter.phase, 0.05) & ~scatter.is_tower_state]
    for index, name in enumerate(COLORS):
        color = COLORS[name]
        ax = fig.add_subplot(agrid[index])
        axes_a.append(ax)
        ax.axvspan((8 - 0.25) / 32, (8 + 0.25) / 32, color="0.5", alpha=0.15)
        ax.scatter(
            background.energy_density,
            background[f"Q_{name}"],
            s=6,
            alpha=0.42,
            color=color,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            [0.25],
            [0],
            marker="*",
            s=65,
            color="#E69F00",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        ax.set_ylabel(rf"$\langle\widehat Q_R^{{{name}}}\rangle_n$")
        ax.set_ylim(bottom=-0.008)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.grid(alpha=0.12)
        if index == 0:
            ax.tick_params(labelbottom=False)
            ax.text(
                0.98,
                0.94,
                r"$8\times4,\ S_y=+1$",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=7,
            )
        else:
            ax.set_xlabel(r"Energy density $e/J=E/(32J)$")
        bx = fig.add_subplot(bgrid[index])
        axes_b.append(bx)
        bx.plot(
            primary.phase,
            primary[f"tau_{name}_mc_raw"],
            "o-",
            color=color,
            markersize=4,
            linewidth=1,
        )
        bx.plot(
            primary.phase,
            primary[f"tau_{name}_can_raw"],
            "o--",
            color=color,
            markerfacecolor="white",
            markersize=4,
            linewidth=1,
        )
        bx.set_ylabel(rf"$\tau_{{{name},8}}$")
        bx.yaxis.set_major_locator(MaxNLocator(3))
        bx.grid(alpha=0.12)
        if index == 0:
            bx.tick_params(labelbottom=False)
        else:
            bx.set_xlabel(r"Checkerboard phase $\varphi$")
            bx.set_xticks(PHASES)
    axes_b[0].legend(
        handles=[
            Line2D([], [], color="0.2", marker="o", label="Raw mc"),
            Line2D(
                [],
                [],
                color="0.2",
                marker="o",
                markerfacecolor="white",
                linestyle="--",
                label="Matched canonical",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(1.02, 1.06),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1,
    )
    lower_axes = []
    for column, (key, ylabel, color) in enumerate(
        [
            ("matching_distance_raw", r"Ensemble mismatch $\Delta_8$", "#7B3294"),
            ("w_raw", r"Raw stripe width $w_8^{\rm raw}$", COLORS["A"]),
        ]
    ):
        ax = fig.add_subplot(grid[1, column])
        lower_axes.append(ax)
        grouped = table.groupby("window_half_width")[key]
        low, high = grouped.min().reindex(WIDTHS), grouped.max().reindex(WIDTHS)
        values = representative[key].to_numpy()
        ax.axvline(0.25, color="0.6", linestyle=":", linewidth=0.8, zorder=0)
        ax.plot(WIDTHS, values, color=color, linestyle=":", linewidth=0.8, zorder=1)
        ax.errorbar(
            WIDTHS,
            values,
            yerr=np.maximum(0, np.vstack([values - low, high - values])),
            fmt="o",
            color=color,
            markersize=4.5,
            capsize=4,
            elinewidth=1.2,
            zorder=3,
        )
        ax.set_xlabel(r"Total-energy half-width $\Delta E/J$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WIDTHS)
        ax.set_ylim(0, float(high.max()) * 1.18)
        ax.grid(alpha=0.12)
        ax.legend(
            handles=[
                Line2D(
                    [], [], color=color, marker="o", linestyle="none", label=r"$\varphi_\star=0.05$"
                ),
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="|",
                    markersize=10,
                    linestyle="none",
                    label=r"Sampled $\varphi$ range",
                ),
            ],
            loc="lower left",
            frameon=False,
        )
    for ax, label in zip(
        [axes_a[0], axes_b[0], *lower_axes], ["(a)", "(b)", "(c)", "(d)"], strict=True
    ):
        ax.text(
            -0.18, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom"
        )
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"qdm_checkerboard_figure9_target_parity_8x4.{suffix}", dpi=300)
    plt.close(fig)
    scatter.to_csv(output / "figure9_scatter_source.csv", index=False)
    table.to_csv(output / "figure9_thermal_source.csv", index=False)
    report = {
        "scope": "8x4 target-parity figure only; not a 12x4 submission gate",
        "source_sha256": sources,
        "symmetry_protocol": PROTOCOL,
        "Lx": 8,
        "Ly": 4,
        "Sy": 1,
        "sector_dimension": 875,
        "phases": PHASES,
        "window_half_widths": WIDTHS,
        "representative_phase": 0.05,
        "primary_half_width": 0.25,
        "raw_population_includes_target": True,
        "four_by_four_thermal_point_plotted": False,
        "twelve_by_four_plotted": False,
        "size_fit_performed": False,
        "eigensolver_called": False,
        "source_means_reconciled": True,
        "primary_values": representative.loc[np.isclose(representative.window_half_width, 0.25)]
        .iloc[0]
        .to_dict(),
        "claim_boundary": (
            "Finite-size local witness mismatch and phase/window checks; "
            "no thermodynamic extrapolation."
        ),
    }
    (output / "figure9_8x4_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.data, args.output), indent=2))
