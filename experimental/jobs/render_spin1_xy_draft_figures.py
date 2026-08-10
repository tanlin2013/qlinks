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
    PRX_SINGLE_PANEL_FIGSIZE,
    PRX_TWO_PANEL_FIGSIZE,
    add_panel_label,
    save_prx_figure,
    set_revtex_matplotlib_style,
    use_integer_ticks,
    write_figure_manifest,
)

REPRESENTATIVE_KAPPA_OVER_J = 0.10
PRINCIPAL_KAPPA_OVER_J = (0.05, 0.10, 0.15, 0.20)
WITNESS_SPECS = [
    ("A", r"$Q_R^A$", "o"),
    ("Z", r"$Q_R^Z$", "s"),
    ("Y", r"$Q_R^Y$", "^"),
]


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _read_first(data: Path, *names: str) -> pd.DataFrame:
    for name in names:
        path = data / name
        if path.is_file():
            return pd.read_csv(path)
    raise FileNotFoundError(f"None of the requested evidence files exist: {names}")


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
    if "kappa_over_J" in sequence:
        mask &= np.isclose(sequence["kappa_over_J"], REPRESENTATIVE_KAPPA_OVER_J)
    return sequence[mask].sort_values("L")


def _principal_interval(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    values = np.asarray(PRINCIPAL_KAPPA_OVER_J, dtype=float)
    mask = np.zeros(len(frame), dtype=bool)
    for value in values:
        mask |= np.isclose(frame["kappa_over_J"], value)
    return frame[mask].copy()


def _tower_mask(scatter: pd.DataFrame) -> np.ndarray:
    """Identify the analytical tower row without relying on Q=0 alone."""
    if "is_tower_state" in scatter:
        return scatter["is_tower_state"].fillna(False).astype(bool).to_numpy()
    if "tower_overlap" in scatter:
        return scatter["tower_overlap"].fillna(0.0).to_numpy(dtype=float) > 1.0 - 1.0e-7
    # Legacy fallback: the translated joint-dark projector is rank one in the
    # representative production data. This is used only for old exports.
    return scatter["is_exceptional"].fillna(False).astype(bool).to_numpy()


def _panel_c_curves(ax_top, ax_bottom, deformation: pd.DataFrame) -> None:
    principal = _principal_interval(deformation)
    lengths = sorted(principal["L"].unique())
    for length in lengths:
        frame = principal[principal["L"] == length].sort_values("kappa_over_J")
        sparse_safe = "grid_role" in frame and frame["grid_role"].notna().any()
        label = rf"$L={int(length)}$" + (r" ($\Delta E=1$)" if sparse_safe else "")
        line = ax_top.plot(
            frame["kappa_over_J"],
            frame["delta_max"],
            marker="o",
            linestyle="--" if sparse_safe else "-",
            label=label,
        )[0]
        ax_bottom.plot(
            frame["kappa_over_J"],
            frame["L"] * frame["delta_max"],
            marker="o",
            color=line.get_color(),
        )
    ax_top.axvline(REPRESENTATIVE_KAPPA_OVER_J, color="0.45", ls=":", lw=0.9)
    ax_bottom.axvline(REPRESENTATIVE_KAPPA_OVER_J, color="0.45", ls=":", lw=0.9)
    ax_top.set_ylabel(r"$\Delta_L(\kappa)$")
    ax_bottom.set_ylabel(r"$L\Delta_L(\kappa)$")
    ax_bottom.set_xlabel(r"Compatible deformation $\kappa/J$")
    ax_top.grid(alpha=0.20)
    ax_bottom.grid(alpha=0.20)
    ax_top.legend(loc="upper right", ncol=2, fontsize=8.5)
    ax_top.tick_params(labelbottom=False)


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

    sequence = _read_first(
        data, "spin1_xy_kappa0p1_sequence.csv", "spin1_xy_cage_excised_sequence.csv"
    )
    scatter = _read_first(
        data,
        "spin1_xy_kappa0p1_eth_scatter_Lmax.csv",
        "spin1_xy_kappa0p1_eth_scatter_all_sizes.csv",
        "spin1_xy_cage_excised_eth_scatter.csv",
    )
    overlap = _read_first(
        data, "spin1_xy_kappa0p1_beta0_overlap.csv", "spin1_xy_beta0_cage_excised_overlap.csv"
    )
    panel_b_sequence = _read_optional(data / "spin1_xy_kappa0p1_panel_b_sequence.csv")
    if not panel_b_sequence.empty:
        overlap = panel_b_sequence
    deformation = pd.read_csv(data / "spin1_xy_kappa_matching_grid.csv")
    large_family_matching = _read_optional(
        data / "spin1_xy_kappa_matching_large_size_safe_window.csv"
    )
    if not large_family_matching.empty:
        large_family_matching = large_family_matching[
            large_family_matching.get(
                "window_coverage_complete", pd.Series(True, index=large_family_matching.index)
            )
            .fillna(False)
            .astype(bool)
        ].copy()
        if "sparse_convergence_passed" in large_family_matching:
            large_family_matching = large_family_matching[
                large_family_matching["sparse_convergence_passed"].fillna(False).astype(bool)
            ].copy()
        if (
            "delta_max" not in large_family_matching
            and "delta_max_raw_raw" in large_family_matching
        ):
            large_family_matching["delta_max"] = large_family_matching["delta_max_raw_raw"]
        deformation = pd.concat([deformation, large_family_matching], ignore_index=True, sort=False)
    concentration = _read_optional(data / "spin1_xy_kappa_concentration_grid.csv")
    large_rep_concentration = _read_optional(data / "spin1_xy_kappa0p1_concentration_L14.csv")
    large_family_concentration = _read_optional(
        data / "spin1_xy_large_size_family_concentration.csv"
    )
    extra_concentration = []
    if not large_rep_concentration.empty:
        rep_raw = large_rep_concentration[
            large_rep_concentration.get("variant", "raw") == "raw"
        ].copy()
        if not rep_raw.empty:
            if "sparse_convergence_passed" in rep_raw:
                rep_raw = rep_raw[rep_raw["sparse_convergence_passed"].fillna(False).astype(bool)]
            extra_concentration.append(rep_raw)
    if not large_family_concentration.empty:
        if "sparse_convergence_passed" in large_family_concentration:
            large_family_concentration = large_family_concentration[
                large_family_concentration["sparse_convergence_passed"].fillna(False).astype(bool)
            ].copy()
        if not large_family_concentration.empty:
            extra_concentration.append(large_family_concentration)
    if extra_concentration:
        concentration = pd.concat(
            [concentration, *extra_concentration], ignore_index=True, sort=False
        )
    exact = _read_optional(data / "exact_fixed_M_activities.csv")
    obstruction = _read_optional(data / "spin1_xy_complex_t2_obstruction_grid.csv")

    central = _primary_sequence(sequence)
    if central.empty:
        raise RuntimeError("No representative-point primary sequence is available")
    complete_scatter_sizes = sorted(scatter["L"].unique())
    largest_scatter = int(max(complete_scatter_sizes))
    scatter_largest = scatter[scatter["L"] == largest_scatter].copy()
    row = central[central["L"] == largest_scatter]
    if row.empty:
        row = central.iloc[[-1]]
    row = row.iloc[0]
    overlap = overlap.sort_values("L")

    # The nested layout preserves readable 9 pt typography at final double-column size.
    fig = plt.figure(figsize=(7.05, 6.85))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.955,
        bottom=0.08,
        top=0.955,
        wspace=0.38,
        hspace=0.34,
    )

    # (a) Three witness-resolved ETH strips at kappa_star.
    gs_a = outer[0, 0].subgridspec(3, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gs_a[i]) for i in range(3)]
    half = float(row["window_energy_density_half_width"])
    tower_rows = _tower_mask(scatter_largest)
    retained_mask = ~scatter_largest["is_exceptional"].fillna(False).astype(bool).to_numpy()
    background = scatter_largest[retained_mask & ~tower_rows]
    for index, ((key, label, _marker), ax) in enumerate(zip(WITNESS_SPECS, axes_a, strict=True)):
        ax.axvspan(-half, half, color="0.5", alpha=0.10, zorder=0)
        ax.axvline(0.0, color="0.45", ls="--", lw=0.8)
        ax.scatter(
            background["energy_density"],
            background[f"Q_{key}"],
            s=10,
            alpha=0.52,
            marker="o",
            linewidths=0,
            label="joint-dark-cleaned background" if index == 0 else None,
        )
        # Draw the analytical tower exactly once. No empty exceptional marker is
        # retained underneath the star.
        ax.scatter(
            [0.0],
            [0.0],
            marker="*",
            s=78,
            edgecolors="black",
            linewidths=0.45,
            label="selected tower" if index == 0 else None,
            zorder=8,
        )
        beta_col = _resolved_beta0_column(overlap, key)
        beta_row = overlap[overlap["L"] == largest_scatter]
        if not beta_row.empty:
            ax.axhline(float(beta_row.iloc[0][beta_col]), color="0.3", ls=":", lw=0.9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/L$")
    axes_a[0].legend(loc="upper left", fontsize=8.3, frameon=False)
    add_panel_label(axes_a[0], "(a)")

    # (b) Representative-point ensemble values and individual distances.
    gs_b = outer[0, 1].subgridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.08)
    ax_b = fig.add_subplot(gs_b[0])
    ax_b_delta = fig.add_subplot(gs_b[1], sharex=ax_b)
    for key, label, marker in WITNESS_SPECS:
        mc_column = (
            f"tau_{key}_mc_raw"
            if f"tau_{key}_mc_raw" in overlap
            else (f"tau_{key}_mc_clean" if f"tau_{key}_mc_clean" in overlap else f"tau_{key}_mc_th")
        )
        beta_column = (
            f"tau_{key}_resolved_beta0_raw"
            if f"tau_{key}_resolved_beta0_raw" in overlap
            else _resolved_beta0_column(overlap, key)
        )
        line = ax_b.plot(overlap["L"], overlap[mc_column], marker=marker, label=rf"{label}: MC")[0]
        ax_b.plot(
            overlap["L"],
            overlap[beta_column],
            marker=marker,
            markerfacecolor="none",
            linestyle="--",
            color=line.get_color(),
            label=rf"{label}: $\beta=0$",
        )
        delta_column = (
            f"delta_{key}_raw_raw"
            if f"delta_{key}_raw_raw" in overlap
            else (
                f"delta_{key}_clean_clean"
                if f"delta_{key}_clean_clean" in overlap
                else f"delta_{key}"
            )
        )
        regular = overlap[overlap["L"] >= 8]
        pre = overlap[overlap["L"] < 8]
        ax_b_delta.plot(regular["L"], regular[delta_column], marker=marker, color=line.get_color())
        if not pre.empty:
            ax_b_delta.scatter(pre["L"], pre[delta_column], marker=marker, color="0.58", zorder=4)
    ax_b.set_ylabel("Local activity")
    if "window_role" in overlap and (overlap["window_role"] == "sparse_safe_fixed_width").any():
        ax_b.text(
            0.98,
            0.03,
            r"$L=14$: sparse, $\Delta E=1$",
            transform=ax_b.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
        )
    ax_b.grid(alpha=0.20)
    ax_b.legend(loc="best", fontsize=7.8, ncol=2)
    ax_b.tick_params(labelbottom=False)
    add_panel_label(ax_b, "(b)")
    ax_b_delta.set_xlabel(r"System size $L$")
    ax_b_delta.set_ylabel(r"$\delta_{\alpha,L}$")
    ax_b_delta.set_ylim(bottom=0.0)
    ax_b_delta.grid(alpha=0.20)
    lengths_b = np.sort(overlap["L"].unique()).astype(int)
    for axis in (ax_b, ax_b_delta):
        use_integer_ticks(axis, axis="x")
        axis.set_xticks(lengths_b)

    # (c) Family-wide matching and scaled matching on a separate lower strip.
    gs_c = outer[1, 0].subgridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.08)
    ax_c = fig.add_subplot(gs_c[0])
    ax_c_scaled = fig.add_subplot(gs_c[1], sharex=ax_c)
    _panel_c_curves(ax_c, ax_c_scaled, deformation)
    add_panel_label(ax_c, "(c)")

    # (d) Complete two-site covariance width over the principal positive interval.
    ax_d = fig.add_subplot(outer[1, 1])
    principal_concentration = _principal_interval(concentration)
    if not principal_concentration.empty:
        # L=6 is explicitly pre-asymptotic and is not used to set the main color scale.
        display_concentration = principal_concentration[principal_concentration["L"] >= 8].copy()
        if display_concentration.empty:
            display_concentration = principal_concentration.copy()
        if (
            "largest_covariance_width" not in display_concentration
            and "largest_covariance_width_raw" in display_concentration
        ):
            display_concentration["largest_covariance_width"] = display_concentration[
                "largest_covariance_width_raw"
            ]
        display_concentration = display_concentration.sort_values(
            ["L", "kappa_over_J"]
        ).drop_duplicates(["L", "kappa_over_J"], keep="last")
        cpivot = display_concentration.pivot(
            index="L", columns="kappa_over_J", values="largest_covariance_width"
        )
        cx = cpivot.columns.to_numpy(dtype=float)
        cy = cpivot.index.to_numpy(dtype=float)
        cmesh = ax_d.pcolormesh(
            _centered_edges(cx, fallback_half_width=0.025),
            _centered_edges(cy, fallback_half_width=0.5),
            cpivot.to_numpy(),
            shading="flat",
        )
        ax_d.axvline(REPRESENTATIVE_KAPPA_OVER_J, color="white", ls=":", lw=1.0)
        ax_d.set_yticks(cy.astype(int))
        colorbar = fig.colorbar(cmesh, ax=ax_d, pad=0.025)
        colorbar.ax.set_title(r"$w_L(\kappa)$", fontsize=9, pad=4)
        colorbar.ax.tick_params(labelsize=9)
    else:
        ax_d.text(0.5, 0.5, "concentration data not computed", ha="center", va="center")
    ax_d.set_xlabel(r"Compatible deformation $\kappa/J$")
    ax_d.set_ylabel(r"System size $L$")
    use_integer_ticks(ax_d, axis="y")
    if not principal_concentration.empty:
        ax_d.set_yticks(np.sort(display_concentration["L"].unique()).astype(int))
    add_panel_label(ax_d, "(d)")

    save_prx_figure(fig, "spin1_xy_figure6_combined", directory=figures, formats=formats)

    # Standalone representative matching and family-wide matching figures.
    fig_delta, (ax0, ax1) = plt.subplots(1, 2, figsize=PRX_TWO_PANEL_FIGSIZE)
    for key, label, marker in WITNESS_SPECS:
        column = (
            f"delta_{key}_raw_raw"
            if f"delta_{key}_raw_raw" in overlap
            else (
                f"delta_{key}_clean_clean"
                if f"delta_{key}_clean_clean" in overlap
                else f"delta_{key}"
            )
        )
        regular = overlap[overlap["L"] >= 8]
        pre = overlap[overlap["L"] < 8]
        ax0.plot(regular["L"], regular[column], marker=marker, label=label)
        if not pre.empty:
            ax0.scatter(pre["L"], pre[column], marker=marker, color="0.58")
    ax0.set_xlabel(r"System size $L$")
    ax0.set_ylabel(r"$\delta_{\alpha,L}$")
    use_integer_ticks(ax0, axis="x")
    ax0.set_xticks(np.sort(overlap["L"].unique()).astype(int))
    ax0.grid(alpha=0.22)
    ax0.legend(loc="upper right")
    add_panel_label(ax0, "(a)")
    principal = _principal_interval(deformation)
    for length, frame in principal.groupby("L"):
        frame = frame.sort_values("kappa_over_J")
        ax1.plot(frame["kappa_over_J"], frame["delta_max"], marker="o", label=rf"$L={int(length)}$")
    ax1.axvline(REPRESENTATIVE_KAPPA_OVER_J, color="0.45", ls=":", lw=0.9)
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

    # Supporting exact fixed-M targets and family-wide matching.
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
        # Avoid overcrowding while keeping every displayed tick integral.
        lengths = exact["length"].to_numpy(dtype=int)
        stride = max(1, int(np.ceil(len(lengths) / 8)))
        ax0.set_xticks(lengths[::stride])
        ax0.grid(alpha=0.22)
        ax0.legend(loc="upper right")
        add_panel_label(ax0, "(a)")
        for length, frame in principal.groupby("L"):
            frame = frame.sort_values("kappa_over_J")
            ax1.plot(
                frame["kappa_over_J"], frame["delta_max"], marker="o", label=rf"$L={int(length)}$"
            )
        ax1.axvline(REPRESENTATIVE_KAPPA_OVER_J, color="0.45", ls=":", lw=0.9)
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

    # The ambient obstruction plane is retained as appendix material.
    if not obstruction.empty:
        pivot = obstruction.pivot(
            index="imag_t2_over_J",
            columns="real_t2_over_J",
            values="normalized_tower_residual",
        )
        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        fig_ob, ax_ob = plt.subplots(figsize=PRX_SINGLE_PANEL_FIGSIZE)
        mesh = ax_ob.pcolormesh(
            _centered_edges(x, fallback_half_width=0.01),
            _centered_edges(y, fallback_half_width=0.01),
            pivot.to_numpy(),
            shading="flat",
        )
        ax_ob.axvline(0.0, linestyle="--", linewidth=1.0, color="white")
        ax_ob.set_xlabel(r"$\operatorname{Re}t_2/J$")
        ax_ob.set_ylabel(r"$\operatorname{Im}t_2/J$")
        colorbar = fig_ob.colorbar(mesh, ax=ax_ob, pad=0.025)
        colorbar.set_label("Normalized tower residual")
        save_prx_figure(
            fig_ob,
            "spin1_xy_complex_t2_obstruction_appendix",
            directory=figures,
            formats=formats,
        )

    write_figure_manifest(data / "figure_manifest.json")


if __name__ == "__main__":
    main()
