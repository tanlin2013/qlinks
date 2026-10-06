"""Follow-up render-only polish for PRX Figs. 6 and 9.

This module sits on top of :mod:`prx_main_thermal_figure_redesign` and keeps its
frozen-evidence table construction.  It changes only presentation and input
robustness: caged-state stars use the manuscript orange, witness rows use
stable colors, representative values use horizontal-bar glyphs, sampled ranges
use visually explicit capped whiskers, panels (b,c) do not connect points, and
panel (d) uses a subordinate dashed guide.  No solver or interpolation is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prx_main_thermal_figure_redesign as _base
from helpers import add_panel_label_margin, save_prx_figure, use_integer_ticks
from matplotlib.lines import Line2D

FULL_WIDTH_IN = _base.FULL_WIDTH_IN
FIG_HEIGHT_IN = _base.FIG_HEIGHT_IN
REPRESENTATIVE_KAPPA_OVER_J = _base.REPRESENTATIVE_KAPPA_OVER_J
RAW_COLOR = _base.RAW_COLOR
CANONICAL_COLOR = _base.CANONICAL_COLOR
TARGET_COLOR = _base.TARGET_COLOR
WITNESS_TARGETS = _base.WITNESS_TARGETS
WITNESS_TARGET_LABELS = _base.WITNESS_TARGET_LABELS

# Okabe-Ito-like witness palette; the star keeps the warm orange used in the
# earlier manuscript artwork and is deliberately not reused for a witness.
WITNESS_COLORS = {
    "A": "#0072B2",
    "Z": "#009E73",
    "Y": "#CC79A7",
}
STAR_COLOR = "#E69F00"
BAR_MARKER_SIZE = 10.0
BAR_MARKER_EDGE_WIDTH = 1.45
WHISKER_CAP_SIZE = 5.0
WHISKER_LINE_WIDTH = 1.4
WHISKER_CAP_THICK = 1.25
GUIDE_LINE_WIDTH = 0.85


def _numeric(series: pd.Series) -> pd.Series:
    """Return numeric values while treating malformed/blank CSV cells as NaN."""

    return pd.to_numeric(series, errors="coerce")


def _padded_ylim(*values: float) -> tuple[float, float]:
    """Tight but readable limits around data and reference values.

    The previous positive-from-zero scale visually collapsed the small
    separation between finite-size data and the reference lines.  These limits
    retain all displayed values while reserving explicit headroom around the
    full span.  A minimum scale-dependent span prevents nearly coincident data
    from producing a singular-looking axis.
    """

    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    scale = max(abs(lo), abs(hi), 0.05)
    span = max(hi - lo, 0.08 * scale)
    pad = 0.22 * span
    return lo - pad, hi + pad


def _bar_point(
    ax,
    *,
    x: float,
    y: float,
    color: str,
    zorder: float = 6,
) -> None:
    ax.plot(
        [x],
        [y],
        linestyle="none",
        marker="_",
        markersize=BAR_MARKER_SIZE,
        markeredgewidth=BAR_MARKER_EDGE_WIDTH,
        color=color,
        zorder=zorder,
    )


def _sampled_whisker(
    ax,
    *,
    x: float,
    center: float,
    minimum: float | None,
    maximum: float | None,
    color: str,
) -> None:
    finite_range = (
        minimum is not None
        and maximum is not None
        and np.isfinite(minimum)
        and np.isfinite(maximum)
    )
    has_range = bool(
        finite_range
        and float(maximum) + 1.0e-15 >= center >= float(minimum) - 1.0e-15
        and float(maximum) - float(minimum) > 1.0e-14
    )
    yerr = None
    if has_range:
        yerr = _base._asymmetric_yerr(center, float(minimum), float(maximum))
    ax.errorbar(
        [x],
        [center],
        yerr=yerr,
        fmt="_",
        color=color,
        markersize=BAR_MARKER_SIZE,
        markeredgewidth=BAR_MARKER_EDGE_WIDTH,
        linewidth=0.0,
        capsize=WHISKER_CAP_SIZE if has_range else 0.0,
        capthick=WHISKER_CAP_THICK,
        elinewidth=WHISKER_LINE_WIDTH,
        zorder=7,
    )


def _draw_spin1_finite_size_panels(
    *,
    axes_b: list,
    axes_c: list,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
) -> None:
    for index, key in enumerate(("A", "Z", "Y")):
        color = WITNESS_COLORS[key]
        b = panel_b[panel_b["witness"].astype(str) == key].sort_values("L")
        c = panel_c[panel_c["witness"].astype(str) == key].sort_values("L")
        target = WITNESS_TARGETS[key]
        ylim = _padded_ylim(
            target,
            *b["tau_mc_raw"].to_numpy(dtype=float),
            *c["tau_min"].to_numpy(dtype=float),
            *c["tau_max"].to_numpy(dtype=float),
        )

        axb = axes_b[index]
        for row in b.itertuples(index=False):
            _bar_point(axb, x=float(row.L), y=float(row.tau_mc_raw), color=color)
        axb.axhline(target, color=TARGET_COLOR, ls="--", lw=0.9, zorder=2)
        axb.text(
            0.98,
            0.88,
            rf"$\tau_{{{key}}}^{{\beta=0}}={WITNESS_TARGET_LABELS[key]}$",
            transform=axb.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
        )
        axb.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_{{\rm mc}}$")
        axb.set_ylim(*ylim)
        axb.grid(alpha=0.13)
        use_integer_ticks(axb, axis="x")
        axb.set_xticks([8, 10, 12, 14])
        if index < 2:
            axb.tick_params(labelbottom=False)
        else:
            axb.set_xlabel(r"System size $L$")

        axc = axes_c[index]
        for row in c.itertuples(index=False):
            _sampled_whisker(
                axc,
                x=float(row.L),
                center=float(row.tau_star),
                minimum=float(row.tau_min),
                maximum=float(row.tau_max),
                color=color,
            )
        axc.axhline(target, color=TARGET_COLOR, ls="--", lw=0.9, zorder=2)
        axc.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_{{\rm mc}}$")
        axc.set_ylim(*ylim)
        axc.grid(alpha=0.13)
        use_integer_ticks(axc, axis="x")
        axc.set_xticks([8, 10, 12])
        if index < 2:
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel(r"System size $L$")


def render_spin1_figure6(
    data: Path,
    figures: Path,
    *,
    allow_incomplete: bool,
    read_csv: Callable[[Path], pd.DataFrame],
    save_figure: Callable[..., list[str]],
) -> list[str]:
    """Render polished Fig. 6 from the frozen handoff tables."""

    del allow_incomplete
    panel_a, panel_b, panel_c, panel_d, grid_path = _base._spin1_tables(
        data=data,
        figures=figures,
        read_csv=read_csv,
    )
    fig = plt.figure(figsize=(FULL_WIDTH_IN, FIG_HEIGHT_IN))
    outer = _base._outer_grid(fig)

    gsa = outer[0, 0].subgridspec(3, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gsa[index]) for index in range(3)]
    tower = panel_a["is_tower_state"].fillna(False).astype(bool).to_numpy()
    background = panel_a[~tower]
    l12 = panel_b[panel_b["L"].astype(int) == 12]
    half = float(l12.iloc[0]["window_energy_density_half_width"])
    for index, key in enumerate(("A", "Z", "Y")):
        ax = axes_a[index]
        ax.axvspan(-half, half, color="0.5", alpha=0.10, zorder=0)
        ax.scatter(
            background["energy_density"],
            background[f"Q_{key}"],
            s=8,
            alpha=0.38,
            color=WITNESS_COLORS[key],
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            [0.0],
            [0.0],
            marker="*",
            s=72,
            color=STAR_COLOR,
            edgecolors="black",
            linewidths=0.4,
            zorder=8,
        )
        mean = l12[l12["witness"].astype(str) == key]
        if len(mean) == 1:
            ax.axhline(float(mean.iloc[0]["tau_mc_raw"]), color="0.35", ls=":", lw=0.8)
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_n$")
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/L$")
    add_panel_label_margin(axes_a[0], "(a)")

    gsb = outer[0, 1].subgridspec(3, 1, hspace=0.10)
    axes_b = [fig.add_subplot(gsb[index]) for index in range(3)]
    gsc = outer[1, 0].subgridspec(3, 1, hspace=0.10)
    axes_c = [fig.add_subplot(gsc[index]) for index in range(3)]
    _draw_spin1_finite_size_panels(
        axes_b=axes_b,
        axes_c=axes_c,
        panel_b=panel_b,
        panel_c=panel_c,
    )
    axes_c[0].text(
        0.98,
        0.88,
        r"bars/whiskers: representative value / sampled $\kappa/J$ range",
        transform=axes_c[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    add_panel_label_margin(axes_b[0], "(b)")
    add_panel_label_margin(axes_c[0], "(c)")

    axd = fig.add_subplot(outer[1, 1])
    axd.plot(
        panel_d["L"],
        panel_d["w_star"],
        color=WITNESS_COLORS["A"],
        linewidth=GUIDE_LINE_WIDTH,
        linestyle="--",
        alpha=0.72,
        zorder=1,
    )
    for row in panel_d.itertuples(index=False):
        minimum = None if pd.isna(row.w_min) else float(row.w_min)
        maximum = None if pd.isna(row.w_max) else float(row.w_max)
        _sampled_whisker(
            axd,
            x=float(row.L),
            center=float(row.w_star),
            minimum=minimum,
            maximum=maximum,
            color=WITNESS_COLORS["A"],
        )
    axd.set_xlabel(r"System size $L$")
    axd.set_ylabel(r"$w_L^{\rm raw}$")
    axd.set_ylim(bottom=0.0)
    use_integer_ticks(axd, axis="x")
    axd.set_xticks([8, 10, 12, 14])
    axd.grid(alpha=0.13)
    marker_handle = Line2D(
        [0], [0], color=WITNESS_COLORS["A"], marker="_", linestyle="none",
        markersize=BAR_MARKER_SIZE, markeredgewidth=BAR_MARKER_EDGE_WIDTH,
        label=r"$\kappa_\star/J=0.1$",
    )
    whisker_handle = Line2D(
        [0], [0], color=WITNESS_COLORS["A"], marker="|", markersize=11,
        lw=WHISKER_LINE_WIDTH, label="sampled range",
    )
    axd.legend(handles=[marker_handle, whisker_handle], loc="upper right", frameon=False, fontsize=7.6)
    add_panel_label_margin(axd, "(d)")

    manifest = {
        "figure": "Fig. 6",
        "source_evidence_directory": str(data),
        "star_color": STAR_COLOR,
        "witness_colors": WITNESS_COLORS,
        "panel_b_connecting_lines": False,
        "panel_c_connecting_lines": False,
        "panel_c_marker": "horizontal_bar",
        "panel_d_guide_line": "dashed",
        "whiskers": "sampled min/max only; enlarged caps and stems",
        "deformation_source": grid_path.name,
        "no_interpolated_deformation_values": True,
        "expensive_recomputation": False,
    }
    _base._write_json(figures / "spin1_xy_figure6_provenance.json", manifest)
    return save_figure(fig, figures, "spin1_xy_figure6_prx", preview=True)


def _qdm_gate_raw_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lengths = _numeric(result["Lx"])
    result = result[lengths.notna()].copy()
    result["Lx"] = lengths[lengths.notna()].astype(int)
    if "window_coverage_complete" in result.columns:
        large = result["Lx"] >= 12
        covered = result["window_coverage_complete"].fillna(False).astype(bool)
        result = result[~large | covered]
    if "converged_vs_previous_budget" in result.columns:
        large = result["Lx"] >= 12
        converged = result["converged_vs_previous_budget"].fillna(False).astype(bool)
        result = result[~large | converged]
    return result


def _qdm_canonical_l12(data: Path, *, phase: float) -> tuple[pd.DataFrame, Path]:
    """Read the persisted Lx=12 canonical target without unsafe int casting."""

    path = data / "qdm_checkerboard_finite_beta_transfer_target.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Fig. 9(b) requires qdm_checkerboard_finite_beta_transfer_target.csv "
            "so the Lx=12 canonical-typicality continuation is visible"
        )
    frame = pd.read_csv(path)
    length_col = _base._first_column(frame, ("Lx", "L_x", "length"))
    if length_col is None:
        raise ValueError(f"canonical target has no strip-length column: {path}")
    lengths = _numeric(frame[length_col])
    selected = frame[lengths.eq(12)].copy()
    phase_col = _base._first_column(frame, ("phase", "varphi", "phi"))
    if phase_col is not None and not selected.empty:
        phases = _numeric(selected[phase_col])
        valid = phases.notna()
        selected = selected[valid].copy()
        phases = phases[valid]
        if not selected.empty:
            distance = np.abs(phases.to_numpy(dtype=float) - phase)
            selected = selected.iloc[[int(np.argmin(distance))]]
    elif len(selected) > 1:
        selected = selected.tail(1)
    if selected.empty:
        raise ValueError(f"canonical target has no finite Lx=12 row: {path}")
    selected = selected.rename(columns={length_col: "Lx"})
    selected["Lx"] = 12
    if phase_col is None:
        selected["phase"] = phase
    return selected, path


def _sanitize_length_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    length_col = _base._first_column(result, ("Lx", "L_x", "length"))
    if length_col is None:
        return result
    lengths = _numeric(result[length_col])
    result = result[lengths.notna()].copy()
    result[length_col] = lengths[lengths.notna()].astype(int)
    return result


def _qdm_ensemble_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=RAW_COLOR, marker="_", linestyle="none", markersize=BAR_MARKER_SIZE,
               markeredgewidth=BAR_MARKER_EDGE_WIDTH, label="raw microcanonical"),
        Line2D([0], [0], color=CANONICAL_COLOR, marker="_", linestyle="none", markersize=BAR_MARKER_SIZE,
               markeredgewidth=BAR_MARKER_EDGE_WIDTH, label="energy-matched canonical"),
    ]


def _draw_qdm_b_c(
    *,
    axes_b: list,
    axes_c: list,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    phase: float,
) -> None:
    for index, key in enumerate(("A", "Z")):
        b_raw = panel_b[(panel_b["witness"] == key) & (panel_b["ensemble"] == "raw_microcanonical")].sort_values("Lx")
        b_can = panel_b[(panel_b["witness"] == key) & (panel_b["ensemble"] == "canonical")].sort_values("Lx")
        c_key = panel_c[panel_c["witness"] == key]
        ylim = _padded_ylim(
            *b_raw["value"].to_numpy(dtype=float),
            *b_can["value"].to_numpy(dtype=float),
            *c_key["value_min"].to_numpy(dtype=float),
            *c_key["value_max"].to_numpy(dtype=float),
        )

        axb = axes_b[index]
        for row in b_raw.itertuples(index=False):
            _bar_point(axb, x=float(row.Lx) - 0.10, y=float(row.value), color=RAW_COLOR)
        for row in b_can.itertuples(index=False):
            _bar_point(axb, x=float(row.Lx) + 0.10, y=float(row.value), color=CANONICAL_COLOR)
        typicality = b_can[b_can["method"] == "canonical_typicality"]
        if not typicality.empty and float(typicality.iloc[0]["stderr"]) > 0.0:
            axb.errorbar(
                typicality["Lx"] + 0.10,
                typicality["value"],
                yerr=typicality["stderr"],
                fmt="none",
                color=CANONICAL_COLOR,
                capsize=WHISKER_CAP_SIZE,
                capthick=WHISKER_CAP_THICK,
                elinewidth=WHISKER_LINE_WIDTH,
                zorder=6,
            )
        axb.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle$")
        axb.set_ylim(*ylim)
        axb.grid(alpha=0.13)
        use_integer_ticks(axb, axis="x")
        axb.set_xticks([4, 8, 12])
        if index == 0:
            axb.tick_params(labelbottom=False)
            axb.text(0.98, 0.88, rf"$\varphi_\star={phase:g}$", transform=axb.transAxes,
                     ha="right", va="top", fontsize=7.6)
        else:
            axb.set_xlabel(r"Strip length $L_x$")

        axc = axes_c[index]
        styles = (("raw_microcanonical", RAW_COLOR, -0.10), ("canonical", CANONICAL_COLOR, 0.10))
        for ensemble, color, offset in styles:
            frame = c_key[c_key["ensemble"] == ensemble].sort_values("Lx")
            for row in frame.itertuples(index=False):
                _sampled_whisker(
                    axc,
                    x=float(row.Lx) + offset,
                    center=float(row.value_star),
                    minimum=float(row.value_min),
                    maximum=float(row.value_max),
                    color=color,
                )
        axc.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle$")
        axc.set_ylim(*ylim)
        axc.grid(alpha=0.13)
        use_integer_ticks(axc, axis="x")
        axc.set_xticks([4, 8, 12])
        if index == 0:
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel(r"Strip length $L_x$")


def render_qdm_figure9(
    *,
    data: Path,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> list[Path]:
    """Render polished Fig. 9 with robust persisted-table parsing."""

    # Make base helpers used by its table builders robust to nullable strip lengths.
    _base._qdm_gate_raw = _qdm_gate_raw_safe

    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    thermal_path = data / "qdm_checkerboard_thermal_overlap.csv"
    if not thermal_path.is_file():
        thermal_path = data / "qdm_checkerboard_beta0_overlap.csv"
    scatter_path = data / "qdm_checkerboard_eth_scatter.csv"
    concentration_path = data / "qdm_checkerboard_concentration_grid.csv"
    rep_path = data / "qdm_checkerboard_representative_phase.csv"
    thermal = _sanitize_length_column(pd.read_csv(thermal_path))
    scatter = _sanitize_length_column(pd.read_csv(scatter_path))
    concentration = _sanitize_length_column(pd.read_csv(concentration_path))
    rep = pd.read_csv(rep_path) if rep_path.is_file() else pd.DataFrame()
    if thermal.empty or scatter.empty:
        raise RuntimeError("Checkerboard thermal/scatter evidence is unavailable")

    primary, prefactor = _base._qdm_primary_thermal(thermal)
    phase = (
        float(rep["phi_star"].iloc[0])
        if not rep.empty
        else float(sorted(primary["phase"].unique())[len(primary["phase"].unique()) // 2])
    )
    raw = _qdm_gate_raw_safe(primary)
    raw = raw[raw["Lx"] < 12].copy()
    canonical_l12, canonical_path = _qdm_canonical_l12(data, phase=phase)
    phase_check, phase_check_path = _base._qdm_phase_check(data)
    phase_check = _sanitize_length_column(phase_check)

    representative_raw = raw[np.isclose(raw["phase"], phase)].sort_values("Lx")
    raw_lengths = set(representative_raw["Lx"].astype(int))
    common = sorted(raw_lengths.intersection(set(scatter["Lx"].astype(int))))
    if not common:
        raise RuntimeError("No verified raw thermal size has matching ETH-scatter data")
    largest = int(common[-1])
    scatter_largest = scatter[scatter["Lx"].astype(int) == largest].copy()
    representative_row = representative_raw[representative_raw["Lx"].astype(int) == largest].iloc[-1]

    panel_b = _base._qdm_panel_b(
        raw=raw,
        canonical_l12=canonical_l12,
        thermal_path=thermal_path,
        canonical_path=canonical_path,
        phase=phase,
    )
    panel_c = _base._qdm_panel_c(
        raw=raw,
        panel_b=panel_b,
        thermal_path=thermal_path,
        phase_check=phase_check,
        phase_check_path=phase_check_path,
        phase=phase,
    )
    panel_d = _base._qdm_panel_d(
        concentration=concentration,
        concentration_path=concentration_path,
        prefactor=prefactor,
        phase=phase,
    )
    panel_a = scatter_largest.copy()
    panel_a["source_file"] = scatter_path.name
    for name, frame in {
        "qdm_checkerboard_figure9_panel_a_plot.csv": panel_a,
        "qdm_checkerboard_figure9_panel_b_plot.csv": panel_b,
        "qdm_checkerboard_figure9_panel_c_plot.csv": panel_c,
        "qdm_checkerboard_figure9_panel_d_plot.csv": panel_d,
    }.items():
        _base._write_csv(figures / name, frame)

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FIG_HEIGHT_IN))
    outer = _base._outer_grid(fig)
    gsa = outer[0, 0].subgridspec(2, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gsa[index]) for index in range(2)]
    lower = representative_row.cage_energy_density - representative_row.window_energy_density_half_width
    upper = representative_row.cage_energy_density + representative_row.window_energy_density_half_width
    for index, (key, column) in enumerate((("A", "Q_A"), ("Z", "Q_Z"))):
        ax = axes_a[index]
        ax.axvspan(lower, upper, color="0.5", alpha=0.10, zorder=0)
        ax.scatter(scatter_largest["energy_density"], scatter_largest[column], s=9, alpha=0.42,
                   color="0.35", linewidths=0, rasterized=True)
        ax.scatter([representative_row.cage_energy_density], [0.0], marker="*", s=76,
                   color=STAR_COLOR, edgecolors="black", linewidths=0.4, zorder=8)
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_n$")
        ax.grid(alpha=0.13)
        if index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/(4L_x)$")
    add_panel_label_margin(axes_a[0], "(a)")

    gsb = outer[0, 1].subgridspec(2, 1, hspace=0.10)
    axes_b = [fig.add_subplot(gsb[index]) for index in range(2)]
    gsc = outer[1, 0].subgridspec(2, 1, hspace=0.10)
    axes_c = [fig.add_subplot(gsc[index]) for index in range(2)]
    _draw_qdm_b_c(axes_b=axes_b, axes_c=axes_c, panel_b=panel_b, panel_c=panel_c, phase=phase)
    handles = _qdm_ensemble_handles()
    legend_kwargs = {
        "handles": handles,
        "loc": "lower left",
        "bbox_to_anchor": (0.0, 1.015),
        "borderaxespad": 0.0,
        "frameon": False,
        "ncol": 2,
        "fontsize": 7.2,
    }
    axes_b[0].legend(**legend_kwargs)
    axes_b[0].text(0.98, 1.02, r"$L_x=12$: canonical typicality", transform=axes_b[0].transAxes,
                   ha="right", va="bottom", fontsize=7.2)
    axes_c[0].legend(**legend_kwargs)
    axes_c[0].text(0.98, 0.88, r"bars/whiskers: representative value / sampled $\varphi$ range",
                   transform=axes_c[0].transAxes, ha="right", va="top", fontsize=7.0)
    add_panel_label_margin(axes_b[0], "(b)")
    add_panel_label_margin(axes_c[0], "(c)")

    axd = fig.add_subplot(outer[1, 1])
    if not panel_d.empty:
        axd.plot(panel_d["Lx"], panel_d["w_star"], color=RAW_COLOR, linewidth=GUIDE_LINE_WIDTH,
                 linestyle="--", alpha=0.72, zorder=1)
    for row in panel_d.itertuples(index=False):
        _sampled_whisker(axd, x=float(row.Lx), center=float(row.w_star),
                         minimum=float(row.w_min), maximum=float(row.w_max), color=RAW_COLOR)
    axd.set_xlabel(r"Strip length $L_x$")
    axd.set_ylabel(r"$w_{L_x}^{\rm raw}$")
    axd.set_ylim(bottom=0.0)
    axd.grid(alpha=0.13)
    use_integer_ticks(axd, axis="x")
    axd.set_xticks(sorted(set(panel_d["Lx"].astype(int))))
    axd.text(0.98, 0.92, r"bars/whiskers: representative value / sampled $\varphi$ range",
             transform=axd.transAxes, ha="right", va="top", fontsize=7.0)
    add_panel_label_margin(axd, "(d)")

    written: list[Path] = []
    for stem in ("qdm_checkerboard_figure7_combined", "qdm_checkerboard_figure9_prx"):
        written.extend(save_prx_figure(fig, stem, directory=figures, formats=formats, close=False))
    preview = figures / "qdm_checkerboard_figure9_prx_preview.png"
    fig.savefig(preview, dpi=300, bbox_inches=None, pad_inches=0.0)
    written.append(preview)
    plt.close(fig)

    manifest = {
        "figure": "Fig. 9",
        "source_evidence_directory": str(data),
        "representative_phase": phase,
        "primary_window_prefactor": prefactor,
        "star_color": STAR_COLOR,
        "panel_b_connecting_lines": False,
        "panel_c_connecting_lines": False,
        "panel_c_marker": "horizontal_bar",
        "panel_d_guide_line": "dashed",
        "whiskers": "sampled min/max only; enlarged caps and stems",
        "raw_12x4_plotted": False,
        "canonical_12x4_plotted": True,
        "dedicated_Delta_panel_removed": True,
        "heatmap_removed": True,
        "no_interpolated_deformation_values": True,
        "expensive_recomputation": False,
    }
    _base._write_json(figures / "qdm_checkerboard_figure9_provenance.json", manifest)
    return written
