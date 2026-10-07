"""Final render-only polish for PRX Figs. 6 and 9.

The module consumes only frozen evidence.  Panels (b) combine the representative
finite-size comparison with the sampled compatible-deformation span, while
panels (c) show the witness values directly across the sampled compatible
deformation.  Panel (d) keeps the concentration/scaling diagnostic.  No solver,
interpolation, or artificial range inflation is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prx_main_thermal_figure_redesign as _base
from helpers import add_panel_label_margin, save_prx_figure, use_integer_ticks
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter

FULL_WIDTH_IN = _base.FULL_WIDTH_IN
FIG_HEIGHT_IN = _base.FIG_HEIGHT_IN
REPRESENTATIVE_KAPPA_OVER_J = _base.REPRESENTATIVE_KAPPA_OVER_J
TARGET_COLOR = _base.TARGET_COLOR
WITNESS_TARGETS = _base.WITNESS_TARGETS

WITNESS_COLORS = {
    "A": "#0072B2",
    "Z": "#009E73",
    "Y": "#CC79A7",
}
STAR_COLOR = "#E69F00"
POINT_MARKER_SIZE = 4.6
POINT_EDGE_WIDTH = 1.0
RANGE_BOX_FACE_ALPHA = 0.16
RANGE_BOX_EDGE_WIDTH = 1.05
RANGE_BOX_CENTER_WIDTH = 1.55
GUIDE_LINE_WIDTH = 0.85
TYPICALITY_CAP_SIZE = 3.5
TYPICALITY_LINE_WIDTH = 1.0


def _numeric(series: pd.Series) -> pd.Series:
    """Return numeric values while treating malformed/blank CSV cells as NaN."""

    return pd.to_numeric(series, errors="coerce")


def _padded_ylim(*values: float) -> tuple[float, float]:
    """Tight but readable limits around data and reference values."""

    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    scale = max(abs(lo), abs(hi), 0.05)
    span = max(hi - lo, 0.08 * scale)
    pad = 0.22 * span
    return lo - pad, hi + pad


def _clean_y_ticks(ax, *, nbins: int = 3) -> None:
    """Use a small set of plain, non-offset ticks on compact stacked axes."""

    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10], min_n_ticks=2))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)


def _panel_label(ax, label: str) -> None:
    """Use manuscript-style bold panel labels without moving their anchor."""

    add_panel_label_margin(ax, rf"\textbf{{{label}}}")


def _top_legend(
    ax,
    *,
    handles: list,
    ncol: int,
    title: str | None = None,
) -> None:
    """Place a compact marker key above a panel instead of over its data."""

    kwargs = {
        "handles": handles,
        "loc": "lower left",
        "bbox_to_anchor": (0.0, 1.02),
        "borderaxespad": 0.0,
        "frameon": False,
        "fontsize": 6.8,
        "handletextpad": 0.45,
        "columnspacing": 0.9,
        "ncol": ncol,
    }
    if title is not None:
        kwargs["title"] = title
        kwargs["title_fontsize"] = 6.8
    ax.legend(**kwargs)


def _range_box(
    ax,
    *,
    x: float,
    center: float,
    minimum: float | None,
    maximum: float | None,
    color: str,
    width: float,
    filled: bool = True,
    linestyle: str = "-",
    zorder: float = 7,
) -> None:
    """Draw the literal sampled min/max box and its representative-value line."""

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
    left = x - 0.5 * width
    right = x + 0.5 * width
    if has_range:
        patch = Rectangle(
            (left, float(minimum)),
            width,
            float(maximum) - float(minimum),
            facecolor=color if filled else "none",
            edgecolor=color,
            linewidth=RANGE_BOX_EDGE_WIDTH,
            linestyle=linestyle,
            alpha=RANGE_BOX_FACE_ALPHA if filled else 1.0,
            zorder=zorder,
        )
        ax.add_patch(patch)
    ax.hlines(
        center,
        left,
        right,
        color=color,
        linewidth=RANGE_BOX_CENTER_WIDTH,
        linestyle=linestyle,
        zorder=zorder + 1,
    )


def _center_marker(
    ax,
    *,
    x: float,
    y: float,
    color: str,
    filled: bool,
) -> None:
    """Make coincident raw/canonical range centers distinguishable in Fig. 9(b)."""

    ax.plot(
        [x],
        [y],
        linestyle="none",
        marker="o",
        markersize=POINT_MARKER_SIZE - 0.5,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=POINT_EDGE_WIDTH,
        color=color,
        zorder=10,
    )


def _range_semantics_handles(
    star_label: str,
    scan_label: str,
    *,
    color: str = "0.35",
) -> list:
    """Compact legend handles for representative value versus sampled span."""

    return [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=RANGE_BOX_CENTER_WIDTH,
            label=star_label,
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color,
            edgecolor=color,
            alpha=RANGE_BOX_FACE_ALPHA,
            linewidth=RANGE_BOX_EDGE_WIDTH,
            label=scan_label,
        ),
    ]


def _spin1_panel_b_merged(
    representative: pd.DataFrame,
    ranges: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []
    for row in representative.itertuples(index=False):
        match = ranges[
            (ranges["L"].astype(int) == int(row.L))
            & (ranges["witness"].astype(str) == str(row.witness))
        ]
        if len(match) == 1:
            sample = match.iloc[0]
            center = float(sample["tau_star"])
            minimum = float(sample["tau_min"])
            maximum = float(sample["tau_max"])
            sampled = True
        else:
            center = float(row.tau_mc_raw)
            minimum = maximum = np.nan
            sampled = False
        records.append(
            {
                "L": int(row.L),
                "witness": str(row.witness),
                "tau_star": center,
                "tau_min": minimum,
                "tau_max": maximum,
                "sampled_range_available": sampled,
                "kappa_star_over_J": REPRESENTATIVE_KAPPA_OVER_J,
                "target_beta0": WITNESS_TARGETS[str(row.witness)],
            }
        )
    return pd.DataFrame(records).sort_values(["witness", "L"])


def _spin1_deformation_scan(grid: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    valid_lengths: list[int] = []
    for length, group in grid.groupby(grid["L"].astype(int)):
        if group["kappa_over_J"].nunique() >= 2:
            valid_lengths.append(int(length))
    if not valid_lengths:
        raise ValueError("Fig. 6(c) requires at least two sampled compatible kappa values")
    length = max(valid_lengths)
    group = grid[grid["L"].astype(int) == length].sort_values("kappa_over_J")
    records: list[dict] = []
    for row in group.itertuples(index=False):
        for key in ("A", "Z", "Y"):
            records.append(
                {
                    "L": length,
                    "kappa_over_J": float(row.kappa_over_J),
                    "witness": key,
                    "value": float(getattr(row, f"tau_{key}_mc_raw")),
                    "target_beta0": WITNESS_TARGETS[key],
                }
            )
    return pd.DataFrame(records), length


def _draw_spin1_panel_b(axes: list, panel_b: pd.DataFrame) -> None:
    for index, key in enumerate(("A", "Z", "Y")):
        frame = panel_b[panel_b["witness"] == key].sort_values("L")
        color = WITNESS_COLORS[key]
        target = WITNESS_TARGETS[key]
        ylim = _padded_ylim(
            target,
            *frame["tau_star"].to_numpy(dtype=float),
            *frame["tau_min"].dropna().to_numpy(dtype=float),
            *frame["tau_max"].dropna().to_numpy(dtype=float),
        )
        ax = axes[index]
        for row in frame.itertuples(index=False):
            _range_box(
                ax,
                x=float(row.L),
                center=float(row.tau_star),
                minimum=None if pd.isna(row.tau_min) else float(row.tau_min),
                maximum=None if pd.isna(row.tau_max) else float(row.tau_max),
                color=color,
                width=0.26,
            )
        ax.axhline(target, color=TARGET_COLOR, ls="--", lw=0.9, zorder=2)
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_{{\rm mc}}$")
        ax.set_ylim(*ylim)
        _clean_y_ticks(ax)
        ax.grid(alpha=0.13)
        use_integer_ticks(ax, axis="x")
        ax.set_xticks([8, 10, 12, 14])
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"System size $L$")
    handles = _range_semantics_handles(r"$\kappa_\star$", r"$\kappa$ scan")
    handles.append(Line2D([0], [0], color=TARGET_COLOR, ls="--", lw=0.9, label=r"$\beta=0$"))
    _top_legend(axes[0], handles=handles, ncol=3)


def _draw_spin1_panel_c(axes: list, scan: pd.DataFrame, length: int) -> None:
    for index, key in enumerate(("A", "Z", "Y")):
        frame = scan[scan["witness"] == key].sort_values("kappa_over_J")
        color = WITNESS_COLORS[key]
        target = WITNESS_TARGETS[key]
        ylim = _padded_ylim(target, *frame["value"].to_numpy(dtype=float))
        ax = axes[index]
        ax.plot(
            frame["kappa_over_J"],
            frame["value"],
            color=color,
            marker="o",
            markersize=POINT_MARKER_SIZE,
            markeredgewidth=POINT_EDGE_WIDTH,
            linewidth=0.9,
        )
        ax.axhline(target, color=TARGET_COLOR, ls="--", lw=0.9, zorder=2)
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_{{\rm mc}}$")
        ax.set_ylim(*ylim)
        _clean_y_ticks(ax)
        ax.grid(alpha=0.13)
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Compatible deformation $\kappa/J$")
    _top_legend(
        axes[0],
        handles=[
            Line2D([0], [0], color="0.35", marker="o", lw=0.9, label=rf"$L={length}$"),
            Line2D([0], [0], color=TARGET_COLOR, ls="--", lw=0.9, label=r"$\beta=0$"),
        ],
        ncol=2,
    )


def render_spin1_figure6(
    data: Path,
    figures: Path,
    *,
    allow_incomplete: bool,
    read_csv: Callable[[Path], pd.DataFrame],
    save_figure: Callable[..., list[str]],
) -> list[str]:
    """Render Fig. 6 from frozen evidence with merged finite-size/range panel."""

    del allow_incomplete
    panel_a, representative_b, ranges_b, panel_d, grid_path = _base._spin1_tables(
        data=data,
        figures=figures,
        read_csv=read_csv,
    )
    grid = read_csv(grid_path)
    panel_b = _spin1_panel_b_merged(representative_b, ranges_b)
    panel_c, scan_length = _spin1_deformation_scan(grid)
    _base._write_csv(figures / "spin1_xy_figure6_panel_b_plot.csv", panel_b)
    _base._write_csv(figures / "spin1_xy_figure6_panel_c_plot.csv", panel_c)

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FIG_HEIGHT_IN))
    outer = _base._outer_grid(fig)

    gsa = outer[0, 0].subgridspec(3, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gsa[index]) for index in range(3)]
    tower = panel_a["is_tower_state"].fillna(False).astype(bool).to_numpy()
    background = panel_a[~tower]
    l12 = representative_b[representative_b["L"].astype(int) == 12]
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
    _panel_label(axes_a[0], "(a)")

    gsb = outer[0, 1].subgridspec(3, 1, hspace=0.10)
    axes_b = [fig.add_subplot(gsb[index]) for index in range(3)]
    _draw_spin1_panel_b(axes_b, panel_b)
    _panel_label(axes_b[0], "(b)")

    gsc = outer[1, 0].subgridspec(3, 1, hspace=0.10)
    axes_c = [fig.add_subplot(gsc[index]) for index in range(3)]
    _draw_spin1_panel_c(axes_c, panel_c, scan_length)
    _panel_label(axes_c[0], "(c)")

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
        _range_box(
            axd,
            x=float(row.L),
            center=float(row.w_star),
            minimum=None if pd.isna(row.w_min) else float(row.w_min),
            maximum=None if pd.isna(row.w_max) else float(row.w_max),
            color=WITNESS_COLORS["A"],
            width=0.28,
        )
    axd.set_xlabel(r"System size $L$")
    axd.set_ylabel(r"$w_L^{\rm raw}$")
    axd.set_ylim(bottom=0.0)
    use_integer_ticks(axd, axis="x")
    axd.set_xticks([8, 10, 12, 14])
    axd.grid(alpha=0.13)
    axd.legend(
        handles=_range_semantics_handles(
            r"$\kappa_\star$",
            r"$\kappa$ scan",
            color=WITNESS_COLORS["A"],
        ),
        loc="upper right",
        frameon=False,
        fontsize=6.8,
        handletextpad=0.45,
        borderaxespad=0.3,
    )
    _panel_label(axd, "(d)")

    manifest = {
        "figure": "Fig. 6",
        "source_evidence_directory": str(data),
        "star_color": STAR_COLOR,
        "witness_colors": WITNESS_COLORS,
        "panel_b_role": "finite size plus compatible-kappa range",
        "panel_b_includes_L14_representative_without_range": True,
        "panel_b_legend": "above axes; kappa_star, kappa scan, beta=0",
        "panel_c_role": "witness versus compatible kappa",
        "panel_c_scan_L": scan_length,
        "panel_c_legend": "above axes",
        "panel_d_legend_color": WITNESS_COLORS["A"],
        "panel_d_marker": "literal_range_box_with_representative_line",
        "panel_d_guide_line": "dashed",
        "range_encoding": "box height = sampled min/max; internal line = representative value",
        "range_display_floor": False,
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


def _qdm_panel_b_merged(panel_b: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for row in panel_b.itertuples(index=False):
        match = ranges[
            (ranges["Lx"].astype(int) == int(row.Lx))
            & (ranges["witness"] == row.witness)
            & (ranges["ensemble"] == row.ensemble)
        ]
        if len(match) == 1:
            sample = match.iloc[0]
            minimum = float(sample["value_min"])
            maximum = float(sample["value_max"])
            phase_grid = sample["sampled_phase_grid"]
        else:
            minimum = maximum = np.nan
            phase_grid = "[]"
        records.append(
            {
                "Lx": int(row.Lx),
                "witness": str(row.witness),
                "ensemble": str(row.ensemble),
                "phase_star": float(row.phase),
                "value": float(row.value),
                "value_min": minimum,
                "value_max": maximum,
                "sampled_phase_grid": phase_grid,
                "stderr": float(row.stderr),
                "method": str(row.method),
                "source_file": str(row.source_file),
            }
        )
    return pd.DataFrame(records).sort_values(["witness", "ensemble", "Lx"])


def _qdm_deformation_scan(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    positive = raw[raw["phase"] > 0].copy()
    valid_lengths = [
        int(length)
        for length, group in positive.groupby(positive["Lx"].astype(int))
        if group["phase"].nunique() >= 2
    ]
    if not valid_lengths:
        raise ValueError("Fig. 9(c) requires at least two sampled compatible phases")
    length = max(valid_lengths)
    group = positive[positive["Lx"].astype(int) == length].sort_values("phase")
    records: list[dict] = []
    for row in group.itertuples(index=False):
        for key in ("A", "Z"):
            ref_col = _base._qdm_reference_column(group, key)
            records.extend(
                [
                    {
                        "Lx": length,
                        "phase": float(row.phase),
                        "witness": key,
                        "ensemble": "raw_microcanonical",
                        "value": float(getattr(row, f"tau_{key}_mc")),
                    },
                    {
                        "Lx": length,
                        "phase": float(row.phase),
                        "witness": key,
                        "ensemble": "canonical",
                        "value": float(getattr(row, ref_col)),
                    },
                ]
            )
    return pd.DataFrame(records), length


def _qdm_ensemble_box_handles() -> list[Line2D]:
    """Use line style plus fill state so the ensemble key stays visible when ranges are tiny."""

    neutral = "0.35"
    return [
        Line2D(
            [0],
            [0],
            color=neutral,
            marker="o",
            markerfacecolor=neutral,
            markeredgecolor=neutral,
            linestyle="-",
            lw=RANGE_BOX_CENTER_WIDTH,
            label="raw MC",
        ),
        Line2D(
            [0],
            [0],
            color=neutral,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=neutral,
            linestyle="--",
            lw=RANGE_BOX_CENTER_WIDTH,
            label="canonical",
        ),
    ]


def _qdm_scan_handles() -> list[Line2D]:
    neutral = "0.35"
    return [
        Line2D(
            [0],
            [0],
            color=neutral,
            marker="o",
            markerfacecolor=neutral,
            markeredgecolor=neutral,
            lw=0.9,
            label="raw MC",
        ),
        Line2D(
            [0],
            [0],
            color=neutral,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=neutral,
            linestyle="--",
            lw=0.9,
            label="canonical",
        ),
    ]


def _draw_qdm_panel_b(axes: list, panel_b: pd.DataFrame) -> None:
    for index, key in enumerate(("A", "Z")):
        color = WITNESS_COLORS[key]
        frame = panel_b[panel_b["witness"] == key]
        ylim = _padded_ylim(
            *frame["value"].to_numpy(dtype=float),
            *frame["value_min"].dropna().to_numpy(dtype=float),
            *frame["value_max"].dropna().to_numpy(dtype=float),
        )
        ax = axes[index]
        for ensemble, filled, linestyle in (
            ("raw_microcanonical", True, "-"),
            ("canonical", False, "--"),
        ):
            selected = frame[frame["ensemble"] == ensemble].sort_values("Lx")
            for row in selected.itertuples(index=False):
                _range_box(
                    ax,
                    x=float(row.Lx),
                    center=float(row.value),
                    minimum=None if pd.isna(row.value_min) else float(row.value_min),
                    maximum=None if pd.isna(row.value_max) else float(row.value_max),
                    color=color,
                    width=0.42,
                    filled=filled,
                    linestyle=linestyle,
                )
                _center_marker(
                    ax,
                    x=float(row.Lx),
                    y=float(row.value),
                    color=color,
                    filled=filled,
                )
        typicality = frame[
            (frame["ensemble"] == "canonical") & (frame["method"] == "canonical_typicality")
        ]
        if not typicality.empty and float(typicality.iloc[0]["stderr"]) > 0.0:
            ax.errorbar(
                typicality["Lx"],
                typicality["value"],
                yerr=typicality["stderr"],
                fmt="none",
                color=color,
                capsize=TYPICALITY_CAP_SIZE,
                capthick=TYPICALITY_LINE_WIDTH,
                elinewidth=TYPICALITY_LINE_WIDTH,
                zorder=8,
            )
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle$")
        ax.set_ylim(*ylim)
        _clean_y_ticks(ax)
        ax.grid(alpha=0.13)
        use_integer_ticks(ax, axis="x")
        ax.set_xticks([4, 8, 12])
        if index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Strip length $L_x$")
    _top_legend(axes[0], handles=_qdm_ensemble_box_handles(), ncol=2)


def _draw_qdm_panel_c(axes: list, scan: pd.DataFrame, length: int) -> None:
    for index, key in enumerate(("A", "Z")):
        color = WITNESS_COLORS[key]
        frame = scan[scan["witness"] == key]
        ylim = _padded_ylim(*frame["value"].to_numpy(dtype=float))
        ax = axes[index]
        for ensemble, filled, linestyle in (
            ("raw_microcanonical", True, "-"),
            ("canonical", False, "--"),
        ):
            selected = frame[frame["ensemble"] == ensemble].sort_values("phase")
            ax.plot(
                selected["phase"],
                selected["value"],
                color=color,
                marker="o",
                markersize=POINT_MARKER_SIZE,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=POINT_EDGE_WIDTH,
                linestyle=linestyle,
                linewidth=0.9,
            )
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle$")
        ax.set_ylim(*ylim)
        _clean_y_ticks(ax)
        ax.grid(alpha=0.13)
        if index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Compatible deformation $\varphi$")
    _top_legend(
        axes[0],
        handles=_qdm_scan_handles(),
        ncol=2,
        title=rf"$L_x={length}$",
    )


def render_qdm_figure9(
    *,
    data: Path,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> list[Path]:
    """Render Fig. 9 with merged finite-size/range and direct phase-scan panels."""

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
    representative_row = representative_raw[representative_raw["Lx"].astype(int) == largest].iloc[
        -1
    ]

    representative_b = _base._qdm_panel_b(
        raw=raw,
        canonical_l12=canonical_l12,
        thermal_path=thermal_path,
        canonical_path=canonical_path,
        phase=phase,
    )
    ranges_b = _base._qdm_panel_c(
        raw=raw,
        panel_b=representative_b,
        thermal_path=thermal_path,
        phase_check=phase_check,
        phase_check_path=phase_check_path,
        phase=phase,
    )
    panel_b = _qdm_panel_b_merged(representative_b, ranges_b)
    panel_c, scan_length = _qdm_deformation_scan(raw)
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
    lower = (
        representative_row.cage_energy_density - representative_row.window_energy_density_half_width
    )
    upper = (
        representative_row.cage_energy_density + representative_row.window_energy_density_half_width
    )
    for index, (key, column) in enumerate((("A", "Q_A"), ("Z", "Q_Z"))):
        color = WITNESS_COLORS[key]
        ax = axes_a[index]
        ax.axvspan(lower, upper, color="0.5", alpha=0.10, zorder=0)
        ax.scatter(
            scatter_largest["energy_density"],
            scatter_largest[column],
            s=9,
            alpha=0.38,
            color=color,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            [representative_row.cage_energy_density],
            [0.0],
            marker="*",
            s=76,
            color=STAR_COLOR,
            edgecolors="black",
            linewidths=0.4,
            zorder=8,
        )
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_n$")
        ax.grid(alpha=0.13)
        if index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/(4L_x)$")
    _panel_label(axes_a[0], "(a)")

    gsb = outer[0, 1].subgridspec(2, 1, hspace=0.10)
    axes_b = [fig.add_subplot(gsb[index]) for index in range(2)]
    _draw_qdm_panel_b(axes_b, panel_b)
    _panel_label(axes_b[0], "(b)")

    gsc = outer[1, 0].subgridspec(2, 1, hspace=0.10)
    axes_c = [fig.add_subplot(gsc[index]) for index in range(2)]
    _draw_qdm_panel_c(axes_c, panel_c, scan_length)
    _panel_label(axes_c[0], "(c)")

    axd = fig.add_subplot(outer[1, 1])
    if not panel_d.empty:
        axd.plot(
            panel_d["Lx"],
            panel_d["w_star"],
            color=WITNESS_COLORS["A"],
            linewidth=GUIDE_LINE_WIDTH,
            linestyle="--",
            alpha=0.72,
            zorder=1,
        )
    for row in panel_d.itertuples(index=False):
        _range_box(
            axd,
            x=float(row.Lx),
            center=float(row.w_star),
            minimum=float(row.w_min),
            maximum=float(row.w_max),
            color=WITNESS_COLORS["A"],
            width=0.34,
        )
    axd.set_xlabel(r"Strip length $L_x$")
    axd.set_ylabel(r"$w_{L_x}^{\rm raw}$")
    if not panel_d.empty:
        upper_values = np.concatenate(
            [
                panel_d["w_star"].to_numpy(dtype=float),
                panel_d["w_max"].to_numpy(dtype=float),
            ]
        )
        finite_upper = upper_values[np.isfinite(upper_values)]
        upper_limit = 1.0 if finite_upper.size == 0 else 1.08 * float(np.max(finite_upper))
        axd.set_ylim(0.0, upper_limit)
    else:
        axd.set_ylim(bottom=0.0)
    axd.grid(alpha=0.13)
    use_integer_ticks(axd, axis="x")
    axd.set_xticks(sorted(set(panel_d["Lx"].astype(int))))
    axd.legend(
        handles=_range_semantics_handles(
            r"$\varphi_\star$",
            r"$\varphi$ scan",
            color=WITNESS_COLORS["A"],
        ),
        loc="upper right",
        frameon=False,
        fontsize=6.8,
        handletextpad=0.45,
        borderaxespad=0.3,
    )
    _panel_label(axd, "(d)")

    written: list[Path] = []
    for stem in ("qdm_checkerboard_figure7_combined", "qdm_checkerboard_figure9_prx"):
        written.extend(
            save_prx_figure(
                fig,
                stem,
                directory=figures,
                formats=formats,
                close=False,
            )
        )
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
        "witness_colors": {"A": WITNESS_COLORS["A"], "Z": WITNESS_COLORS["Z"]},
        "panel_b_role": "finite size plus compatible-phase range",
        "panel_b_ensemble_encoding": "raw filled solid circle; canonical open dashed circle",
        "panel_b_horizontal_displacement": False,
        "panel_b_legend": "above axes",
        "panel_c_role": "witness versus compatible phase",
        "panel_c_scan_Lx": scan_length,
        "panel_c_ensemble_encoding": "raw filled solid; canonical open dashed",
        "panel_c_legend": "above axes",
        "panel_d_marker": "literal_range_box_with_representative_line",
        "panel_d_box_width": 0.34,
        "panel_d_guide_line": "dashed",
        "panel_d_legend_color": WITNESS_COLORS["A"],
        "range_encoding": "box height = sampled min/max; internal line = representative value",
        "range_display_floor": False,
        "raw_12x4_plotted": False,
        "canonical_12x4_plotted": True,
        "no_interpolated_deformation_values": True,
        "expensive_recomputation": False,
    }
    _base._write_json(figures / "qdm_checkerboard_figure9_provenance.json", manifest)
    return written