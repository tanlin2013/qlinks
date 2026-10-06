"""Shared render-only redesign for PRX Figs. 6 and 9.

This module implements the handoff visual grammar without launching any physics
calculation.  It consumes only frozen CSV/JSON evidence and emits the figure,
per-panel machine-readable plot tables, and a provenance manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from helpers import add_panel_label_margin, save_prx_figure, use_integer_ticks

FULL_WIDTH_IN = 7.05
FIG_HEIGHT_IN = 5.75
LINE_WIDTH = 1.0
MARKER_SIZE = 4.5
CAP_SIZE = 3.0
REPRESENTATIVE_KAPPA_OVER_J = 0.10
RAW_COLOR = "tab:blue"
CANONICAL_COLOR = "tab:orange"
SPIN1_COLOR = "tab:blue"
TARGET_COLOR = "0.45"
WITNESS_TARGETS = {"A": 1.0 / 9.0, "Z": 2.0 / 9.0, "Y": 1.0 / 3.0}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _asymmetric_yerr(center: float, minimum: float, maximum: float) -> np.ndarray:
    return np.asarray(
        [[max(0.0, center - minimum)], [max(0.0, maximum - center)]],
        dtype=float,
    )


def _sampled_errorbar(
    ax,
    *,
    x: float,
    center: float,
    minimum: float | None,
    maximum: float | None,
    color: str,
    marker: str = "o",
    markerfacecolor: str | None = None,
    zorder: int = 5,
) -> None:
    has_range = (
        minimum is not None
        and maximum is not None
        and np.isfinite(minimum)
        and np.isfinite(maximum)
        and maximum + 1.0e-15 >= center >= minimum - 1.0e-15
        and maximum - minimum > 1.0e-14
    )
    kwargs = {
        "fmt": marker,
        "color": color,
        "markersize": MARKER_SIZE,
        "linewidth": LINE_WIDTH,
        "capsize": CAP_SIZE if has_range else 0.0,
        "capthick": 0.9,
        "elinewidth": 0.9,
        "zorder": zorder,
    }
    if markerfacecolor is not None:
        kwargs["markerfacecolor"] = markerfacecolor
        kwargs["markeredgecolor"] = color
    ax.errorbar(
        [x],
        [center],
        yerr=(
            _asymmetric_yerr(center, float(minimum), float(maximum)) if has_range else None
        ),
        **kwargs,
    )


def _outer_grid(fig: plt.Figure):
    return fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.095,
        top=0.925,
        wspace=0.34,
        hspace=0.46,
    )


def _shared_positive_ylim(*values: float) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    upper = 1.0 if finite.size == 0 else float(np.max(finite))
    return 0.0, max(0.05, 1.15 * upper)


def _spin1_grid_source(data: Path, read_csv: Callable[[Path], pd.DataFrame]) -> tuple[pd.DataFrame, Path]:
    candidates = (
        data / "spin1_xy_sec6_p1_kappa_refinement_rows.csv",
        data.parent / "p1" / "spin1_xy_sec6_p1_kappa_refinement_rows.csv",
        data / "spin1_xy_sec6_deformation_grid_rows.csv",
    )
    for path in candidates:
        if path.is_file():
            return read_csv(path), path
    raise FileNotFoundError(
        "Fig. 6(c,d) requires sampled deformation rows. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _spin1_plot_tables(
    *,
    data: Path,
    figures: Path,
    read_csv: Callable[[Path], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    scatter_path = data / "spin1_xy_figure6_panel_a_scatter.csv"
    sequence_path = data / "spin1_xy_figure6_panel_b_witness_sequence.csv"
    concentration_path = data / "spin1_xy_kappa0p1_concentration_common_windows.csv"
    scatter = read_csv(scatter_path)
    sequence = read_csv(sequence_path)
    concentration = read_csv(concentration_path)
    grid, grid_path = _spin1_grid_source(data, read_csv)

    panel_b = sequence[["L", "witness", "tau_mc_raw"]].copy()
    panel_b["target_beta0"] = panel_b["witness"].map(WITNESS_TARGETS)
    panel_b["kappa_over_J"] = REPRESENTATIVE_KAPPA_OVER_J
    panel_b["source_file"] = sequence_path.name

    records_c: list[dict] = []
    for length in (8, 10, 12):
        group = grid[grid["L"].astype(int) == length].sort_values("kappa_over_J")
        if group.empty:
            continue
        kappas = group["kappa_over_J"].to_numpy(dtype=float)
        rep = group[np.isclose(group["kappa_over_J"].to_numpy(dtype=float), REPRESENTATIVE_KAPPA_OVER_J)]
        if len(rep) != 1:
            raise ValueError(f"Fig. 6(c) needs one representative kappa row at L={length}")
        for key in ("A", "Z", "Y"):
            column = f"tau_{key}_mc_raw"
            values = group[column].to_numpy(dtype=float)
            records_c.append(
                {
                    "L": length,
                    "witness": key,
                    "kappa_star_over_J": REPRESENTATIVE_KAPPA_OVER_J,
                    "tau_star": float(rep.iloc[0][column]),
                    "tau_min": float(np.min(values)),
                    "tau_max": float(np.max(values)),
                    "sampled_kappa_grid": json.dumps([float(value) for value in kappas]),
                    "sampled_kappa_count": int(len(kappas)),
                    "target_beta0": WITNESS_TARGETS[key],
                    "source_file": grid_path.name,
                    "source_evidence_directory": str(grid_path.parent),
                }
            )
    panel_c = pd.DataFrame(records_c)

    primary = concentration[
        (concentration["variant"].astype(str) == "raw")
        & np.isclose(
            concentration["kappa_over_J"].to_numpy(dtype=float),
            REPRESENTATIVE_KAPPA_OVER_J,
        )
    ].copy()
    if "window_protocol" in primary.columns:
        preferred = primary[primary["window_protocol"].astype(str).str.contains("quarter_power")]
        if not preferred.empty:
            primary = preferred
    primary = primary.sort_values("L")
    if set(primary["L"].astype(int)) != {8, 10, 12, 14}:
        raise ValueError("Fig. 6(d) requires representative L=8,10,12,14 concentration rows")

    records_d: list[dict] = []
    for row in primary.itertuples(index=False):
        length = int(row.L)
        group = grid[grid["L"].astype(int) == length].sort_values("kappa_over_J")
        if length == 14 or group.empty:
            minimum = maximum = None
            kappas: list[float] = []
        else:
            values = group["w_L_raw"].to_numpy(dtype=float)
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            kappas = [float(value) for value in group["kappa_over_J"].to_numpy(dtype=float)]
        records_d.append(
            {
                "L": length,
                "kappa_star_over_J": REPRESENTATIVE_KAPPA_OVER_J,
                "w_star": float(row.w_L),
                "w_min": minimum,
                "w_max": maximum,
                "sampled_kappa_grid": json.dumps(kappas),
                "sampled_kappa_count": len(kappas),
                "source_file": (
                    concentration_path.name if length == 14 else grid_path.name
                ),
                "source_evidence_directory": str(data if length == 14 else grid_path.parent),
            }
        )
    panel_d = pd.DataFrame(records_d)

    panel_a = scatter.copy()
    panel_a["source_file"] = scatter_path.name

    for name, frame in (
        ("spin1_xy_figure6_panel_a_plot.csv", panel_a),
        ("spin1_xy_figure6_panel_b_plot.csv", panel_b),
        ("spin1_xy_figure6_panel_c_plot.csv", panel_c),
        ("spin1_xy_figure6_panel_d_plot.csv", panel_d),
    ):
        _write_csv(figures / name, frame)
    return panel_a, panel_b, panel_c, panel_d, grid_path


def render_spin1_figure6(
    data: Path,
    figures: Path,
    *,
    allow_incomplete: bool,
    read_csv: Callable[[Path], pd.DataFrame],
    save_figure: Callable[..., list[str]],
) -> list[str]:
    del allow_incomplete  # missing sampled data is reported explicitly; no inferred ranges are allowed.
    panel_a, panel_b, panel_c, panel_d, grid_path = _spin1_plot_tables(
        data=data,
        figures=figures,
        read_csv=read_csv,
    )

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FIG_HEIGHT_IN))
    outer = _outer_grid(fig)

    # (a) Eigenstate-level local exceptionalness.
    gsa = outer[0, 0].subgridspec(3, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gsa[i]) for i in range(3)]
    tower = panel_a["is_tower_state"].fillna(False).astype(bool).to_numpy()
    background = panel_a[~tower]
    sequence = panel_b.copy()
    l12 = sequence[sequence["L"].astype(int) == 12]
    source_sequence = read_csv(data / "spin1_xy_figure6_panel_b_witness_sequence.csv")
    half = float(source_sequence[source_sequence["L"].astype(int) == 12].iloc[0]["window_energy_density_half_width"])
    for index, key in enumerate(("A", "Z", "Y")):
        ax = axes_a[index]
        ax.axvspan(-half, half, color="0.5", alpha=0.10, zorder=0)
        ax.scatter(
            background["energy_density"],
            background[f"Q_{key}"],
            s=8,
            alpha=0.38,
            color=SPIN1_COLOR,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter([0.0], [0.0], marker="*", s=72, color="firebrick", edgecolors="black", linewidths=0.4, zorder=8)
        mean = l12[l12["witness"].astype(str) == key]
        if len(mean) == 1:
            ax.axhline(float(mean.iloc[0]["tau_mc_raw"]), color="0.35", ls=":", lw=0.8)
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_n$")
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/L$")
    add_panel_label_margin(axes_a[0], "(a)")

    # (b) Representative finite-size trend, one witness per mini-axis.
    gsb = outer[0, 1].subgridspec(3, 1, hspace=0.10)
    axes_b = [fig.add_subplot(gsb[i]) for i in range(3)]
    # (c) Finite-size trend versus sampled deformation range, same grammar/limits.
    gsc = outer[1, 0].subgridspec(3, 1, hspace=0.10)
    axes_c = [fig.add_subplot(gsc[i]) for i in range(3)]
    for index, key in enumerate(("A", "Z", "Y")):
        b = panel_b[panel_b["witness"].astype(str) == key].sort_values("L")
        c = panel_c[panel_c["witness"].astype(str) == key].sort_values("L")
        target = WITNESS_TARGETS[key]
        ylim = _shared_positive_ylim(
            target,
            *b["tau_mc_raw"].to_numpy(dtype=float),
            *c["tau_max"].to_numpy(dtype=float),
        )

        axb = axes_b[index]
        axb.plot(
            b["L"],
            b["tau_mc_raw"],
            color=SPIN1_COLOR,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
        )
        axb.axhline(target, color=TARGET_COLOR, ls="--", lw=0.8)
        axb.text(0.98, 0.88, rf"$\tau_{{{key}}}^{{\beta=0}}={target:.3g}$", transform=axb.transAxes, ha="right", va="top", fontsize=7.6)
        axb.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_{{\rm mc}}$")
        axb.set_ylim(*ylim)
        axb.grid(alpha=0.15)
        use_integer_ticks(axb, axis="x")
        axb.set_xticks([8, 10, 12, 14])
        if index < 2:
            axb.tick_params(labelbottom=False)
        else:
            axb.set_xlabel(r"System size $L$")

        axc = axes_c[index]
        for row in c.itertuples(index=False):
            _sampled_errorbar(
                axc,
                x=float(row.L),
                center=float(row.tau_star),
                minimum=float(row.tau_min),
                maximum=float(row.tau_max),
                color=SPIN1_COLOR,
            )
        if not c.empty:
            axc.plot(c["L"], c["tau_star"], color=SPIN1_COLOR, linewidth=LINE_WIDTH, zorder=3)
        axc.axhline(target, color=TARGET_COLOR, ls="--", lw=0.8)
        axc.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_{{\rm mc}}$")
        axc.set_ylim(*ylim)
        axc.grid(alpha=0.15)
        use_integer_ticks(axc, axis="x")
        axc.set_xticks([8, 10, 12])
        if index < 2:
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel(r"System size $L$")

    axes_c[0].text(
        0.98,
        0.88,
        r"whiskers: sampled $\kappa/J$ range",
        transform=axes_c[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
    )
    add_panel_label_margin(axes_b[0], "(b)")
    add_panel_label_margin(axes_c[0], "(c)")

    # (d) Background concentration with sampled-range whiskers, never a shaded envelope.
    axd = fig.add_subplot(outer[1, 1])
    for row in panel_d.itertuples(index=False):
        minimum = None if pd.isna(row.w_min) else float(row.w_min)
        maximum = None if pd.isna(row.w_max) else float(row.w_max)
        _sampled_errorbar(
            axd,
            x=float(row.L),
            center=float(row.w_star),
            minimum=minimum,
            maximum=maximum,
            color=SPIN1_COLOR,
        )
    axd.plot(panel_d["L"], panel_d["w_star"], color=SPIN1_COLOR, linewidth=LINE_WIDTH, zorder=3)
    axd.set_xlabel(r"System size $L$")
    axd.set_ylabel(r"$w_L^{\rm raw}$")
    axd.set_ylim(bottom=0.0)
    use_integer_ticks(axd, axis="x")
    axd.set_xticks([8, 10, 12, 14])
    axd.grid(alpha=0.15)
    marker_handle = Line2D([0], [0], color=SPIN1_COLOR, marker="o", lw=LINE_WIDTH, label=r"$\kappa_\star/J=0.1$")
    whisker_handle = Line2D([0], [0], color=SPIN1_COLOR, marker="|", markersize=10, lw=0.9, label="sampled range")
    axd.legend(handles=[marker_handle, whisker_handle], loc="upper right", frameon=False, fontsize=7.6)
    add_panel_label_margin(axd, "(d)")

    manifest = {
        "figure": "Fig. 6",
        "design": "local exception -> representative finite-size trend -> sampled-deformation robustness -> background concentration",
        "source_evidence_directory": str(data),
        "panels": {
            "a": {"source_file": "spin1_xy_figure6_panel_a_scatter.csv", "ensemble": "eigenstate"},
            "b": {"source_file": "spin1_xy_figure6_panel_b_witness_sequence.csv", "ensemble": "raw microcanonical", "representative_kappa_over_J": REPRESENTATIVE_KAPPA_OVER_J},
            "c": {"source_file": grid_path.name, "source_directory": str(grid_path.parent), "ensemble": "raw microcanonical", "whiskers": "sampled min/max only"},
            "d": {"representative_source": "spin1_xy_kappa0p1_concentration_common_windows.csv", "range_source": grid_path.name, "range_source_directory": str(grid_path.parent), "whiskers": "sampled min/max only", "L14_whisker": False},
        },
        "no_interpolated_deformation_values": True,
        "expensive_recomputation": False,
    }
    _write_json(figures / "spin1_xy_figure6_provenance.json", manifest)
    return save_figure(fig, figures, "spin1_xy_figure6_prx", preview=True)


def _qdm_gate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "window_coverage_complete" in result.columns:
        large = result["Lx"].astype(int) >= 12
        result = result[~large | result["window_coverage_complete"].fillna(False).astype(bool)]
    if "converged_vs_previous_budget" in result.columns:
        large = result["Lx"].astype(int) >= 12
        result = result[~large | result["converged_vs_previous_budget"].fillna(False).astype(bool)]
    return result


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _qdm_canonical_target(data: Path, *, phi: float) -> tuple[pd.DataFrame, Path]:
    path = data / "qdm_checkerboard_finite_beta_transfer_target.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Fig. 9(b) requires qdm_checkerboard_finite_beta_transfer_target.csv so the "
            "Lx=12 canonical-typicality continuation is visible."
        )
    frame = pd.read_csv(path)
    length_col = _first_column(frame, ("Lx", "L_x", "length"))
    if length_col is None:
        raise ValueError(f"canonical target has no strip-length column: {path}")
    phase_col = _first_column(frame, ("phase", "varphi", "phi"))
    selected = frame[frame[length_col].astype(int) == 12].copy()
    if phase_col is not None and not selected.empty:
        distances = np.abs(selected[phase_col].to_numpy(dtype=float) - float(phi))
        selected = selected.iloc[[int(np.argmin(distances))]]
    elif len(selected) > 1:
        selected = selected.tail(1)
    if selected.empty:
        raise ValueError(f"canonical target has no Lx=12 row: {path}")
    selected = selected.rename(columns={length_col: "Lx"})
    selected["phase"] = float(phi) if phase_col is None else selected[phase_col].astype(float)
    return selected, path


def _qdm_phase_check(data: Path) -> tuple[pd.DataFrame, Path | None]:
    path = data / "qdm_checkerboard_finite_beta_transfer_phase_check.csv"
    if not path.is_file():
        return pd.DataFrame(), None
    return pd.read_csv(path), path


def _qdm_reference_column(frame: pd.DataFrame, key: str) -> str:
    for name in (f"tau_{key}_reference_physical", f"tau_{key}_reference"):
        if name in frame.columns:
            return name
    raise ValueError(f"thermal table has no canonical/reference column for {key}")


def render_qdm_figure9(
    *,
    data: Path,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> list[Path]:
    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    thermal_path = data / "qdm_checkerboard_thermal_overlap.csv"
    if not thermal_path.is_file():
        thermal_path = data / "qdm_checkerboard_beta0_overlap.csv"
    thermal = pd.read_csv(thermal_path)
    scatter_path = data / "qdm_checkerboard_eth_scatter.csv"
    concentration_path = data / "qdm_checkerboard_concentration_grid.csv"
    rep_path = data / "qdm_checkerboard_representative_phase.csv"
    scatter = pd.read_csv(scatter_path)
    concentration = pd.read_csv(concentration_path)
    rep = pd.read_csv(rep_path) if rep_path.is_file() else pd.DataFrame()
    if thermal.empty or scatter.empty:
        raise RuntimeError("Checkerboard thermal/scatter evidence is unavailable")

    primary_pref = float(thermal.window_prefactor.iloc[(thermal.window_prefactor - 0.75).abs().argmin()])
    primary_all = thermal[np.isclose(thermal.window_prefactor, primary_pref)].copy()
    phi = (
        float(rep.phi_star.iloc[0])
        if not rep.empty
        else float(sorted(primary_all.phase.unique())[len(primary_all.phase.unique()) // 2])
    )
    raw = _qdm_gate_raw(primary_all)
    canonical_l12, canonical_path = _qdm_canonical_target(data, phi=phi)
    phase_check, phase_check_path = _qdm_phase_check(data)

    representative_raw = raw[np.isclose(raw.phase, phi)].sort_values("Lx")
    raw_lengths = sorted(set(representative_raw["Lx"].astype(int)))
    if 12 in raw_lengths:
        # This is allowed only if the strict coverage gates above certify it.
        raw_12_verified = True
    else:
        raw_12_verified = False
    scatter_lengths = set(scatter["Lx"].astype(int))
    common = sorted(set(raw_lengths).intersection(scatter_lengths))
    if not common:
        raise RuntimeError("No verified raw thermal size has matching ETH-scatter data")
    largest = int(common[-1])
    scatter_largest = scatter[scatter.Lx.astype(int) == largest].copy()
    representative_row = representative_raw[representative_raw.Lx.astype(int) == largest].iloc[-1]

    # Build representative ensemble table for panel (b).
    records_b: list[dict] = []
    for row in representative_raw.itertuples(index=False):
        lx = int(row.Lx)
        for key in ("A", "Z"):
            records_b.append(
                {
                    "Lx": lx,
                    "witness": key,
                    "ensemble": "raw_microcanonical",
                    "phase": phi,
                    "value": float(getattr(row, f"tau_{key}_mc")),
                    "stderr": 0.0,
                    "method": "raw_microcanonical_window",
                    "source_file": thermal_path.name,
                }
            )
            reference_col = _qdm_reference_column(primary_all, key)
            records_b.append(
                {
                    "Lx": lx,
                    "witness": key,
                    "ensemble": "canonical",
                    "phase": phi,
                    "value": float(getattr(row, reference_col)),
                    "stderr": 0.0,
                    "method": "canonical_exact_or_persisted_finite_size",
                    "source_file": thermal_path.name,
                }
            )
    ct = canonical_l12.iloc[-1]
    for key in ("A", "Z"):
        value_col = f"tau_{key}_target"
        stderr_col = f"tau_{key}_stderr"
        if value_col not in canonical_l12.columns:
            raise ValueError(f"Lx=12 canonical target is missing {value_col}")
        records_b.append(
            {
                "Lx": 12,
                "witness": key,
                "ensemble": "canonical",
                "phase": phi,
                "value": float(ct[value_col]),
                "stderr": float(ct[stderr_col]) if stderr_col in canonical_l12.columns and pd.notna(ct[stderr_col]) else 0.0,
                "method": "canonical_typicality",
                "source_file": canonical_path.name,
            }
        )
    panel_b = pd.DataFrame(records_b).drop_duplicates(["Lx", "witness", "ensemble"], keep="last")

    # Panel (c): actual sampled phase min/max only, separately for each ensemble.
    records_c: list[dict] = []
    for lx, group in raw[raw.phase > 0].groupby("Lx"):
        lx = int(lx)
        rep_group = group[np.isclose(group.phase, phi)]
        if len(rep_group) != 1:
            continue
        for key in ("A", "Z"):
            raw_values = group[f"tau_{key}_mc"].to_numpy(dtype=float)
            ref_col = _qdm_reference_column(group, key)
            can_values = group[ref_col].to_numpy(dtype=float)
            grid = [float(value) for value in sorted(group.phase.astype(float).unique())]
            records_c.extend(
                [
                    {
                        "Lx": lx,
                        "witness": key,
                        "ensemble": "raw_microcanonical",
                        "phase_star": phi,
                        "value_star": float(rep_group.iloc[0][f"tau_{key}_mc"]),
                        "value_min": float(np.min(raw_values)),
                        "value_max": float(np.max(raw_values)),
                        "sampled_phase_grid": json.dumps(grid),
                        "source_file": thermal_path.name,
                    },
                    {
                        "Lx": lx,
                        "witness": key,
                        "ensemble": "canonical",
                        "phase_star": phi,
                        "value_star": float(rep_group.iloc[0][ref_col]),
                        "value_min": float(np.min(can_values)),
                        "value_max": float(np.max(can_values)),
                        "sampled_phase_grid": json.dumps(grid),
                        "source_file": thermal_path.name,
                    },
                ]
            )

    # A genuine stored Lx=12 canonical phase check can contribute a range; a single point cannot.
    if not phase_check.empty:
        length_col = _first_column(phase_check, ("Lx", "L_x", "length"))
        phase_col = _first_column(phase_check, ("phase", "varphi", "phi"))
        if length_col is not None and phase_col is not None:
            group12 = phase_check[phase_check[length_col].astype(int) == 12].copy()
            positive = group12[group12[phase_col].astype(float) > 0].copy()
            if positive[phase_col].nunique() >= 2:
                for key in ("A", "Z"):
                    value_col = _first_column(positive, (f"tau_{key}_target", f"tau_{key}_raw", f"tau_{key}"))
                    if value_col is None:
                        continue
                    values = positive[value_col].to_numpy(dtype=float)
                    star = panel_b[(panel_b.Lx == 12) & (panel_b.witness == key) & (panel_b.ensemble == "canonical")]
                    if len(star) != 1:
                        continue
                    records_c.append(
                        {
                            "Lx": 12,
                            "witness": key,
                            "ensemble": "canonical",
                            "phase_star": phi,
                            "value_star": float(star.iloc[0].value),
                            "value_min": float(np.min(values)),
                            "value_max": float(np.max(values)),
                            "sampled_phase_grid": json.dumps([float(value) for value in sorted(positive[phase_col].astype(float).unique())]),
                            "source_file": phase_check_path.name if phase_check_path is not None else "",
                        }
                    )
    panel_c = pd.DataFrame(records_c)

    # Panel (d): raw concentration only, with phase min/max at each covered size.
    c = _qdm_gate_raw(concentration.copy())
    c = c[c.phase > 0].copy()
    value_col = "w_raw" if "w_raw" in c.columns else "w"
    records_d: list[dict] = []
    for lx, group in c.groupby("Lx"):
        lx = int(lx)
        rep_group = group[np.isclose(group.phase, phi)]
        if len(rep_group) != 1:
            continue
        values = group[value_col].to_numpy(dtype=float)
        records_d.append(
            {
                "Lx": lx,
                "phase_star": phi,
                "w_star": float(rep_group.iloc[0][value_col]),
                "w_min": float(np.min(values)),
                "w_max": float(np.max(values)),
                "sampled_phase_grid": json.dumps([float(value) for value in sorted(group.phase.astype(float).unique())]),
                "source_file": concentration_path.name,
            }
        )
    panel_d = pd.DataFrame(records_d).sort_values("Lx")

    panel_a = scatter_largest.copy()
    panel_a["source_file"] = scatter_path.name
    for name, frame in (
        ("qdm_checkerboard_figure9_panel_a_plot.csv", panel_a),
        ("qdm_checkerboard_figure9_panel_b_plot.csv", panel_b),
        ("qdm_checkerboard_figure9_panel_c_plot.csv", panel_c),
        ("qdm_checkerboard_figure9_panel_d_plot.csv", panel_d),
    ):
        _write_csv(figures / name, frame)

    fig = plt.figure(figsize=(FULL_WIDTH_IN, FIG_HEIGHT_IN))
    outer = _outer_grid(fig)

    gsa = outer[0, 0].subgridspec(2, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gsa[i]) for i in range(2)]
    for index, (key, column) in enumerate((("A", "Q_A"), ("Z", "Q_Z"))):
        ax = axes_a[index]
        ax.axvspan(
            representative_row.cage_energy_density - representative_row.window_energy_density_half_width,
            representative_row.cage_energy_density + representative_row.window_energy_density_half_width,
            color="0.5",
            alpha=0.10,
            zorder=0,
        )
        ax.scatter(
            scatter_largest.energy_density,
            scatter_largest[column],
            s=9,
            alpha=0.42,
            color="0.35",
            linewidths=0,
            rasterized=True,
        )
        ax.scatter([representative_row.cage_energy_density], [0.0], marker="*", s=76, color=RAW_COLOR, edgecolors="black", linewidths=0.4, zorder=8)
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle_n$")
        ax.grid(alpha=0.15)
        if index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/(4L_x)$")
    add_panel_label_margin(axes_a[0], "(a)")

    gsb = outer[0, 1].subgridspec(2, 1, hspace=0.10)
    axes_b = [fig.add_subplot(gsb[i]) for i in range(2)]
    gsc = outer[1, 0].subgridspec(2, 1, hspace=0.10)
    axes_c = [fig.add_subplot(gsc[i]) for i in range(2)]

    for index, key in enumerate(("A", "Z")):
        b_raw = panel_b[(panel_b.witness == key) & (panel_b.ensemble == "raw_microcanonical")].sort_values("Lx")
        b_can = panel_b[(panel_b.witness == key) & (panel_b.ensemble == "canonical")].sort_values("Lx")
        c_key = panel_c[panel_c.witness == key]
        ylim = _shared_positive_ylim(
            *b_raw.value.to_numpy(dtype=float),
            *b_can.value.to_numpy(dtype=float),
            *c_key.value_max.to_numpy(dtype=float),
        )

        axb = axes_b[index]
        if not b_raw.empty:
            axb.plot(b_raw.Lx, b_raw.value, color=RAW_COLOR, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
        if not b_can.empty:
            axb.plot(b_can.Lx, b_can.value, color=CANONICAL_COLOR, marker="o", markerfacecolor="white", markeredgecolor=CANONICAL_COLOR, markersize=MARKER_SIZE + 0.5, linewidth=LINE_WIDTH)
            typicality = b_can[b_can.method == "canonical_typicality"]
            if not typicality.empty and float(typicality.stderr.iloc[0]) > 0.0:
                axb.errorbar(typicality.Lx, typicality.value, yerr=typicality.stderr, fmt="none", color=CANONICAL_COLOR, capsize=CAP_SIZE, elinewidth=0.9)
        axb.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle$")
        axb.set_ylim(*ylim)
        axb.grid(alpha=0.15)
        use_integer_ticks(axb, axis="x")
        axb.set_xticks([4, 8, 12])
        if index == 0:
            axb.tick_params(labelbottom=False)
            axb.text(0.98, 0.88, rf"$\varphi_\star={phi:g}$", transform=axb.transAxes, ha="right", va="top", fontsize=7.6)
        else:
            axb.set_xlabel(r"Strip length $L_x$")

        axc = axes_c[index]
        for ensemble, color, face, offset in (
            ("raw_microcanonical", RAW_COLOR, RAW_COLOR, -0.10),
            ("canonical", CANONICAL_COLOR, "white", 0.10),
        ):
            frame = c_key[c_key.ensemble == ensemble].sort_values("Lx")
            for row in frame.itertuples(index=False):
                _sampled_errorbar(
                    axc,
                    x=float(row.Lx) + offset,
                    center=float(row.value_star),
                    minimum=float(row.value_min),
                    maximum=float(row.value_max),
                    color=color,
                    markerfacecolor=face,
                )
            if not frame.empty:
                axc.plot(frame.Lx + offset, frame.value_star, color=color, linewidth=LINE_WIDTH, zorder=2)
        axc.set_ylabel(rf"$\langle \widehat Q_R^{{{key}}}\rangle$")
        axc.set_ylim(*ylim)
        axc.grid(alpha=0.15)
        use_integer_ticks(axc, axis="x")
        axc.set_xticks([4, 8, 12])
        if index == 0:
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel(r"Strip length $L_x$")

    legend_handles = [
        Line2D([0], [0], color=RAW_COLOR, marker="o", lw=LINE_WIDTH, label="raw microcanonical"),
        Line2D([0], [0], color=CANONICAL_COLOR, marker="o", markerfacecolor="white", markeredgecolor=CANONICAL_COLOR, lw=LINE_WIDTH, label="energy-matched canonical"),
    ]
    axes_b[0].legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.015), borderaxespad=0.0, frameon=False, ncol=2, fontsize=7.2)
    axes_b[0].text(0.98, 1.02, r"$L_x=12$: canonical typicality", transform=axes_b[0].transAxes, ha="right", va="bottom", fontsize=7.2)
    axes_c[0].legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.015), borderaxespad=0.0, frameon=False, ncol=2, fontsize=7.2)
    axes_c[0].text(0.98, 0.88, r"whiskers: sampled $\varphi$ range", transform=axes_c[0].transAxes, ha="right", va="top", fontsize=7.2)
    add_panel_label_margin(axes_b[0], "(b)")
    add_panel_label_margin(axes_c[0], "(c)")

    axd = fig.add_subplot(outer[1, 1])
    for row in panel_d.itertuples(index=False):
        _sampled_errorbar(
            axd,
            x=float(row.Lx),
            center=float(row.w_star),
            minimum=float(row.w_min),
            maximum=float(row.w_max),
            color=RAW_COLOR,
        )
    if not panel_d.empty:
        axd.plot(panel_d.Lx, panel_d.w_star, color=RAW_COLOR, linewidth=LINE_WIDTH, zorder=2)
    axd.set_xlabel(r"Strip length $L_x$")
    axd.set_ylabel(r"$w_{L_x}^{\rm raw}$")
    axd.set_ylim(bottom=0.0)
    axd.grid(alpha=0.15)
    use_integer_ticks(axd, axis="x")
    axd.set_xticks(sorted(set(panel_d.Lx.astype(int))))
    axd.text(0.98, 0.92, r"whiskers: sampled $\varphi$ range", transform=axd.transAxes, ha="right", va="top", fontsize=7.2)
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
        "design": "local exception -> representative ensemble bridge -> sampled-deformation robustness -> raw background concentration",
        "source_evidence_directory": str(data),
        "representative_phase": phi,
        "primary_window_prefactor": primary_pref,
        "panels": {
            "a": {"source_file": scatter_path.name, "largest_verified_raw_size": largest},
            "b": {"raw_source": thermal_path.name, "canonical_L12_source": canonical_path.name, "canonical_L12_method": "canonical_typicality"},
            "c": {"raw_and_small_size_canonical_source": thermal_path.name, "L12_phase_range_source": None if phase_check_path is None else phase_check_path.name, "whiskers": "sampled min/max only"},
            "d": {"source_file": concentration_path.name, "whiskers": "sampled min/max only"},
        },
        "raw_12x4_plotted": bool(raw_12_verified),
        "canonical_12x4_plotted": True,
        "dedicated_Delta_panel_removed": True,
        "heatmap_removed": True,
        "no_interpolated_deformation_values": True,
        "expensive_recomputation": False,
    }
    _write_json(figures / "qdm_checkerboard_figure9_provenance.json", manifest)
    return written
