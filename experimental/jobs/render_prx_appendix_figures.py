#!/usr/bin/env python
"""Render PRX Appendix-D/E numerical figures from validated cached tables.

This module is intentionally render-only. It consumes existing CSV products and
never calls an eigensolver, reconstruction job, or evidence-generation notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

NOTEBOOKS = ROOT / "experimental" / "notebooks"
for path in (NOTEBOOKS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers import (  # noqa: E402
    PRX_COLUMN_WIDTH,
    PRX_TEXT_WIDTH,
    add_panel_label,
    add_panel_label_margin,
    save_prx_figure,
    set_revtex_matplotlib_style,
    use_integer_ticks,
)
from spin1_exchange_convention import (  # noqa: E402
    CURRENT_EXCHANGE_CONVENTION,
    EXCHANGE_CONVENTION_METADATA_KEY,
    FIXED_WINDOW_PROTOCOL,
    PRIMARY_WINDOW_PROTOCOL,
)

BASE_FONT_SIZE = 9.0
LINE_WIDTH = 1.0
MARKER_SIZE = 4.5
NUMERICAL_TOLERANCE = 1.0e-10

SPIN1_SOURCE_NAMES = (
    "spin1_xy_appendix_beta0_bridges_data.csv",
    "spin1_xy_kappa0p1_concentration_common_windows.csv",
)
QDM_SOURCE_NAMES = (
    "qdm_4x4_minimum_annihilator_radius.csv",
    "qdm_4N_by_4_exact_sequence.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty cached figure table: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], *, source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _require_current_spin1_convention(frame: pd.DataFrame, *, source: Path) -> None:
    if EXCHANGE_CONVENTION_METADATA_KEY not in frame.columns:
        raise ValueError(f"unstamped Spin-1 cached figure table: {source}")
    conventions = set(frame[EXCHANGE_CONVENTION_METADATA_KEY].dropna().astype(str))
    if conventions != {CURRENT_EXCHANGE_CONVENTION}:
        raise ValueError(f"Spin-1 convention mismatch in {source}: {sorted(conventions)!r}")


def _configure_style(*, use_tex: bool) -> None:
    set_revtex_matplotlib_style(base_font_size=BASE_FONT_SIZE, prefer_tex=use_tex)
    if use_tex and not bool(plt.rcParams.get("text.usetex", False)):
        raise RuntimeError(
            "--use-tex requires a working LaTeX executable; refusing mathtext fallback"
        )


def _spin1_bridge_figure(frame: pd.DataFrame) -> plt.Figure:
    _require_columns(frame, {"L", "bridge", "trace_distance"}, source=Path("bridge table"))
    fig = plt.figure(figsize=(PRX_COLUMN_WIDTH, 4.45))
    grid = fig.add_gridspec(
        2,
        1,
        left=0.22,
        right=0.975,
        bottom=0.11,
        top=0.98,
        hspace=0.36,
    )
    ax0 = fig.add_subplot(grid[0])
    ax1 = fig.add_subplot(grid[1])

    bridge_labels = {
        "mc_to_beta0_resolved": r"MC $\leftrightarrow$ resolved $\beta=0$",
        "beta0_resolved_to_fixedM": r"resolved $\beta=0\leftrightarrow$ fixed-$M$",
    }
    for bridge, group in frame.groupby("bridge", sort=True):
        group = group.sort_values("L")
        ax0.plot(
            group["L"],
            group["trace_distance"],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=bridge_labels.get(str(bridge), str(bridge)),
        )
    ax0.set_yscale("log")
    ax0.set_ylabel("Two-site trace distance")
    ax0.legend(loc="best", fontsize=7.4)
    ax0.tick_params(labelbottom=False)
    ax0.grid(alpha=0.18)
    add_panel_label_margin(ax0, "(a)")

    first = frame[frame["bridge"].astype(str) == "mc_to_beta0_resolved"].sort_values("L")
    specs = (("A", r"$Q_R^A$", "o"), ("Z", r"$Q_R^Z$", "s"), ("Y", r"$Q_R^Y$", "^"))
    for key, label, marker in specs:
        column = f"abs_delta_tau_{key}"
        if column not in first.columns:
            continue
        ax1.plot(
            first["L"],
            first[column],
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=label,
        )
    ax1.set_xlabel(r"System size $L$")
    ax1.set_ylabel(r"$|\Delta\tau_\alpha|$")
    ax1.legend(loc="best", fontsize=8.0)
    ax1.grid(alpha=0.18)
    add_panel_label_margin(ax1, "(b)")

    ticks = sorted(set(frame["L"].astype(int)))
    for axis in (ax0, ax1):
        use_integer_ticks(axis, axis="x")
        axis.set_xticks(ticks)
    return fig


def _spin1_concentration_figure(frame: pd.DataFrame) -> plt.Figure:
    _require_columns(
        frame,
        {"L", "variant", "window_protocol", "w_L"},
        source=Path("concentration table"),
    )
    raw = frame[frame["variant"].astype(str) == "raw"].copy()
    if raw.empty:
        raise ValueError("Spin-1 concentration table has no raw rows")

    fig = plt.figure(figsize=(PRX_COLUMN_WIDTH, 4.35))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(3.0, 2.0),
        left=0.22,
        right=0.975,
        bottom=0.11,
        top=0.98,
        hspace=0.36,
    )
    ax0 = fig.add_subplot(grid[0])
    ax1 = fig.add_subplot(grid[1])

    labels = {
        PRIMARY_WINDOW_PROTOCOL: r"$\Delta E=(J/2)L^{1/4}$",
        FIXED_WINDOW_PROTOCOL: r"$\Delta E=J/2$",
    }
    for protocol, group in raw.groupby("window_protocol", sort=True):
        group = group.sort_values("L")
        label = labels.get(str(protocol), str(protocol))
        line = ax0.plot(
            group["L"],
            group["w_L"],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=label,
        )[0]
        if "window_state_count" in group.columns:
            ax1.plot(
                group["L"],
                np.log(group["window_state_count"].to_numpy(dtype=float))
                / group["L"].to_numpy(dtype=float),
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                color=line.get_color(),
            )

    ax0.set_ylabel(r"$w_L^{\rm raw}$")
    ax0.set_ylim(bottom=0.0)
    ax0.legend(loc="best", fontsize=7.6)
    ax0.tick_params(labelbottom=False)
    ax0.grid(alpha=0.18)
    add_panel_label(ax0, "(a)")

    ax1.set_xlabel(r"System size $L$")
    ax1.set_ylabel(r"$\log N_{\rm win}/L$")
    ax1.grid(alpha=0.18)
    add_panel_label(ax1, "(b)")

    ticks = sorted(set(raw["L"].astype(int)))
    for axis in (ax0, ax1):
        use_integer_ticks(axis, axis="x")
        axis.set_xticks(ticks)
    return fig


def render_spin1_appendix(
    data_dir: Path,
    *,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> dict[str, object]:
    data = Path(data_dir).resolve(strict=False)
    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    bridge_path = data / SPIN1_SOURCE_NAMES[0]
    concentration_path = data / SPIN1_SOURCE_NAMES[1]
    bridge = _read(bridge_path)
    concentration = _read(concentration_path)
    _require_current_spin1_convention(bridge, source=bridge_path)
    _require_current_spin1_convention(concentration, source=concentration_path)

    written = []
    fig10 = _spin1_bridge_figure(bridge)
    written.extend(
        save_prx_figure(
            fig10,
            "spin1_xy_appendix_beta0_bridges",
            directory=figures,
            formats=formats,
            close=True,
        )
    )
    fig11 = _spin1_concentration_figure(concentration)
    written.extend(
        save_prx_figure(
            fig11,
            "spin1_xy_appendix_concentration_windows",
            directory=figures,
            formats=formats,
            close=True,
        )
    )
    return {
        "data_dir": str(data),
        "sources": {path.name: _sha256(path) for path in (bridge_path, concentration_path)},
        "written": [str(path) for path in written],
    }


def _qdm_radius_figure(frame: pd.DataFrame) -> plt.Figure:
    _require_columns(
        frame,
        {"state", "radius", "minimum_residual"},
        source=Path("QDM radius table"),
    )
    fig, ax = plt.subplots(figsize=(PRX_COLUMN_WIDTH, 2.62))
    label_map = {
        "compact record 0": "compact",
        "collective record 8": "collective",
    }
    for state, group in frame.groupby("state", sort=True):
        group = group.sort_values("radius")
        ax.semilogy(
            group["radius"],
            np.maximum(group["minimum_residual"].to_numpy(dtype=float), 1.0e-16),
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=label_map.get(str(state), str(state)),
        )
    ax.axhline(
        NUMERICAL_TOLERANCE,
        linestyle="--",
        linewidth=0.8,
        label=r"tolerance $10^{-10}$",
    )
    ax.set_xlabel("Allowed Chebyshev radius")
    ax.set_ylabel("Minimum annihilation residual")
    ticks = sorted(set(frame["radius"].astype(int)))
    ax.set_xticks(ticks)
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=7.6)
    fig.subplots_adjust(left=0.22, right=0.975, bottom=0.20, top=0.96)
    return fig


def _qdm_compatibility_figure(frame: pd.DataFrame) -> plt.Figure:
    _require_columns(
        frame,
        {"repeats", "kinetic_constraint_rank", "kinetic_compatible_dimension"},
        source=Path("QDM repeated-strip table"),
    )
    ordered = frame.sort_values("repeats").copy()
    ordered["kinetic_parameter_count"] = ordered["kinetic_constraint_rank"].astype(float) + ordered[
        "kinetic_compatible_dimension"
    ].astype(float)

    fig, ax = plt.subplots(figsize=(PRX_COLUMN_WIDTH, 2.62))
    ax.plot(
        ordered["repeats"],
        ordered["kinetic_parameter_count"],
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="local kinetic parameters",
    )
    ax.plot(
        ordered["repeats"],
        ordered["kinetic_constraint_rank"],
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="compatibility constraints",
    )
    per_cell = ordered["kinetic_constraint_rank"].to_numpy(dtype=float) / ordered[
        "repeats"
    ].to_numpy(dtype=float)
    if np.allclose(per_cell, per_cell[0], rtol=0.0, atol=1.0e-12):
        annotation = "$" + f"{per_cell[0]:g}" + r"$ constraints/cell"
        ax.text(
            0.97,
            0.08,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
        )
    ax.set_xlabel(r"Repeated four-column cells $N$")
    ax.set_ylabel("Coefficient-space dimension")
    use_integer_ticks(ax, axis="both")
    ax.set_xticks(ordered["repeats"].astype(int))
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=7.6)
    fig.subplots_adjust(left=0.22, right=0.975, bottom=0.20, top=0.96)
    return fig


def _qdm_locality_scaling_figure(
    radius: pd.DataFrame,
    sequence: pd.DataFrame,
) -> plt.Figure:
    """Compose the two numerical Appendix-E certificates as final Fig. 15."""

    _require_columns(
        radius,
        {"state", "radius", "minimum_residual"},
        source=Path("QDM radius table"),
    )
    _require_columns(
        sequence,
        {"repeats", "kinetic_constraint_rank", "kinetic_compatible_dimension"},
        source=Path("QDM repeated-strip table"),
    )

    fig = plt.figure(figsize=(PRX_TEXT_WIDTH, 3.05))
    grid = fig.add_gridspec(
        1,
        2,
        left=0.09,
        right=0.985,
        bottom=0.19,
        top=0.90,
        wspace=0.34,
    )
    ax0 = fig.add_subplot(grid[0])
    ax1 = fig.add_subplot(grid[1])

    label_map = {
        "compact record 0": "compact",
        "collective record 8": "collective",
    }
    for state, group in radius.groupby("state", sort=True):
        group = group.sort_values("radius")
        ax0.semilogy(
            group["radius"],
            np.maximum(group["minimum_residual"].to_numpy(dtype=float), 1.0e-16),
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=label_map.get(str(state), str(state)),
        )
    ax0.axhline(
        NUMERICAL_TOLERANCE,
        linestyle="--",
        linewidth=0.8,
        label=r"tolerance $10^{-10}$",
    )
    ax0.set_xlabel("Allowed Chebyshev radius")
    ax0.set_ylabel("Minimum annihilation residual")
    ax0.set_xticks(sorted(set(radius["radius"].astype(int))))
    ax0.grid(alpha=0.18)
    ax0.legend(loc="best", fontsize=7.6)
    add_panel_label_margin(ax0, "(a)")

    ordered = sequence.sort_values("repeats").copy()
    ordered["kinetic_parameter_count"] = ordered["kinetic_constraint_rank"].astype(
        float
    ) + ordered["kinetic_compatible_dimension"].astype(float)
    ax1.plot(
        ordered["repeats"],
        ordered["kinetic_parameter_count"],
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="local kinetic parameters",
    )
    ax1.plot(
        ordered["repeats"],
        ordered["kinetic_constraint_rank"],
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="compatibility constraints",
    )
    per_cell = ordered["kinetic_constraint_rank"].to_numpy(dtype=float) / ordered[
        "repeats"
    ].to_numpy(dtype=float)
    if np.allclose(per_cell, per_cell[0], rtol=0.0, atol=1.0e-12):
        ax1.text(
            0.97,
            0.08,
            "$" + f"{per_cell[0]:g}" + r"$ constraints/cell",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
        )
    ax1.set_xlabel(r"Repeated four-column cells $N$")
    ax1.set_ylabel("Coefficient-space dimension")
    use_integer_ticks(ax1, axis="both")
    ax1.set_xticks(ordered["repeats"].astype(int))
    ax1.set_ylim(bottom=0.0)
    ax1.grid(alpha=0.18)
    ax1.legend(loc="upper left", fontsize=7.6)
    add_panel_label_margin(ax1, "(b)")
    return fig


def render_qdm_appendix(
    data_dir: Path,
    *,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> dict[str, object]:
    data = Path(data_dir).resolve(strict=False)
    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    radius_path = data / QDM_SOURCE_NAMES[0]
    sequence_path = data / QDM_SOURCE_NAMES[1]
    radius = _read(radius_path)
    sequence = _read(sequence_path)

    written = []
    fig14b = _qdm_radius_figure(radius)
    written.extend(
        save_prx_figure(
            fig14b,
            "qdm_4x4_annihilator_radius",
            directory=figures,
            formats=formats,
            close=True,
        )
    )
    fig14c = _qdm_compatibility_figure(sequence)
    written.extend(
        save_prx_figure(
            fig14c,
            "qdm_strip_compatibility_scaling",
            directory=figures,
            formats=formats,
            close=True,
        )
    )
    fig15 = _qdm_locality_scaling_figure(radius, sequence)
    written.extend(
        save_prx_figure(
            fig15,
            "qdm_appendix_locality_scaling_certificates",
            directory=figures,
            formats=formats,
            close=True,
        )
    )
    return {
        "data_dir": str(data),
        "sources": {path.name: _sha256(path) for path in (radius_path, sequence_path)},
        "written": [str(path) for path in written],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin1-data-dir", type=Path)
    parser.add_argument("--qdm-data-dir", type=Path)
    parser.add_argument("--figure-formats", default="pdf,svg")
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.spin1_data_dir is None and args.qdm_data_dir is None:
        parser.error("at least one of --spin1-data-dir or --qdm-data-dir is required")

    formats = tuple(part.strip() for part in args.figure_formats.split(",") if part.strip())
    _configure_style(use_tex=args.use_tex)

    payload: dict[str, object] = {
        "schema_version": 1,
        "render_only": True,
        "numerical_recomputation": False,
        "base_font_size_pt": BASE_FONT_SIZE,
        "text_usetex": bool(plt.rcParams.get("text.usetex", False)),
        "physical_column_width_in": PRX_COLUMN_WIDTH,
    }
    if args.spin1_data_dir is not None:
        payload["spin1"] = render_spin1_appendix(args.spin1_data_dir, formats=formats)
    if args.qdm_data_dir is not None:
        payload["qdm"] = render_qdm_appendix(args.qdm_data_dir, formats=formats)

    manifest = args.manifest
    if manifest is None:
        base = args.spin1_data_dir if args.spin1_data_dir is not None else args.qdm_data_dir
        assert base is not None
        manifest = Path(base) / "prx_appendix_render_manifest.json"
    manifest = manifest.resolve(strict=False)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
