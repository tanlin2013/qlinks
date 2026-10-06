#!/usr/bin/env python
"""Render current-convention PRX Spin-1 Sec. VI figures.

The pre-migration renderer is preserved in
``render_spin1_xy_sec6_integration_figures_legacy``. This adapter requires
convention-stamped figure data, aliases protocol names only inside the preserved
plotting logic, and replaces only the main Fig. 6 artwork with the current
finite-size/deformation handoff design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import render_spin1_xy_sec6_integration_figures_legacy as _legacy
import spin1_exchange_convention as _convention
from helpers import (
    PRX_TEXT_WIDTH,
    save_prx_figure,
    write_figure_manifest,
)
from prx_main_thermal_figure_polish import render_spin1_figure6

_ORIGINAL_READ = _legacy._read
_ORIGINAL_RENDER = _legacy.render
_ORIGINAL_WRITE_AUDIT = _legacy._write_audit
_ORIGINAL_SAVE = _legacy._save

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

PRIMARY_WINDOW_PROTOCOL = _convention.PRIMARY_WINDOW_PROTOCOL
FIXED_WINDOW_PROTOCOL = _convention.FIXED_WINDOW_PROTOCOL
CURRENT_EXCHANGE_CONVENTION = _convention.CURRENT_EXCHANGE_CONVENTION
EXCHANGE_CONVENTION_METADATA_KEY = _convention.EXCHANGE_CONVENTION_METADATA_KEY

# Render at the physical widths of the active REVTeX manuscript.
_legacy.FULL_WIDTH_IN = PRX_TEXT_WIDTH
FULL_WIDTH_IN = PRX_TEXT_WIDTH


def _save_final_size(
    fig: _legacy.plt.Figure,
    directory: Path,
    stem: str,
    *,
    preview: bool = False,
) -> list[str]:
    """Save vector artwork without tight-bbox resizing the physical canvas."""
    paths = save_prx_figure(
        fig,
        stem,
        directory=directory,
        formats=("svg", "pdf"),
        close=False,
    )
    written = [path.name for path in paths]
    if preview:
        path = directory / f"{stem}_preview.png"
        fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0.0)
        written.append(path.name)
    _legacy.plt.close(fig)
    return written


_legacy._save = _save_final_size


def _read_current(path: Path) -> pd.DataFrame:
    frame = _ORIGINAL_READ(path)
    if EXCHANGE_CONVENTION_METADATA_KEY not in frame.columns:
        raise ValueError(f"unstamped Spin-1 figure data: {path}")
    conventions = set(frame[EXCHANGE_CONVENTION_METADATA_KEY].dropna().astype(str))
    if conventions != {CURRENT_EXCHANGE_CONVENTION}:
        raise ValueError(
            f"Spin-1 figure-data convention mismatch in {path}: {sorted(conventions)!r}"
        )
    return frame


def _read(path: Path) -> pd.DataFrame:
    """Return current data, with temporary protocol aliases for legacy plotting code."""

    frame = _read_current(path).copy()
    if "window_protocol" in frame.columns:
        frame["window_protocol"] = frame["window_protocol"].replace(
            {
                PRIMARY_WINDOW_PROTOCOL: "quarter_power_c1",
                FIXED_WINDOW_PROTOCOL: "fixed_width_1",
            }
        )
    return frame


_legacy._read = _read


def _figure6(data: Path, figures: Path, *, allow_incomplete: bool) -> list[str]:
    """Render the main figure from frozen evidence with no inferred values."""

    return render_spin1_figure6(
        data,
        figures,
        allow_incomplete=allow_incomplete,
        read_csv=_read_current,
        save_figure=_legacy._save,
    )


_legacy._figure6 = _figure6


def _appendix_concentration(data: Path, figures: Path) -> list[str]:
    """Render Fig. 11 as a final-width horizontal two-panel figure."""

    source = data / "spin1_xy_kappa0p1_concentration_common_windows.csv"
    concentration = _read_current(source)
    raw = concentration[concentration["variant"].astype(str) == "raw"].copy()
    fig = _legacy.plt.figure(figsize=(PRX_TEXT_WIDTH, 2.72))
    grid = fig.add_gridspec(
        1,
        2,
        left=0.09,
        right=0.985,
        bottom=0.21,
        top=0.91,
        wspace=0.34,
    )
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    labels = {
        PRIMARY_WINDOW_PROTOCOL: r"$\Delta E=(J/2)L^{1/4}$",
        FIXED_WINDOW_PROTOCOL: r"$\Delta E=J/2$",
    }
    for protocol, frame in raw.groupby("window_protocol", sort=True):
        frame = frame.sort_values("L")
        label = labels.get(str(protocol), str(protocol))
        line = ax0.plot(
            frame["L"],
            frame["w_L"],
            marker="o",
            markersize=_legacy.MARKER_SIZE,
            linewidth=_legacy.LINE_WIDTH,
            label=label,
        )[0]
        if "window_state_count" in frame.columns:
            ax1.plot(
                frame["L"],
                np.log(frame["window_state_count"].to_numpy(dtype=float))
                / frame["L"].to_numpy(dtype=float),
                marker="o",
                markersize=_legacy.MARKER_SIZE,
                linewidth=_legacy.LINE_WIDTH,
                color=line.get_color(),
                label=label,
            )
    ax0.set_xlabel(r"System size $L$")
    ax0.set_ylabel(r"$w_L^{\rm raw}$")
    ax0.set_ylim(bottom=0.0)
    ax0.legend(frameon=False, fontsize=7.6, loc="best")
    ax1.set_xlabel(r"System size $L$")
    ax1.set_ylabel(r"$\log N_{\rm win}/L$")
    ax1.legend(frameon=False, fontsize=7.6, loc="best")
    for axis in (ax0, ax1):
        _legacy.use_integer_ticks(axis, axis="x")
        axis.set_xticks(sorted(set(raw["L"].astype(int))))
        axis.grid(alpha=0.18)
    _legacy.add_panel_label_margin(ax0, "(a)")
    _legacy.add_panel_label_margin(ax1, "(b)")
    return _legacy._save(fig, figures, "spin1_xy_appendix_concentration_windows")


_legacy._appendix_concentration = _appendix_concentration


def _appendix_beta0(data: Path, figures: Path) -> list[str]:
    """Render Fig. 10 as a final-width horizontal two-panel figure."""

    frame = _read_current(data / "spin1_xy_appendix_beta0_bridges_data.csv")
    fig = _legacy.plt.figure(figsize=(PRX_TEXT_WIDTH, 2.72))
    grid = fig.add_gridspec(
        1,
        2,
        left=0.09,
        right=0.985,
        bottom=0.21,
        top=0.91,
        wspace=0.34,
    )
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    bridge_labels = {
        "mc_to_beta0_resolved": (
            r"$\rho_{\rm mc}^{(M,k)}\leftrightarrow\rho_{\beta=0}^{(M,k)}$"
        ),
        "beta0_resolved_to_fixedM": (
            r"$\rho_{\beta=0}^{(M,k)}\leftrightarrow\rho_{\beta=0}^{M}$"
        ),
    }
    for bridge, group in frame.groupby("bridge", sort=True):
        group = group.sort_values("L")
        ax0.plot(
            group["L"],
            group["trace_distance"],
            marker="o",
            markersize=_legacy.MARKER_SIZE,
            linewidth=_legacy.LINE_WIDTH,
            label=bridge_labels.get(str(bridge), str(bridge)),
        )
    ax0.set_yscale("log")
    ax0.set_xlabel(r"System size $L$")
    ax0.set_ylabel("Two-site RDM distance")
    ax0.legend(frameon=False, fontsize=7.4, loc="best")
    ax0.grid(alpha=0.18, which="both")

    first = frame[frame["bridge"].astype(str) == "mc_to_beta0_resolved"].sort_values("L")
    for key, spec in _legacy.WITNESS_SPECS.items():
        column = f"abs_delta_tau_{key}"
        if column in first.columns:
            ax1.plot(
                first["L"],
                first[column],
                marker=spec["marker"],
                markersize=_legacy.MARKER_SIZE,
                linewidth=_legacy.LINE_WIDTH,
                label=spec["label"],
            )
    ax1.set_xlabel(r"System size $L$")
    ax1.set_ylabel(r"$|\Delta\tau_\alpha|$")
    ax1.legend(frameon=False, fontsize=8.0, ncol=3, loc="best")
    ax1.grid(alpha=0.18)
    for axis in (ax0, ax1):
        _legacy.use_integer_ticks(axis, axis="x")
        axis.set_xticks(sorted(set(frame["L"].astype(int))))
    _legacy.add_panel_label_margin(ax0, "(a)")
    _legacy.add_panel_label_margin(ax1, "(b)")
    return _legacy._save(fig, figures, "spin1_xy_appendix_beta0_bridges")


_legacy._appendix_beta0 = _appendix_beta0


def _write_audit(data: Path, figures: Path, written: list[str]) -> None:
    _ORIGINAL_WRITE_AUDIT(data, figures, written)
    json_path = figures / "spin1_xy_figure6_prx_audit.json"
    audit = json.loads(json_path.read_text(encoding="utf-8"))
    audit[EXCHANGE_CONVENTION_METADATA_KEY] = CURRENT_EXCHANGE_CONVENTION
    audit["primary_window_protocol"] = PRIMARY_WINDOW_PROTOCOL
    audit["primary_window_label"] = "Delta E=(J/2)L^(1/4)"
    audit["fixed_window_protocol"] = FIXED_WINDOW_PROTOCOL
    audit["fixed_window_label"] = "Delta E=J/2"
    audit["energy_density_rescaled_in_renderer"] = False
    audit["main_figure_panel_logic"] = (
        "local exceptionalness -> representative finite-size trend -> "
        "sampled-deformation min/max robustness -> background concentration "
        "min/max robustness"
    )
    audit["deformation_ranges_are_statistical_errors"] = False
    audit["deformation_ranges_use_interpolation"] = False
    audit["fig10_layout"] = "horizontal_1x2_full_text_width"
    audit["fig11_layout"] = "horizontal_1x2_full_text_width"
    json_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    json_path.write_text(json_text, encoding="utf-8")

    markdown_path = figures / "spin1_xy_figure6_prx_audit.md"
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- Exchange convention: `{CURRENT_EXCHANGE_CONVENTION}`.\n")
        handle.write("- Window labels: $\\Delta E=(J/2)L^{1/4}$ and $\\Delta E=J/2$.\n")
        handle.write(
            "- Energy density is consumed from mapped figure data without a second rescaling.\n"
        )
        handle.write("- Fig. 6(c,d) whiskers are sampled deformation min/max ranges, not errors.\n")
        handle.write(
            "- Fig. 6(b,c) use unconnected representative bars; "
            "Fig. 6(d) uses a dashed guide.\n"
        )
        handle.write("- No interpolation or L=14 deformation whisker is introduced.\n")
        handle.write("- Figs. 10 and 11 are final-width horizontal 1x2 figures.\n")


_legacy._write_audit = _write_audit


def render(data_dir: Path, *, use_tex: bool, allow_incomplete: bool) -> list[str]:
    """Render only convention-stamped current Sec. VI figure products."""

    written = _ORIGINAL_RENDER(
        data_dir,
        use_tex=use_tex,
        allow_incomplete=allow_incomplete,
    )
    write_figure_manifest(Path(data_dir) / "figure_manifest.json")
    return written


if __name__ == "__main__":
    _legacy.render = render
    _legacy.main()
