#!/usr/bin/env python
"""Render current-convention PRX Spin-1 Sec. VI figures.

The pre-migration renderer is preserved in
``render_spin1_xy_sec6_integration_figures_legacy``. This adapter requires
convention-stamped figure data, aliases protocol names only inside the preserved
plotting logic, and updates displayed window labels to the permanent J/2 ladder
convention. Energy-density columns are consumed as stored and are never rescaled
again in the renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import render_spin1_xy_sec6_integration_figures_legacy as _legacy
import spin1_exchange_convention as _convention

_ORIGINAL_READ = _legacy._read
_ORIGINAL_RENDER = _legacy.render
_ORIGINAL_WRITE_AUDIT = _legacy._write_audit

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

PRIMARY_WINDOW_PROTOCOL = _convention.PRIMARY_WINDOW_PROTOCOL
FIXED_WINDOW_PROTOCOL = _convention.FIXED_WINDOW_PROTOCOL
CURRENT_EXCHANGE_CONVENTION = _convention.CURRENT_EXCHANGE_CONVENTION
EXCHANGE_CONVENTION_METADATA_KEY = _convention.EXCHANGE_CONVENTION_METADATA_KEY


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


def _appendix_concentration(data: Path, figures: Path) -> list[str]:
    concentration = _read_current(data / "spin1_xy_kappa0p1_concentration_common_windows.csv")
    raw = concentration[concentration["variant"].astype(str) == "raw"].copy()
    fig, (ax0, ax1) = _legacy.plt.subplots(1, 2, figsize=(_legacy.FULL_WIDTH_IN, 2.65))
    labels = {
        PRIMARY_WINDOW_PROTOCOL: r"$\Delta E=(J/2)L^{1/4}$",
        FIXED_WINDOW_PROTOCOL: r"$\Delta E=J/2$",
    }
    for protocol, frame in raw.groupby("window_protocol", sort=True):
        frame = frame.sort_values("L")
        label = labels.get(str(protocol), str(protocol))
        ax0.plot(
            frame["L"],
            frame["w_L"],
            marker="o",
            markersize=_legacy.MARKER_SIZE,
            linewidth=_legacy.LINE_WIDTH,
            label=label,
        )
        if "window_state_count" in frame.columns:
            ax1.plot(
                frame["L"],
                np.log(frame["window_state_count"].to_numpy(dtype=float))
                / frame["L"].to_numpy(dtype=float),
                marker="o",
                markersize=_legacy.MARKER_SIZE,
                linewidth=_legacy.LINE_WIDTH,
                label=label,
            )
    ax0.set_xlabel(r"System size $L$")
    ax0.set_ylabel(r"$w_L^{\rm raw}$")
    ax0.set_ylim(bottom=0.0)
    ax0.legend(frameon=False)
    ax1.set_xlabel(r"System size $L$")
    ax1.set_ylabel(r"$\log N_{\rm win}/L$")
    for axis in (ax0, ax1):
        _legacy.use_integer_ticks(axis, axis="x")
        axis.set_xticks([8, 10, 12, 14])
    _legacy.add_panel_label(ax0, "(a)")
    _legacy.add_panel_label(ax1, "(b)")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.95, wspace=0.32)
    return _legacy._save(fig, figures, "spin1_xy_appendix_concentration_windows")


_legacy._appendix_concentration = _appendix_concentration


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
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown_path = figures / "spin1_xy_figure6_prx_audit.md"
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- Exchange convention: `{CURRENT_EXCHANGE_CONVENTION}`.\n")
        handle.write("- Window labels: $\\Delta E=(J/2)L^{1/4}$ and $\\Delta E=J/2$.\n")
        handle.write("- Energy density is consumed from mapped figure data without a second rescaling.\n")


_legacy._write_audit = _write_audit


def render(data_dir: Path, *, use_tex: bool, allow_incomplete: bool) -> list[str]:
    """Render only convention-stamped current Sec. VI figure products."""

    return _ORIGINAL_RENDER(data_dir, use_tex=use_tex, allow_incomplete=allow_incomplete)


if __name__ == "__main__":
    _legacy.render = render
    _legacy.main()
