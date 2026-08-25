#!/usr/bin/env python
"""Render PRX Spin-1 Sec. VI integration and Appendix-D support figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (candidate / "qlinks").is_dir():
        ROOT = candidate
        break
else:
    ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "experimental" / "notebooks"
for path in (NOTEBOOKS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers import add_panel_label, set_revtex_matplotlib_style, use_integer_ticks  # noqa: E402

FULL_WIDTH_IN = 7.05
BASE_FONT_SIZE = 9.0
LINE_WIDTH = 1.0
MARKER_SIZE = 4.5
REPRESENTATIVE_KAPPA_OVER_J = 0.10
WITNESS_SPECS = {
    "A": {"label": r"$Q_R^A$", "marker": "o", "target": 1.0 / 9.0},
    "Z": {"label": r"$Q_R^Z$", "marker": "s", "target": 2.0 / 9.0},
    "Y": {"label": r"$Q_R^Y$", "marker": "^", "target": 1.0 / 3.0},
}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty figure-data table: {path}")
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(
    fig: plt.Figure, directory: Path, stem: str, *, preview: bool = False
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for suffix in ("svg", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path.name)
    if preview:
        path = directory / f"{stem}_preview.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(path.name)
    plt.close(fig)
    return written


def _family_band(data: Path) -> pd.DataFrame:
    path = data / "spin1_xy_figure6_panel_d_family_band.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = _read(path)
    missing = {"L", "w_min", "w_max"}.difference(frame.columns)
    if missing:
        raise ValueError(f"panel-d family band is missing columns: {sorted(missing)}")
    return frame.sort_values("L")


def _figure6(data: Path, figures: Path, *, allow_incomplete: bool) -> list[str]:
    scatter = _read(data / "spin1_xy_figure6_panel_a_scatter.csv")
    sequence = _read(data / "spin1_xy_figure6_panel_b_witness_sequence.csv")
    deformation = _read(data / "spin1_xy_figure6_panel_c_deformation.csv")
    concentration = _read(data / "spin1_xy_kappa0p1_concentration_common_windows.csv")
    band = _family_band(data)
    primary = concentration[
        (concentration["variant"].astype(str) == "raw")
        & (concentration["window_protocol"].astype(str) == "quarter_power_c1")
        & np.isclose(
            concentration["kappa_over_J"].to_numpy(dtype=float),
            REPRESENTATIVE_KAPPA_OVER_J,
        )
    ].sort_values("L")
    if set(primary["L"].astype(int)) != {8, 10, 12, 14}:
        raise ValueError("Fig. 6(d) requires primary-window L=8,10,12,14")
    if band.empty and not allow_incomplete:
        raise ValueError(
            "Fig. 6(d) family band is missing; --allow-incomplete is preview-only"
        )

    fig = plt.figure(figsize=(FULL_WIDTH_IN, 5.75))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.095,
        top=0.975,
        wspace=0.34,
        hspace=0.36,
    )

    gs_a = outer[0, 0].subgridspec(3, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gs_a[index]) for index in range(3)]
    tower = scatter["is_tower_state"].fillna(False).astype(bool).to_numpy()
    background = scatter[~tower]
    sequence_l12 = sequence[sequence["L"].astype(int) == 12]
    if sequence_l12.empty:
        raise ValueError("panel (a) needs the L=12 primary-window means")
    half = float(sequence_l12.iloc[0]["window_energy_density_half_width"])
    for index, (key, spec) in enumerate(WITNESS_SPECS.items()):
        ax = axes_a[index]
        ax.axvspan(-half, half, alpha=0.10, zorder=0)
        ax.scatter(
            background["energy_density"],
            background[f"Q_{key}"],
            s=8,
            alpha=0.38,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            [0.0],
            [0.0],
            marker="*",
            s=72,
            edgecolors="black",
            linewidths=0.45,
            zorder=8,
        )
        mean_row = sequence_l12[sequence_l12["witness"] == key]
        if len(mean_row) != 1:
            raise ValueError(f"panel (a) has no unique L=12 mean for witness {key}")
        ax.axhline(float(mean_row.iloc[0]["tau_mc_raw"]), ls=":", lw=0.9)
        ax.set_ylabel(spec["label"])
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/L$")
    add_panel_label(axes_a[0], "(a)")

    ax_b = fig.add_subplot(outer[0, 1])
    for key, spec in WITNESS_SPECS.items():
        frame = sequence[sequence["witness"] == key].sort_values("L")
        line = ax_b.plot(
            frame["L"],
            frame["tau_mc_raw"],
            marker=spec["marker"],
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=spec["label"],
        )[0]
        ax_b.axhline(
            spec["target"], ls="--", lw=0.75, color=line.get_color(), alpha=0.55
        )
    ax_b.set_xlabel(r"System size $L$")
    ax_b.set_ylabel("Raw microcanonical activity")
    use_integer_ticks(ax_b, axis="x")
    ax_b.set_xticks([8, 10, 12, 14])
    ax_b.legend(loc="best", frameon=False)
    add_panel_label(ax_b, "(b)")

    ax_c = fig.add_subplot(outer[1, 0])
    for key, spec in WITNESS_SPECS.items():
        frame = deformation[deformation["witness"] == key].sort_values("kappa_over_J")
        ax_c.plot(
            frame["kappa_over_J"],
            frame["tau_mc_raw"],
            marker=spec["marker"],
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=spec["label"],
        )
    ax_c.axhline(0.0, lw=0.75, alpha=0.55)
    ax_c.axvline(REPRESENTATIVE_KAPPA_OVER_J, ls=":", lw=0.9, alpha=0.7)
    ax_c.set_xlabel(r"Compatible deformation $\kappa/J$")
    ax_c.set_ylabel("Raw microcanonical activity")
    ax_c.legend(loc="best", frameon=False)
    add_panel_label(ax_c, "(c)")

    ax_d = fig.add_subplot(outer[1, 1])
    if not band.empty:
        ax_d.fill_between(
            band["L"].to_numpy(dtype=float),
            band["w_min"].to_numpy(dtype=float),
            band["w_max"].to_numpy(dtype=float),
            alpha=0.16,
            linewidth=0,
            label="sampled positive-$\\kappa$ range",
        )
    ax_d.plot(
        primary["L"],
        primary["w_L"],
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label=r"$\kappa/J=0.1$",
    )
    ax_d.set_xlabel(r"System size $L$")
    ax_d.set_ylabel(r"Complete two-site width $w_L$")
    use_integer_ticks(ax_d, axis="x")
    ax_d.set_xticks([8, 10, 12, 14])
    ax_d.set_ylim(bottom=0.0)
    ax_d.legend(loc="best", frameon=False)
    add_panel_label(ax_d, "(d)")
    return _save(fig, figures, "spin1_xy_figure6_prx", preview=True)


def _appendix_concentration(data: Path, figures: Path) -> list[str]:
    concentration = _read(data / "spin1_xy_kappa0p1_concentration_common_windows.csv")
    raw = concentration[concentration["variant"].astype(str) == "raw"].copy()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.65))
    labels = {
        "quarter_power_c1": r"$\Delta E=L^{1/4}$",
        "fixed_width_1": r"$\Delta E=1$",
    }
    for protocol, frame in raw.groupby("window_protocol", sort=True):
        frame = frame.sort_values("L")
        label = labels.get(str(protocol), str(protocol))
        ax0.plot(
            frame["L"],
            frame["w_L"],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=label,
        )
        if "window_state_count" in frame.columns:
            ax1.plot(
                frame["L"],
                np.log(frame["window_state_count"].to_numpy(dtype=float))
                / frame["L"].to_numpy(dtype=float),
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=label,
            )
    ax0.set_xlabel(r"System size $L$")
    ax0.set_ylabel(r"$w_L^{\rm raw}$")
    ax0.set_ylim(bottom=0.0)
    ax0.legend(frameon=False)
    ax1.set_xlabel(r"System size $L$")
    ax1.set_ylabel(r"$\log N_{\rm win}/L$")
    for axis in (ax0, ax1):
        use_integer_ticks(axis, axis="x")
        axis.set_xticks([8, 10, 12, 14])
    add_panel_label(ax0, "(a)")
    add_panel_label(ax1, "(b)")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.95, wspace=0.32)
    return _save(fig, figures, "spin1_xy_appendix_concentration_windows")


def _appendix_beta0(data: Path, figures: Path) -> list[str]:
    frame = _read(data / "spin1_xy_appendix_beta0_bridges_data.csv")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.65))
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
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=bridge_labels.get(str(bridge), str(bridge)),
        )
    ax0.set_yscale("log")
    ax0.set_xlabel(r"System size $L$")
    ax0.set_ylabel("Two-site RDM trace distance")
    ax0.legend(frameon=False, fontsize=8.0)
    first = frame[frame["bridge"] == "mc_to_beta0_resolved"].sort_values("L")
    for key, spec in WITNESS_SPECS.items():
        column = f"abs_delta_tau_{key}"
        if column in first.columns:
            ax1.plot(
                first["L"],
                first[column],
                marker=spec["marker"],
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=spec["label"],
            )
    ax1.set_xlabel(r"System size $L$")
    ax1.set_ylabel(r"$|\Delta\tau_\alpha|$")
    ax1.legend(frameon=False)
    for axis in (ax0, ax1):
        use_integer_ticks(axis, axis="x")
        axis.set_xticks(sorted(set(frame["L"].astype(int))))
    add_panel_label(ax0, "(a)")
    add_panel_label(ax1, "(b)")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.95, wspace=0.34)
    return _save(fig, figures, "spin1_xy_appendix_beta0_bridges")


def _centered_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.01, values[0] + 0.01])
    midpoint = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            [values[0] - (midpoint[0] - values[0])],
            midpoint,
            [values[-1] + (values[-1] - midpoint[-1])],
        )
    )


def _appendix_obstruction(data: Path, figures: Path) -> list[str]:
    frame = _read(data / "spin1_xy_appendix_complex_t2_obstruction_data.csv")
    pivot = frame.pivot(
        index="imag_t2_over_J",
        columns="real_t2_over_J",
        values="normalized_tower_residual",
    )
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    residual = pivot.to_numpy(dtype=float)
    finite_positive = residual[np.isfinite(residual) & (residual > 0.0)]
    floor = (
        max(1.0e-16, float(np.min(finite_positive)) * 0.1)
        if finite_positive.size
        else 1.0e-16
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    mesh = ax.pcolormesh(
        _centered_edges(x),
        _centered_edges(y),
        np.log10(np.maximum(residual, floor)),
        shading="flat",
    )
    ax.axvline(0.0, ls="--", lw=1.0)
    ax.scatter([0.0], [0.10], marker="*", s=62, edgecolors="black", linewidths=0.45)
    ax.set_xlabel(r"$\operatorname{Re}(t_2/J)$")
    ax.set_ylabel(r"$\operatorname{Im}(t_2/J)$")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    colorbar.set_label(r"$\log_{10}$ normalized tower residual")
    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.18, top=0.96)
    return _save(fig, figures, "spin1_xy_appendix_complex_t2_obstruction")


def _write_audit(data: Path, figures: Path, written: list[str]) -> None:
    source_names = (
        "spin1_xy_figure6_panel_a_scatter.csv",
        "spin1_xy_figure6_panel_b_witness_sequence.csv",
        "spin1_xy_figure6_panel_c_deformation.csv",
        "spin1_xy_kappa0p1_concentration_common_windows.csv",
        "spin1_xy_figure6_panel_d_family_band.csv",
        "spin1_xy_appendix_beta0_bridges_data.csv",
        "spin1_xy_appendix_complex_t2_obstruction_data.csv",
    )
    sources = {
        name: _sha256(data / name) for name in source_names if (data / name).is_file()
    }
    audit: dict[str, Any] = {
        "physical_width_inches": FULL_WIDTH_IN,
        "base_font_size_pt": BASE_FONT_SIZE,
        "line_width_pt": LINE_WIDTH,
        "marker_size_pt": MARKER_SIZE,
        "panel_layout": "2x2; panel (a) contains three stacked witness axes",
        "written_files": written,
        "source_csv_sha256": sources,
        "power_law_fit_displayed": False,
    }
    (figures / "spin1_xy_figure6_prx_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (figures / "spin1_xy_figure6_prx_audit.md").write_text(
        "\n".join(
            (
                "# Spin-1 Sec. VI figure audit",
                "",
                f"- Physical width: {FULL_WIDTH_IN:.2f} in",
                f"- Base font: {BASE_FONT_SIZE:.1f} pt",
                f"- Line width: {LINE_WIDTH:.1f} pt",
                f"- Marker size: {MARKER_SIZE:.1f} pt",
                "- Main concentration panel: linear y-axis; no fitted exponent.",
                "- Source CSV hashes are recorded in the JSON audit.",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def render(data_dir: Path, *, use_tex: bool, allow_incomplete: bool) -> list[str]:
    data = Path(data_dir).resolve(strict=False)
    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    set_revtex_matplotlib_style(base_font_size=BASE_FONT_SIZE, prefer_tex=use_tex)
    written: list[str] = []
    written.extend(_figure6(data, figures, allow_incomplete=allow_incomplete))
    written.extend(_appendix_concentration(data, figures))
    written.extend(_appendix_beta0(data, figures))
    written.extend(_appendix_obstruction(data, figures))
    _write_audit(data, figures, written)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a preview without the positive-kappa family concentration band.",
    )
    args = parser.parse_args()
    written = render(
        args.data_dir, use_tex=args.use_tex, allow_incomplete=args.allow_incomplete
    )
    print(json.dumps({"written": written}, indent=2), flush=True)


if __name__ == "__main__":
    main()
