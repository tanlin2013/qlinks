"""Render the frozen 8x4 target-parity comparison without eigensolves or 12x4 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qdm_parity_protocol import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experimental" / "notebooks"), str(ROOT)]

import prx_main_thermal_figure_polish as style  # noqa: E402
from helpers import (  # noqa: E402
    add_panel_label_margin,
    save_prx_figure,
    set_revtex_matplotlib_style,
    write_figure_manifest,
)

PHASES = (0.025, 0.05, 0.075, 0.10)
WIDTHS = (0.10, 0.20, 0.25, 0.50)
COLORS = {name: style.WITNESS_COLORS[name] for name in ("A", "Z")}


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


def render(
    data: Path,
    output: Path,
    *,
    use_tex: bool = False,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
):
    """Feed target-parity evidence into the pre-PR157 Fig. 9 presentation."""
    scatter, table, sources = load_evidence(data)
    set_revtex_matplotlib_style(base_font_size=9.0, prefer_tex=use_tex)
    if use_tex and not plt.rcParams["text.usetex"]:
        raise RuntimeError("--use-tex requires LaTeX; refusing a font substitution")
    primary = table.loc[np.isclose(table.window_half_width, 0.25)].sort_values("phase")
    representative = primary.loc[np.isclose(primary.phase, 0.05)].iloc[0]
    background = scatter.loc[np.isclose(scatter.phase, 0.05)]
    records_b, records_c = [], []
    for name in COLORS:
        for ensemble, source in (("raw_microcanonical", "mc_raw"), ("canonical", "can_raw")):
            values = primary[f"tau_{name}_{source}"]
            records_b.append(
                {
                    "Lx": 8,
                    "witness": name,
                    "ensemble": ensemble,
                    "value": float(representative[f"tau_{name}_{source}"]),
                    "value_min": float(values.min()),
                    "value_max": float(values.max()),
                    "method": "exact_target_parity",
                    "stderr": 0.0,
                }
            )
            records_c.extend(
                {
                    "Lx": 8,
                    "witness": name,
                    "ensemble": ensemble,
                    "phase": float(row.phase),
                    "value": float(getattr(row, f"tau_{name}_{source}")),
                }
                for row in primary.itertuples()
            )
    panel_b, panel_c = pd.DataFrame(records_b), pd.DataFrame(records_c)
    panel_d = pd.DataFrame(
        [
            {
                "Lx": 8,
                "w_star": float(representative.w_raw),
                "w_min": float(primary.w_raw.min()),
                "w_max": float(primary.w_raw.max()),
            }
        ]
    )

    fig = plt.figure(figsize=(style.FULL_WIDTH_IN, style.FIG_HEIGHT_IN))
    outer = style._base._outer_grid(fig)
    gsa = outer[0, 0].subgridspec(2, 1, hspace=0.08)
    axes_a = [fig.add_subplot(gsa[index]) for index in range(2)]
    for index, name in enumerate(COLORS):
        ax = axes_a[index]
        ax.axvspan((8 - 0.25) / 32, (8 + 0.25) / 32, color="0.5", alpha=0.10, zorder=0)
        ax.scatter(
            background.energy_density,
            background[f"Q_{name}"],
            s=9,
            alpha=0.38,
            color=COLORS[name],
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            [0.25],
            [0.0],
            marker="*",
            s=76,
            color=style.STAR_COLOR,
            edgecolors="black",
            linewidths=0.4,
            zorder=8,
        )
        ax.set_ylabel(rf"$\langle \widehat Q_R^{{{name}}}\rangle_n$")
        ax.grid(alpha=0.13)
        if index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Energy density $e=E/(4L_x)$")

    gsb = outer[0, 1].subgridspec(2, 1, hspace=0.10)
    axes_b = [fig.add_subplot(gsb[index]) for index in range(2)]
    style._draw_qdm_panel_b(axes_b, panel_b)
    # Only one nontrivial completed target-parity thermal size remains.
    # Keep the original size axes and range glyph widths without inventing points.
    for ax in axes_b:
        ax.set_xticks([8])
        ax.set_xlim(6, 10)
    style._top_legend(
        axes_b[0],
        handles=[
            *style._qdm_ensemble_box_handles(),
            *style._range_semantics_handles(r"$\varphi_\star$", r"$\varphi$ scan"),
        ],
        ncol=4,
    )

    gsc = outer[1, 0].subgridspec(2, 1, hspace=0.10)
    axes_c = [fig.add_subplot(gsc[index]) for index in range(2)]
    style._draw_qdm_panel_c(axes_c, panel_c, 8)
    for ax in axes_c:
        # Keep the original grid while accommodating the new three-decimal ticks.
        ax.yaxis.labelpad = 0

    axd = fig.add_subplot(outer[1, 1])
    row = panel_d.iloc[0]
    style._range_box(
        axd,
        x=8,
        center=float(row.w_star),
        minimum=float(row.w_min),
        maximum=float(row.w_max),
        color=COLORS["A"],
        width=0.34,
    )
    axd.set_xlabel(r"Strip length $L_x$")
    axd.set_ylabel(r"$w_{L_x}^{\rm raw}$")
    axd.set_xticks([8])
    axd.set_xlim(6, 10)
    axd.set_ylim(0.0, 1.08 * float(row.w_max))
    axd.grid(alpha=0.13)
    axd.legend(
        handles=style._range_semantics_handles(
            r"$\varphi_\star$",
            r"$\varphi$ scan",
            color=COLORS["A"],
        ),
        loc="upper right",
        frameon=False,
        fontsize=6.8,
        handletextpad=0.45,
        borderaxespad=0.3,
    )
    for ax, label in zip(
        [axes_a[0], axes_b[0], axes_c[0], axd], ["(a)", "(b)", "(c)", "(d)"], strict=True
    ):
        if use_tex:
            style._panel_label(ax, label)
        else:
            add_panel_label_margin(ax, label)

    output.mkdir(parents=True, exist_ok=True)
    save_prx_figure(
        fig,
        "qdm_checkerboard_figure9_prx",
        directory=output,
        formats=formats,
        close=True,
    )
    write_figure_manifest(output / "figure_manifest.json")
    background.to_csv(output / "qdm_checkerboard_figure9_panel_a_plot.csv", index=False)
    panel_b.to_csv(output / "qdm_checkerboard_figure9_panel_b_plot.csv", index=False)
    panel_c.to_csv(output / "qdm_checkerboard_figure9_panel_c_plot.csv", index=False)
    panel_d.to_csv(output / "qdm_checkerboard_figure9_panel_d_plot.csv", index=False)
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
        "primary_values": representative.to_dict(),
        "presentation_source": "pre-PR157 prx_main_thermal_figure_polish.py",
        "canvas_inches": [style.FULL_WIDTH_IN, style.FIG_HEIGHT_IN],
        "panel_roles": {
            "a": "8x4 eigenstate A/Z scatter",
            "b": "raw/canonical A/Z means versus Lx (8 only)",
            "c": "raw/canonical A/Z means versus phase",
            "d": "raw stripe width versus Lx (8 only)",
        },
        "size_axes_are_single_points": True,
        "native_tex": bool(plt.rcParams["text.usetex"]),
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
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument("--figure-formats", default="pdf,svg,png")
    args = parser.parse_args()
    formats = tuple(value.strip() for value in args.figure_formats.split(",") if value.strip())
    if not formats or not set(formats).issubset({"pdf", "svg", "png"}):
        parser.error("--figure-formats must select pdf, svg and/or png")
    print(
        json.dumps(render(args.data, args.output, use_tex=args.use_tex, formats=formats), indent=2)
    )
