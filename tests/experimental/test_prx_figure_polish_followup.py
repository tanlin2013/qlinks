"""Regression contracts for the Oct. 6-7 PRX figure-polish follow-up."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

POLISH = JOBS / "prx_main_thermal_figure_polish.py"
SPIN1_RENDERER = JOBS / "render_spin1_xy_sec6_integration_figures.py"
QDM_RENDERER = JOBS / "render_square_qdm_draft_figures.py"
APPENDIX_RENDERER = JOBS / "render_prx_appendix_figures.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qdm_canonical_l12_ignores_nonfinite_length_rows(tmp_path: Path) -> None:
    module = _load(POLISH, "prx_main_thermal_figure_polish_nullable_test")
    pd.DataFrame(
        [
            {"Lx": None, "phase": 0.05, "tau_A_target": 0.1, "tau_Z_target": 0.2},
            {"Lx": 12, "phase": 0.05, "tau_A_target": 0.11, "tau_Z_target": 0.21},
        ]
    ).to_csv(tmp_path / "qdm_checkerboard_finite_beta_transfer_target.csv", index=False)

    selected, _ = module._qdm_canonical_l12(tmp_path, phase=0.05)
    assert len(selected) == 1
    assert int(selected.iloc[0]["Lx"]) == 12
    assert float(selected.iloc[0]["tau_A_target"]) == 0.11


def test_spin1_merged_panel_keeps_large_size_without_inventing_range() -> None:
    module = _load(POLISH, "prx_main_thermal_figure_polish_spin1_merge_test")
    representative = pd.DataFrame(
        [
            {"L": 12, "witness": "A", "tau_mc_raw": 0.12},
            {"L": 14, "witness": "A", "tau_mc_raw": 0.11},
        ]
    )
    ranges = pd.DataFrame(
        [
            {
                "L": 12,
                "witness": "A",
                "tau_star": 0.12,
                "tau_min": 0.119,
                "tau_max": 0.121,
            }
        ]
    )

    merged = module._spin1_panel_b_merged(representative, ranges)
    row14 = merged[merged["L"] == 14].iloc[0]
    assert float(row14["tau_star"]) == 0.11
    assert pd.isna(row14["tau_min"])
    assert pd.isna(row14["tau_max"])
    assert not bool(row14["sampled_range_available"])


def test_qdm_deformation_scan_uses_largest_sampled_raw_size() -> None:
    module = _load(POLISH, "prx_main_thermal_figure_polish_qdm_scan_test")
    rows = []
    for lx in (4, 8):
        for phase in (0.025, 0.05, 0.075):
            rows.append(
                {
                    "Lx": lx,
                    "phase": phase,
                    "tau_A_mc": 0.05,
                    "tau_Z_mc": 0.10,
                    "tau_A_reference": 0.06,
                    "tau_Z_reference": 0.12,
                }
            )
    scan, length = module._qdm_deformation_scan(pd.DataFrame(rows))
    assert length == 8
    assert set(scan["Lx"].astype(int)) == {8}
    assert set(scan["ensemble"]) == {"raw_microcanonical", "canonical"}
    assert set(scan["witness"]) == {"A", "Z"}


def test_main_figure_roles_and_literal_range_contract() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert 'STAR_COLOR = "#E69F00"' in source
    assert '"panel_b_role": "finite size plus compatible-kappa range"' in source
    assert '"panel_c_role": "witness versus compatible kappa"' in source
    assert '"panel_b_role": "finite size plus compatible-phase range"' in source
    assert '"panel_c_role": "witness versus compatible phase"' in source
    assert '"range_display_floor": False' in source
    assert '"panel_d_box_width": 0.34' in source
    assert "float(maximum) - float(minimum)" in source
    assert "display floor" not in source.lower()
    assert 'rf"\\textbf{{{label}}}"' in source


def test_legends_are_outside_and_compact_panel_ticks_are_cleaned() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert '"bbox_to_anchor": (0.0, 1.02)' in source
    assert "from matplotlib.ticker import MaxNLocator, ScalarFormatter" in source
    assert "MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10], min_n_ticks=2)" in source
    assert source.count("_clean_y_ticks(ax)") >= 4
    assert 'label=r"$\\beta=0$"' in source
    assert '"panel_b_legend": "above axes; kappa_star, kappa scan, beta=0"' in source
    assert '"panel_c_legend": "above axes"' in source


def test_qdm_has_clear_ensemble_markers_and_no_horizontal_offset() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert "_qdm_ensemble_box_handles" in source
    assert "_qdm_scan_handles" in source
    assert "_center_marker(" in source
    assert 'markerfacecolor=color if filled else "white"' in source
    assert 'label="raw MC"' in source
    assert 'label="canonical"' in source
    assert '"panel_b_horizontal_displacement": False' in source
    assert '"panel_b_ensemble_encoding": "raw filled solid circle; canonical open dashed circle"' in source
    assert "x=float(row.Lx) - 0.10" not in source
    assert "x=float(row.Lx) + 0.10" not in source


def test_panel_d_legends_use_plotted_blue_and_qdm_width_stays_narrow() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert source.count('color=WITNESS_COLORS["A"]') >= 6
    assert source.count('"panel_d_legend_color": WITNESS_COLORS["A"]') == 2
    assert '"panel_d_box_width": 0.34' in source


def test_qdm_panel_d_keeps_zero_floor_with_data_driven_ceiling() -> None:
    source = POLISH.read_text(encoding="utf-8")
    ceiling = "upper_limit = 1.0 if finite_upper.size == 0 else 1.08 * float(np.max(finite_upper))"
    assert ceiling in source
    assert "axd.set_ylim(0.0, upper_limit)" in source


def test_renderers_use_followup_polish_module() -> None:
    spin1 = SPIN1_RENDERER.read_text(encoding="utf-8")
    qdm = QDM_RENDERER.read_text(encoding="utf-8")
    assert "from prx_main_thermal_figure_polish import render_spin1_figure6" in spin1
    assert "from prx_main_thermal_figure_polish import render_qdm_figure9" in qdm
    assert "_install_qdm_panel_c_legend" not in qdm


def test_spin1_witness_support_notation_is_explicit() -> None:
    source = SPIN1_RENDERER.read_text(encoding="utf-8")
    assert '"A": r"\\widehat Q^A_{R_r}"' in source
    assert '"Z": r"\\widehat Q^Z_{R_r}"' in source
    assert '"Y": r"\\widehat Q^Y_r"' in source
    assert "_save_figure6_with_support_notation" in source
    assert 'label=rf"${SPIN1_WITNESS_OPERATOR_LABELS[key]}$"' in source
    assert 'audit["spin1_witness_support_notation"] = SPIN1_WITNESS_OPERATOR_LABELS' in source


def test_spin1_appendix_figures_are_horizontal_full_width() -> None:
    source = SPIN1_RENDERER.read_text(encoding="utf-8")
    assert source.count("figsize=(PRX_TEXT_WIDTH, 2.72)") == 2
    assert source.count("fig.add_gridspec(\n        1,\n        2,") == 2
    assert 'audit["fig10_layout"] = "horizontal_1x2_full_text_width"' in source
    assert 'audit["fig11_layout"] = "horizontal_1x2_full_text_width"' in source


def test_fig15_remains_one_code_generated_horizontal_asset() -> None:
    source = APPENDIX_RENDERER.read_text(encoding="utf-8")
    assert "_qdm_locality_scaling_figure" in source
    assert "fig = plt.figure(figsize=(PRX_TEXT_WIDTH, 3.05))" in source
    assert '"qdm_appendix_locality_scaling_certificates"' in source
