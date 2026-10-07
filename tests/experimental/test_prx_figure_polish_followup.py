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


def test_main_figure_polish_uses_points_boxes_and_clean_annotations() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert 'marker="o"' in source
    assert "Rectangle(" in source
    assert '"panel_b_marker": "circle"' in source
    assert '"panel_c_marker": "range_box_with_representative_line"' in source
    assert '"panel_d_marker": "range_box_with_representative_line"' in source
    assert '"panel_d_guide_line": "dashed"' in source
    assert '"in_panel_encoding_text": False' in source
    assert "bars/whiskers:" not in source
    assert "tau_{{{key}}}" not in source
    assert 'STAR_COLOR = "#E69F00"' in source
    assert '"A": "#0072B2"' in source
    assert '"Z": "#009E73"' in source
    assert '"Y": "#CC79A7"' in source
    assert 'rf"\\textbf{{{label}}}"' in source


def test_qdm_uses_witness_color_and_no_horizontal_ensemble_offset() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert 'color = WITNESS_COLORS[key]' in source
    assert 'markerfacecolor=color if filled else "white"' in source
    assert '("raw_microcanonical", True, "-")' in source
    assert '("canonical", False, "--")' in source
    assert '"panel_b_horizontal_displacement": False' in source
    assert '"panel_c_horizontal_displacement": False' in source
    assert 'x=float(row.Lx) - 0.10' not in source
    assert 'x=float(row.Lx) + 0.10' not in source


def test_qdm_panel_d_keeps_zero_floor_with_data_driven_ceiling() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert "upper_limit = 1.0 if finite_upper.size == 0 else 1.08 * float(np.max(finite_upper))" in source
    assert "axd.set_ylim(0.0, upper_limit)" in source


def test_renderers_use_followup_polish_module() -> None:
    spin1 = SPIN1_RENDERER.read_text(encoding="utf-8")
    qdm = QDM_RENDERER.read_text(encoding="utf-8")
    assert "from prx_main_thermal_figure_polish import render_spin1_figure6" in spin1
    assert "from prx_main_thermal_figure_polish import render_qdm_figure9" in qdm


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
