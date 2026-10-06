"""Regression contracts for the Oct. 6 PRX figure-polish follow-up."""

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


def test_main_figure_polish_uses_bar_whiskers_and_no_bc_connecting_lines() -> None:
    source = POLISH.read_text(encoding="utf-8")
    assert 'marker="_"' in source
    assert "WHISKER_CAP_SIZE = 5.0" in source
    assert "WHISKER_LINE_WIDTH = 1.4" in source
    assert '"panel_b_connecting_lines": False' in source
    assert '"panel_c_connecting_lines": False' in source
    assert '"panel_d_guide_line": "dashed"' in source
    assert 'STAR_COLOR = "#E69F00"' in source
    assert '"A": "#0072B2"' in source
    assert '"Z": "#009E73"' in source
    assert '"Y": "#CC79A7"' in source


def test_renderers_use_followup_polish_module() -> None:
    spin1 = SPIN1_RENDERER.read_text(encoding="utf-8")
    qdm = QDM_RENDERER.read_text(encoding="utf-8")
    assert "from prx_main_thermal_figure_polish import render_spin1_figure6" in spin1
    assert "from prx_main_thermal_figure_polish import render_qdm_figure9" in qdm


def test_spin1_appendix_figures_are_horizontal_full_width() -> None:
    source = SPIN1_RENDERER.read_text(encoding="utf-8")
    assert source.count("figsize=(PRX_TEXT_WIDTH, 2.72)") == 2
    assert source.count("fig.add_gridspec(\n        1,\n        2,") == 2
    assert 'audit["fig10_layout"] = "horizontal_1x2_full_text_width"' in source
    assert 'audit["fig11_layout"] = "horizontal_1x2_full_text_width"' in source


def test_fig15_remains_one_code_generated_horizontal_asset() -> None:
    source = APPENDIX_RENDERER.read_text(encoding="utf-8")
    assert "_qdm_locality_scaling_figure" in source
    assert 'fig = plt.figure(figsize=(PRX_TEXT_WIDTH, 3.05))' in source
    assert '"qdm_appendix_locality_scaling_certificates"' in source
