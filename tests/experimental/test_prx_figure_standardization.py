"""Contracts for the render-only PRX figure-standardization pass."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

RENDERER = JOBS / "render_prx_appendix_figures.py"
AUDIT = JOBS / "audit_prx_figure_standardization.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_appendix_renderer_uses_final_prx_column_width() -> None:
    module = _load(RENDERER, "render_prx_appendix_figures_test")
    module._configure_style(use_tex=False)

    bridge = pd.DataFrame(
        [
            {
                "L": length,
                "bridge": bridge_name,
                "trace_distance": value,
                "abs_delta_tau_A": 0.02,
                "abs_delta_tau_Z": 0.01,
                "abs_delta_tau_Y": 0.015,
            }
            for length, value in ((8, 0.04), (10, 0.03))
            for bridge_name in ("mc_to_beta0_resolved", "beta0_resolved_to_fixedM")
        ]
    )
    concentration = pd.DataFrame(
        [
            {
                "L": length,
                "variant": "raw",
                "window_protocol": protocol,
                "w_L": width,
                "window_state_count": count,
            }
            for length, width, count in ((8, 0.15, 30), (10, 0.08, 150))
            for protocol in (
                module.PRIMARY_WINDOW_PROTOCOL,
                module.FIXED_WINDOW_PROTOCOL,
            )
        ]
    )
    radius = pd.DataFrame(
        [
            {"state": state, "radius": radius_value, "minimum_residual": residual}
            for state in ("compact record 0", "collective record 8")
            for radius_value, residual in ((0, 1.0), (1, 0.5), (2, 1.0e-12))
        ]
    )
    compatibility = pd.DataFrame(
        [
            {
                "repeats": repeats,
                "kinetic_constraint_rank": 2 * repeats,
                "kinetic_compatible_dimension": 14 * repeats,
            }
            for repeats in (1, 2, 3)
        ]
    )

    figures = (
        module._spin1_bridge_figure(bridge),
        module._spin1_concentration_figure(concentration),
        module._qdm_radius_figure(radius),
        module._qdm_compatibility_figure(compatibility),
    )
    combined = module._qdm_locality_scaling_figure(radius, compatibility)
    try:
        for figure in figures:
            assert figure.get_size_inches()[0] == pytest.approx(module.PRX_COLUMN_WIDTH)
        assert combined.get_size_inches()[0] == pytest.approx(module.PRX_TEXT_WIDTH)
        assert len(figures[0].axes) == 2
        assert len(figures[1].axes) == 2
        assert len(figures[2].axes) == 1
        assert len(figures[3].axes) == 1
        assert len(combined.axes) == 2

        for figure in (figures[0], figures[1], combined):
            panel_labels = [
                text
                for axis in figure.axes
                for text in axis.texts
                if text.get_text() in {"(a)", "(b)"}
            ]
            assert panel_labels
            assert all(text.get_position()[0] < 0.0 for text in panel_labels)
            assert all(text.get_position()[1] > 1.0 for text in panel_labels)
            assert all(text.get_clip_on() is False for text in panel_labels)
    finally:
        for figure in (*figures, combined):
            plt.close(figure)


def test_appendix_renderer_is_render_only() -> None:
    source = RENDERER.read_text(encoding="utf-8")
    assert "qdm_4x4_minimum_annihilator_radius.csv" in source
    assert "qdm_4N_by_4_exact_sequence.csv" in source
    assert "spin1_xy_appendix_beta0_bridges_data.csv" in source
    assert "spin1_xy_kappa0p1_concentration_common_windows.csv" in source
    assert "eigsh" not in source
    assert "eigh(" not in source
    assert "scan_windowed_operator_annihilators" not in source
    assert "scan_square_qdm_periodic_product_cancellation_scaling" not in source


def test_polished_main_figure_renderers_make_expectation_semantics_explicit() -> None:
    spin1 = (JOBS / "render_spin1_xy_sec6_integration_figures_legacy.py").read_text(
        encoding="utf-8"
    )
    qdm = (JOBS / "render_square_qdm_draft_figures.py").read_text(encoding="utf-8")

    assert r"\langle \widehat Q_R^{{{key}}}\rangle_n" in spin1
    assert r"\langle \widehat Q_R^\alpha\rangle_{\rm mc}" in spin1
    assert 'bbox_to_anchor=(0.0, 1.015)' in spin1
    assert "add_panel_label_margin" in spin1

    assert r"\langle \widehat Q_R^{{{key}}}\rangle_n" in qdm
    assert r"\langle \widehat Q_R^\alpha\rangle" in qdm
    assert 'title="ensemble"' in qdm
    assert 'title=r"strip size"' in qdm
    assert "add_panel_label_margin" in qdm


def test_appendix_renderer_emits_combined_fig15_asset() -> None:
    source = RENDERER.read_text(encoding="utf-8")
    assert "_qdm_locality_scaling_figure" in source
    assert "qdm_appendix_locality_scaling_certificates" in source
    assert "PRX_TEXT_WIDTH" in source
    assert "add_panel_label_margin" in source


def test_figure_audit_gates_unverified_qdm_12x4(tmp_path: Path) -> None:
    module = _load(AUDIT, "audit_prx_figure_standardization_test")
    rows = []
    for length, verified in ((4, True), (8, True), (12, False)):
        rows.append(
            {
                "Lx": length,
                "phase": 0.1,
                "window_prefactor": 0.75,
                "window_coverage_complete": verified,
                "converged_vs_previous_budget": verified,
            }
        )
    path = tmp_path / "qdm_checkerboard_thermal_overlap.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    assert module._verified_qdm_lengths(tmp_path) == [4, 8]

    frame = pd.read_csv(path)
    frame.loc[frame["Lx"] == 12, "window_coverage_complete"] = True
    frame.loc[frame["Lx"] == 12, "converged_vs_previous_budget"] = True
    frame.to_csv(path, index=False)
    assert module._verified_qdm_lengths(tmp_path) == [4, 8, 12]


def test_figure_audit_uses_convention_mapped_spin1_p0_contract() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    assert "spin1_exchange_convention_render_p0.py" in source
    assert "spin1_xy_figure6_panel_a_scatter.csv" in source
    assert "spin1_xy_figure6_panel_d_family_band.csv" in source
    assert "spin1_exchange_convention_migration_manifest.json" in source
    assert "render_spin1_xy_draft_figures.py --data-dir {data}" not in source
    assert "qdm_appendix_locality_scaling_certificates" in source
    assert '"figure": "Fig. 15"' in source
    assert "prx_figure_polish_followup_audit.md" in source


def test_figure_audit_validates_mapped_spin1_p0_manifest_hashes(tmp_path: Path) -> None:
    module = _load(AUDIT, "audit_prx_figure_standardization_p0_test")
    fig6 = next(spec for spec in module.ASSETS if spec["figure"] == "Fig. 6")
    required = [str(name) for name in fig6["sources"] if name != module.SPIN1_MIGRATION_MANIFEST]
    converted = []
    for index, name in enumerate(required):
        path = tmp_path / name
        path.write_text(f"payload-{index}\n", encoding="utf-8")
        converted.append(
            {
                "path": name,
                "derived_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    manifest = {
        "source_run_id": "spin1_sec6_integration_20260825T073925Z",
        "spin1_xy_exchange_convention": module.SPIN1_CURRENT_EXCHANGE_CONVENTION,
        "rescaled_from_exchange_convention": module.SPIN1_LEGACY_EXCHANGE_CONVENTION,
        "converted_files": converted,
    }
    (tmp_path / module.SPIN1_MIGRATION_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    provenance = module._spin1_mapped_p0_provenance(tmp_path)
    assert provenance["valid"] is True
    assert provenance["errors"] == []
    assert provenance["verified_inputs"] == required

    first = tmp_path / required[0]
    first.write_text("tampered\n", encoding="utf-8")
    provenance = module._spin1_mapped_p0_provenance(tmp_path)
    assert provenance["valid"] is False
    assert any("hash mismatch" in str(error) for error in provenance["errors"])


def test_notebook_image_contains_tex_and_pdf_font_audit_tools() -> None:
    source = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "texlive-latex-base" in source
    assert "texlive-latex-recommended" in source
    assert "cm-super" in source
    assert "dvipng" in source
    assert "poppler-utils" in source
