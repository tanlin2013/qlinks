"""Contracts for the evidence-first PRX P1 strengthening handoff."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))
ENTROPY = JOBS / "spin1_prx_p1_resolved_entropy.py"
OBSTRUCTION = JOBS / "spin1_prx_p1_obstruction_hierarchy.py"
THERMO = JOBS / "spin1_prx_p1_thermodynamic_summary.py"
RUNNER = JOBS / "run_prx_p1_strengthening.py"
SPIN_RENDER = JOBS / "render_spin1_xy_draft_figures.py"
QDM_RENDER = JOBS / "render_square_qdm_draft_figures.py"
STYLE_AUDIT = JOBS / "audit_prx_p1_figure_style.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolved_entropy_exact_translation_character_counts() -> None:
    module = _load(ENTROPY, "spin1_prx_p1_resolved_entropy_test")
    assert module.momentum_dimensions(4, -2) == [3, 2, 3, 2]
    assert module.momentum_dimensions(6, -2) == [16, 14, 16, 14, 16, 14]
    assert sum(module.momentum_dimensions(8, -2)) == 784
    assert module.staggered_tower_momentum_index(8, -2) == 4
    assert module.staggered_tower_momentum_index(10, -2) == 0


def test_resolved_entropy_run_uses_even_tower_sequence(tmp_path: Path) -> None:
    module = _load(ENTROPY, "spin1_prx_p1_resolved_entropy_run_test")
    module.run(tmp_path, minimum_length=4, maximum_length=8, magnetization=-2)
    frame = pd.read_csv(tmp_path / "spin1_resolved_sector_counts.csv")
    assert set(frame["L"].astype(int)) == {4, 6, 8}
    selected = frame[frame["is_selected_tower_momentum"].astype(bool)]
    assert selected.groupby("L").size().to_dict() == {4: 1, 6: 1, 8: 1}
    summary = (tmp_path / "spin1_resolved_sector_entropy.md").read_text(encoding="utf-8")
    assert "ordinary inversion is broken" in summary


def test_thermodynamic_lane_preserves_stop_rules() -> None:
    source = THERMO.read_text(encoding="utf-8")
    assert '"track": "A1"' in source
    assert '"status": "proved"' in source
    assert '"track": "A2"' in source
    assert '"status": "inconclusive"' in source
    assert '"track": "A3"' in source
    assert '"status": "not attempted"' in source
    assert "local-limit theorem" in source
    assert "large_L_bruteforce_launched" in source
    assert "missing the spin-1 exchange-convention stamp" in source


def test_blind_obstruction_chart_does_not_encode_tower_rule_in_input() -> None:
    source = OBSTRUCTION.read_text(encoding="utf-8")
    for coordinate in (
        "Re_t1",
        "Im_t1",
        "Re_t2",
        "Im_t2",
        "Re_t3",
        "Im_t3",
    ):
        assert coordinate in source
    assert 'default="8,10"' in source
    assert '"T_joint"' in source
    assert '"not_implemented"' in source
    assert "CURRENT_EXCHANGE_CONVENTION" in source
    assert "spin1_obstruction_singular_values.csv" in source
    assert "spin1_obstruction_spot_checks.csv" in source
    assert "eigsh" not in source
    assert "eigh(" not in source


@pytest.mark.integration
def test_blind_obstruction_runtime_smoke_uses_public_stability_api() -> None:
    module = _load(OBSTRUCTION, "spin1_prx_p1_obstruction_runtime_test")
    result = module.analyze(8, -2)
    layers = pd.DataFrame(result["layers"])
    assert set(layers["layer"]) == {"T_cage", "T_fixed", "T_tower", "T_joint"}
    joint = layers[layers["layer"] == "T_joint"].iloc[0]
    assert joint["status"] == "not_implemented"
    assert result["summary"]["T_joint_status"] == "not_implemented"
    assert result["spot_checks"]


def test_a4_requires_current_convention_and_reports_finite_size_only(tmp_path: Path) -> None:
    module = _load(THERMO, "spin1_prx_p1_thermodynamic_summary_test")
    key = module.EXCHANGE_CONVENTION_METADATA_KEY
    convention = module.CURRENT_EXCHANGE_CONVENTION
    rows = []
    for length in (8, 10):
        for bridge, distance in (
            ("mc_to_beta0_resolved", 0.02 / length),
            ("beta0_resolved_to_fixedM", 0.001 / length),
        ):
            rows.append(
                {
                    "L": length,
                    "bridge": bridge,
                    "trace_distance": distance,
                    key: convention,
                }
            )
    path = tmp_path / "bridges.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    verdict, metadata = module.evaluate_a4(path)
    assert verdict["status"] == "controlled numerical strengthening"
    assert metadata["largest_common_L"] == 10
    assert (
        "not an asymptotic ensemble-equivalence theorem"
        in verdict["strongest_manuscript_safe_sentence"]
    )

    unstamped = tmp_path / "unstamped.csv"
    pd.DataFrame(
        [{key_: value for key_, value in row.items() if key_ != key} for row in rows]
    ).to_csv(unstamped, index=False)
    with pytest.raises(ValueError, match="exchange-convention stamp"):
        module.evaluate_a4(unstamped)


def test_final_renderers_require_real_tex_and_keep_manuscript_stems() -> None:
    spin = SPIN_RENDER.read_text(encoding="utf-8")
    qdm = QDM_RENDER.read_text(encoding="utf-8")
    assert "spin1_xy_figure6_prx" in spin
    assert "text.usetex" in spin
    assert "refusing mathtext fallback" in spin
    assert "qdm_checkerboard_figure7_combined" in qdm
    assert "qdm_checkerboard_figure9_prx" in qdm
    assert "text.usetex" in qdm
    assert "refusing mathtext fallback" in qdm


def test_qdm_figure_uses_nested_fig6_grammar_and_optional_l12_gate() -> None:
    source = QDM_RENDER.read_text(encoding="utf-8")
    assert "subgridspec(2, 1, height_ratios=(2.2, 1.0)" in source
    assert "subgridspec(2, 1, hspace=0.08)" in source
    assert 'cb.ax.set_title(r"$w_{L_x}(\\varphi)$"' in source
    assert "window_coverage_complete" in source
    assert "fig.legend(" not in source


@pytest.mark.integration
def test_qdm_renderer_ignores_unverified_optional_l12(tmp_path: Path) -> None:
    thermal_rows = []
    concentration_rows = []
    scatter_rows = []
    for length in (4, 8, 12):
        verified = length != 12
        for phase in (0.10, 0.20):
            thermal_rows.append(
                {
                    "Lx": length,
                    "phase": phase,
                    "window_prefactor": 0.75,
                    "thermal_protocol": "finite-beta",
                    "window_coverage_complete": verified,
                    "cage_energy_density": 0.0,
                    "window_energy_density_half_width": 0.02,
                    "tau_A_mc": 0.10 + 0.001 * length,
                    "tau_A_reference": 0.11,
                    "delta_A": 0.01 / length,
                    "tau_Z_mc": 0.20 + 0.001 * length,
                    "tau_Z_reference": 0.21,
                    "delta_Z": 0.02 / length,
                    "Delta": 0.03 / length,
                }
            )
            concentration_rows.append(
                {
                    "Lx": length,
                    "phase": phase,
                    "window_coverage_complete": verified,
                    "w_raw": 0.2 / length,
                }
            )
        for energy in (-0.08, 0.0, 0.08):
            scatter_rows.append(
                {
                    "Lx": length,
                    "energy_density": energy,
                    "Q_A": 0.10 + energy,
                    "Q_Z": 0.20 - energy,
                }
            )

    pd.DataFrame(thermal_rows).to_csv(
        tmp_path / "qdm_checkerboard_thermal_overlap.csv", index=False
    )
    pd.DataFrame(concentration_rows).to_csv(
        tmp_path / "qdm_checkerboard_concentration_grid.csv", index=False
    )
    pd.DataFrame(scatter_rows).to_csv(tmp_path / "qdm_checkerboard_eth_scatter.csv", index=False)
    pd.DataFrame([{"phi_star": 0.10}]).to_csv(
        tmp_path / "qdm_checkerboard_representative_phase.csv", index=False
    )

    ipython_stub = tmp_path / "ipython_stub" / "IPython"
    ipython_stub.mkdir(parents=True)
    (ipython_stub / "__init__.py").write_text(
        "version_info = (0, 0)\n__version__ = '0.0'\ndef get_ipython():\n    return None\n",
        encoding="utf-8",
    )
    (ipython_stub / "display.py").write_text(
        "def display(*objects, **kwargs):\n    return None\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ipython_stub.parent), inherited_pythonpath) if part
    )
    subprocess.run(
        [
            sys.executable,
            str(QDM_RENDER),
            "--data-dir",
            str(tmp_path),
            "--figure-formats",
            "svg",
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )

    assert (tmp_path / "figures" / "qdm_checkerboard_figure9_prx.svg").is_file()
    manifest = (tmp_path / "figure_manifest.json").read_text(encoding="utf-8")
    assert "qdm_checkerboard_figure9_prx" in manifest


def test_style_audit_checks_fonts_tex_dimensions_and_qdm_l12() -> None:
    source = STYLE_AUDIT.read_text(encoding="utf-8")
    assert "pdffonts" in source
    assert "contains_dejavu" in source
    assert "usetex" in source
    assert "dimension_ok" in source
    assert "qdm_12x4_verified_row_present" in source


def test_runner_is_solver_free_and_render_only_for_figures() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"spectral_solver_launched": False' in source
    assert "--render-figures" in source
    assert "render_spin1_xy_draft_figures.py" in source
    assert "render_square_qdm_draft_figures.py" in source
