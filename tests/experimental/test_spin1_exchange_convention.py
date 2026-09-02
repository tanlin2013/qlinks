from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

import spin1_exchange_convention as convention  # noqa: E402


def _load_converter():
    path = JOBS / "spin1_exchange_convention_migrate_evidence.py"
    spec = importlib.util.spec_from_file_location("spin1_exchange_convention_migrate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exchange_convention_contract_and_windows() -> None:
    assert convention.exchange_convention_from_metadata({}) == convention.LEGACY_EXCHANGE_CONVENTION
    with pytest.raises(ValueError, match="explicit legacy rescaling"):
        convention.require_current_exchange_convention({})

    current = convention.current_metadata(rescaled_from=convention.LEGACY_EXCHANGE_CONVENTION)
    convention.require_current_exchange_convention(current)
    assert convention.current_window_half_width(16, convention.PRIMARY_WINDOW_PROTOCOL) == 1.0
    assert convention.current_window_half_width(16, convention.FIXED_WINDOW_PROTOCOL) == 0.5
    assert convention.map_legacy_window_protocol("quarter_power_c1") == "quarter_power_c0p5"
    assert convention.map_legacy_window_protocol("fixed_width_1") == "fixed_width_0p5"


def test_converter_rescales_energy_beta_and_protocol_without_touching_source(
    tmp_path: Path,
) -> None:
    converter = _load_converter()
    source = tmp_path / "legacy-run"
    output = tmp_path / "derived-run"
    source.mkdir()

    table = pd.DataFrame(
        [
            {
                "L": 14,
                "window_protocol": "quarter_power_c1",
                "window_half_width": 1.93351,
                "window_prefactor": 1.0,
                "covered_spectral_half_width": 2.08384,
                "energy_density": 0.2,
                "window_max_eigenpair_residual": 2.0e-7,
                "energy_block_tolerance": 1.0e-10,
                "beta": -0.0568,
                "kappa_over_J": 0.1,
                "tau_A_mc_raw": 0.113204,
                "w_L": 0.0174573,
                "state_count": 7615,
            }
        ]
    )
    source_csv = source / "rows.csv"
    table.to_csv(source_csv, index=False)
    source_json = source / "metadata.json"
    source_json.write_text(
        json.dumps(
            {
                "window_protocol": "fixed_width_1",
                "window_half_width": 1.0,
                "covered_spectral_half_width": 2.08384,
                "tower_residual": 4.0e-12,
                "interference_gap": 1.59991,
                "beta_J": -0.0568,
                "D_over_J": 0.63,
                "kappa_over_J": 0.1,
                "returned_eigenpairs": 8192,
            }
        ),
        encoding="utf-8",
    )
    source_snapshot = source_csv.read_bytes(), source_json.read_bytes()

    manifest = converter.convert_evidence_directory(source_dir=source, output_dir=output)

    converted = pd.read_csv(output / "rows.csv").iloc[0]
    assert converted.window_protocol == "quarter_power_c0p5"
    assert np.isclose(converted.window_half_width, 0.966755)
    assert np.isclose(converted.window_prefactor, 0.5)
    assert np.isclose(converted.covered_spectral_half_width, 1.04192)
    assert np.isclose(converted.energy_density, 0.1)
    assert np.isclose(converted.window_max_eigenpair_residual, 1.0e-7)
    assert np.isclose(converted.energy_block_tolerance, 5.0e-11)
    assert np.isclose(converted.beta, -0.1136)
    assert np.isclose(converted.kappa_over_J, 0.1)
    assert np.isclose(converted.tau_A_mc_raw, 0.113204)
    assert np.isclose(converted.w_L, 0.0174573)
    assert int(converted.state_count) == 7615
    assert converted.spin1_xy_exchange_convention == convention.CURRENT_EXCHANGE_CONVENTION

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["window_protocol"] == "fixed_width_0p5"
    assert np.isclose(metadata["window_half_width"], 0.5)
    assert np.isclose(metadata["covered_spectral_half_width"], 1.04192)
    assert np.isclose(metadata["tower_residual"], 2.0e-12)
    assert np.isclose(metadata["interference_gap"], 0.799955)
    assert np.isclose(metadata["beta_J"], -0.1136)
    assert np.isclose(metadata["D_over_J"], 0.315)
    assert np.isclose(metadata["kappa_over_J"], 0.1)
    assert metadata["returned_eigenpairs"] == 8192
    assert manifest["energy_scale"] == 0.5
    assert manifest["beta_J_scale"] == 2.0

    assert (source_csv.read_bytes(), source_json.read_bytes()) == source_snapshot
