"""Contracts for the analytic spin-1 model-realization closure package."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

RUNNER = JOBS / "run_prx_model_realization_closure.py"
PROOF = ROOT / "experimental" / "PRX_MODEL_REALIZATION_CLOSURE.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wide_window_bound_constants_match_locked_family() -> None:
    module = _load(RUNNER, "run_prx_model_realization_closure_constants_test")
    assert module.WINDOW_POWER == pytest.approx(0.75)
    assert module.CHEBYSHEV_DECAY_POWER == pytest.approx(0.5)
    assert module.FIXED_M_VARIANCE_COEFFICIENT_MAX == pytest.approx(2.1)
    assert module.energy_variance_identity_coefficient(0.20) == pytest.approx(2.1)
    assert module.Y_BETA0_LIMIT == pytest.approx(1.0 / 3.0)
    assert module.Y_EVENTUAL_SAFE_LOWER_BOUND > 0.0


def test_fixed_m_local_pattern_probability_is_exact_combinatorics() -> None:
    module = _load(RUNNER, "run_prx_model_realization_closure_probability_test")
    for length in (8, 10, 20, 40):
        probability = module.fixed_m_local_pattern_probability(length, (0,))
        expected = module.coefficient_for_cycles(
            length - 1, 0, module.TOTAL_SZ
        ) / module.coefficient_for_cycles(length, 0, module.TOTAL_SZ)
        assert probability == pytest.approx(expected)
    assert module.fixed_m_local_pattern_probability(40, (0,)) == pytest.approx(1.0 / 3.0, abs=0.02)


def test_model_closure_runner_emits_required_solver_free_package(tmp_path: Path) -> None:
    module = _load(RUNNER, "run_prx_model_realization_closure_run_test")
    result = module.run(tmp_path)
    assert result["metadata"]["spectral_solver_launched"] is False
    assert result["metadata"]["new_large_L_diagonalization"] is False
    assert result["metadata"]["qdm_12x4_required"] is False

    required = {
        "framework_model_closure_matrix.md",
        "framework_model_closure_matrix.json",
        "spin1_wide_window_thermodynamic_proof.md",
        "spin1_wide_window_bounds.json",
        "spin1_wide_window_exact_combinatorial_checks.csv",
        "deformation_uniformity_audit.md",
        "qdm_framework_layer_audit.md",
        "manuscript_claim_upgrade_handoff.md",
        "job_metadata.json",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})

    matrix = json.loads(
        (tmp_path / "framework_model_closure_matrix.json").read_text(encoding="utf-8")
    )
    assert matrix["verdicts"] == {
        "formal_framework_closed": True,
        "model_instantiation_closed": True,
        "literal_icqmbs_realization_closed": True,
        "deformation_stable_icqmbs_closed": True,
    }
    assert "L^(1/4)" in matrix["narrow_L_quarter_window_status"]


def test_proof_keeps_narrow_window_and_qdm_boundaries_explicit() -> None:
    text = PROOF.read_text(encoding="utf-8")
    assert "does not retroactively prove" in text
    assert "L^{3/4}" in text
    assert "L^{7/2}3^{-L/2}" in text
    assert "arbitrary-fixed-bounded-region background-concentration gate" in text
    assert "QDM" in text
    assert "fixed-width" in text
    assert "thermodynamic ICQMBS classification remains open" in text
    assert not [char for char in text if ord(char) < 32 and char not in "\n\t"]

    source = RUNNER.read_text(encoding="utf-8")
    assert "eigsh" not in source
    assert "eigh(" not in source
    assert "spectral_solver_launched" in source


def test_locked_spin1_family_satisfies_center_and_row_norm_sanity() -> None:
    import numpy as np
    import scipy.sparse as sp

    from qlinks.models import (
        spin_one_xy_hxy_h3_imaginary_j2_model,
        spin_one_xy_scar_tower_states,
    )

    length = 8
    model = spin_one_xy_hxy_h3_imaginary_j2_model(
        length=length,
        j=1.0,
        j3=0.1,
        kappa=0.2,
        total_sz=-2,
    )
    build = model.build(builder="optimized", basis_solver="dfs", sort_basis=True)
    configs = np.asarray(build.basis.states)
    hamiltonian = sp.csr_array(build.hamiltonian, dtype=np.complex128)

    charges = np.sum(configs[:, 0::2] + 1, axis=1).astype(np.int64)
    signs = np.where(charges % 2 == 0, 1.0, -1.0)
    grading = sp.diags(signs.astype(np.complex128), format="csr")
    transformed = grading @ hamiltonian.conjugate() @ grading
    scale = max(float(sp.linalg.norm(hamiltonian)), np.finfo(float).tiny)
    assert float(sp.linalg.norm(transformed + hamiltonian)) / scale < 1.0e-12

    translated_charges = np.sum(np.roll(configs, 1, axis=1)[:, 0::2] + 1, axis=1)
    translated_signs = np.where(translated_charges % 2 == 0, 1.0, -1.0)
    np.testing.assert_array_equal(signs, translated_signs)

    row_norm_squared = (
        np.asarray(hamiltonian.multiply(hamiltonian.conjugate()).sum(axis=1)).ravel().real
    )
    assert float(np.max(row_norm_squared)) <= 2.1 * length + 1.0e-12

    tower, labels = spin_one_xy_scar_tower_states(
        basis_configs=configs,
        length=length,
        normalize=True,
    )
    assert tower.shape[1] == 1, labels
    assert float(np.linalg.norm(hamiltonian @ tower[:, 0])) < 1.0e-12
