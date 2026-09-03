from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

import spin1_exchange_convention_validate as validation  # noqa: E402


def test_five_site_shell_has_zero_and_minus_two_modes() -> None:
    report = validation._five_site_shell()

    assert report["S_0"]["support_size"] == 20
    assert report["S_0"]["expectation"] == 0.0
    assert report["S_0"]["eigenpair_residual"] < 1.0e-10
    assert report["S_0"]["boundary_residual"] < 1.0e-10

    assert report["S_-2"]["support_size"] == 20
    assert math.isclose(report["S_-2"]["expectation"], -2.0, abs_tol=1.0e-10)
    assert report["S_-2"]["eigenpair_residual"] < 1.0e-10
    assert report["S_-2"]["boundary_residual"] < 1.0e-10


def test_decorated_periodic_counterexample_has_sqrt_two_residual() -> None:
    report = validation._decorated_pbc_counterexample()
    assert math.isclose(report["normalized_residual"], math.sqrt(2.0), abs_tol=1.0e-10)


def test_current_model_is_half_legacy_model_at_fixed_ratios() -> None:
    report = validation._matrix_scaling_check(length=6)
    assert report["maximum_matrix_scaling_residual"] < 1.0e-10


def test_finite_d_rescales_with_exchange_convention() -> None:
    report = validation._matrix_scaling_check(length=6, d_new=0.315)
    assert report["D_over_J_current"] == 0.315
    assert report["D_over_J_legacy_display"] == 0.63
    assert report["maximum_matrix_scaling_residual"] < 1.0e-10
