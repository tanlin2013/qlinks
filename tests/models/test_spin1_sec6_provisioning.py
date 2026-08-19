from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT / "experimental" / "jobs"
NOTEBOOK_DIR = ROOT / "experimental" / "notebooks"
for path in (JOBS_DIR, NOTEBOOK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spin1_sec6_provisioning import (  # noqa: E402
    Sec6ProvisioningConfig,
    run_sec6_provisioning,
)


@pytest.mark.integration
def test_sec6_two_bridge_smoke(tmp_path: Path) -> None:
    """Protect the compact Sec. VI two-bridge/RDM export workflow at L=8."""

    products = run_sec6_provisioning(
        Sec6ProvisioningConfig(
            output_dir=tmp_path,
            dense_sizes=(8,),
            fixed_half_widths=(1.0,),
            include_quarter_window=False,
            include_sqrt_window_for_dense=False,
            run_large_representative=False,
            run_family_large_size=False,
        )
    )

    bridges = products["bridge_distances"]
    assert set(bridges["bridge"]) == {
        "mc_to_beta0_resolved",
        "beta0_resolved_to_fixedM",
    }
    assert set(bridges["L"]) == {8}
    assert np.all(np.isfinite(bridges["trace_distance"]))
    assert np.all(bridges["trace_distance"] >= 0.0)

    coefficients = products["residual_coefficients"]
    for bridge in ("mc_to_beta0_resolved", "beta0_resolved_to_fixedM"):
        selected = coefficients[coefficients["bridge"] == bridge]
        assert len(selected) == 19
        norm = np.linalg.norm(
            selected["worst_hs_operator_coefficient_real"].to_numpy()
            + 1.0j * selected["worst_hs_operator_coefficient_imag"].to_numpy()
        )
        assert norm == pytest.approx(1.0, abs=1.0e-10)

    for filename in (
        "spin1_xy_kappa0p1_two_bridge_rdm_distance.csv",
        "spin1_xy_kappa0p1_residual_operator_spectrum.csv",
        "spin1_xy_kappa0p1_residual_operator_coefficients.csv",
    ):
        assert (tmp_path / filename).is_file()
