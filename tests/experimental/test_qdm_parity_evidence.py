"""Behavioural contracts for measured target parity and bounded character counts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experimental/jobs"))
from qdm_character_dimensions import checkerboard_character_dimensions  # noqa: E402
from qdm_parity_protocol import (  # noqa: E402
    PROTOCOL,
    require_parity_protocol,
    validate_partial_eigensystem,
)
from qdm_sec7_fixed_o1 import (  # noqa: E402
    build_context,
    checkerboard_instance,
    recover_reference_geometry,
)
from qdm_target_parity import compact_target_parity  # noqa: E402


@pytest.fixture(scope="module")
def reference():
    return recover_reference_geometry()


@pytest.mark.integration
@pytest.mark.parametrize("repeats,parity", [(1, -1), (2, 1), (3, -1)])
def test_target_irrep_is_measured_from_closed_support(reference, repeats, parity):
    """Only 16/64/256 support-orbit states are needed, including on the 12x4 torus."""
    result = compact_target_parity(checkerboard_instance(reference, repeats, 0.05), repeats=repeats)
    assert result["Sy"] == parity
    assert result["target_Sy_residual"] < 1e-12
    assert result["support_orbit_size"] <= 256


@pytest.mark.integration
@pytest.mark.parametrize("lx,dimensions", [(4, (14, 1)), (8, (875, 250)), (12, (74816, 39667))])
def test_integer_character_count_recovers_both_irrep_multiplicities(lx, dimensions):
    """Exact-cover counting performs no full basis generation or eigensolve."""
    result = checkerboard_character_dimensions(lx)
    assert result["Sy_dimensions"] == dict(zip(("1", "-1"), dimensions, strict=True))
    assert result["translation_dimension"] == sum(dimensions)


@pytest.mark.integration
def test_tiny_target_sector_is_one_dimensional_and_rejects_wrong_parity(reference):
    context = build_context(reference=reference, repeats=1, sy_character="target")
    assert context.sector.sector_dimension == 1
    assert context.sector.labels["Sy_character"] == -1
    assert context.tower_residual < 1e-12
    assert max(context.cage_q.values()) < 1e-12
    with pytest.raises(RuntimeError, match="zero weight"):
        build_context(reference=reference, repeats=1, sy_character=1)
    with pytest.raises(ValueError, match="phase"):
        build_context(reference=reference, repeats=1, phase=0, sy_character="target")


def test_partial_validation_recomputes_all_vectors_and_cross_chunk_gram_entries():
    h = np.zeros((12, 12))
    vectors = np.eye(12)[:, :8]
    energies = np.zeros(8)
    validate_partial_eigensystem(h, energies, vectors, chunk_size=3)
    # Degeneracy keeps residuals zero; only the cross-chunk Gram test detects this.
    vectors[:, 7] = vectors[:, 0]
    with pytest.raises(ValueError, match="Gram"):
        validate_partial_eigensystem(h, energies, vectors, chunk_size=3)
    vectors = np.eye(12)[:, :8]
    with pytest.raises(ValueError, match="residual"):
        validate_partial_eigensystem(np.diag(np.arange(12.0)), energies, vectors, chunk_size=3)


@pytest.mark.parametrize(
    "payload", [{}, {"symmetry_protocol": PROTOCOL}, {"Sy": 1, "symmetry_protocol": "legacy"}]
)
def test_protocol_rejects_legacy_and_unverified_parity(payload):
    with pytest.raises(ValueError, match="target-parity"):
        require_parity_protocol(payload, description="canonical source")
    require_parity_protocol(
        {"symmetry_protocol": PROTOCOL, "parity_policy": "measured_target"},
        description="measured target source",
    )


@pytest.mark.parametrize(
    "failure", [None, "missing_control", "wrong_parity", "bad_vector", "unstable_width"]
)
def test_submission_controls_require_all_windows_and_two_valid_budgets(failure):
    import pandas as pd
    from qdm_parity_protocol import window_controls_stable

    rows = [
        {
            "requested_subspace_size": budget,
            "window_half_width": width,
            "Sy": -1,
            "symmetry_protocol": PROTOCOL,
            "raw_window_state_count": 30,
            "maximum_eigenpair_residual": 1e-10,
            "gram_frobenius_residual": 1e-10,
            "cage_window_projector_weight": 1.0,
            "tau_A_mc_raw": 0.1,
            "tau_Z_mc_raw": 0.2,
            "w_raw": 0.03,
        }
        for budget in (768, 1024)
        for width in (0.1, 0.2, 0.25, 0.5)
    ]
    frame = pd.DataFrame(rows)
    if failure == "missing_control":
        frame = frame[frame.window_half_width < 0.5]
    elif failure == "wrong_parity":
        frame.loc[0, "Sy"] = 1
    elif failure == "bad_vector":
        frame.loc[0, "maximum_eigenpair_residual"] = 0.01
    elif failure == "unstable_width":
        frame.loc[7, "w_raw"] = 0.3
    assert window_controls_stable(frame) is (failure is None)
