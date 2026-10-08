"""Regression contracts for missed torus reflection and provenance interpretation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experimental/jobs"))
from qdm_checkerboard_large_strip import packed_binary_basis_index  # noqa: E402
from qdm_checkerboard_symmetry import (  # noqa: E402
    checkerboard_fully_resolved_sector,
    checkerboard_translation_sector,
)
from qdm_followup_provenance import compare_populations, validate_eigensystem  # noqa: E402

from qlinks.models import SquareQDMModel


@pytest.mark.integration
def test_shifted_reflection_blocks_complete_tiny_torus_translation_irrep():
    """4x4 is the smallest torus supporting Tdiag=i and the shifted mirrors."""
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
    )
    build = model.build(basis_solver="dfs", builder="bitmask", sort_basis=True)
    index = packed_binary_basis_index(build.basis)
    legacy, perms = checkerboard_translation_sector(
        model, build.basis, packed_index=index, repeats=1
    )
    b = legacy.sector.basis
    # Sy Tdiag Sy = Tdiag Ty2^-1, which preserves this character.
    sy, td, ty2 = (perms[name] for name in ("Sy", "Tdiag", "Ty2"))
    np.testing.assert_array_equal(sy[td[sy]], td[ty2])
    projector = sp.csr_array((b.shape[0], b.shape[0]), dtype=complex)
    dims = []
    for parity in (1, -1):
        resolved, _ = checkerboard_fully_resolved_sector(
            model, build.basis, packed_index=index, repeats=1, sy_character=parity
        )
        c = resolved.sector.basis
        # U|i> = |p[i]>; permutation action on rows uses the inverse.
        np.testing.assert_allclose(c[np.argsort(sy)].toarray(), parity * c.toarray(), atol=1.0e-12)
        dims.append(c.shape[1])
        projector += c @ c.conj().T
    assert sum(dims) == b.shape[1]
    assert all(dims)
    assert float(sp.linalg.norm(projector - b @ b.conj().T)) < 1.0e-12
    assert legacy.sector.labels["fully_symmetry_resolved"] is False
    with pytest.raises(ValueError, match="requires sy_character"):
        checkerboard_fully_resolved_sector(model, build.basis, packed_index=index, repeats=1)


def test_degenerate_population_rotation_is_distinguished_from_changed_operator():
    source = pd.DataFrame(
        {"energy": [0.0, 0.0, 1.0], "Q_A": [0.0, 1.0, 2.0], "Q_Z": [0.0, 2.0, 3.0]}
    )
    rotated = pd.DataFrame(
        {"energy": [0.0, 0.0, 1.0], "Q_A": [0.5, 0.5, 2.0], "Q_Z": [1.0, 1.0, 3.0]}
    )
    result = compare_populations(source, rotated)
    assert result["population_matches"] is False
    assert result["interpretation"] == "compatible_with_degenerate_basis_rotation"
    rotated.loc[0, "Q_A"] += 0.1
    assert compare_populations(source, rotated)["interpretation"] == "witness_or_exclusion_mismatch"
    rotated.loc[0, "energy"] += 0.1
    assert compare_populations(source, rotated)["interpretation"] == "spectrum_mismatch"
    assert compare_populations(None, source)["population_matches"] is None


def test_complete_validation_catches_an_unsampled_bad_eigenvector():
    h = np.diag(np.arange(16.0))
    e, v = np.arange(16.0), np.eye(16)
    validate_eigensystem(h, e, v)
    v[:, 7] = v[:, 6]
    with pytest.raises(ValueError, match="validation failed"):
        validate_eigensystem(h, e, v)
