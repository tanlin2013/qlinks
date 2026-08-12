"""Behaviour and compatibility checks for neutral local-structure primitives."""

from __future__ import annotations

import numpy as np

import qlinks.open_system as open_system
from qlinks.local_structure import (
    LocalMatrixUnitTerm,
    LocalReducedDensityMatrix,
    embed_local_pattern_operator,
    local_operator_matrix_unit_expansion,
    local_reduced_density_matrix_from_state,
)


def test_open_system_keeps_compatibility_aliases_for_moved_local_primitives() -> None:
    assert open_system.LocalReducedDensityMatrix is LocalReducedDensityMatrix
    assert open_system.LocalMatrixUnitTerm is LocalMatrixUnitTerm
    assert open_system.embed_local_pattern_operator is embed_local_pattern_operator
    assert open_system.local_operator_matrix_unit_expansion is local_operator_matrix_unit_expansion
    assert (
        open_system.local_reduced_density_matrix_from_state
        is local_reduced_density_matrix_from_state
    )


def test_local_rdm_and_embedding_share_the_same_constrained_pattern_convention() -> None:
    basis_configs = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )
    state = np.asarray([1.0, 1.0j, 0.0], dtype=np.complex128)

    rdm = local_reduced_density_matrix_from_state(
        basis_configs=basis_configs,
        state=state,
        variable_indices=(0,),
    )
    operator = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    embedded = embed_local_pattern_operator(
        basis_configs=basis_configs,
        variable_indices=(0,),
        local_patterns=rdm.local_patterns,
        local_operator=operator,
    )

    assert rdm.local_patterns == ((0,), (1,))
    np.testing.assert_allclose(np.trace(rdm.density_matrix), 1.0)
    np.testing.assert_allclose(
        embedded.toarray(),
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        ),
    )
