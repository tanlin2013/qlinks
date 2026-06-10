import numpy as np

from qlinks.open_system.local_recycling import (
    local_reduced_density_matrix_from_state,
    scan_local_recycling_candidates,
)


def test_local_reduced_density_matrix_product_state_has_local_nullspace():
    # Two qubits, constrained basis is full product basis.
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )

    # |00>
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    rdm = local_reduced_density_matrix_from_state(
        basis_configs=basis_configs,
        state=state,
        variable_indices=(0,),
    )

    assert rdm.local_dim == 2
    assert rdm.support_rank == 1
    assert rdm.nullity == 1


def test_scan_local_recycling_candidates_finds_product_state_pump():
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )

    # |00>
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    result = scan_local_recycling_candidates(
        basis_configs=basis_configs,
        target_state=state,
        variable_indices=(0,),
        inflow_tolerance=1e-12,
    )

    assert result.reduced_density_matrix.support_rank == 1
    assert result.reduced_density_matrix.nullity == 1
    assert result.n_candidates >= 1

    candidate = result.best_candidates[0]
    assert candidate.target_residual < 1e-12
    assert candidate.inflow_norm > 0.0


def test_local_reduced_density_matrix_bell_state_has_full_one_site_support():
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )

    # (|00> + |11>) / sqrt(2)
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    state = state / np.linalg.norm(state)

    rdm = local_reduced_density_matrix_from_state(
        basis_configs=basis_configs,
        state=state,
        variable_indices=(0,),
    )

    assert rdm.support_rank == 2
    assert rdm.nullity == 0
