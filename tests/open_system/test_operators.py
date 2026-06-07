import numpy as np

from qlinks.open_system import (
    build_liouvillian,
    initial_density_matrix,
    lindblad_rhs_density_matrix,
    vectorize_density_matrix,
)


def test_lindblad_rhs_preserves_trace_for_density_matrix():
    hamiltonian = np.array(
        [[1.0, 0.2], [0.2, -1.0]],
        dtype=np.complex128,
    )
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    density_matrix = initial_density_matrix(2, kind="mixed", rng=0)

    derivative = lindblad_rhs_density_matrix(
        density_matrix,
        hamiltonian=hamiltonian,
        jumps=[jump],
    )

    assert abs(np.trace(derivative)) < 1e-12


def test_build_liouvillian_matches_matrix_rhs():
    hamiltonian = np.array(
        [[1.0, 0.2], [0.2, -1.0]],
        dtype=np.complex128,
    )
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    density_matrix = initial_density_matrix(2, kind="mixed", rng=0)

    liouvillian = build_liouvillian(
        hamiltonian,
        [jump],
        backend="scipy",
    )

    rhs_matrix = lindblad_rhs_density_matrix(
        density_matrix,
        hamiltonian=hamiltonian,
        jumps=[jump],
    )
    rhs_vector = liouvillian @ vectorize_density_matrix(density_matrix)

    np.testing.assert_allclose(
        rhs_vector,
        vectorize_density_matrix(rhs_matrix),
        atol=1e-12,
    )


def test_lindblad_rhs_accepts_sparse_hamiltonian_and_jumps():
    import scipy.sparse as sp

    from qlinks.open_system import lindblad_rhs_density_matrix
    from qlinks.open_system.states import density_matrix_from_state

    hamiltonian = sp.csr_array(
        np.array(
            [[1.0, 0.2], [0.2, -1.0]],
            dtype=np.complex128,
        )
    )
    jump = sp.csr_array(
        np.array(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=np.complex128,
        )
    )

    state = np.array([1.0, 0.0], dtype=np.complex128)
    density_matrix = density_matrix_from_state(state)

    rhs = lindblad_rhs_density_matrix(
        density_matrix,
        hamiltonian=hamiltonian,
        jumps=[jump],
        backend="scipy",
    )

    assert rhs.shape == (2, 2)
    assert rhs.dtype == np.complex128
    assert abs(np.trace(rhs)) < 1e-12
