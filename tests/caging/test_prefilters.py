import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.caging import (
    boundary_nullity,
    extract_subblocks,
    has_uniform_diagonal,
    passes_basic_prefilters,
)


def test_extract_subblocks_dense() -> None:
    hamiltonian = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.complex128,
    )
    support_indices = np.array([0, 1])

    internal_matrix, boundary_matrix, outside_indices = extract_subblocks(
        hamiltonian,
        support_indices,
    )

    np.testing.assert_array_equal(outside_indices, np.array([2]))
    np.testing.assert_allclose(
        internal_matrix,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    np.testing.assert_allclose(
        boundary_matrix,
        np.array([[2.0, 3.0]]),
    )


def test_extract_subblocks_sparse() -> None:
    hamiltonian = scipy_sparse.csr_matrix(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.complex128,
    )
    support_indices = np.array([0, 1])

    internal_matrix, boundary_matrix, outside_indices = extract_subblocks(
        hamiltonian,
        support_indices,
    )

    np.testing.assert_array_equal(outside_indices, np.array([2]))
    np.testing.assert_allclose(
        internal_matrix.toarray(),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    np.testing.assert_allclose(
        boundary_matrix.toarray(),
        np.array([[2.0, 3.0]]),
    )


def test_has_uniform_diagonal() -> None:
    hamiltonian = np.diag([2.0, 2.0, 5.0]).astype(np.complex128)

    assert has_uniform_diagonal(
        hamiltonian,
        np.array([0, 1]),
        tolerance=1e-12,
    )

    assert not has_uniform_diagonal(
        hamiltonian,
        np.array([0, 2]),
        tolerance=1e-12,
    )


def test_boundary_nullity() -> None:
    boundary_matrix = np.array([[1.0, 1.0]], dtype=np.complex128)

    assert boundary_nullity(boundary_matrix, tolerance=1e-12) == 1


def test_passes_basic_prefilters_accepts_valid_candidate() -> None:
    hamiltonian = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )

    assert passes_basic_prefilters(
        hamiltonian,
        np.array([0, 1]),
        tolerance=1e-12,
        require_uniform_diagonal=True,
    )


def test_passes_basic_prefilters_rejects_zero_boundary_nullity() -> None:
    hamiltonian = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    assert not passes_basic_prefilters(
        hamiltonian,
        np.array([0, 1]),
        tolerance=1e-12,
    )
