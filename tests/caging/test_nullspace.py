import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.caging import nullspace_svd


def test_nullspace_svd_finds_one_dimensional_nullspace() -> None:
    matrix = np.array([[1.0, 1.0]], dtype=np.complex128)

    nullspace_basis = nullspace_svd(matrix, tolerance=1e-12)

    assert nullspace_basis.shape == (2, 1)
    np.testing.assert_allclose(matrix @ nullspace_basis, 0.0, atol=1e-12)


def test_nullspace_svd_returns_empty_when_full_rank() -> None:
    matrix = np.eye(3, dtype=np.complex128)

    nullspace_basis = nullspace_svd(matrix, tolerance=1e-12)

    assert nullspace_basis.shape == (3, 0)


def test_nullspace_svd_handles_sparse_input() -> None:
    matrix = scipy_sparse.csr_matrix([[1.0, 1.0]])

    nullspace_basis = nullspace_svd(matrix, tolerance=1e-12)

    assert nullspace_basis.shape == (2, 1)
    np.testing.assert_allclose(
        matrix @ nullspace_basis,
        0.0,
        atol=1e-12,
    )


def test_nullspace_svd_empty_row_matrix_has_full_nullspace() -> None:
    matrix = np.zeros((0, 3), dtype=np.complex128)

    nullspace_basis = nullspace_svd(matrix, tolerance=1e-12)

    assert nullspace_basis.shape == (3, 3)
    np.testing.assert_allclose(
        nullspace_basis.conj().T @ nullspace_basis,
        np.eye(3),
        atol=1e-12,
    )
