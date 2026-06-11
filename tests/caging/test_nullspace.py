import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.caging import nullspace_from_gram, nullspace_svd


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


def test_nullspace_from_gram_matches_svd_nullspace() -> None:
    matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.complex128,
    )
    gram_matrix = matrix.conj().T @ matrix

    svd_basis = nullspace_svd(matrix, tolerance=1e-12)
    gram_basis = nullspace_from_gram(gram_matrix, tolerance=1e-12)

    assert gram_basis.shape == (3, 1)
    np.testing.assert_allclose(matrix @ gram_basis, 0.0, atol=1e-12)

    overlap = abs(np.vdot(svd_basis[:, 0], gram_basis[:, 0]))
    np.testing.assert_allclose(overlap, 1.0, atol=1e-12)


def test_nullspace_from_gram_rejects_nonsquare_matrix() -> None:
    matrix = np.zeros((2, 3), dtype=np.complex128)

    with pytest.raises(ValueError, match="square"):
        nullspace_from_gram(matrix)
