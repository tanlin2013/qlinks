import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.caging import invariant_boundary_nullspace


def test_invariant_boundary_nullspace_keeps_true_cage() -> None:
    internal_matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    boundary_matrix = np.array([[1.0, 1.0]], dtype=np.complex128)

    subspace_basis = invariant_boundary_nullspace(
        internal_matrix,
        boundary_matrix,
        tolerance=1e-12,
    )

    assert subspace_basis.shape == (2, 1)
    np.testing.assert_allclose(
        boundary_matrix @ subspace_basis,
        0.0,
        atol=1e-12,
    )


def test_invariant_boundary_nullspace_removes_non_invariant_vector() -> None:
    internal_matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    boundary_matrix = np.array([[1.0, 0.0]], dtype=np.complex128)

    subspace_basis = invariant_boundary_nullspace(
        internal_matrix,
        boundary_matrix,
        tolerance=1e-12,
    )

    assert subspace_basis.shape == (2, 0)


def test_invariant_boundary_nullspace_handles_sparse_input() -> None:
    internal_matrix = scipy_sparse.csr_matrix(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    boundary_matrix = scipy_sparse.csr_matrix(
        [[1.0, 1.0]],
        dtype=np.complex128,
    )

    subspace_basis = invariant_boundary_nullspace(
        internal_matrix,
        boundary_matrix,
        tolerance=1e-12,
    )

    assert subspace_basis.shape == (2, 1)
    np.testing.assert_allclose(
        boundary_matrix @ subspace_basis,
        0.0,
        atol=1e-12,
    )


def test_invariant_boundary_nullspace_rejects_shape_mismatch() -> None:
    internal_matrix = np.eye(2, dtype=np.complex128)
    boundary_matrix = np.ones((1, 3), dtype=np.complex128)

    try:
        invariant_boundary_nullspace(
            internal_matrix,
            boundary_matrix,
            tolerance=1e-12,
        )
    except ValueError as error:
        assert "same number of columns" in str(error)
    else:
        raise AssertionError("Expected ValueError.")
    