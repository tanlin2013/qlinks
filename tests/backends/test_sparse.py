import numpy as np
import pytest

from qlinks.backends import CupySparseBackend, ScipySparseBackend, get_sparse_backend


def test_get_scipy_backend() -> None:
    backend = get_sparse_backend("scipy")

    assert isinstance(backend, ScipySparseBackend)
    assert backend.name == "scipy"


def test_get_auto_backend() -> None:
    backend = get_sparse_backend("auto")

    assert isinstance(backend, ScipySparseBackend)
    assert backend.name == "scipy"


def test_get_invalid_backend() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        get_sparse_backend("bad")  # type: ignore[arg-type]


def test_scipy_backend_builds_csr() -> None:
    backend = ScipySparseBackend()

    data = backend.as_data_array([1.0, 2.0], dtype=np.complex128)
    rows = backend.as_index_array([0, 1])
    cols = backend.as_index_array([1, 0])

    matrix = backend.coo_matrix(
        data=data,
        rows=rows,
        cols=cols,
        shape=(2, 2),
        dtype=np.complex128,
    )

    expected = np.array(
        [
            [0, 1],
            [2, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(matrix.toarray(), expected)


def test_scipy_backend_empty_csr() -> None:
    backend = ScipySparseBackend()

    matrix = backend.empty_csr((3, 3), dtype=np.complex128)

    assert matrix.shape == (3, 3)
    assert matrix.nnz == 0


def test_scipy_backend_max_abs_data() -> None:
    backend = ScipySparseBackend()

    data = backend.as_data_array([1.0, -3.0], dtype=np.complex128)
    rows = backend.as_index_array([0, 1])
    cols = backend.as_index_array([0, 1])

    matrix = backend.coo_matrix(
        data=data,
        rows=rows,
        cols=cols,
        shape=(2, 2),
        dtype=np.complex128,
    )

    assert backend.max_abs_data(matrix) == 3.0


def test_cupy_backend_import_or_skip() -> None:
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cupyx.scipy.sparse")

    backend = CupySparseBackend()

    data = backend.as_data_array([1.0, 2.0], dtype=cp.complex128)
    rows = backend.as_index_array([0, 1])
    cols = backend.as_index_array([1, 0])

    matrix = backend.coo_matrix(
        data=data,
        rows=rows,
        cols=cols,
        shape=(2, 2),
        dtype=cp.complex128,
    )

    expected = np.array(
        [
            [0, 1],
            [2, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(cp.asnumpy(matrix.toarray()), expected)
