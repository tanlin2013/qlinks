import sys
import types

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

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


def test_scipy_backend_conversions_and_empty_max_abs() -> None:
    backend = ScipySparseBackend()

    array = backend.as_data_array([1.0, 2.0], dtype=np.float64)

    np.testing.assert_array_equal(backend.to_cpu_array(array), np.array([1.0, 2.0]))
    assert backend.max_abs_data(backend.empty_csr((2, 2), dtype=np.complex128)) == 0.0

    empty_data_matrix = scipy_sparse.csr_array(
        (np.asarray([], dtype=np.complex128), np.asarray([], dtype=np.int64), np.array([0, 0])),
        shape=(1, 1),
    )
    assert backend.max_abs_data(empty_data_matrix) == 0.0


def test_get_sparse_backend_accepts_existing_backend_object() -> None:
    backend = ScipySparseBackend()

    assert get_sparse_backend(backend) is backend


def test_cupy_backend_reports_missing_modules(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", None)

    with pytest.raises(ImportError, match="requires CuPy"):
        CupySparseBackend().cp

    fake_cupy = types.ModuleType("cupy")
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse", None)

    with pytest.raises(ImportError, match="cupyx.scipy.sparse"):
        CupySparseBackend().cupyx_sparse


def test_cupy_backend_with_fake_modules(monkeypatch) -> None:
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.asarray = np.asarray
    fake_cupy.asnumpy = np.asarray
    fake_cupy.abs = np.abs
    fake_cupy.max = np.max
    fake_cupy.int64 = np.int64
    fake_cupy.complex128 = np.complex128

    fake_sparse = types.ModuleType("cupyx.scipy.sparse")
    fake_sparse.coo_matrix = scipy_sparse.coo_array
    fake_sparse.csr_matrix = scipy_sparse.csr_array

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx", types.ModuleType("cupyx"))
    monkeypatch.setitem(sys.modules, "cupyx.scipy", types.ModuleType("cupyx.scipy"))
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse", fake_sparse)

    backend = get_sparse_backend("cupy")
    assert isinstance(backend, CupySparseBackend)

    data = backend.as_data_array([1.0 + 1.0j, -3.0], dtype=np.complex128)
    rows = backend.as_index_array([0, 1])
    cols = backend.as_index_array([1, 0])
    matrix = backend.coo_matrix(
        data=data,
        rows=rows,
        cols=cols,
        shape=(2, 2),
        dtype=np.complex128,
    )

    np.testing.assert_allclose(
        backend.to_cpu_array(matrix.toarray()),
        np.asarray([[0.0, 1.0 + 1.0j], [-3.0, 0.0]], dtype=np.complex128),
    )
    assert backend.max_abs_data(matrix) == pytest.approx(3.0)
    assert backend.max_abs_data(backend.empty_csr((2, 2), dtype=np.complex128)) == 0.0
