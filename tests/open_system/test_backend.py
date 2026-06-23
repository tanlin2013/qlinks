import sys
import types

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.open_system import (
    as_backend_dense_array,
    as_backend_sparse_matrix,
    get_open_system_backend,
    initial_density_matrix,
    solve_lindblad,
)


def _require_functional_cupy():
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("cupyx.scipy.sparse")

    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
        test_array = cupy.asarray([1.0])
        cupy.asnumpy(test_array)
    except Exception as exc:  # pragma: no cover - depends on CI GPU runtime.
        pytest.skip(f"CuPy is installed but no functional CUDA runtime is available: {exc}")

    if device_count <= 0:  # pragma: no cover - depends on CI GPU runtime.
        pytest.skip("CuPy is installed but no CUDA device is available.")

    return cupy


def test_get_open_system_backend_accepts_existing_backend_object() -> None:
    backend = get_open_system_backend("scipy")

    assert get_open_system_backend(backend) is backend


def test_get_open_system_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported open-system backend"):
        get_open_system_backend("bad-backend")  # type: ignore[arg-type]


def test_scipy_backend_converts_dense_and_sparse_inputs() -> None:
    backend = get_open_system_backend("scipy")
    matrix = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    sparse_matrix = as_backend_sparse_matrix(matrix, backend=backend)
    dense_matrix = as_backend_dense_array(scipy_sparse.csr_array(matrix), backend=backend)

    assert scipy_sparse.issparse(sparse_matrix)
    np.testing.assert_allclose(sparse_matrix.toarray(), matrix)
    np.testing.assert_allclose(dense_matrix, matrix)


def test_cupy_backend_resolution_and_converters_with_fake_modules(monkeypatch) -> None:
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.asarray = np.asarray
    fake_cupy.asnumpy = np.asarray
    fake_cupy.linalg = np.linalg

    fake_sparse = types.ModuleType("cupyx.scipy.sparse")
    fake_sparse.csr_matrix = scipy_sparse.csr_array
    fake_sparse.identity = scipy_sparse.identity
    fake_sparse.kron = scipy_sparse.kron

    fake_sparse_linalg = types.ModuleType("cupyx.scipy.sparse.linalg")
    fake_sparse_linalg.expm_multiply = object()

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx", types.ModuleType("cupyx"))
    monkeypatch.setitem(sys.modules, "cupyx.scipy", types.ModuleType("cupyx.scipy"))
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse", fake_sparse)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse.linalg", fake_sparse_linalg)

    backend = get_open_system_backend("cupy")
    matrix = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    assert backend.name == "cupy"
    assert backend.supports_expm_multiply is True
    np.testing.assert_allclose(backend.to_numpy(backend.asarray([1.0, 2.0])), [1.0, 2.0])
    assert backend.norm(np.asarray([3.0, 4.0])) == pytest.approx(5.0)
    np.testing.assert_allclose(backend.sparse_identity(2).toarray(), np.eye(2))
    np.testing.assert_allclose(backend.sparse_kron(np.eye(1), np.eye(2)).toarray(), np.eye(2))

    sparse_matrix = as_backend_sparse_matrix(matrix, backend=backend)
    dense_matrix = as_backend_dense_array(scipy_sparse.csr_array(matrix), backend=backend)

    np.testing.assert_allclose(sparse_matrix.toarray(), matrix)
    np.testing.assert_allclose(dense_matrix, matrix)


def test_solve_lindblad_rk4_matrix_cupy_backend():
    _require_functional_cupy()

    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)
    density_matrix_initial = initial_density_matrix(2, kind="mixed", rng=0)
    times = np.linspace(0.0, 0.1, 3)

    result = solve_lindblad(
        hamiltonian=hamiltonian,
        jumps=[],
        density_matrix_initial=density_matrix_initial,
        times=times,
        method="rk4_matrix",
        backend="cupy",
    )

    assert len(result.density_matrices) == len(times)
