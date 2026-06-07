import numpy as np
import pytest

from qlinks.open_system import initial_density_matrix, solve_lindblad


def test_solve_lindblad_rk4_matrix_cupy_backend():
    pytest.importorskip("cupy")
    pytest.importorskip("cupyx.scipy.sparse")

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
