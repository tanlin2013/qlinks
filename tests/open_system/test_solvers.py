import numpy as np
import pytest

from qlinks.open_system import (
    LindbladEvolutionOptions,
    density_matrix_from_state,
    initial_density_matrix,
    solve_lindblad,
    verify_density_matrix,
)


def test_solve_lindblad_krylov_preserves_hamiltonian_eigenstate():
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)
    state = np.array([1.0, 0.0], dtype=np.complex128)
    density_matrix_initial = density_matrix_from_state(state)

    times = np.linspace(0.0, 1.0, 5)

    result = solve_lindblad(
        hamiltonian=hamiltonian,
        jumps=[],
        density_matrix_initial=density_matrix_initial,
        times=times,
        method="krylov",
        backend="scipy",
    )

    for density_matrix in result.density_matrices:
        np.testing.assert_allclose(
            density_matrix,
            density_matrix_initial,
            atol=1e-12,
        )


def test_solve_lindblad_rk4_liouville_preserves_trace():
    hamiltonian = np.array(
        [[1.0, 0.2], [0.2, -1.0]],
        dtype=np.complex128,
    )
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    density_matrix_initial = initial_density_matrix(2, kind="mixed", rng=0)

    times = np.linspace(0.0, 0.1, 6)

    result = solve_lindblad(
        hamiltonian=hamiltonian,
        jumps=[jump],
        density_matrix_initial=density_matrix_initial,
        times=times,
        method="rk4_liouville",
        backend="scipy",
        options=LindbladEvolutionOptions(
            method="rk4_liouville",
            backend="scipy",
            rk4_step_policy="adaptive",
            adaptive_tolerance=1e-10,
        ),
    )

    for density_matrix in result.density_matrices:
        verification = verify_density_matrix(density_matrix, atol=1e-8)
        assert verification.trace_error < 1e-8
        assert verification.hermiticity_error < 1e-8


def test_rk4_step_policy_raise_rejects_large_step():
    hamiltonian = 100.0 * np.array(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=np.complex128,
    )
    density_matrix_initial = initial_density_matrix(2, kind="mixed", rng=0)
    times = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="RK4 time step may be too large"):
        solve_lindblad(
            hamiltonian=hamiltonian,
            jumps=[],
            density_matrix_initial=density_matrix_initial,
            times=times,
            method="rk4_matrix",
            backend="scipy",
            options=LindbladEvolutionOptions(
                method="rk4_matrix",
                backend="scipy",
                rk4_step_policy="raise",
                max_rk4_step_scale=0.05,
            ),
        )


def test_solve_lindblad_auto_returns_result():
    hamiltonian = np.diag([1.0, -1.0]).astype(np.complex128)
    density_matrix_initial = initial_density_matrix(2, kind="mixed", rng=0)
    times = np.linspace(0.0, 0.1, 3)

    result = solve_lindblad(
        hamiltonian=hamiltonian,
        jumps=[],
        density_matrix_initial=density_matrix_initial,
        times=times,
        method="auto",
        backend="scipy",
    )

    assert len(result.density_matrices) == len(times)
    assert result.method in {"krylov", "rk4_liouville", "rk4_matrix"}
