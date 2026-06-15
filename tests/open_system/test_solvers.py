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


def test_lindblad_problem_reuses_prepared_dense_operators(monkeypatch):
    import qlinks.open_system.solvers as solvers

    hamiltonian = np.array(
        [[1.0, 0.2], [0.2, -1.0]],
        dtype=np.complex128,
    )
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    density_matrix = initial_density_matrix(2, kind="mixed", rng=0)

    call_count = 0
    original_prepare = solvers.prepare_dense_lindblad_operators

    def counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(solvers, "prepare_dense_lindblad_operators", counted_prepare)

    problem = solvers.LindbladProblem(
        hamiltonian=hamiltonian,
        jumps=[jump],
        backend="scipy",
    )

    problem.rhs(density_matrix)
    problem.rhs(density_matrix)
    _ = problem.rk4_scale

    assert call_count == 1


def test_lindblad_problem_reuses_prepared_sparse_operators_and_liouvillian(monkeypatch):
    import qlinks.open_system.solvers as solvers

    hamiltonian = np.array(
        [[1.0, 0.2], [0.2, -1.0]],
        dtype=np.complex128,
    )
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )

    call_count = 0
    original_prepare = solvers.prepare_sparse_lindblad_operators

    def counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(solvers, "prepare_sparse_lindblad_operators", counted_prepare)

    problem = solvers.LindbladProblem(
        hamiltonian=hamiltonian,
        jumps=[jump],
        backend="scipy",
    )

    liouvillian_1 = problem.build_liouvillian(sparse_format="csr")
    liouvillian_2 = problem.build_liouvillian(sparse_format="csr")
    _ = problem.sparse_operators(sparse_format="csr")

    assert call_count == 1
    assert liouvillian_1 is liouvillian_2


def test_solve_lindblad_rk4_sparse_matrix_matches_liouville():
    hamiltonian = np.array(
        [[1.0, 0.2], [0.2, -1.0]],
        dtype=np.complex128,
    )
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    density_matrix_initial = initial_density_matrix(2, kind="mixed", rng=0)
    times = np.linspace(0.0, 0.05, 4)

    sparse_result = solve_lindblad(
        hamiltonian=hamiltonian,
        jumps=[jump],
        density_matrix_initial=density_matrix_initial,
        times=times,
        method="rk4_sparse_matrix",
        backend="scipy",
        options=LindbladEvolutionOptions(
            method="rk4_sparse_matrix",
            backend="scipy",
            rk4_step_policy="ignore",
        ),
    )
    liouville_result = solve_lindblad(
        hamiltonian=hamiltonian,
        jumps=[jump],
        density_matrix_initial=density_matrix_initial,
        times=times,
        method="rk4_liouville",
        backend="scipy",
        options=LindbladEvolutionOptions(
            method="rk4_liouville",
            backend="scipy",
            rk4_step_policy="ignore",
        ),
    )

    assert sparse_result.method == "rk4_sparse_matrix"
    np.testing.assert_allclose(
        sparse_result.density_matrices[-1],
        liouville_result.density_matrices[-1],
        atol=1e-12,
    )
