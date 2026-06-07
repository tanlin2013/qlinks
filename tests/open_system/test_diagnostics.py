import numpy as np
import pytest

from qlinks.open_system import (
    analyze_lindblad_evolution,
    density_matrix_from_state,
    verify_density_matrix,
    verify_lindblad_final_state,
)


def test_verify_density_matrix_for_target_pure_state():
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())

    result = verify_density_matrix(rho, target_state=psi)

    assert result.is_density_matrix
    assert result.fidelity_with_target == pytest.approx(1.0)
    assert result.purity == pytest.approx(1.0)
    assert result.trace_error < 1e-12
    assert result.hermiticity_error < 1e-12


def test_verify_lindblad_final_state_reports_lindblad_residual():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())

    result = verify_lindblad_final_state(
        rho,
        hamiltonian=sx,
        jumps=[],
        target_state=psi,
    )

    assert result.density_matrix.is_density_matrix
    assert result.lindblad_residual > 0.0


def test_verify_lindblad_final_state_accepts_hamiltonian_eigenstate():
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())

    result = verify_lindblad_final_state(
        rho,
        hamiltonian=sz,
        jumps=[],
        target_state=psi,
    )

    assert result.lindblad_residual < 1e-12
    assert result.density_matrix.fidelity_with_target == pytest.approx(1.0)


def test_analyze_lindblad_evolution_reports_fidelity():
    state = np.array([1.0, 0.0], dtype=np.complex128)
    density_matrix = density_matrix_from_state(state)

    diagnostics = analyze_lindblad_evolution(
        [density_matrix, density_matrix],
        target_state=state,
    )

    np.testing.assert_allclose(diagnostics.fidelities, [1.0, 1.0])
    np.testing.assert_allclose(diagnostics.purities, [1.0, 1.0])
