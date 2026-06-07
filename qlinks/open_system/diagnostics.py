from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.open_system.operators import lindblad_rhs_density_matrix


@dataclass(frozen=True, slots=True)
class EvolutionDiagnostics:
    trace_errors: np.ndarray
    hermiticity_errors: np.ndarray
    min_eigenvalues: np.ndarray
    purities: np.ndarray
    fidelities: np.ndarray | None
    lindblad_residuals: np.ndarray | None


def analyze_lindblad_evolution(
    density_matrices: list[np.ndarray],
    *,
    target_state: np.ndarray | None = None,
    hamiltonian=None,
    jumps=None,
    atol: float = 1e-10,
) -> EvolutionDiagnostics:
    density_diagnostics = [
        verify_density_matrix(
            density_matrix,
            target_state=target_state,
            atol=atol,
        )
        for density_matrix in density_matrices
    ]

    lindblad_residuals = None
    if hamiltonian is not None and jumps is not None:
        lindblad_residuals = np.array(
            [
                np.linalg.norm(
                    lindblad_rhs_density_matrix(
                        density_matrix,
                        hamiltonian=hamiltonian,
                        jumps=jumps,
                    )
                )
                for density_matrix in density_matrices
            ],
            dtype=np.float64,
        )

    fidelities = None
    if target_state is not None:
        fidelities = np.array(
            [diagnostic.fidelity_with_target for diagnostic in density_diagnostics],
            dtype=np.float64,
        )

    return EvolutionDiagnostics(
        trace_errors=np.array([d.trace_error for d in density_diagnostics]),
        hermiticity_errors=np.array([d.hermiticity_error for d in density_diagnostics]),
        min_eigenvalues=np.array([d.min_eigenvalue for d in density_diagnostics]),
        purities=np.array([d.purity for d in density_diagnostics]),
        fidelities=fidelities,
        lindblad_residuals=lindblad_residuals,
    )


@dataclass(frozen=True, slots=True)
class DensityMatrixVerification:
    trace: complex
    trace_error: float
    hermiticity_error: float
    min_eigenvalue: float
    purity: float
    fidelity_with_target: float | None
    is_hermitian: bool
    is_trace_one: bool
    is_positive_semidefinite: bool
    is_density_matrix: bool


def verify_density_matrix(
    rho: npt.ArrayLike,
    *,
    target_state: npt.ArrayLike | None = None,
    atol: float = 1e-10,
) -> DensityMatrixVerification:
    rho_array = np.asarray(rho, dtype=np.complex128)

    if rho_array.ndim != 2 or rho_array.shape[0] != rho_array.shape[1]:
        raise ValueError("rho must be a square matrix.")

    trace = np.trace(rho_array)
    trace_error = float(abs(trace - 1.0))

    hermitian_part = 0.5 * (rho_array + rho_array.conj().T)
    hermiticity_error = float(np.linalg.norm(rho_array - rho_array.conj().T))

    eigenvalues = np.linalg.eigvalsh(hermitian_part)
    min_eigenvalue = float(np.min(eigenvalues))

    purity_value = float(np.real(np.trace(rho_array @ rho_array)))

    fidelity = None
    if target_state is not None:
        psi = np.asarray(target_state, dtype=np.complex128)

        if psi.ndim != 1:
            raise ValueError("target_state must be one-dimensional.")

        norm = np.linalg.norm(psi)
        if norm == 0:
            raise ValueError("target_state must be nonzero.")

        psi = psi / norm
        fidelity = float(np.real(np.vdot(psi, rho_array @ psi)))

    is_hermitian = hermiticity_error <= atol
    is_trace_one = trace_error <= atol
    is_positive = min_eigenvalue >= -atol

    return DensityMatrixVerification(
        trace=complex(trace),
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        min_eigenvalue=min_eigenvalue,
        purity=purity_value,
        fidelity_with_target=fidelity,
        is_hermitian=is_hermitian,
        is_trace_one=is_trace_one,
        is_positive_semidefinite=is_positive,
        is_density_matrix=(is_hermitian and is_trace_one and is_positive),
    )


@dataclass(frozen=True, slots=True)
class LindbladFinalStateVerification:
    density_matrix: DensityMatrixVerification
    lindblad_residual: float
    relative_lindblad_residual: float


def verify_lindblad_final_state(
    rho: npt.ArrayLike,
    *,
    hamiltonian,
    jumps: list,
    target_state: npt.ArrayLike | None = None,
    atol: float = 1e-10,
) -> LindbladFinalStateVerification:
    rho_array = np.asarray(rho, dtype=np.complex128)

    density = verify_density_matrix(
        rho_array,
        target_state=target_state,
        atol=atol,
    )

    rhs = lindblad_rhs_density_matrix(
        rho_array,
        hamiltonian=hamiltonian,
        jumps=jumps,
    )

    residual = float(np.linalg.norm(rhs))
    relative = residual / max(1.0, float(np.linalg.norm(rho_array)))

    return LindbladFinalStateVerification(
        density_matrix=density,
        lindblad_residual=residual,
        relative_lindblad_residual=relative,
    )
