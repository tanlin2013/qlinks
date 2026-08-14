from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.open_system.backend import OpenSystemBackend, OpenSystemBackendName
from qlinks.open_system.operators import lindblad_rhs_density_matrix


@dataclass(frozen=True, slots=True)
class DensityMatrixVerification:
    """Numerical checks for a candidate density matrix.

    Attributes:
        trace: Matrix trace.
        trace_error: Absolute error of the trace from one.
        hermiticity_error: Frobenius norm of ``rho - rho^dagger``.
        min_eigenvalue: Smallest Hermitian eigenvalue after symmetrization.
        purity: ``Tr(rho^2)``.
        fidelity_with_target: Optional fidelity ``<psi|rho|psi>``.
        is_hermitian: Whether the Hermiticity check passes.
        is_trace_one: Whether the trace-one check passes.
        is_positive_semidefinite: Whether the minimum eigenvalue is above
            ``-atol``.
        is_density_matrix: Combined validity flag.
    """

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
    """Check whether an array is a valid density matrix.

    Args:
        rho: Candidate density matrix.
        target_state: Optional pure state used to compute ``<psi|rho|psi>``.
        atol: Absolute tolerance for trace, Hermiticity, and positivity checks.

    Returns:
        Verification record with scalar diagnostics and boolean flags.

    Raises:
        ValueError: If ``rho`` is not square or ``target_state`` is invalid.
    """
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
    """Verification of a final Lindblad density matrix.

    Attributes:
        density_matrix: Basic density-matrix validity diagnostics.
        lindblad_residual: Norm of the Lindblad RHS evaluated at the final
            state.
        relative_lindblad_residual: Residual normalized by the density-matrix
            norm scale.
    """

    density_matrix: DensityMatrixVerification
    lindblad_residual: float
    relative_lindblad_residual: float


def verify_lindblad_final_state(
    rho: npt.ArrayLike,
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_state: npt.ArrayLike | None = None,
    atol: float = 1e-10,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
) -> LindbladFinalStateVerification:
    """Verify density-matrix validity and Lindblad stationarity.

    Args:
        rho: Candidate final density matrix.
        hamiltonian: Hamiltonian matrix.
        jumps: Lindblad jump operators.
        target_state: Optional pure state for fidelity diagnostics.
        atol: Absolute tolerance passed to :func:`verify_density_matrix`.
        backend: Open-system backend name or object.

    Returns:
        Final-state verification record.
    """
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
        backend=backend,
    )

    residual = float(np.linalg.norm(rhs))
    relative = residual / max(1.0, float(np.linalg.norm(rho_array)))

    return LindbladFinalStateVerification(
        density_matrix=density,
        lindblad_residual=residual,
        relative_lindblad_residual=relative,
    )
