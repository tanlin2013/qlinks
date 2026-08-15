from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.open_system._subspace import (
    _as_scipy_csr_matrix,
    _common_kernel_basis_from_sparse_operators,
    _nullspace_basis,
    _orthonormal_column_basis,
    _orthonormal_target_state_matrix,
)
from qlinks.open_system.diagnostics.dark import diagnose_common_kernel_h_invariant_sector

_ATTRACTIVE_SUBSPACE_ALGORITHM = "no_inflow_common_invariant_fixed_point_v1"


@dataclass(frozen=True, slots=True)
class AttractiveSubspaceDiagnostics:
    """Finite-dimensional certificate for attraction into a dark target subspace.

    The target must be invariant under ``H`` and dark under every jump.  The
    diagnostic then forms the no-inflow space

    ``N = cap_mu ker(P_T J_mu (I-P_T)) cap T^perp``

    and computes the largest subspace of ``N`` invariant under every jump and
    the no-jump generator ``G = -i H - 1/2 sum_mu J_mu^dagger J_mu``.
    Attraction is certified when the target preconditions pass and this
    invariant obstruction is empty.
    """

    hilbert_dimension: int
    target_dimension: int
    n_jumps: int
    tolerance: float
    algorithm: str

    hamiltonian_target_invariance_residual: float
    jump_darkness_residuals: tuple[float, ...]
    max_jump_darkness_residual: float
    target_directed_block_norms: tuple[float, ...]
    total_target_directed_inflow_norm: float

    no_inflow_dimension: int
    invariant_obstruction_dimension: int
    invariant_fixed_point_iterations: int
    jump_invariance_residuals: tuple[float, ...]
    no_jump_generator_invariance_residual: float
    max_invariant_obstruction_residual: float

    common_jump_kernel_dimension: int
    old_h_invariant_kernel_dimension: int

    target_hamiltonian_invariant: bool
    target_dark: bool
    obstruction_free: bool
    target_attractive_certified: bool

    no_inflow_basis: npt.NDArray[np.complex128]
    invariant_obstruction_basis: npt.NDArray[np.complex128]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "target_dimension": self.target_dimension,
            "n_jumps": self.n_jumps,
            "tolerance": self.tolerance,
            "algorithm": self.algorithm,
            "hamiltonian_target_invariance_residual": (self.hamiltonian_target_invariance_residual),
            "jump_darkness_residuals": self.jump_darkness_residuals,
            "max_jump_darkness_residual": self.max_jump_darkness_residual,
            "target_directed_block_norms": self.target_directed_block_norms,
            "total_target_directed_inflow_norm": self.total_target_directed_inflow_norm,
            "no_inflow_dimension": self.no_inflow_dimension,
            "invariant_obstruction_dimension": self.invariant_obstruction_dimension,
            "invariant_fixed_point_iterations": self.invariant_fixed_point_iterations,
            "jump_invariance_residuals": self.jump_invariance_residuals,
            "no_jump_generator_invariance_residual": (self.no_jump_generator_invariance_residual),
            "max_invariant_obstruction_residual": self.max_invariant_obstruction_residual,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "old_h_invariant_kernel_dimension": self.old_h_invariant_kernel_dimension,
            "target_hamiltonian_invariant": self.target_hamiltonian_invariant,
            "target_dark": self.target_dark,
            "obstruction_free": self.obstruction_free,
            "target_attractive_certified": self.target_attractive_certified,
            "no_inflow_basis_shape": tuple(int(value) for value in self.no_inflow_basis.shape),
            "invariant_obstruction_basis_shape": tuple(
                int(value) for value in self.invariant_obstruction_basis.shape
            ),
        }


def _orthogonal_complement_basis(
    target_basis: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    dim = int(target_basis.shape[0])
    if target_basis.shape[1] == dim:
        return np.zeros((dim, 0), dtype=np.complex128)
    coefficients = _nullspace_basis(target_basis.conj().T, tolerance=tolerance)
    return _orthonormal_column_basis(coefficients, tolerance=tolerance)


def _target_basis_from_projector(
    projector: npt.ArrayLike,
    *,
    tolerance: float,
) -> np.ndarray:
    matrix = np.asarray(projector, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("projector must be a square matrix.")
    hermiticity_residual = float(np.linalg.norm(matrix - matrix.conj().T))
    idempotency_residual = float(np.linalg.norm(matrix @ matrix - matrix))
    if hermiticity_residual > tolerance or idempotency_residual > tolerance:
        raise ValueError("projector must be Hermitian and idempotent within tolerance.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    keep = eigenvalues > 0.5
    if not np.any(keep):
        raise ValueError("projector must have positive rank.")
    return eigenvectors[:, keep].astype(np.complex128, copy=False)


def _subspace_leakage(
    operator: sp.csr_array,
    basis: np.ndarray,
) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((basis.shape[0], 0), dtype=np.complex128)
    action = np.asarray(operator @ basis, dtype=np.complex128)
    return action - basis @ (basis.conj().T @ action)


def _largest_common_invariant_subspace(
    *,
    initial_basis: np.ndarray,
    operators: tuple[sp.csr_array, ...],
    tolerance: float,
) -> tuple[np.ndarray, int]:
    basis = _orthonormal_column_basis(initial_basis, tolerance=tolerance)
    iterations = 0
    while basis.shape[1] > 0:
        leakage_blocks = [_subspace_leakage(operator, basis) for operator in operators]
        stacked = np.vstack(leakage_blocks)
        coefficients = _nullspace_basis(stacked, tolerance=tolerance)
        next_basis = _orthonormal_column_basis(basis @ coefficients, tolerance=tolerance)
        iterations += 1
        if next_basis.shape[1] == basis.shape[1]:
            return next_basis, iterations
        basis = next_basis
    return basis, iterations


def diagnose_attractive_subspace(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_basis: npt.ArrayLike | None = None,
    projector: npt.ArrayLike | None = None,
    tolerance: float = 1.0e-10,
) -> AttractiveSubspaceDiagnostics:
    """Certify attraction into an invariant dark target subspace.

    Exactly one of ``target_basis`` or ``projector`` must be supplied.  The
    theorem-based Hilbert-space certificate is independent of a Liouvillian
    eigensolve and is intended to be the primary finite-size P0 diagnostic for
    caging-generated jump families.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if (target_basis is None) == (projector is None):
        raise ValueError("Pass exactly one of target_basis or projector.")

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    dim = int(hamiltonian_sparse.shape[0])
    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    if target_basis is not None:
        target = _orthonormal_target_state_matrix(
            target_basis,
            dim=dim,
            tolerance=tolerance,
        )
    else:
        assert projector is not None
        target = _target_basis_from_projector(projector, tolerance=tolerance)
        if target.shape[0] != dim:
            raise ValueError("projector has incompatible Hilbert-space dimension.")

    jump_matrices = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)
    for jump in jump_matrices:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    target_action = np.asarray(hamiltonian_sparse @ target, dtype=np.complex128)
    h_target_leakage = target_action - target @ (target.conj().T @ target_action)
    h_target_residual = float(np.linalg.norm(h_target_leakage))

    jump_darkness = tuple(
        float(np.linalg.norm(np.asarray(jump @ target, dtype=np.complex128)))
        for jump in jump_matrices
    )
    max_jump_darkness = max(jump_darkness, default=0.0)

    complement = _orthogonal_complement_basis(target, tolerance=tolerance)
    target_blocks: list[np.ndarray] = []
    target_block_norms: list[float] = []
    for jump in jump_matrices:
        block = target.conj().T @ np.asarray(jump @ complement, dtype=np.complex128)
        target_blocks.append(block)
        target_block_norms.append(float(np.linalg.norm(block)))
    total_inflow = float(np.sqrt(np.sum(np.square(target_block_norms))))

    if complement.shape[1] == 0:
        no_inflow_basis = complement
    elif len(target_blocks) == 0:
        no_inflow_basis = complement
    else:
        no_inflow_coefficients = _nullspace_basis(
            np.vstack(target_blocks),
            tolerance=tolerance,
        )
        no_inflow_basis = _orthonormal_column_basis(
            complement @ no_inflow_coefficients,
            tolerance=tolerance,
        )

    rate_operator = sp.csr_array((dim, dim), dtype=np.complex128)
    for jump in jump_matrices:
        rate_operator = rate_operator + jump.conj().T @ jump
    no_jump_generator = ((-1.0j) * hamiltonian_sparse - 0.5 * rate_operator).tocsr()
    invariant_operators = (*jump_matrices, no_jump_generator)
    obstruction_basis, iterations = _largest_common_invariant_subspace(
        initial_basis=no_inflow_basis,
        operators=invariant_operators,
        tolerance=tolerance,
    )

    jump_invariance_residuals = tuple(
        float(np.linalg.norm(_subspace_leakage(jump, obstruction_basis))) for jump in jump_matrices
    )
    generator_residual = float(
        np.linalg.norm(_subspace_leakage(no_jump_generator, obstruction_basis))
    )
    max_obstruction_residual = max((*jump_invariance_residuals, generator_residual), default=0.0)

    common_kernel = _common_kernel_basis_from_sparse_operators(
        operators=jump_matrices,
        dim=dim,
        tolerance=tolerance,
    )
    old_report = diagnose_common_kernel_h_invariant_sector(
        hamiltonian=hamiltonian_sparse,
        jumps=jump_matrices,
        target_states=target,
        kernel_tolerance=tolerance,
    )

    target_hamiltonian_invariant = h_target_residual <= tolerance
    target_dark = max_jump_darkness <= tolerance
    obstruction_free = obstruction_basis.shape[1] == 0
    certified = target_hamiltonian_invariant and target_dark and obstruction_free

    return AttractiveSubspaceDiagnostics(
        hilbert_dimension=dim,
        target_dimension=int(target.shape[1]),
        n_jumps=len(jump_matrices),
        tolerance=float(tolerance),
        algorithm=_ATTRACTIVE_SUBSPACE_ALGORITHM,
        hamiltonian_target_invariance_residual=h_target_residual,
        jump_darkness_residuals=jump_darkness,
        max_jump_darkness_residual=max_jump_darkness,
        target_directed_block_norms=tuple(target_block_norms),
        total_target_directed_inflow_norm=total_inflow,
        no_inflow_dimension=int(no_inflow_basis.shape[1]),
        invariant_obstruction_dimension=int(obstruction_basis.shape[1]),
        invariant_fixed_point_iterations=iterations,
        jump_invariance_residuals=jump_invariance_residuals,
        no_jump_generator_invariance_residual=generator_residual,
        max_invariant_obstruction_residual=max_obstruction_residual,
        common_jump_kernel_dimension=int(common_kernel.shape[1]),
        old_h_invariant_kernel_dimension=int(old_report.bad_h_invariant_kernel_dimension),
        target_hamiltonian_invariant=target_hamiltonian_invariant,
        target_dark=target_dark,
        obstruction_free=obstruction_free,
        target_attractive_certified=certified,
        no_inflow_basis=no_inflow_basis,
        invariant_obstruction_basis=obstruction_basis,
    )
