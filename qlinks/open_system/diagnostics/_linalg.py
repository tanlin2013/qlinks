from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.open_system._subspace import (
    _nullspace_basis_by_gram,
    _orthonormal_column_basis,
)


def _monitor_hamiltonian_leakage_norms(
    *,
    hamiltonian: scipy_sparse.csr_array,
    monitors: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    basis: np.ndarray,
) -> np.ndarray:
    if basis.size == 0 or basis.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)

    hamiltonian_basis = hamiltonian @ basis
    squared_norms = np.zeros(basis.shape[1], dtype=np.float64)
    for monitor in monitors:
        image = monitor @ hamiltonian_basis
        squared_norms += np.sum(np.abs(image) ** 2, axis=0).real

    return np.sqrt(np.maximum(squared_norms, 0.0)).astype(np.float64, copy=False)


def _rank_one_lindblad_rhs_norm(
    *,
    hamiltonian: scipy_sparse.spmatrix | scipy_sparse.sparray,
    jumps: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    target: np.ndarray,
    precomputed_jump_targets: tuple[np.ndarray, ...] | None = None,
) -> float:
    hamiltonian_target = hamiltonian @ target

    # Evaluate the Hamiltonian commutator using only the component orthogonal
    # to the rank-one projector.  Writing the two commutator terms as
    # ``-i H|psi><psi| + i |psi><psi|H`` suffers from catastrophic
    # cancellation when ``|psi>`` is an eigenstate with a nonzero energy.  In
    # that case the two large rank-one terms cancel exactly, but the low-rank
    # Frobenius contraction can leave a spurious residual around sqrt(eps).
    # Subtracting the Rayleigh quotient first preserves the same commutator and
    # makes exact eigenstates numerically dark.
    target_energy = np.vdot(target, hamiltonian_target)
    hamiltonian_target_perp = hamiltonian_target - target_energy * target
    terms: list[tuple[complex, np.ndarray, np.ndarray]] = [
        (-1j, hamiltonian_target_perp, target),
        (1j, target, hamiltonian_target_perp),
    ]

    if precomputed_jump_targets is None:
        jump_targets = tuple(jump @ target for jump in jumps)
    else:
        jump_targets = precomputed_jump_targets

    for jump, jump_target in zip(jumps, jump_targets):
        jump_dagger_jump_target = jump.conj().T @ jump_target
        terms.extend(
            (
                (1.0, jump_target, jump_target),
                (-0.5, jump_dagger_jump_target, target),
                (-0.5, target, jump_dagger_jump_target),
            )
        )

    return _low_rank_operator_frobenius_norm(tuple(terms))


def _manifold_inflow_norm(
    *,
    jumps: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    manifold_basis: np.ndarray,
) -> float:
    total = 0.0
    for jump in jumps:
        adjoint_action = np.asarray(jump.conj().T @ manifold_basis, dtype=np.complex128)
        projected_out = adjoint_action - manifold_basis @ (manifold_basis.conj().T @ adjoint_action)
        total += float(np.linalg.norm(projected_out) ** 2)
    return float(np.sqrt(max(total, 0.0)))


def _internal_liouvillian_eigenvalues(
    internal_hamiltonian_eigenvalues: Sequence[complex],
) -> tuple[complex, ...]:
    energies = tuple(complex(value) for value in internal_hamiltonian_eigenvalues)
    return tuple(-1j * (left - right) for left in energies for right in energies)


def _match_expected_internal_nondecaying_modes(
    *,
    observed: Sequence[complex],
    expected: Sequence[complex],
    tolerance: float,
) -> dict[str, object]:
    observed_values = [complex(value) for value in observed]
    expected_values = [complex(value) for value in expected]
    unmatched = set(range(len(observed_values)))
    matched_indices: list[int] = []

    for expected_value in expected_values:
        best_index = None
        best_distance = float("inf")
        for observed_index in unmatched:
            distance = abs(observed_values[observed_index] - expected_value)
            if distance < best_distance:
                best_distance = float(distance)
                best_index = observed_index
        if best_index is not None and best_distance <= tolerance:
            unmatched.remove(best_index)
            matched_indices.append(best_index)

    matched = len(matched_indices)
    return {
        "matched": matched,
        "missing": max(0, len(expected_values) - matched),
        "extra": max(0, len(observed_values) - matched),
        "matched_observed_indices": tuple(sorted(matched_indices)),
    }


def _external_decay_gap_from_spectrum(
    *,
    eigenvalues: npt.ArrayLike,
    matched_observed_indices: Sequence[int],
    zero_tolerance: float,
) -> float | None:
    values = np.asarray(eigenvalues, dtype=np.complex128)
    nondecaying_indices = [
        index for index, value in enumerate(values) if abs(complex(value).real) <= zero_tolerance
    ]
    matched_nondecaying_indices = set(matched_observed_indices)
    external_indices = [
        index for index in range(values.size) if index not in matched_nondecaying_indices
    ]
    if not external_indices:
        return None

    external_real_parts = np.real(values[external_indices])
    decaying = external_real_parts < -zero_tolerance
    if not np.any(decaying):
        return None
    if any(index in external_indices for index in nondecaying_indices):
        return 0.0
    return float(-np.max(external_real_parts[decaying]))


def _orthogonal_component_norm(vector: np.ndarray, basis_vector: np.ndarray) -> float:
    vector_norm_squared = float(np.real(np.vdot(vector, vector)))
    projection = np.vdot(basis_vector, vector)
    return float(np.sqrt(max(0.0, vector_norm_squared - abs(projection) ** 2)))


def _low_rank_operator_frobenius_norm(
    terms: Sequence[tuple[complex, np.ndarray, np.ndarray]],
) -> float:
    if len(terms) == 0:
        return 0.0

    norm_squared = 0.0 + 0.0j
    for coefficient_i, left_i, right_i in terms:
        for coefficient_j, left_j, right_j in terms:
            norm_squared += (
                np.conj(coefficient_i)
                * coefficient_j
                * np.vdot(left_i, left_j)
                * np.vdot(right_j, right_i)
            )

    return float(np.sqrt(max(0.0, float(np.real(norm_squared)))))


def _largest_h_invariant_subspace_inside_leakage_kernel(
    *,
    leakage: np.ndarray,
    bad_block: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return bad-coordinate vectors in ``ker(leakage)`` invariant under ``bad_block``.

    The direct Krylov certificate stacks ``leakage @ bad_block**n`` for every
    ``n`` and then computes one large dense nullspace.  That is fragile for the
    triangular-QDM production cases: the stacked matrix is very tall and can
    trigger LAPACK ``SVD did not converge`` failures.  This computes the same
    largest invariant subspace by fixed-point intersections,

    ``S <- {x in S : bad_block @ x in S}``, starting from ``S = ker(leakage)``.

    Each nullspace problem has only ``bad_dimension`` rows and the current
    subspace dimension columns, so the dense linear algebra remains bounded by
    the selected jump set rather than by the full Krylov stack.
    """
    bad_dimension = int(bad_block.shape[0])
    if bad_dimension == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    if not np.all(np.isfinite(leakage)) or not np.all(np.isfinite(bad_block)):
        raise ValueError("H-invariant kernel diagnostic received non-finite matrix entries.")

    current = _nullspace_basis_by_gram(leakage, tolerance=tolerance)
    current = _orthonormal_column_basis(current, tolerance=tolerance)
    if current.shape[1] == 0:
        return np.zeros((bad_dimension, 0), dtype=np.complex128)

    for _iteration in range(bad_dimension):
        image = bad_block @ current
        projected = current @ (current.conj().T @ image)
        escape = image - projected
        if float(np.linalg.norm(escape)) <= tolerance:
            return current.astype(np.complex128, copy=False)

        surviving_coordinates = _nullspace_basis_by_gram(
            escape,
            tolerance=tolerance,
        )
        if surviving_coordinates.shape[1] == 0:
            return np.zeros((bad_dimension, 0), dtype=np.complex128)

        next_current = current @ surviving_coordinates
        next_current = _orthonormal_column_basis(next_current, tolerance=tolerance)
        if next_current.shape[1] == 0:
            return np.zeros((bad_dimension, 0), dtype=np.complex128)

        if next_current.shape[1] == current.shape[1]:
            current = next_current
            continue
        current = next_current

    return current.astype(np.complex128, copy=False)
