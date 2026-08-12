"""Local reduced-density-matrix primitives for constrained bases.

This module is intentionally neutral: both caging diagnostics and open-system
construction may depend on it, while it must not depend on either layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class LocalReducedDensityMatrix:
    """Reduced density matrix of a pure state or subspace on selected variables."""

    variable_indices: tuple[int, ...]
    local_patterns: tuple[tuple[int, ...], ...]
    density_matrix: npt.NDArray[np.complex128]
    eigenvalues: npt.NDArray[np.float64]
    support_basis: npt.NDArray[np.complex128]
    null_basis: npt.NDArray[np.complex128]

    @property
    def local_dim(self) -> int:
        return len(self.local_patterns)

    @property
    def support_rank(self) -> int:
        return int(self.support_basis.shape[1])

    @property
    def nullity(self) -> int:
        return int(self.null_basis.shape[1])


@dataclass(frozen=True, slots=True)
class _LocalPatternBasisContext:
    """Grouped constrained-basis data for one local region."""

    variable_indices: tuple[int, ...]
    local_patterns: tuple[tuple[int, ...], ...]
    environment_groups: tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]], ...]
    dim: int

    @property
    def local_dim(self) -> int:
        return len(self.local_patterns)


def _local_pattern_basis_context_from_basis(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...] | list[int],
    local_patterns: tuple[tuple[int, ...], ...] | None = None,
) -> _LocalPatternBasisContext:
    configs = np.asarray(basis_configs)
    variable_indices = tuple(int(index) for index in variable_indices)

    if len(variable_indices) == 0:
        raise ValueError("variable_indices must be nonempty.")

    if configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_basis, n_variables).")

    n_basis, n_variables = configs.shape

    if any(index < 0 or index >= n_variables for index in variable_indices):
        raise ValueError("variable_indices contains out-of-range entries.")

    variable_index_set = set(variable_indices)
    environment_indices = tuple(
        index for index in range(n_variables) if index not in variable_index_set
    )
    variable_index_array = np.asarray(variable_indices, dtype=np.int64)
    environment_index_array = np.asarray(environment_indices, dtype=np.int64)

    if local_patterns is None:
        local_patterns = tuple(
            sorted(
                {tuple(int(value) for value in config[variable_index_array]) for config in configs}
            )
        )
    else:
        local_patterns = tuple(tuple(int(value) for value in pattern) for pattern in local_patterns)

    if len(local_patterns) == 0:
        raise ValueError("local_patterns must be nonempty.")

    if any(len(pattern) != len(variable_indices) for pattern in local_patterns):
        raise ValueError("local pattern length must match variable_indices.")

    local_pattern_to_index = {pattern: index for index, pattern in enumerate(local_patterns)}
    environment_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}

    for basis_index, config in enumerate(configs):
        local_pattern = tuple(int(value) for value in config[variable_index_array])
        local_index = local_pattern_to_index.get(local_pattern)

        if local_index is None:
            continue

        environment_pattern = tuple(int(value) for value in config[environment_index_array])
        environment_groups.setdefault(environment_pattern, []).append(
            (int(basis_index), int(local_index))
        )

    grouped = tuple(
        (
            np.asarray([basis_index for basis_index, _ in group], dtype=np.int64),
            np.asarray([local_index for _, local_index in group], dtype=np.int64),
        )
        for group in environment_groups.values()
    )

    return _LocalPatternBasisContext(
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        environment_groups=grouped,
        dim=int(n_basis),
    )


def _local_reduced_density_matrix_from_basis_context(
    *,
    context: _LocalPatternBasisContext,
    state: npt.ArrayLike,
    tolerance: float = 1e-10,
) -> LocalReducedDensityMatrix:
    amplitudes = np.asarray(state, dtype=np.complex128)

    if amplitudes.ndim != 1:
        raise ValueError("state must be one-dimensional.")

    if context.dim != amplitudes.size:
        raise ValueError("basis_configs and state have incompatible sizes.")

    norm = np.linalg.norm(amplitudes)
    if norm == 0.0:
        raise ValueError("state must be nonzero.")

    amplitudes = amplitudes / norm
    density_matrix = np.zeros((context.local_dim, context.local_dim), dtype=np.complex128)

    for basis_indices, local_indices in context.environment_groups:
        group_amplitudes = amplitudes[basis_indices]
        density_matrix[np.ix_(local_indices, local_indices)] += np.outer(
            group_amplitudes,
            group_amplitudes.conj(),
        )

    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    support_mask = eigenvalues > tolerance
    null_mask = ~support_mask

    return LocalReducedDensityMatrix(
        variable_indices=context.variable_indices,
        local_patterns=context.local_patterns,
        density_matrix=density_matrix,
        eigenvalues=eigenvalues,
        support_basis=eigenvectors[:, support_mask].astype(np.complex128),
        null_basis=eigenvectors[:, null_mask].astype(np.complex128),
    )


def local_reduced_density_matrix_from_state(
    *,
    basis_configs: npt.NDArray[np.integer],
    state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    tolerance: float = 1e-10,
) -> LocalReducedDensityMatrix:
    """Compute the local RDM of a state represented in a constrained basis."""
    context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    return _local_reduced_density_matrix_from_basis_context(
        context=context,
        state=state,
        tolerance=tolerance,
    )


def _normalize_state_matrix_columns(
    states: npt.ArrayLike,
    *,
    dim: int,
    tolerance: float = 1e-10,
) -> npt.NDArray[np.complex128]:
    matrix = np.asarray(states, dtype=np.complex128)

    if matrix.ndim == 1:
        if matrix.size != dim:
            raise ValueError("state vector has incompatible dimension.")
        matrix = matrix.reshape(dim, 1)
    elif matrix.ndim == 2:
        if matrix.shape[0] == dim:
            pass
        elif matrix.shape[1] == dim:
            matrix = matrix.T
        else:
            raise ValueError("state matrix must have shape (dim, n_states) or (n_states, dim).")
    else:
        raise ValueError("states must be a vector or a two-dimensional matrix.")

    if matrix.shape[1] == 0:
        raise ValueError("state matrix must contain at least one state.")

    q, r = np.linalg.qr(matrix)
    rank_mask = np.abs(np.diag(r)) > tolerance
    rank = int(np.count_nonzero(rank_mask))

    if rank == 0:
        raise ValueError("state matrix has numerical rank zero.")

    return q[:, :rank].astype(np.complex128, copy=False)


def _local_reduced_density_matrix_from_basis_context_and_states(
    *,
    context: _LocalPatternBasisContext,
    states: npt.ArrayLike,
    tolerance: float = 1e-10,
) -> LocalReducedDensityMatrix:
    """Return the local RDM of the normalized projector onto a state subspace."""
    state_basis = _normalize_state_matrix_columns(
        states,
        dim=context.dim,
        tolerance=tolerance,
    )
    density_matrix = np.zeros((context.local_dim, context.local_dim), dtype=np.complex128)

    for state_index in range(state_basis.shape[1]):
        amplitudes = state_basis[:, state_index]
        for basis_indices, local_indices in context.environment_groups:
            group_amplitudes = amplitudes[basis_indices]
            density_matrix[np.ix_(local_indices, local_indices)] += np.outer(
                group_amplitudes,
                group_amplitudes.conj(),
            )

    density_matrix /= float(state_basis.shape[1])
    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    support_mask = eigenvalues > tolerance
    null_mask = ~support_mask

    return LocalReducedDensityMatrix(
        variable_indices=context.variable_indices,
        local_patterns=context.local_patterns,
        density_matrix=density_matrix,
        eigenvalues=eigenvalues,
        support_basis=eigenvectors[:, support_mask].astype(np.complex128),
        null_basis=eigenvectors[:, null_mask].astype(np.complex128),
    )


def local_reduced_density_matrix_from_state_matrix(
    *,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    tolerance: float = 1e-10,
) -> LocalReducedDensityMatrix:
    """Compute the local RDM of the normalized projector onto a state subspace.

    ``states`` may have shape ``(dim, n_states)`` or ``(n_states, dim)``. The
    columns are orthonormalized before tracing out the environment, so callers
    may pass linearly dependent representatives without changing local support.
    """
    context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    return _local_reduced_density_matrix_from_basis_context_and_states(
        context=context,
        states=states,
        tolerance=tolerance,
    )
