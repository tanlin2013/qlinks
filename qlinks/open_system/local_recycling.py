from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


@dataclass(frozen=True, slots=True)
class LocalReducedDensityMatrix:
    """Reduced density matrix of a state on selected configuration variables."""

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
class LocalRecyclingCandidate:
    """One embedded local recycling jump candidate."""

    variable_indices: tuple[int, ...]
    alpha_index: int
    beta_index: int
    jump: sp.csr_array
    target_residual: float
    inflow_norm: float
    outflow_norm: float
    projector_commutator_norm: float
    local_alpha_vector: npt.NDArray[np.complex128]
    local_beta_vector: npt.NDArray[np.complex128]

    @property
    def is_dark(self) -> bool:
        return self.target_residual < 1e-10

    @property
    def has_inflow(self) -> bool:
        return self.inflow_norm > 1e-10


@dataclass(frozen=True, slots=True)
class LocalRecyclingScanResult:
    """Candidate jumps from one local region."""

    reduced_density_matrix: LocalReducedDensityMatrix
    candidates: tuple[LocalRecyclingCandidate, ...]

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def best_candidates(self) -> tuple[LocalRecyclingCandidate, ...]:
        return tuple(
            sorted(
                self.candidates,
                key=lambda candidate: candidate.inflow_norm,
                reverse=True,
            )
        )


def local_reduced_density_matrix_from_state(
    *,
    basis_configs: npt.NDArray[np.integer],
    state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    tolerance: float = 1e-10,
) -> LocalReducedDensityMatrix:
    """Compute rho_Omega for a state represented in a constrained basis.

    The local basis is the set of local patterns that appear in the constrained
    Hilbert basis. The environment is the complement of variable_indices.
    """
    variable_indices = tuple(int(index) for index in variable_indices)

    if len(variable_indices) == 0:
        raise ValueError("variable_indices must be nonempty.")

    configs = np.asarray(basis_configs)
    amplitudes = np.asarray(state, dtype=np.complex128)

    if configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_basis, n_variables).")

    if amplitudes.ndim != 1:
        raise ValueError("state must be one-dimensional.")

    if configs.shape[0] != amplitudes.size:
        raise ValueError("basis_configs and state have incompatible sizes.")

    norm = np.linalg.norm(amplitudes)
    if norm == 0.0:
        raise ValueError("state must be nonzero.")

    amplitudes = amplitudes / norm

    n_variables = configs.shape[1]
    variable_set = set(variable_indices)
    environment_indices = tuple(index for index in range(n_variables) if index not in variable_set)

    local_patterns = sorted(
        {tuple(int(value) for value in config[list(variable_indices)]) for config in configs}
    )
    local_to_index = {pattern: index for index, pattern in enumerate(local_patterns)}

    environment_groups: dict[tuple[int, ...], list[tuple[int, complex]]] = {}

    for basis_index, config in enumerate(configs):
        local_pattern = tuple(int(value) for value in config[list(variable_indices)])
        environment_pattern = tuple(int(value) for value in config[list(environment_indices)])

        local_index = local_to_index[local_pattern]
        environment_groups.setdefault(environment_pattern, []).append(
            (local_index, amplitudes[basis_index])
        )

    local_dim = len(local_patterns)
    density_matrix = np.zeros(
        (local_dim, local_dim),
        dtype=np.complex128,
    )

    for local_amplitudes in environment_groups.values():
        for row_index, row_amplitude in local_amplitudes:
            for col_index, col_amplitude in local_amplitudes:
                density_matrix[row_index, col_index] += row_amplitude * col_amplitude.conj()

    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    support_mask = eigenvalues > tolerance
    null_mask = ~support_mask

    support_basis = eigenvectors[:, support_mask].astype(
        np.complex128,
        copy=False,
    )
    null_basis = eigenvectors[:, null_mask].astype(
        np.complex128,
        copy=False,
    )

    return LocalReducedDensityMatrix(
        variable_indices=variable_indices,
        local_patterns=tuple(local_patterns),
        density_matrix=density_matrix,
        eigenvalues=eigenvalues,
        support_basis=support_basis,
        null_basis=null_basis,
    )


def embed_local_pattern_operator(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...],
    local_patterns: tuple[tuple[int, ...], ...],
    local_operator: npt.NDArray[np.complex128],
) -> sp.csr_array:
    """Embed a local pattern operator into the constrained full basis."""
    configs = np.asarray(basis_configs)
    variable_indices = tuple(int(index) for index in variable_indices)

    local_dim = len(local_patterns)

    if local_operator.shape != (local_dim, local_dim):
        raise ValueError("local_operator must have shape " f"({local_dim}, {local_dim}).")

    config_to_index = {
        tuple(int(value) for value in config): index for index, config in enumerate(configs)
    }

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for source_index, source_config in enumerate(configs):
        source_local_pattern = tuple(int(value) for value in source_config[list(variable_indices)])

        try:
            source_local_index = local_patterns.index(source_local_pattern)
        except ValueError:
            continue

        for target_local_index, target_local_pattern in enumerate(local_patterns):
            matrix_element = local_operator[
                target_local_index,
                source_local_index,
            ]

            if abs(matrix_element) == 0.0:
                continue

            target_config = np.array(source_config, copy=True)
            target_config[list(variable_indices)] = np.asarray(
                target_local_pattern,
                dtype=target_config.dtype,
            )

            target_index = config_to_index.get(tuple(int(value) for value in target_config))

            if target_index is None:
                continue

            rows.append(int(target_index))
            cols.append(int(source_index))
            data.append(complex(matrix_element))

    dim = configs.shape[0]

    return sp.csr_array(
        (
            np.asarray(data, dtype=np.complex128),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )


def score_recycling_jump(
    *,
    jump: Any,
    target_state: npt.ArrayLike,
) -> tuple[float, float, float, float]:
    """Return target residual, inflow, outflow, and projector commutator."""
    jump_dense = jump.toarray() if hasattr(jump, "toarray") else np.asarray(jump)

    state = np.asarray(target_state, dtype=np.complex128)
    state = state / np.linalg.norm(state)

    projector_target = np.outer(state, state.conj())
    identity = np.eye(state.size, dtype=np.complex128)
    projector_orthogonal = identity - projector_target

    target_residual = float(np.linalg.norm(jump_dense @ state))
    inflow_norm = float(np.linalg.norm(projector_target @ jump_dense @ projector_orthogonal))
    outflow_norm = float(np.linalg.norm(projector_orthogonal @ jump_dense @ projector_target))
    projector_commutator_norm = float(
        np.linalg.norm(jump_dense @ projector_target - projector_target @ jump_dense)
    )

    return (
        target_residual,
        inflow_norm,
        outflow_norm,
        projector_commutator_norm,
    )


def scan_local_recycling_candidates(
    *,
    basis_configs: npt.NDArray[np.integer],
    target_state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-10,
    max_candidates: int | None = None,
) -> LocalRecyclingScanResult:
    """Scan local parent/recycling jumps from rho_Omega support/nullspace."""
    reduced_density_matrix = local_reduced_density_matrix_from_state(
        basis_configs=basis_configs,
        state=target_state,
        variable_indices=variable_indices,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis

    candidates: list[LocalRecyclingCandidate] = []

    if support_basis.shape[1] == 0 or null_basis.shape[1] == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    for alpha_index in range(support_basis.shape[1]):
        alpha_vector = support_basis[:, alpha_index]

        for beta_index in range(null_basis.shape[1]):
            beta_vector = null_basis[:, beta_index]

            local_operator = np.outer(alpha_vector, beta_vector.conj())

            jump = embed_local_pattern_operator(
                basis_configs=basis_configs,
                variable_indices=reduced_density_matrix.variable_indices,
                local_patterns=reduced_density_matrix.local_patterns,
                local_operator=local_operator,
            )

            (
                target_residual,
                inflow_norm,
                outflow_norm,
                projector_commutator_norm,
            ) = score_recycling_jump(
                jump=jump,
                target_state=target_state,
            )

            if target_residual > dark_tolerance:
                continue

            if inflow_norm <= inflow_tolerance:
                continue

            candidates.append(
                LocalRecyclingCandidate(
                    variable_indices=reduced_density_matrix.variable_indices,
                    alpha_index=alpha_index,
                    beta_index=beta_index,
                    jump=jump,
                    target_residual=target_residual,
                    inflow_norm=inflow_norm,
                    outflow_norm=outflow_norm,
                    projector_commutator_norm=projector_commutator_norm,
                    local_alpha_vector=alpha_vector,
                    local_beta_vector=beta_vector,
                )
            )

    candidates = sorted(
        candidates,
        key=lambda candidate: candidate.inflow_norm,
        reverse=True,
    )

    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )
