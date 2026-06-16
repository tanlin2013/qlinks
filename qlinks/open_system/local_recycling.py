from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

RecyclingJumpSource = Literal[
    "none",
    "local_rdm_rank_one",
    "local_rdm_two_pattern",
    "local_rdm_null_basis",
]


@dataclass(frozen=True, slots=True)
class LocalReducedDensityMatrix:
    """Reduced density matrix of a pure state on selected basis variables."""

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


@dataclass(frozen=True, slots=True)
class _LocalPatternEmbeddingContext:
    """Precomputed constrained-basis embedding data for one local region."""

    variable_indices: tuple[int, ...]
    local_patterns: tuple[tuple[int, ...], ...]
    source_full_indices: npt.NDArray[np.int64]
    target_full_indices: npt.NDArray[np.int64]
    source_local_indices: npt.NDArray[np.int64]
    target_local_indices: npt.NDArray[np.int64]
    dim: int

    @property
    def local_dim(self) -> int:
        return len(self.local_patterns)


@dataclass(frozen=True, slots=True)
class LocalMatrixUnitTerm:
    """One local matrix-unit term coefficient * |target><source|."""

    coefficient: complex
    target_pattern: tuple[int, ...]
    source_pattern: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TwoPatternRecyclingStructure:
    """Detected local two-pattern recycling structure.

    This represents a local jump of the form |minus><plus|, up to phase
    and convention.
    """

    variable_indices: tuple[int, ...]
    pattern_a: tuple[int, ...]
    pattern_b: tuple[int, ...]
    alpha_index: int
    beta_index: int
    phase: complex
    residual: float
    matrix_unit_terms: tuple[LocalMatrixUnitTerm, ...]

    @property
    def n_variables(self) -> int:
        return len(self.variable_indices)


@dataclass(frozen=True, slots=True)
class LocalRecyclingCandidate:
    """One embedded local RDM recycling jump candidate."""

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
                key=lambda candidate: (
                    -float(candidate.inflow_norm),
                    float(candidate.target_residual),
                    int(candidate.jump.nnz),
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class LocalRecyclingSelection:
    """Selected recycling candidate plus optional detected structure."""

    candidate: LocalRecyclingCandidate
    two_pattern_structure: TwoPatternRecyclingStructure | None
    score: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LocalRecyclingBuildResult:
    """Selected recycling jumps from several local regions."""

    scan_results: tuple[LocalRecyclingScanResult, ...]
    selections: tuple[LocalRecyclingSelection, ...]

    @property
    def jumps(self) -> tuple[sp.csr_array, ...]:
        return tuple(selection.candidate.jump for selection in self.selections)

    @property
    def n_jumps(self) -> int:
        return len(self.selections)

    @property
    def variable_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(selection.candidate.variable_indices for selection in self.selections)

    @property
    def alpha_beta_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                int(selection.candidate.alpha_index),
                int(selection.candidate.beta_index),
            )
            for selection in self.selections
        )


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
    """Compute rho_Omega for a state represented in a constrained basis."""
    context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    return _local_reduced_density_matrix_from_basis_context(
        context=context,
        state=state,
        tolerance=tolerance,
    )


def _embedding_context_from_basis_context(
    context: _LocalPatternBasisContext,
) -> _LocalPatternEmbeddingContext:
    source_full_chunks: list[npt.NDArray[np.int64]] = []
    target_full_chunks: list[npt.NDArray[np.int64]] = []
    source_local_chunks: list[npt.NDArray[np.int64]] = []
    target_local_chunks: list[npt.NDArray[np.int64]] = []

    for full_indices, local_indices in context.environment_groups:
        group_size = int(full_indices.size)

        if group_size == 0:
            continue

        source_full_chunks.append(np.repeat(full_indices, group_size))
        target_full_chunks.append(np.tile(full_indices, group_size))
        source_local_chunks.append(np.repeat(local_indices, group_size))
        target_local_chunks.append(np.tile(local_indices, group_size))

    if len(source_full_chunks) == 0:
        source_full_indices = np.asarray((), dtype=np.int64)
        target_full_indices = np.asarray((), dtype=np.int64)
        source_local_indices = np.asarray((), dtype=np.int64)
        target_local_indices = np.asarray((), dtype=np.int64)
    else:
        source_full_indices = np.concatenate(source_full_chunks).astype(np.int64, copy=False)
        target_full_indices = np.concatenate(target_full_chunks).astype(np.int64, copy=False)
        source_local_indices = np.concatenate(source_local_chunks).astype(np.int64, copy=False)
        target_local_indices = np.concatenate(target_local_chunks).astype(np.int64, copy=False)

    return _LocalPatternEmbeddingContext(
        variable_indices=context.variable_indices,
        local_patterns=context.local_patterns,
        source_full_indices=source_full_indices,
        target_full_indices=target_full_indices,
        source_local_indices=source_local_indices,
        target_local_indices=target_local_indices,
        dim=context.dim,
    )


def _embedding_context_from_basis(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...] | list[int],
    local_patterns: tuple[tuple[int, ...], ...],
) -> _LocalPatternEmbeddingContext:
    """Precompute constrained-basis transitions induced by local pattern changes."""
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
        local_patterns=local_patterns,
    )
    return _embedding_context_from_basis_context(basis_context)


def _embed_local_pattern_operator_from_context(
    *,
    context: _LocalPatternEmbeddingContext,
    local_operator: npt.NDArray[np.complex128],
) -> sp.csr_array:
    local_dim = context.local_dim

    if local_operator.shape != (local_dim, local_dim):
        raise ValueError(
            "local_operator has incompatible shape: "
            f"{local_operator.shape} != {(local_dim, local_dim)}."
        )

    if context.source_full_indices.size == 0:
        return sp.csr_array((context.dim, context.dim), dtype=np.complex128)

    data = np.asarray(
        local_operator[context.target_local_indices, context.source_local_indices],
        dtype=np.complex128,
    )
    nonzero_mask = data != 0.0

    return sp.csr_array(
        (
            data[nonzero_mask],
            (
                context.target_full_indices[nonzero_mask],
                context.source_full_indices[nonzero_mask],
            ),
        ),
        shape=(context.dim, context.dim),
        dtype=np.complex128,
    )


def embed_local_pattern_operator(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...],
    local_patterns: tuple[tuple[int, ...], ...],
    local_operator: npt.NDArray[np.complex128],
) -> sp.csr_array:
    """Embed a local operator into the constrained full basis."""
    context = _embedding_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
        local_patterns=local_patterns,
    )
    return _embed_local_pattern_operator_from_context(
        context=context,
        local_operator=local_operator,
    )


def score_recycling_jump(
    *,
    jump: Any,
    target_state: npt.ArrayLike,
) -> tuple[float, float, float, float]:
    """Return target residual, inflow, outflow, and projector commutator.

    The diagnostics are Frobenius norms of the corresponding projected
    operators.  They can be evaluated from ``J|psi>`` and ``J^dagger|psi>``
    without materializing the dense projectors ``|psi><psi|`` and
    ``I-|psi><psi|``.
    """
    state = np.asarray(target_state, dtype=np.complex128)
    norm = np.linalg.norm(state)

    if norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    state = state / norm

    if sp.issparse(jump):
        target_vector = np.asarray(jump @ state, dtype=np.complex128)
        adjoint_target_vector = np.asarray(jump.conj().T @ state, dtype=np.complex128)
    else:
        jump_array = np.asarray(jump, dtype=np.complex128)
        target_vector = np.asarray(jump_array @ state, dtype=np.complex128)
        adjoint_target_vector = np.asarray(jump_array.conj().T @ state, dtype=np.complex128)

    expectation = complex(np.vdot(state, target_vector))
    expectation_norm_sq = abs(expectation) ** 2
    target_norm_sq = float(np.vdot(target_vector, target_vector).real)
    adjoint_target_norm_sq = float(np.vdot(adjoint_target_vector, adjoint_target_vector).real)

    target_residual = float(np.sqrt(max(target_norm_sq, 0.0)))
    inflow_norm = float(np.sqrt(max(adjoint_target_norm_sq - expectation_norm_sq, 0.0)))
    outflow_norm = float(np.sqrt(max(target_norm_sq - expectation_norm_sq, 0.0)))
    projector_commutator_norm = float(
        np.sqrt(max(target_norm_sq + adjoint_target_norm_sq - 2.0 * expectation_norm_sq, 0.0))
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
    """Scan local rank-one recycling jumps from rho_Omega."""
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    reduced_density_matrix = _local_reduced_density_matrix_from_basis_context(
        context=basis_context,
        state=target_state,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis
    support_eigenvalues = reduced_density_matrix.eigenvalues[
        reduced_density_matrix.eigenvalues > rdm_tolerance
    ]

    candidates: list[LocalRecyclingCandidate] = []

    if support_basis.shape[1] == 0 or null_basis.shape[1] == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    embedding_context = _embedding_context_from_basis_context(basis_context)

    for alpha_index in range(support_basis.shape[1]):
        alpha_vector = support_basis[:, alpha_index]

        for beta_index in range(null_basis.shape[1]):
            beta_vector = null_basis[:, beta_index]
            local_operator = np.outer(alpha_vector, beta_vector.conj())

            jump = _embed_local_pattern_operator_from_context(
                context=embedding_context,
                local_operator=local_operator,
            )

            inflow_norm = float(np.sqrt(max(float(support_eigenvalues[alpha_index]), 0.0)))
            target_residual = 0.0
            outflow_norm = 0.0
            projector_commutator_norm = inflow_norm

            if target_residual > dark_tolerance:
                continue

            if inflow_norm <= inflow_tolerance:
                continue

            candidates.append(
                LocalRecyclingCandidate(
                    variable_indices=reduced_density_matrix.variable_indices,
                    alpha_index=int(alpha_index),
                    beta_index=int(beta_index),
                    jump=jump,
                    target_residual=target_residual,
                    inflow_norm=inflow_norm,
                    outflow_norm=outflow_norm,
                    projector_commutator_norm=projector_commutator_norm,
                    local_alpha_vector=alpha_vector.astype(np.complex128),
                    local_beta_vector=beta_vector.astype(np.complex128),
                )
            )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.inflow_norm),
            float(candidate.target_residual),
            int(candidate.jump.nnz),
        ),
    )

    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )


def local_rank_one_matrix_unit_expansion(
    *,
    local_patterns: tuple[tuple[int, ...], ...],
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    tolerance: float = 1e-10,
) -> tuple[LocalMatrixUnitTerm, ...]:
    """Expand |alpha><beta| into local matrix units |a><b|."""
    alpha_array = np.asarray(alpha, dtype=np.complex128)
    beta_array = np.asarray(beta, dtype=np.complex128)

    if alpha_array.ndim != 1 or beta_array.ndim != 1:
        raise ValueError("alpha and beta must be one-dimensional.")

    if alpha_array.shape != beta_array.shape:
        raise ValueError("alpha and beta must have the same shape.")

    if alpha_array.size != len(local_patterns):
        raise ValueError("alpha/beta size must match the number of local patterns.")

    terms: list[LocalMatrixUnitTerm] = []

    for target_index, target_pattern in enumerate(local_patterns):
        for source_index, source_pattern in enumerate(local_patterns):
            coefficient = alpha_array[target_index] * beta_array[source_index].conj()

            if abs(coefficient) <= tolerance:
                continue

            terms.append(
                LocalMatrixUnitTerm(
                    coefficient=complex(coefficient),
                    target_pattern=tuple(int(value) for value in target_pattern),
                    source_pattern=tuple(int(value) for value in source_pattern),
                )
            )

    return tuple(terms)


def _detect_two_pattern_recycling_structure_from_vectors(
    *,
    variable_indices: tuple[int, ...],
    alpha_index: int,
    beta_index: int,
    local_patterns: tuple[tuple[int, ...], ...],
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    tolerance: float = 1e-8,
) -> TwoPatternRecyclingStructure | None:
    alpha_array = np.asarray(alpha, dtype=np.complex128)
    beta_array = np.asarray(beta, dtype=np.complex128)

    if alpha_array.ndim != 1 or beta_array.ndim != 1:
        raise ValueError("alpha and beta must be one-dimensional.")

    if alpha_array.shape != beta_array.shape:
        raise ValueError("alpha and beta must have the same shape.")

    if alpha_array.size != len(local_patterns):
        raise ValueError("alpha/beta size must match the number of local patterns.")

    alpha_support = np.flatnonzero(np.abs(alpha_array) > tolerance)
    beta_support = np.flatnonzero(np.abs(beta_array) > tolerance)

    if alpha_support.size != 2 or beta_support.size != 2:
        return None

    if set(int(index) for index in alpha_support) != set(int(index) for index in beta_support):
        return None

    pattern_indices = tuple(
        sorted((int(index) for index in alpha_support), key=lambda index: local_patterns[index])
    )
    pattern_a = local_patterns[pattern_indices[0]]
    pattern_b = local_patterns[pattern_indices[1]]

    coefficients = np.asarray(
        [
            alpha_array[pattern_indices[0]] * beta_array[pattern_indices[0]].conj(),
            alpha_array[pattern_indices[0]] * beta_array[pattern_indices[1]].conj(),
            alpha_array[pattern_indices[1]] * beta_array[pattern_indices[0]].conj(),
            alpha_array[pattern_indices[1]] * beta_array[pattern_indices[1]].conj(),
        ],
        dtype=np.complex128,
    )

    templates = (
        np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.complex128) / 2.0,
        np.asarray([-1.0, -1.0, 1.0, 1.0], dtype=np.complex128) / 2.0,
        np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.complex128) / 2.0,
        np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.complex128) / 2.0,
    )

    best_phase = 0.0 + 0.0j
    best_residual = np.inf

    for template in templates:
        overlap = np.vdot(template, coefficients)

        if abs(overlap) <= tolerance:
            continue

        phase = overlap / abs(overlap)
        residual = float(np.linalg.norm(coefficients - phase * template))

        if residual < best_residual:
            best_residual = residual
            best_phase = complex(phase)

    if best_residual > tolerance:
        return None

    terms = tuple(
        LocalMatrixUnitTerm(
            coefficient=complex(alpha_array[target_index] * beta_array[source_index].conj()),
            target_pattern=tuple(int(value) for value in local_patterns[target_index]),
            source_pattern=tuple(int(value) for value in local_patterns[source_index]),
        )
        for target_index in pattern_indices
        for source_index in pattern_indices
    )

    return TwoPatternRecyclingStructure(
        variable_indices=tuple(int(index) for index in variable_indices),
        pattern_a=pattern_a,
        pattern_b=pattern_b,
        alpha_index=int(alpha_index),
        beta_index=int(beta_index),
        phase=best_phase,
        residual=best_residual,
        matrix_unit_terms=terms,
    )


def detect_two_pattern_recycling_structure(
    *,
    candidate: LocalRecyclingCandidate,
    local_patterns: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-8,
) -> TwoPatternRecyclingStructure | None:
    """Detect whether a candidate is a two-pattern |minus><plus| jump."""
    return _detect_two_pattern_recycling_structure_from_vectors(
        variable_indices=candidate.variable_indices,
        alpha_index=candidate.alpha_index,
        beta_index=candidate.beta_index,
        local_patterns=local_patterns,
        alpha=candidate.local_alpha_vector,
        beta=candidate.local_beta_vector,
        tolerance=tolerance,
    )


def select_local_recycling_candidates(
    *,
    scan_result: LocalRecyclingScanResult,
    source: RecyclingJumpSource = "local_rdm_two_pattern",
    max_candidates: int = 1,
    prefer_sparse: bool = True,
    two_pattern_tolerance: float = 1e-8,
) -> tuple[LocalRecyclingSelection, ...]:
    """Select recycling candidates from one scan result.

    ``local_rdm_rank_one`` and ``local_rdm_two_pattern`` keep the historical
    behavior: they choose the best few rank-one reset maps ``|alpha><beta|``.

    ``local_rdm_null_basis`` is designed for monitor-recycler jumps
    ``L=V P``.  A single rank-one recycler can make ``V P`` much more singular
    than the monitor ``P`` itself, because it only tests one local ``beta``
    direction.  This source instead selects one good target-support vector
    ``alpha`` for every local-RDM null vector ``beta``.  The resulting jump
    family preserves the full local null-space information detected by the
    monitor, while still recycling into the target local support.  The
    ``max_candidates`` argument is intentionally ignored for this source; the
    number of selected jumps is the local-RDM nullity.
    """
    if source == "none":
        return ()

    if source == "local_rdm_null_basis":
        best_by_beta: dict[int, LocalRecyclingSelection] = {}

        for candidate in scan_result.candidates:
            structure = detect_two_pattern_recycling_structure(
                candidate=candidate,
                local_patterns=scan_result.reduced_density_matrix.local_patterns,
                tolerance=two_pattern_tolerance,
            )
            nnz = candidate.jump.nnz if hasattr(candidate.jump, "nnz") else np.inf
            score = (
                -float(candidate.inflow_norm),
                float(candidate.target_residual),
                float(nnz) if prefer_sparse else 0.0,
                int(candidate.alpha_index),
            )
            selection = LocalRecyclingSelection(
                candidate=candidate,
                two_pattern_structure=structure,
                score=score,
            )
            previous = best_by_beta.get(int(candidate.beta_index))
            if previous is None or selection.score < previous.score:
                best_by_beta[int(candidate.beta_index)] = selection

        return tuple(best_by_beta[index] for index in sorted(best_by_beta))

    selections: list[LocalRecyclingSelection] = []

    for candidate in scan_result.candidates:
        structure = detect_two_pattern_recycling_structure(
            candidate=candidate,
            local_patterns=scan_result.reduced_density_matrix.local_patterns,
            tolerance=two_pattern_tolerance,
        )

        if source == "local_rdm_two_pattern" and structure is None:
            continue

        nnz = candidate.jump.nnz if hasattr(candidate.jump, "nnz") else np.inf

        score = (
            0.0 if structure is not None else 1.0,
            float(nnz) if prefer_sparse else 0.0,
            -float(candidate.inflow_norm),
            float(candidate.target_residual),
        )

        selections.append(
            LocalRecyclingSelection(
                candidate=candidate,
                two_pattern_structure=structure,
                score=score,
            )
        )

    selections = sorted(selections, key=lambda selection: selection.score)

    return tuple(selections[:max_candidates])


def _two_pattern_support_indices(
    vector: npt.ArrayLike,
    *,
    tolerance: float,
) -> tuple[int, int] | None:
    support = np.flatnonzero(np.abs(np.asarray(vector, dtype=np.complex128)) > tolerance)

    if support.size != 2:
        return None

    return int(support[0]), int(support[1])


def _scan_local_two_pattern_recycling_candidates(
    *,
    basis_configs: npt.NDArray[np.integer],
    target_state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-10,
    two_pattern_tolerance: float = 1e-8,
) -> LocalRecyclingScanResult:
    """Scan only two-pattern local RDM recycling candidates.

    This avoids embedding/scoring every rank-one support-null pair when the
    caller will discard all non-two-pattern candidates anyway.
    """
    basis_context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    reduced_density_matrix = _local_reduced_density_matrix_from_basis_context(
        context=basis_context,
        state=target_state,
        tolerance=rdm_tolerance,
    )

    support_basis = reduced_density_matrix.support_basis
    null_basis = reduced_density_matrix.null_basis
    support_eigenvalues = reduced_density_matrix.eigenvalues[
        reduced_density_matrix.eigenvalues > rdm_tolerance
    ]

    candidates: list[LocalRecyclingCandidate] = []

    if support_basis.shape[1] == 0 or null_basis.shape[1] == 0:
        return LocalRecyclingScanResult(
            reduced_density_matrix=reduced_density_matrix,
            candidates=(),
        )

    alpha_two_pattern_supports = tuple(
        _two_pattern_support_indices(
            support_basis[:, alpha_index],
            tolerance=two_pattern_tolerance,
        )
        for alpha_index in range(support_basis.shape[1])
    )
    beta_two_pattern_supports = tuple(
        _two_pattern_support_indices(
            null_basis[:, beta_index],
            tolerance=two_pattern_tolerance,
        )
        for beta_index in range(null_basis.shape[1])
    )

    embedding_context: _LocalPatternEmbeddingContext | None = None

    for alpha_index in range(support_basis.shape[1]):
        alpha_support = alpha_two_pattern_supports[alpha_index]

        if alpha_support is None:
            continue

        alpha_vector = support_basis[:, alpha_index]
        inflow_norm = float(np.sqrt(max(float(support_eigenvalues[alpha_index]), 0.0)))
        target_residual = 0.0
        outflow_norm = 0.0
        projector_commutator_norm = inflow_norm

        if target_residual > dark_tolerance or inflow_norm <= inflow_tolerance:
            continue

        for beta_index in range(null_basis.shape[1]):
            if beta_two_pattern_supports[beta_index] != alpha_support:
                continue

            beta_vector = null_basis[:, beta_index]
            structure = _detect_two_pattern_recycling_structure_from_vectors(
                variable_indices=reduced_density_matrix.variable_indices,
                alpha_index=int(alpha_index),
                beta_index=int(beta_index),
                local_patterns=reduced_density_matrix.local_patterns,
                alpha=alpha_vector,
                beta=beta_vector,
                tolerance=two_pattern_tolerance,
            )

            if structure is None:
                continue

            if embedding_context is None:
                embedding_context = _embedding_context_from_basis_context(basis_context)

            local_operator = np.outer(alpha_vector, beta_vector.conj())
            jump = _embed_local_pattern_operator_from_context(
                context=embedding_context,
                local_operator=local_operator,
            )
            candidates.append(
                LocalRecyclingCandidate(
                    variable_indices=reduced_density_matrix.variable_indices,
                    alpha_index=int(alpha_index),
                    beta_index=int(beta_index),
                    jump=jump,
                    target_residual=target_residual,
                    inflow_norm=inflow_norm,
                    outflow_norm=outflow_norm,
                    projector_commutator_norm=projector_commutator_norm,
                    local_alpha_vector=alpha_vector.astype(np.complex128),
                    local_beta_vector=beta_vector.astype(np.complex128),
                )
            )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.inflow_norm),
            float(candidate.target_residual),
            int(candidate.jump.nnz),
        ),
    )

    return LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=tuple(candidates),
    )


def build_local_recycling_jumps_from_regions(
    *,
    basis_configs: npt.NDArray[np.integer],
    target_state: npt.ArrayLike,
    regions: tuple[tuple[int, ...], ...],
    source: RecyclingJumpSource = "local_rdm_two_pattern",
    max_jumps_per_region: int = 1,
    rdm_tolerance: float = 1e-10,
    dark_tolerance: float = 1e-10,
    inflow_tolerance: float = 1e-12,
    max_candidates_per_region: int | None = None,
    prefer_sparse: bool = True,
    two_pattern_tolerance: float = 1e-8,
) -> LocalRecyclingBuildResult:
    """Scan several regions and return selected local recycling jumps."""
    scan_results: list[LocalRecyclingScanResult] = []
    selections: list[LocalRecyclingSelection] = []
    scan_result_cache: dict[tuple[int, ...], LocalRecyclingScanResult] = {}

    if source == "none":
        return LocalRecyclingBuildResult(scan_results=(), selections=())

    for region in regions:
        region_key = tuple(int(index) for index in region)
        scan_result = scan_result_cache.get(region_key)

        if scan_result is None:
            if source == "local_rdm_two_pattern":
                scan_result = _scan_local_two_pattern_recycling_candidates(
                    basis_configs=basis_configs,
                    target_state=target_state,
                    variable_indices=region_key,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                    two_pattern_tolerance=two_pattern_tolerance,
                )
            else:
                scan_result = scan_local_recycling_candidates(
                    basis_configs=basis_configs,
                    target_state=target_state,
                    variable_indices=region_key,
                    rdm_tolerance=rdm_tolerance,
                    dark_tolerance=dark_tolerance,
                    inflow_tolerance=inflow_tolerance,
                    max_candidates=max_candidates_per_region,
                )
            scan_result_cache[region_key] = scan_result

        scan_results.append(scan_result)

        selections.extend(
            select_local_recycling_candidates(
                scan_result=scan_result,
                source=source,
                max_candidates=max_jumps_per_region,
                prefer_sparse=prefer_sparse,
                two_pattern_tolerance=two_pattern_tolerance,
            )
        )

    return LocalRecyclingBuildResult(
        scan_results=tuple(scan_results),
        selections=tuple(selections),
    )
