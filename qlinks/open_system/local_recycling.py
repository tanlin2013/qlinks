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


def local_reduced_density_matrix_from_state(
    *,
    basis_configs: npt.NDArray[np.integer],
    state: npt.ArrayLike,
    variable_indices: tuple[int, ...] | list[int],
    tolerance: float = 1e-10,
) -> LocalReducedDensityMatrix:
    """Compute rho_Omega for a state represented in a constrained basis."""
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

    n_variables = configs.shape[1]

    if any(index < 0 or index >= n_variables for index in variable_indices):
        raise ValueError("variable_indices contains out-of-range entries.")

    norm = np.linalg.norm(amplitudes)
    if norm == 0.0:
        raise ValueError("state must be nonzero.")

    amplitudes = amplitudes / norm

    variable_set = set(variable_indices)
    environment_indices = tuple(index for index in range(n_variables) if index not in variable_set)

    local_patterns = tuple(
        sorted(
            {tuple(int(value) for value in config[list(variable_indices)]) for config in configs}
        )
    )
    local_to_index = {pattern: index for index, pattern in enumerate(local_patterns)}

    environment_groups: dict[
        tuple[int, ...],
        list[tuple[int, np.complex128]],
    ] = {}

    for basis_index, config in enumerate(configs):
        local_pattern = tuple(int(value) for value in config[list(variable_indices)])
        environment_pattern = tuple(int(value) for value in config[list(environment_indices)])
        local_index = local_to_index[local_pattern]
        environment_groups.setdefault(environment_pattern, []).append(
            (local_index, amplitudes[basis_index])
        )

    local_dim = len(local_patterns)
    density_matrix = np.zeros((local_dim, local_dim), dtype=np.complex128)

    for local_amplitudes in environment_groups.values():
        for row_index, row_amplitude in local_amplitudes:
            for col_index, col_amplitude in local_amplitudes:
                density_matrix[row_index, col_index] += row_amplitude * col_amplitude.conj()

    density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    support_mask = eigenvalues > tolerance
    null_mask = ~support_mask

    return LocalReducedDensityMatrix(
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        density_matrix=density_matrix,
        eigenvalues=eigenvalues,
        support_basis=eigenvectors[:, support_mask].astype(np.complex128),
        null_basis=eigenvectors[:, null_mask].astype(np.complex128),
    )


def embed_local_pattern_operator(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...],
    local_patterns: tuple[tuple[int, ...], ...],
    local_operator: npt.NDArray[np.complex128],
) -> sp.csr_array:
    """Embed a local operator into the constrained full basis."""
    configs = np.asarray(basis_configs)
    variable_indices = tuple(int(index) for index in variable_indices)

    if configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_basis, n_variables).")

    local_dim = len(local_patterns)

    if local_operator.shape != (local_dim, local_dim):
        raise ValueError(
            "local_operator has incompatible shape: "
            f"{local_operator.shape} != {(local_dim, local_dim)}."
        )

    local_pattern_to_index = {pattern: index for index, pattern in enumerate(local_patterns)}
    config_to_index = {
        tuple(int(value) for value in config): index for index, config in enumerate(configs)
    }

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for source_index, source_config in enumerate(configs):
        source_local_pattern = tuple(int(value) for value in source_config[list(variable_indices)])
        source_local_index = local_pattern_to_index.get(source_local_pattern)

        if source_local_index is None:
            continue

        for target_local_index, target_local_pattern in enumerate(local_patterns):
            matrix_element = complex(local_operator[target_local_index, source_local_index])

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
            data.append(matrix_element)

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
    norm = np.linalg.norm(state)

    if norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    state = state / norm

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
    """Scan local rank-one recycling jumps from rho_Omega."""
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


def detect_two_pattern_recycling_structure(
    *,
    candidate: LocalRecyclingCandidate,
    local_patterns: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-8,
) -> TwoPatternRecyclingStructure | None:
    """Detect whether a candidate is a two-pattern |minus><plus| jump."""
    terms = local_rank_one_matrix_unit_expansion(
        local_patterns=local_patterns,
        alpha=candidate.local_alpha_vector,
        beta=candidate.local_beta_vector,
        tolerance=tolerance,
    )

    if len(terms) != 4:
        return None

    target_patterns = {term.target_pattern for term in terms}
    source_patterns = {term.source_pattern for term in terms}

    if len(target_patterns) != 2 or target_patterns != source_patterns:
        return None

    pattern_a, pattern_b = tuple(sorted(target_patterns))

    coefficient_by_pair = {
        (term.target_pattern, term.source_pattern): term.coefficient for term in terms
    }

    try:
        coefficients = np.asarray(
            [
                coefficient_by_pair[(pattern_a, pattern_a)],
                coefficient_by_pair[(pattern_a, pattern_b)],
                coefficient_by_pair[(pattern_b, pattern_a)],
                coefficient_by_pair[(pattern_b, pattern_b)],
            ],
            dtype=np.complex128,
        )
    except KeyError:
        return None

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

    return TwoPatternRecyclingStructure(
        variable_indices=tuple(int(index) for index in candidate.variable_indices),
        pattern_a=pattern_a,
        pattern_b=pattern_b,
        alpha_index=int(candidate.alpha_index),
        beta_index=int(candidate.beta_index),
        phase=best_phase,
        residual=best_residual,
        matrix_unit_terms=terms,
    )


def select_local_recycling_candidates(
    *,
    scan_result: LocalRecyclingScanResult,
    source: RecyclingJumpSource = "local_rdm_two_pattern",
    max_candidates: int = 1,
    prefer_sparse: bool = True,
    two_pattern_tolerance: float = 1e-8,
) -> tuple[LocalRecyclingSelection, ...]:
    """Select recycling candidates from one scan result."""
    if source == "none":
        return ()

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

    if source == "none":
        return LocalRecyclingBuildResult(scan_results=(), selections=())

    for region in regions:
        scan_result = scan_local_recycling_candidates(
            basis_configs=basis_configs,
            target_state=target_state,
            variable_indices=region,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_candidates=max_candidates_per_region,
        )
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
