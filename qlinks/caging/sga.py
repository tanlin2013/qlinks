"""Spectrum-generating-algebra diagnostics for cage manifolds.

The routines in this module are intentionally diagnostic-level tools.  They do
not assume a particular real-space model.  Given a candidate manifold of exact
or approximate eigenstates and a basis of candidate operators, they test whether
an operator acts like a single-frequency ladder inside the manifold,

    [H, Q] P_M ~= omega Q P_M,

and whether the action leaks outside the supplied manifold.  This is the first
step toward reverse engineering SGA-like structures before using them in a
Lindblad construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.caging.search import CageRecord
from qlinks.models.local_terms import (
    LocalOperatorKind,
    LocalTermDescriptor,
    LocalTermKind,
)


@dataclass(frozen=True, slots=True)
class SGATransition:
    """One sizeable projected transition induced by a candidate ladder operator."""

    target_index: int
    source_index: int
    energy_difference: complex
    amplitude: complex
    weight: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "target_index": self.target_index,
            "source_index": self.source_index,
            "energy_difference": self.energy_difference,
            "amplitude": self.amplitude,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class SGAOperatorDiagnostic:
    """Diagnostic for one operator as an approximate SGA ladder on a manifold."""

    operator_name: str
    frequency: complex
    energies: npt.NDArray[np.complex128]
    state_eigen_residuals: npt.NDArray[np.float64]
    gram_residual: float
    projected_action: npt.NDArray[np.complex128]
    projected_action_norm: float
    total_action_norm: float
    in_frequency_action_norm: float
    off_frequency_action_norm: float
    leakage_norm: float
    relative_leakage_norm: float
    relative_off_frequency_norm: float
    algebra_residual_norm: float
    relative_algebra_residual_norm: float
    transitions: tuple[SGATransition, ...]
    tolerance: float
    frequency_tolerance: float

    @property
    def has_nonzero_action(self) -> bool:
        return self.total_action_norm > self.tolerance

    @property
    def closes_on_manifold(self) -> bool:
        return self.relative_leakage_norm <= self.tolerance

    @property
    def has_single_frequency_action(self) -> bool:
        return self.relative_off_frequency_norm <= self.tolerance

    @property
    def is_sga_like(self) -> bool:
        return (
            self.has_nonzero_action
            and self.closes_on_manifold
            and self.has_single_frequency_action
            and self.relative_algebra_residual_norm <= self.tolerance
        )

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "operator_name": self.operator_name,
            "frequency": self.frequency,
            "n_states": int(self.energies.size),
            "energies": tuple(complex(value) for value in self.energies),
            "state_eigen_residuals": tuple(float(value) for value in self.state_eigen_residuals),
            "gram_residual": self.gram_residual,
            "projected_action_norm": self.projected_action_norm,
            "total_action_norm": self.total_action_norm,
            "in_frequency_action_norm": self.in_frequency_action_norm,
            "off_frequency_action_norm": self.off_frequency_action_norm,
            "leakage_norm": self.leakage_norm,
            "relative_leakage_norm": self.relative_leakage_norm,
            "relative_off_frequency_norm": self.relative_off_frequency_norm,
            "algebra_residual_norm": self.algebra_residual_norm,
            "relative_algebra_residual_norm": self.relative_algebra_residual_norm,
            "n_transitions": self.n_transitions,
            "has_nonzero_action": self.has_nonzero_action,
            "closes_on_manifold": self.closes_on_manifold,
            "has_single_frequency_action": self.has_single_frequency_action,
            "is_sga_like": self.is_sga_like,
            "transitions": tuple(item.to_summary_dict() for item in self.transitions),
        }


@dataclass(frozen=True, slots=True)
class SGALadderBasisDiagnostic:
    """Nullspace search for SGA-like ladders in an operator basis.

    The constraint matrix encodes two requirements for ``Q=sum_a c_a O_a``:

    * ``Q`` maps the supplied manifold back into itself.
    * the projected action only connects states with ``E_target-E_source`` equal
      to ``frequency``.

    The returned coefficient columns are candidate analytical ladder operators
    in the supplied basis.
    """

    frequency: complex
    operator_names: tuple[str, ...]
    energies: npt.NDArray[np.complex128]
    state_eigen_residuals: npt.NDArray[np.float64]
    gram_residual: float
    constraint_matrix_shape: tuple[int, int]
    constraint_rank: int
    ladder_nullity: int
    singular_values: npt.NDArray[np.float64]
    ladder_coefficients: npt.NDArray[np.complex128]
    ladder_residuals: npt.NDArray[np.float64]
    candidate_diagnostics: tuple[SGAOperatorDiagnostic, ...]
    tolerance: float
    frequency_tolerance: float

    @property
    def has_ladder_candidates(self) -> bool:
        return self.ladder_nullity > 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "frequency": self.frequency,
            "operator_names": self.operator_names,
            "energies": tuple(complex(value) for value in self.energies),
            "state_eigen_residuals": tuple(float(value) for value in self.state_eigen_residuals),
            "gram_residual": self.gram_residual,
            "constraint_matrix_shape": self.constraint_matrix_shape,
            "constraint_rank": self.constraint_rank,
            "ladder_nullity": self.ladder_nullity,
            "singular_values": tuple(float(value) for value in self.singular_values),
            "ladder_residuals": tuple(float(value) for value in self.ladder_residuals),
            "has_ladder_candidates": self.has_ladder_candidates,
            "candidate_diagnostics": tuple(
                diagnostic.to_summary_dict() for diagnostic in self.candidate_diagnostics
            ),
        }


@dataclass(frozen=True, slots=True)
class LocalTermOperatorBasis:
    """Built local-term matrices suitable for SGA operator-basis diagnostics."""

    descriptors: tuple[LocalTermDescriptor, ...]
    operator_names: tuple[str, ...]
    operators: tuple[Any, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "operator_names": self.operator_names,
            "n_operators": len(self.operators),
            "descriptors": tuple(
                {
                    "term_id": descriptor.term_id,
                    "term_kind": descriptor.term_kind,
                    "operator_kind": descriptor.operator_kind,
                    "support_variables": descriptor.support_variables,
                    "support_links": descriptor.support_links,
                    "label": descriptor.label,
                }
                for descriptor in self.descriptors
            ),
        }


def sga_operator_diagnostic(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    operator: Any,
    operator_name: str = "Q",
    frequency: complex | None = None,
    energies: Sequence[complex] | None = None,
    tolerance: float = 1.0e-10,
    frequency_tolerance: float | None = None,
    max_transitions: int | None = 32,
) -> SGAOperatorDiagnostic:
    """Test whether one operator acts as a ladder on a candidate manifold.

    Args:
        hamiltonian: Full Hamiltonian matrix in the same basis as ``states``.
        states: Candidate manifold vectors.  Rows are interpreted as states; a
            one-dimensional input is treated as a one-state manifold.
        operator: Candidate ladder operator matrix.
        operator_name: Human-readable operator label.
        frequency: Target SGA frequency.  If omitted, the dominant frequency of
            the projected action is inferred from ``E_target-E_source``.
        energies: Optional eigenvalues for the supplied states.  If omitted,
            expectation values of ``hamiltonian`` are used.
        tolerance: Numerical tolerance for residual classification.
        frequency_tolerance: Tolerance used to group energy differences.  The
            default is ``10 * tolerance``.
        max_transitions: Maximum number of projected transitions to store.

    Returns:
        A diagnostic report with projected action, leakage, commutator residual,
        and prominent transitions.
    """
    freq_tol = 10.0 * tolerance if frequency_tolerance is None else float(frequency_tolerance)
    state_columns, gram_residual = _normalized_state_columns(states, tolerance=tolerance)
    h_matrix = _as_operator_matrix(hamiltonian)
    o_matrix = _as_operator_matrix(operator)
    _validate_square_matrix(h_matrix, name="hamiltonian")
    _validate_square_matrix(o_matrix, name="operator")
    if h_matrix.shape != o_matrix.shape:
        raise ValueError("hamiltonian and operator must have matching shapes.")
    if h_matrix.shape[0] != state_columns.shape[0]:
        raise ValueError("states have incompatible Hilbert-space dimension.")

    h_states = _matmul(h_matrix, state_columns)
    resolved_energies = _resolve_energies(
        state_columns=state_columns,
        h_states=h_states,
        energies=energies,
    )
    state_eigen_residuals = _state_eigen_residuals(
        state_columns=state_columns,
        h_states=h_states,
        energies=resolved_energies,
    )

    operator_states = _matmul(o_matrix, state_columns)
    projected_action = state_columns.conj().T @ operator_states
    total_action_norm = float(np.linalg.norm(operator_states))
    projected_action_norm = float(np.linalg.norm(projected_action))
    projected_states = state_columns @ projected_action
    leakage_norm = float(np.linalg.norm(operator_states - projected_states))
    relative_leakage_norm = leakage_norm / max(total_action_norm, tolerance)

    energy_differences = _energy_difference_matrix(resolved_energies)
    resolved_frequency = (
        _dominant_frequency_from_action(
            projected_action=projected_action,
            energy_differences=energy_differences,
            tolerance=tolerance,
            frequency_tolerance=freq_tol,
        )
        if frequency is None
        else complex(frequency)
    )
    in_frequency_mask = np.abs(energy_differences - resolved_frequency) <= freq_tol
    in_frequency_action = np.where(in_frequency_mask, projected_action, 0.0)
    off_frequency_action = np.where(in_frequency_mask, 0.0, projected_action)
    in_frequency_norm = float(np.linalg.norm(in_frequency_action))
    off_frequency_norm = float(np.linalg.norm(off_frequency_action))
    relative_off_frequency_norm = off_frequency_norm / max(projected_action_norm, tolerance)

    commutator_states = (
        _matmul(h_matrix, operator_states)
        - _matmul(o_matrix, h_states)
        - resolved_frequency * operator_states
    )
    algebra_residual_norm = float(np.linalg.norm(commutator_states))
    relative_algebra_residual_norm = algebra_residual_norm / max(total_action_norm, tolerance)

    transitions = _prominent_transitions(
        projected_action=projected_action,
        energy_differences=energy_differences,
        tolerance=tolerance,
        max_transitions=max_transitions,
    )

    return SGAOperatorDiagnostic(
        operator_name=str(operator_name),
        frequency=resolved_frequency,
        energies=resolved_energies,
        state_eigen_residuals=state_eigen_residuals,
        gram_residual=gram_residual,
        projected_action=projected_action,
        projected_action_norm=projected_action_norm,
        total_action_norm=total_action_norm,
        in_frequency_action_norm=in_frequency_norm,
        off_frequency_action_norm=off_frequency_norm,
        leakage_norm=leakage_norm,
        relative_leakage_norm=relative_leakage_norm,
        relative_off_frequency_norm=relative_off_frequency_norm,
        algebra_residual_norm=algebra_residual_norm,
        relative_algebra_residual_norm=relative_algebra_residual_norm,
        transitions=transitions,
        tolerance=float(tolerance),
        frequency_tolerance=freq_tol,
    )


def sga_ladder_basis_diagnostic(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    operators: Sequence[Any],
    frequency: complex,
    operator_names: Sequence[str] | None = None,
    energies: Sequence[complex] | None = None,
    tolerance: float = 1.0e-10,
    frequency_tolerance: float | None = None,
    max_ladder_candidates: int | None = 16,
    include_leakage_constraints: bool = True,
) -> SGALadderBasisDiagnostic:
    """Search for SGA-like ladder combinations in a supplied operator basis.

    This solves a right-nullspace problem for coefficients ``c_a`` such that
    ``Q=sum_a c_a O_a`` both closes on the supplied manifold and has no
    projected matrix elements outside the requested SGA frequency.
    """
    if not operators:
        raise ValueError("operators must contain at least one candidate operator.")

    names = _default_operator_names(operator_names, len(operators))
    freq_tol = 10.0 * tolerance if frequency_tolerance is None else float(frequency_tolerance)
    state_columns, gram_residual = _normalized_state_columns(states, tolerance=tolerance)
    h_matrix = _as_operator_matrix(hamiltonian)
    _validate_square_matrix(h_matrix, name="hamiltonian")
    if h_matrix.shape[0] != state_columns.shape[0]:
        raise ValueError("states have incompatible Hilbert-space dimension.")

    h_states = _matmul(h_matrix, state_columns)
    resolved_energies = _resolve_energies(
        state_columns=state_columns,
        h_states=h_states,
        energies=energies,
    )
    state_eigen_residuals = _state_eigen_residuals(
        state_columns=state_columns,
        h_states=h_states,
        energies=resolved_energies,
    )
    energy_differences = _energy_difference_matrix(resolved_energies)
    off_frequency_mask = np.abs(energy_differences - complex(frequency)) > freq_tol

    constraint_columns: list[npt.NDArray[np.complex128]] = []
    for operator in operators:
        o_matrix = _as_operator_matrix(operator)
        _validate_square_matrix(o_matrix, name="operator")
        if o_matrix.shape != h_matrix.shape:
            raise ValueError("all operators must have the same shape as hamiltonian.")
        operator_states = _matmul(o_matrix, state_columns)
        projected_action = state_columns.conj().T @ operator_states
        constraints = [projected_action[off_frequency_mask].reshape(-1)]
        if include_leakage_constraints:
            leakage = operator_states - state_columns @ projected_action
            constraints.append(leakage.reshape(-1))
        constraint_columns.append(np.concatenate(constraints))

    constraint_matrix = np.column_stack(constraint_columns)
    rank, singular_values, null_vectors = _right_nullspace(
        constraint_matrix,
        tolerance=tolerance,
    )
    if max_ladder_candidates is not None:
        null_vectors = null_vectors[:, : int(max_ladder_candidates)]

    residuals = np.asarray(
        [
            float(np.linalg.norm(constraint_matrix @ null_vectors[:, index]))
            for index in range(null_vectors.shape[1])
        ],
        dtype=np.float64,
    )

    candidate_diagnostics = tuple(
        sga_operator_diagnostic(
            hamiltonian=h_matrix,
            states=state_columns.T,
            operator=_operator_linear_combination(operators, null_vectors[:, index]),
            operator_name=_format_ladder_name(names, null_vectors[:, index], tolerance=tolerance),
            frequency=frequency,
            energies=resolved_energies,
            tolerance=tolerance,
            frequency_tolerance=freq_tol,
        )
        for index in range(null_vectors.shape[1])
    )

    return SGALadderBasisDiagnostic(
        frequency=complex(frequency),
        operator_names=names,
        energies=resolved_energies,
        state_eigen_residuals=state_eigen_residuals,
        gram_residual=gram_residual,
        constraint_matrix_shape=(int(constraint_matrix.shape[0]), int(constraint_matrix.shape[1])),
        constraint_rank=rank,
        ladder_nullity=int(null_vectors.shape[1]),
        singular_values=singular_values,
        ladder_coefficients=np.asarray(null_vectors, dtype=np.complex128),
        ladder_residuals=residuals,
        candidate_diagnostics=candidate_diagnostics,
        tolerance=float(tolerance),
        frequency_tolerance=freq_tol,
    )


def scan_sga_ladder_basis_frequencies(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    operators: Sequence[Any],
    operator_names: Sequence[str] | None = None,
    energies: Sequence[complex] | None = None,
    tolerance: float = 1.0e-10,
    frequency_tolerance: float | None = None,
    max_frequencies: int | None = None,
    include_zero_frequency: bool = False,
    max_ladder_candidates: int | None = 16,
) -> tuple[SGALadderBasisDiagnostic, ...]:
    """Scan distinct manifold energy gaps for SGA-like ladder candidates.

    This convenience wrapper is intended for QDM/QLM cage manifolds where the
    relevant SGA spacing is not known a priori.  Frequencies are inferred from
    all distinct ``E_target-E_source`` values of the supplied states.
    """
    freq_tol = 10.0 * tolerance if frequency_tolerance is None else float(frequency_tolerance)
    state_columns, _gram_residual = _normalized_state_columns(states, tolerance=tolerance)
    h_matrix = _as_operator_matrix(hamiltonian)
    _validate_square_matrix(h_matrix, name="hamiltonian")
    h_states = _matmul(h_matrix, state_columns)
    resolved_energies = _resolve_energies(
        state_columns=state_columns,
        h_states=h_states,
        energies=energies,
    )
    frequencies = _unique_energy_differences(
        resolved_energies,
        tolerance=tolerance,
        frequency_tolerance=freq_tol,
        include_zero_frequency=include_zero_frequency,
    )
    if max_frequencies is not None:
        frequencies = frequencies[: int(max_frequencies)]

    return tuple(
        sga_ladder_basis_diagnostic(
            hamiltonian=hamiltonian,
            states=states,
            operators=operators,
            operator_names=operator_names,
            frequency=frequency,
            energies=resolved_energies,
            tolerance=tolerance,
            frequency_tolerance=freq_tol,
            max_ladder_candidates=max_ladder_candidates,
        )
        for frequency in frequencies
    )


def sga_ladder_basis_diagnostic_from_cage_records(
    *,
    hamiltonian: Any,
    records: Sequence[CageRecord],
    operators: Sequence[Any],
    frequency: complex,
    operator_names: Sequence[str] | None = None,
    tolerance: float = 1.0e-10,
    frequency_tolerance: float | None = None,
    max_ladder_candidates: int | None = 16,
) -> SGALadderBasisDiagnostic:
    """Run :func:`sga_ladder_basis_diagnostic` on full states from cage records."""
    if not records:
        raise ValueError("records must contain at least one cage record.")
    hilbert_size = int(_as_operator_matrix(hamiltonian).shape[0])
    states = _full_state_matrix_from_cage_records(records, hilbert_size=hilbert_size)
    energies = tuple(record.cage_state.energy for record in records)
    return sga_ladder_basis_diagnostic(
        hamiltonian=hamiltonian,
        states=states,
        operators=operators,
        frequency=frequency,
        operator_names=operator_names,
        energies=energies,
        tolerance=tolerance,
        frequency_tolerance=frequency_tolerance,
        max_ladder_candidates=max_ladder_candidates,
    )


def local_term_operator_basis(
    *,
    model: Any,
    build_result: Any,
    operator_kind: LocalOperatorKind | None = "kinetic",
    term_kind: LocalTermKind | None = None,
    builder: str = "sparse",
    backend: str = "scipy",
    on_missing: str = "skip",
) -> LocalTermOperatorBasis:
    """Build local Hamiltonian terms as an operator basis for SGA diagnostics.

    This is the model-generic entry point for QDM/QLM/disk/spin-chain scans.  It
    relies only on ``model.local_term_descriptors`` and ``model.build_local_term``.
    """
    if not hasattr(model, "local_term_descriptors") or not hasattr(model, "build_local_term"):
        raise TypeError("model must expose local_term_descriptors() and build_local_term().")

    descriptors = tuple(
        model.local_term_descriptors(
            operator_kind=operator_kind,
            term_kind=term_kind,
        )
    )
    names = tuple(
        descriptor.label
        or f"{descriptor.operator_kind}_{descriptor.term_kind}_{descriptor.term_id}"
        for descriptor in descriptors
    )
    operators = tuple(
        model.build_local_term(
            descriptor,
            build_result,
            builder=builder,
            backend=backend,
            on_missing=on_missing,
        )
        for descriptor in descriptors
    )
    return LocalTermOperatorBasis(
        descriptors=descriptors,
        operator_names=names,
        operators=operators,
    )


def _unique_energy_differences(
    energies: npt.NDArray[np.complex128],
    *,
    tolerance: float,
    frequency_tolerance: float,
    include_zero_frequency: bool,
) -> tuple[complex, ...]:
    differences = []
    for target_energy in energies:
        for source_energy in energies:
            difference = complex(target_energy - source_energy)
            if not include_zero_frequency and abs(difference) <= tolerance:
                continue
            differences.append(difference)

    groups: list[complex] = []
    sorted_differences = sorted(
        differences,
        key=lambda item: (float(np.real(item)), float(np.imag(item))),
    )
    for difference in sorted_differences:
        if any(abs(difference - existing) <= frequency_tolerance for existing in groups):
            continue
        groups.append(difference)
    groups.sort(key=lambda item: (abs(item), float(np.real(item)), float(np.imag(item))))
    return tuple(groups)


def _full_state_matrix_from_cage_records(
    records: Sequence[CageRecord],
    *,
    hilbert_size: int,
) -> npt.NDArray[np.complex128]:
    rows = []
    for record in records:
        if record.full_state is not None:
            rows.append(np.asarray(record.full_state, dtype=np.complex128).reshape(-1))
            continue
        vector = np.zeros(hilbert_size, dtype=np.complex128)
        vector[np.asarray(record.support, dtype=np.int64)] = record.local_state
        rows.append(vector)
    return np.vstack(rows)


def _normalized_state_columns(
    states: npt.ArrayLike,
    *,
    tolerance: float,
) -> tuple[npt.NDArray[np.complex128], float]:
    matrix = np.asarray(states, dtype=np.complex128)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("states must be a one- or two-dimensional array.")
    if matrix.shape[0] > matrix.shape[1]:
        # The public cage APIs return row-major state matrices.  Still accept
        # column-major matrices when that is the only plausible orientation.
        matrix = matrix.T

    rows = matrix.copy()
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= tolerance):
        raise ValueError("all states must have nonzero norm.")
    rows = rows / norms[:, None]
    columns = rows.T
    gram = columns.conj().T @ columns
    gram_residual = float(np.linalg.norm(gram - np.eye(gram.shape[0], dtype=np.complex128)))
    return np.asarray(columns, dtype=np.complex128), gram_residual


def _resolve_energies(
    *,
    state_columns: npt.NDArray[np.complex128],
    h_states: npt.NDArray[np.complex128],
    energies: Sequence[complex] | None,
) -> npt.NDArray[np.complex128]:
    if energies is not None:
        result = np.asarray(tuple(complex(value) for value in energies), dtype=np.complex128)
        if result.size != state_columns.shape[1]:
            raise ValueError("energies must have the same length as states.")
        return result
    return np.asarray(
        [
            complex(np.vdot(state_columns[:, index], h_states[:, index]))
            for index in range(state_columns.shape[1])
        ],
        dtype=np.complex128,
    )


def _state_eigen_residuals(
    *,
    state_columns: npt.NDArray[np.complex128],
    h_states: npt.NDArray[np.complex128],
    energies: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [
            float(np.linalg.norm(h_states[:, index] - energies[index] * state_columns[:, index]))
            for index in range(state_columns.shape[1])
        ],
        dtype=np.float64,
    )


def _energy_difference_matrix(
    energies: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    return energies[:, None] - energies[None, :]


def _dominant_frequency_from_action(
    *,
    projected_action: npt.NDArray[np.complex128],
    energy_differences: npt.NDArray[np.complex128],
    tolerance: float,
    frequency_tolerance: float,
) -> complex:
    weights = np.abs(projected_action) ** 2
    active = weights > tolerance**2
    if not np.any(active):
        return 0.0 + 0.0j

    frequencies = energy_differences[active]
    active_weights = weights[active]
    groups: list[tuple[complex, float]] = []
    for frequency, weight in zip(frequencies, active_weights, strict=True):
        for group_index, (center, total_weight) in enumerate(groups):
            if abs(frequency - center) <= frequency_tolerance:
                new_weight = total_weight + float(weight)
                new_center = (center * total_weight + frequency * float(weight)) / new_weight
                groups[group_index] = (complex(new_center), new_weight)
                break
        else:
            groups.append((complex(frequency), float(weight)))

    return max(groups, key=lambda item: item[1])[0]


def _prominent_transitions(
    *,
    projected_action: npt.NDArray[np.complex128],
    energy_differences: npt.NDArray[np.complex128],
    tolerance: float,
    max_transitions: int | None,
) -> tuple[SGATransition, ...]:
    transitions: list[SGATransition] = []
    for target_index in range(projected_action.shape[0]):
        for source_index in range(projected_action.shape[1]):
            amplitude = complex(projected_action[target_index, source_index])
            weight = float(abs(amplitude) ** 2)
            if abs(amplitude) <= tolerance:
                continue
            transitions.append(
                SGATransition(
                    target_index=int(target_index),
                    source_index=int(source_index),
                    energy_difference=complex(energy_differences[target_index, source_index]),
                    amplitude=amplitude,
                    weight=weight,
                )
            )
    transitions.sort(key=lambda item: item.weight, reverse=True)
    if max_transitions is not None:
        transitions = transitions[: int(max_transitions)]
    return tuple(transitions)


def _as_operator_matrix(operator: Any) -> Any:
    if sp.issparse(operator):
        return sp.csr_array(operator, dtype=np.complex128)
    return np.asarray(operator, dtype=np.complex128)


def _validate_square_matrix(matrix: Any, *, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")


def _matmul(operator: Any, matrix: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    return np.asarray(operator @ matrix, dtype=np.complex128)


def _right_nullspace(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > tolerance))
    null_vectors = vh[rank:].conj().T
    return rank, np.asarray(singular_values, dtype=np.float64), np.asarray(null_vectors)


def _operator_linear_combination(
    operators: Sequence[Any],
    coefficients: npt.NDArray[np.complex128],
) -> Any:
    if len(operators) != coefficients.size:
        raise ValueError("operators and coefficients have incompatible lengths.")
    result = None
    for operator, coefficient in zip(operators, coefficients, strict=True):
        if abs(coefficient) == 0.0:
            continue
        matrix = _as_operator_matrix(operator)
        term = coefficient * matrix
        result = term if result is None else result + term
    if result is None:
        shape = _as_operator_matrix(operators[0]).shape
        return sp.csr_array(shape, dtype=np.complex128)
    return result


def _format_ladder_name(
    names: Sequence[str],
    coefficients: npt.NDArray[np.complex128],
    *,
    tolerance: float,
    max_terms: int = 4,
) -> str:
    terms = [
        (name, complex(coefficient))
        for name, coefficient in zip(names, coefficients, strict=True)
        if abs(coefficient) > tolerance
    ]
    terms.sort(key=lambda item: abs(item[1]), reverse=True)
    if not terms:
        return "0"
    pieces = [f"({coefficient:.3g})*{name}" for name, coefficient in terms[:max_terms]]
    if len(terms) > max_terms:
        pieces.append(f"... {len(terms) - max_terms} more")
    return " + ".join(pieces)


def _default_operator_names(
    names: Sequence[str] | None,
    count: int,
) -> tuple[str, ...]:
    if names is None:
        return tuple(f"O{index}" for index in range(count))
    result = tuple(str(name) for name in names)
    if len(result) != count:
        raise ValueError("operator_names must have the same length as operators.")
    return result
