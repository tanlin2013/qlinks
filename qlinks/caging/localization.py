from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as scipy_linalg
from numpy.typing import NDArray

from qlinks.caging.nullspace import nullspace_svd


@dataclass(frozen=True)
class IPRLocalizationConfig:
    """Configuration for many-start IPR localization."""

    n_restarts: int = 128
    max_iter: int = 1000
    step_size: float = 0.1
    convergence_tolerance: float = 1e-12
    candidate_count: int = 64
    amplitude_tolerance: float = 1e-8
    rank_tolerance: float = 1e-8
    minimum_gap_ratio: float = 10.0
    random_seed: int | None = None


@dataclass(frozen=True)
class LocalizedState:
    """One localized vector found inside a degenerate eigenspace."""

    local_state: NDArray[np.complex128]
    coefficients: NDArray[np.complex128]
    support_mask: NDArray[np.bool_]
    ipr_value: float


def inverse_participation_ratio(
    vector: NDArray[np.complex128],
) -> float:
    """Return sum_i |v_i|^4."""
    return float(np.sum(np.abs(vector) ** 4))


def _normalize_vector(
    vector: NDArray[np.complex128],
    *,
    tolerance: float,
) -> NDArray[np.complex128]:
    norm = scipy_linalg.norm(vector)
    if norm <= tolerance:
        raise ValueError("Cannot normalize a near-zero vector.")
    return np.asarray(vector / norm, dtype=np.complex128)


def _orthonormalize_columns(
    matrix: NDArray[np.complex128],
    *,
    tolerance: float,
) -> NDArray[np.complex128]:
    """Return an orthonormal basis for the span of matrix columns."""
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)

    q_matrix, r_matrix = scipy_linalg.qr(matrix, mode="economic")
    keep_columns = np.abs(np.diag(r_matrix)) > tolerance
    return np.asarray(q_matrix[:, keep_columns], dtype=np.complex128)


def _random_unit_vector(
    dimension: int,
    *,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    real_part = rng.normal(size=dimension)
    imag_part = rng.normal(size=dimension)
    vector = real_part + 1j * imag_part
    return _normalize_vector(
        np.asarray(vector, dtype=np.complex128),
        tolerance=1e-14,
    )


def support_mask_by_largest_gap(
    vector: NDArray[np.complex128],
    *,
    minimum_amplitude: float,
    minimum_gap_ratio: float = 10.0,
) -> NDArray[np.bool_]:
    """
    Infer compact support from a localized trial vector.

    This is more robust than a fixed threshold because IPR ascent often leaves
    tiny tails on other cages. We sort amplitudes and cut at the largest gap.
    """
    amplitudes = np.abs(vector)
    support_mask = amplitudes > minimum_amplitude

    active_indices = np.flatnonzero(support_mask)
    if active_indices.size <= 1:
        return support_mask

    active_amplitudes = amplitudes[active_indices]
    order = np.argsort(active_amplitudes)[::-1]

    sorted_indices = active_indices[order]
    sorted_amplitudes = active_amplitudes[order]

    ratios = sorted_amplitudes[:-1] / np.maximum(
        sorted_amplitudes[1:],
        1e-300,
    )

    if ratios.size == 0:
        return support_mask

    gap_position = int(np.argmax(ratios))
    largest_ratio = ratios[gap_position]

    if largest_ratio < minimum_gap_ratio:
        return support_mask

    selected_indices = sorted_indices[: gap_position + 1]

    gap_support_mask = np.zeros_like(support_mask)
    gap_support_mask[selected_indices] = True

    return gap_support_mask


def maximize_ipr_once(
    eigenspace_basis: NDArray[np.complex128],
    *,
    config: IPRLocalizationConfig,
    rng: np.random.Generator,
) -> LocalizedState:
    """Find one local IPR maximum inside span(eigenspace_basis)."""
    basis = _orthonormalize_columns(
        eigenspace_basis,
        tolerance=config.rank_tolerance,
    )
    dimension = basis.shape[1]
    if dimension == 0:
        raise ValueError("Cannot localize an empty eigenspace.")

    coefficients = _random_unit_vector(dimension, rng=rng)
    step_size = config.step_size
    previous_value = -np.inf

    for _iteration_index in range(config.max_iter):
        local_state = basis @ coefficients
        ipr_value = inverse_participation_ratio(local_state)

        if abs(ipr_value - previous_value) <= config.convergence_tolerance:
            break
        previous_value = ipr_value

        gradient = 2.0 * (basis.conj().T @ ((np.abs(local_state) ** 2) * local_state))

        # Tangent-space projection for ||coefficients||_2 = 1.
        gradient = gradient - coefficients * np.real(np.vdot(coefficients, gradient))

        trial_coefficients = coefficients + step_size * gradient
        trial_coefficients = _normalize_vector(
            trial_coefficients,
            tolerance=config.rank_tolerance,
        )

        trial_state = basis @ trial_coefficients
        trial_value = inverse_participation_ratio(trial_state)

        if trial_value >= ipr_value:
            coefficients = trial_coefficients
            step_size *= 1.02
        else:
            step_size *= 0.5

        if step_size <= 1e-14:
            break

    local_state = basis @ coefficients
    local_state = _normalize_vector(
        local_state,
        tolerance=config.rank_tolerance,
    )

    support_mask = support_mask_by_largest_gap(
        local_state,
        minimum_amplitude=config.amplitude_tolerance,
        minimum_gap_ratio=config.minimum_gap_ratio,
    )

    return LocalizedState(
        local_state=np.asarray(local_state, dtype=np.complex128),
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        support_mask=support_mask,
        ipr_value=inverse_participation_ratio(local_state),
    )


def _support_key(
    support_mask: NDArray[np.bool_],
) -> tuple[int, ...]:
    return tuple(int(index) for index in np.flatnonzero(support_mask))


def exact_compact_states_from_support(
    eigenspace_basis: NDArray[np.complex128],
    support_mask: NDArray[np.bool_],
    *,
    tolerance: float,
) -> NDArray[np.complex128]:
    """
    Given a trial support C, find exact vectors in span(V)
    that vanish outside C.

    Returns columns with shape (support_size_of_parent, n_exact_states).
    """
    outside_mask = ~support_mask
    outside_block = eigenspace_basis[outside_mask, :]

    coefficient_basis = nullspace_svd(
        outside_block,
        tolerance=tolerance,
    )

    if coefficient_basis.shape[1] == 0:
        return np.zeros(
            (eigenspace_basis.shape[0], 0),
            dtype=np.complex128,
        )

    compact_states = eigenspace_basis @ coefficient_basis

    for state_index in range(compact_states.shape[1]):
        state_norm = scipy_linalg.norm(compact_states[:, state_index])
        if state_norm > tolerance:
            compact_states[:, state_index] /= state_norm

    return np.asarray(compact_states, dtype=np.complex128)


def _is_independent_column(
    selected_states: list[NDArray[np.complex128]],
    candidate_state: NDArray[np.complex128],
    *,
    tolerance: float,
) -> bool:
    candidate_state = _normalize_vector(
        candidate_state,
        tolerance=tolerance,
    )

    if len(selected_states) == 0:
        return True

    old_matrix = np.column_stack(selected_states)
    old_rank = np.linalg.matrix_rank(old_matrix, tol=tolerance)

    new_matrix = np.column_stack([old_matrix, candidate_state])
    new_rank = np.linalg.matrix_rank(new_matrix, tol=tolerance)

    return bool(new_rank > old_rank)


def localized_basis_by_many_start_ipr(
    eigenspace_basis: NDArray[np.complex128],
    *,
    config: IPRLocalizationConfig,
) -> NDArray[np.complex128]:
    """
    Build a localized basis inside a degenerate eigenspace.

    The workflow is:
    1. many-start IPR maximization,
    2. infer candidate supports,
    3. enforce exact compactness using a nullspace outside each support,
    4. rank-select independent compact states.
    """
    basis = _orthonormalize_columns(
        eigenspace_basis,
        tolerance=config.rank_tolerance,
    )
    target_dimension = basis.shape[1]

    if target_dimension <= 1:
        return basis

    rng = np.random.default_rng(config.random_seed)

    best_by_support: dict[tuple[int, ...], LocalizedState] = {}

    for _candidate_index in range(config.candidate_count):
        localized_state = maximize_ipr_once(
            basis,
            config=config,
            rng=rng,
        )

        key = _support_key(localized_state.support_mask)
        if len(key) == 0:
            continue

        old_state = best_by_support.get(key)
        if old_state is None or localized_state.ipr_value > old_state.ipr_value:
            best_by_support[key] = localized_state

    selected_states: list[NDArray[np.complex128]] = []

    candidates = sorted(
        best_by_support.values(),
        key=lambda state: state.ipr_value,
        reverse=True,
    )

    for localized_state in candidates:
        compact_states = exact_compact_states_from_support(
            basis,
            localized_state.support_mask,
            tolerance=config.rank_tolerance,
        )

        for state_index in range(compact_states.shape[1]):
            candidate_state = compact_states[:, state_index]

            if _is_independent_column(
                selected_states,
                candidate_state,
                tolerance=config.rank_tolerance,
            ):
                selected_states.append(candidate_state)

            if len(selected_states) == target_dimension:
                break

        if len(selected_states) == target_dimension:
            break

    if len(selected_states) == 0:
        return basis

    localized_basis = np.column_stack(selected_states)

    # If IPR did not recover the full degenerate space, return the localized
    # states plus a completion basis, rather than silently dropping states.
    if localized_basis.shape[1] < target_dimension:
        projector = localized_basis @ localized_basis.conj().T
        residual_basis = basis - projector @ basis
        residual_basis = _orthonormalize_columns(
            residual_basis,
            tolerance=config.rank_tolerance,
        )

        missing_dimension = target_dimension - localized_basis.shape[1]
        if residual_basis.shape[1] >= missing_dimension:
            localized_basis = np.column_stack(
                [
                    localized_basis,
                    residual_basis[:, :missing_dimension],
                ]
            )

    return _orthonormalize_columns(
        localized_basis,
        tolerance=config.rank_tolerance,
    )
