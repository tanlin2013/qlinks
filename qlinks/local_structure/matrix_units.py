"""Local matrix-unit representations shared by caging and open-system code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class LocalMatrixUnitTerm:
    """One local matrix-unit term ``coefficient * |target><source|``."""

    coefficient: complex
    target_pattern: tuple[int, ...]
    source_pattern: tuple[int, ...]


def local_rank_one_matrix_unit_expansion(
    *,
    local_patterns: tuple[tuple[int, ...], ...],
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    tolerance: float = 1e-10,
) -> tuple[LocalMatrixUnitTerm, ...]:
    """Expand ``|alpha><beta|`` into local matrix units ``|a><b|``."""
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


def local_operator_matrix_unit_expansion(
    *,
    local_patterns: tuple[tuple[int, ...], ...],
    local_operator: npt.ArrayLike,
    tolerance: float = 1e-10,
) -> tuple[LocalMatrixUnitTerm, ...]:
    """Expand a local operator into matrix-unit terms."""
    operator = np.asarray(local_operator, dtype=np.complex128)

    local_dim = len(local_patterns)
    if operator.shape != (local_dim, local_dim):
        raise ValueError(
            "local_operator shape must match the number of local patterns: "
            f"{operator.shape} != {(local_dim, local_dim)}."
        )

    terms: list[LocalMatrixUnitTerm] = []

    for target_index, target_pattern in enumerate(local_patterns):
        for source_index, source_pattern in enumerate(local_patterns):
            coefficient = operator[target_index, source_index]

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
