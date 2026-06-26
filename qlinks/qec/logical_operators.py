from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.qec.code_space import CodeSpace
from qlinks.qec.error_sets import ErrorOperator, LocalErrorSet
from qlinks.qec.knill_laflamme import apply_error_to_code


@dataclass(frozen=True, slots=True)
class ProjectedLogicalOperator:
    """Projected action of a physical operator inside the code space."""

    name: str
    support_variables: tuple[int, ...]
    projected_matrix: npt.NDArray[np.complex128]
    scalar: complex
    traceless_matrix: npt.NDArray[np.complex128]
    traceless_frobenius_norm: float
    traceless_spectral_norm: float
    relative_traceless_frobenius_norm: float
    leakage_frobenius_norm: float
    leakage_spectral_norm: float

    @property
    def weight(self) -> int:
        return len(set(self.support_variables))


@dataclass(frozen=True, slots=True)
class LogicalOperatorReport:
    """Projected logical action of candidate physical operators."""

    code_dimension: int
    candidates: tuple[ProjectedLogicalOperator, ...]

    @property
    def best_candidate(self) -> ProjectedLogicalOperator | None:
        if len(self.candidates) == 0:
            return None
        return max(self.candidates, key=lambda candidate: candidate.traceless_spectral_norm)

    def sorted_by_logical_strength(
        self, *, descending: bool = True
    ) -> tuple[ProjectedLogicalOperator, ...]:
        return tuple(
            sorted(
                self.candidates,
                key=lambda candidate: candidate.traceless_spectral_norm,
                reverse=descending,
            )
        )

    def low_leakage_candidates(
        self, *, max_leakage: float = 1e-10
    ) -> tuple[ProjectedLogicalOperator, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.leakage_frobenius_norm <= max_leakage
        )


def search_projected_logical_operators(
    code_space: CodeSpace,
    candidate_operators: LocalErrorSet | Sequence[ErrorOperator] | Sequence[Any],
) -> LogicalOperatorReport:
    """Project candidate physical operators into the code space.

    Operators with large traceless projected action are possible logical gates
    or dangerous logical errors.  Operators with large leakage need additional
    recovery if they are meant to implement controlled gates.
    """
    candidates = _normalize_candidates(candidate_operators)
    reports: list[ProjectedLogicalOperator] = []
    eye = np.eye(code_space.dimension, dtype=np.complex128)

    for candidate in candidates:
        image = apply_error_to_code(code_space, candidate.operator)
        projected = code_space.vectors.conj().T @ image
        scalar = complex(np.trace(projected) / code_space.dimension)
        traceless = projected - scalar * eye
        leakage = code_space.leakage_image(image)
        projected_norm = float(np.linalg.norm(projected, ord="fro"))
        traceless_frobenius = float(np.linalg.norm(traceless, ord="fro"))
        relative_traceless = 0.0 if projected_norm == 0.0 else traceless_frobenius / projected_norm

        reports.append(
            ProjectedLogicalOperator(
                name=candidate.name,
                support_variables=candidate.support_variables,
                projected_matrix=projected,
                scalar=scalar,
                traceless_matrix=traceless,
                traceless_frobenius_norm=traceless_frobenius,
                traceless_spectral_norm=_spectral_norm(traceless),
                relative_traceless_frobenius_norm=relative_traceless,
                leakage_frobenius_norm=float(np.linalg.norm(leakage, ord="fro")),
                leakage_spectral_norm=_spectral_norm(leakage),
            )
        )

    return LogicalOperatorReport(
        code_dimension=code_space.dimension,
        candidates=tuple(reports),
    )


def _normalize_candidates(
    candidate_operators: LocalErrorSet | Sequence[ErrorOperator] | Sequence[Any],
) -> tuple[ErrorOperator, ...]:
    if isinstance(candidate_operators, LocalErrorSet):
        return candidate_operators.errors

    normalized: list[ErrorOperator] = []
    for operator in candidate_operators:
        if isinstance(operator, ErrorOperator):
            normalized.append(operator)
        else:
            normalized.append(ErrorOperator.from_operator(operator))
    return tuple(normalized)


def _spectral_norm(matrix: npt.ArrayLike) -> float:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr, ord=2))
