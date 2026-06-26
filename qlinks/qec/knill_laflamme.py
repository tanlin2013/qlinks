from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from qlinks.operators import BasisOperator
from qlinks.qec.code_space import CodeSpace
from qlinks.qec.error_sets import ErrorOperator, LocalErrorSet

DominantFailure = Literal["scalar", "mixing", "distinguishability"]


@dataclass(frozen=True, slots=True)
class ErrorImageReport:
    """How one error maps the code space into the ambient Hilbert space."""

    name: str
    support_variables: tuple[int, ...]
    leakage_frobenius_norm: float
    leakage_spectral_norm: float
    image_frobenius_norm: float

    @property
    def relative_leakage_frobenius_norm(self) -> float:
        if self.image_frobenius_norm == 0.0:
            return 0.0
        return self.leakage_frobenius_norm / self.image_frobenius_norm


@dataclass(frozen=True, slots=True)
class KnillLaflammePairReport:
    """KL residual for one pair ``E_a^dagger E_b``."""

    left_name: str
    right_name: str
    matrix: npt.NDArray[np.complex128]
    scalar: complex
    residual_matrix: npt.NDArray[np.complex128]
    frobenius_residual: float
    spectral_residual: float
    relative_frobenius_residual: float
    max_offdiagonal_abs: float
    diagonal_spread: float
    dominant_failure: DominantFailure

    @property
    def names(self) -> tuple[str, str]:
        return (self.left_name, self.right_name)


@dataclass(frozen=True, slots=True)
class KnillLaflammeReport:
    """Complete KL diagnostic for one code space and one error set."""

    code_dimension: int
    ambient_dimension: int
    error_set_name: str
    pair_reports: tuple[KnillLaflammePairReport, ...]
    error_image_reports: tuple[ErrorImageReport, ...]
    tolerance: float = 1e-10

    @property
    def max_frobenius_residual(self) -> float:
        return max((pair.frobenius_residual for pair in self.pair_reports), default=0.0)

    @property
    def max_spectral_residual(self) -> float:
        return max((pair.spectral_residual for pair in self.pair_reports), default=0.0)

    @property
    def max_relative_frobenius_residual(self) -> float:
        return max((pair.relative_frobenius_residual for pair in self.pair_reports), default=0.0)

    @property
    def worst_pair(self) -> KnillLaflammePairReport | None:
        if len(self.pair_reports) == 0:
            return None
        return max(self.pair_reports, key=lambda pair: pair.relative_frobenius_residual)

    @property
    def worst_error_image(self) -> ErrorImageReport | None:
        if len(self.error_image_reports) == 0:
            return None
        return max(
            self.error_image_reports,
            key=lambda image: image.relative_leakage_frobenius_norm,
        )

    @property
    def passes_exact_kl(self) -> bool:
        return self.max_frobenius_residual <= self.tolerance

    def pairs_by_failure(self, failure: DominantFailure) -> tuple[KnillLaflammePairReport, ...]:
        return tuple(pair for pair in self.pair_reports if pair.dominant_failure == failure)


def diagnose_knill_laflamme(
    code_space: CodeSpace,
    errors: LocalErrorSet | list[ErrorOperator] | tuple[ErrorOperator, ...],
    *,
    tolerance: float = 1e-10,
) -> KnillLaflammeReport:
    """Check ``P E_a^dagger E_b P = alpha_ab P`` for a candidate code.

    The computation uses error images ``E_a V`` instead of constructing
    explicit products.  This works with sparse matrices, dense matrices,
    :class:`BasisOperator`, and configuration-space ``LocalOperator`` objects.
    """
    if isinstance(errors, LocalErrorSet):
        error_set_name = errors.name
        error_list = errors.errors
    else:
        error_set_name = "local_errors"
        error_list = tuple(errors)

    error_images: list[npt.NDArray[np.complex128]] = []
    image_reports: list[ErrorImageReport] = []

    for error in error_list:
        image = apply_error_to_code(code_space, error.operator)
        error_images.append(image)

        leakage = code_space.leakage_image(image)
        image_frobenius = float(np.linalg.norm(image, ord="fro"))
        image_reports.append(
            ErrorImageReport(
                name=error.name,
                support_variables=error.support_variables,
                leakage_frobenius_norm=float(np.linalg.norm(leakage, ord="fro")),
                leakage_spectral_norm=_spectral_norm(leakage),
                image_frobenius_norm=image_frobenius,
            )
        )

    pair_reports: list[KnillLaflammePairReport] = []
    eye = np.eye(code_space.dimension, dtype=np.complex128)

    for left_error, left_image in zip(error_list, error_images, strict=True):
        for right_error, right_image in zip(error_list, error_images, strict=True):
            matrix = left_image.conj().T @ right_image
            scalar = complex(np.trace(matrix) / code_space.dimension)
            residual_matrix = matrix - scalar * eye
            frobenius_residual = float(np.linalg.norm(residual_matrix, ord="fro"))
            spectral_residual = _spectral_norm(residual_matrix)
            matrix_norm = float(np.linalg.norm(matrix, ord="fro"))
            relative = 0.0 if matrix_norm == 0.0 else frobenius_residual / matrix_norm
            max_offdiagonal_abs = _max_offdiagonal_abs(matrix)
            diagonal_spread = _diagonal_spread(matrix)
            dominant_failure = _dominant_failure(
                frobenius_residual=frobenius_residual,
                tolerance=tolerance,
                max_offdiagonal_abs=max_offdiagonal_abs,
                diagonal_spread=diagonal_spread,
            )

            pair_reports.append(
                KnillLaflammePairReport(
                    left_name=left_error.name,
                    right_name=right_error.name,
                    matrix=matrix,
                    scalar=scalar,
                    residual_matrix=residual_matrix,
                    frobenius_residual=frobenius_residual,
                    spectral_residual=spectral_residual,
                    relative_frobenius_residual=relative,
                    max_offdiagonal_abs=max_offdiagonal_abs,
                    diagonal_spread=diagonal_spread,
                    dominant_failure=dominant_failure,
                )
            )

    return KnillLaflammeReport(
        code_dimension=code_space.dimension,
        ambient_dimension=code_space.ambient_dimension,
        error_set_name=error_set_name,
        pair_reports=tuple(pair_reports),
        error_image_reports=tuple(image_reports),
        tolerance=tolerance,
    )


def apply_error_to_code(code_space: CodeSpace, operator: Any) -> npt.NDArray[np.complex128]:
    """Apply one matrix-like or configuration-space operator to code vectors."""
    matrix_like = _as_matrix_like_operator(code_space, operator)
    return np.asarray(matrix_like @ code_space.vectors, dtype=np.complex128)


def _as_matrix_like_operator(code_space: CodeSpace, operator: Any) -> Any:
    if isinstance(operator, BasisOperator):
        return operator

    if _is_local_operator(operator):
        return BasisOperator.from_operator(
            code_space.basis,
            operator,
        )

    return operator


def _is_local_operator(operator: Any) -> bool:
    affected_variables = getattr(operator, "affected_variables", None)
    apply = getattr(operator, "apply", None)
    return callable(affected_variables) and callable(apply)


def _spectral_norm(matrix: npt.ArrayLike) -> float:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr, ord=2))


def _max_offdiagonal_abs(matrix: npt.NDArray[np.complex128]) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    offdiag = matrix.copy()
    np.fill_diagonal(offdiag, 0.0)
    return float(np.max(np.abs(offdiag)))


def _diagonal_spread(matrix: npt.NDArray[np.complex128]) -> float:
    diagonal = np.diag(matrix)
    if diagonal.size <= 1:
        return 0.0
    return float(np.max(np.abs(diagonal - np.mean(diagonal))))


def _dominant_failure(
    *,
    frobenius_residual: float,
    tolerance: float,
    max_offdiagonal_abs: float,
    diagonal_spread: float,
) -> DominantFailure:
    if frobenius_residual <= tolerance:
        return "scalar"
    if max_offdiagonal_abs >= diagonal_spread:
        return "mixing"
    return "distinguishability"
