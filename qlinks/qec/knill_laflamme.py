from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.encoded.binary_basis import BinaryEncodedBasis, encode_binary_config
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
    """Apply one matrix-like or configuration-space operator to code vectors.

    The error-set layer can receive operators from several builder paths:

    * dense/sparse matrices,
    * :class:`BasisOperator`,
    * configuration-space local operators with ``apply(config)``, and
    * bitmask local operators with ``apply_code(code)``.

    This helper normalizes those representations before applying the operator
    to the code-space basis vectors.
    """
    matrix_like = _as_matrix_like_operator(code_space, operator)

    try:
        image = matrix_like @ code_space.vectors
    except ValueError as exc:
        if np.asarray(matrix_like, dtype=object).ndim == 0:
            raise TypeError(
                "Could not apply QEC error operator as a matrix. "
                f"Got scalar-like object of type {type(operator).__name__}. "
                "If this is a custom operator, wrap it as a matrix-like object "
                "or implement affected_variables() together with apply(config) "
                "or apply_code(code)."
            ) from exc
        raise

    return np.asarray(image, dtype=np.complex128)


def _as_matrix_like_operator(code_space: CodeSpace, operator: Any) -> Any:
    if isinstance(operator, BasisOperator):
        return operator

    if _is_local_operator(operator):
        basis = _array_basis_for_code_space(code_space)
        return BasisOperator.from_operator(
            basis,
            operator,
        )

    if _is_update_operator(operator):
        basis = _array_basis_for_code_space(code_space)
        return _UpdateBasisOperator(basis=basis, operators=(operator,))

    if _is_bitmask_operator(operator):
        basis = _binary_basis_for_code_space(code_space)
        return _BitmaskBasisOperator(basis=basis, operators=(operator,))

    return operator


def _is_local_operator(operator: Any) -> bool:
    affected_variables = getattr(operator, "affected_variables", None)
    apply = getattr(operator, "apply", None)
    return callable(affected_variables) and callable(apply)


def _is_update_operator(operator: Any) -> bool:
    affected_variables = getattr(operator, "affected_variables", None)
    apply_update = getattr(operator, "apply_update", None)
    return callable(affected_variables) and callable(apply_update)


def _is_bitmask_operator(operator: Any) -> bool:
    affected_variables = getattr(operator, "affected_variables", None)
    apply_code = getattr(operator, "apply_code", None)
    return callable(affected_variables) and callable(apply_code)


def _array_basis_for_code_space(code_space: CodeSpace) -> Basis:
    basis = code_space.basis
    if isinstance(basis, Basis):
        return basis

    to_array_basis = getattr(basis, "to_array_basis", None)
    if callable(to_array_basis):
        return to_array_basis()

    raise TypeError(
        "Configuration-space local operators require a Basis-like code_space.basis "
        "or a basis object exposing to_array_basis()."
    )


def _binary_basis_for_code_space(code_space: CodeSpace) -> BinaryEncodedBasis:
    basis = code_space.basis
    if isinstance(basis, BinaryEncodedBasis):
        return basis

    if isinstance(basis, Basis):
        # Preserve the current basis ordering so code_space.vectors and bitmask
        # matrix-vector products use the same row/column convention.
        return BinaryEncodedBasis.from_codes(
            basis.layout,
            [encode_binary_config(config) for config in basis.iter_states(copy=False)],
            sort=False,
        )

    raise TypeError(
        "Bitmask operators require BinaryEncodedBasis, or a binary array Basis "
        "that can be encoded without changing the basis order."
    )


@dataclass(frozen=True, slots=True)
class _UpdateBasisOperator:
    """Matrix-like wrapper for update-style local operators on an array Basis."""

    basis: Basis
    operators: tuple[Any, ...]
    drop_zero_atol: float = 0.0
    dtype: npt.DTypeLike = np.complex128

    @property
    def shape(self) -> tuple[int, int]:
        return (self.basis.n_states, self.basis.n_states)

    def matvec(self, vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        x = np.asarray(vector, dtype=self.dtype)
        if x.ndim != 1:
            raise ValueError("matvec expects a one-dimensional vector.")
        if x.shape[0] != self.basis.n_states:
            raise ValueError(f"Expected vector length {self.basis.n_states}, got {x.shape[0]}.")

        y = np.zeros(self.basis.n_states, dtype=self.dtype)
        scratch = np.empty(self.basis.layout.n_variables, dtype=np.int64)

        for col, config in enumerate(self.basis.iter_states(copy=False)):
            amplitude = x[col]
            if amplitude == 0:
                continue

            for operator in self.operators:
                for action in operator.apply_update(config):
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue
                    scratch[:] = config
                    scratch[action.variable_indices] = action.new_values
                    row = self.basis.get_index(scratch)
                    if row is None:
                        continue
                    y[row] += action.coefficient * amplitude

        return y

    def matmat(self, matrix: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        x = np.asarray(matrix, dtype=self.dtype)
        if x.ndim != 2:
            raise ValueError("matmat expects a two-dimensional array.")
        if x.shape[0] != self.basis.n_states:
            raise ValueError(f"Expected matrix shape ({self.basis.n_states}, k), got {x.shape}.")

        y = np.zeros_like(x, dtype=self.dtype)
        scratch = np.empty(self.basis.layout.n_variables, dtype=np.int64)

        for col, config in enumerate(self.basis.iter_states(copy=False)):
            amplitudes = x[col, :]
            if np.all(amplitudes == 0):
                continue

            for operator in self.operators:
                for action in operator.apply_update(config):
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue
                    scratch[:] = config
                    scratch[action.variable_indices] = action.new_values
                    row = self.basis.get_index(scratch)
                    if row is None:
                        continue
                    y[row, :] += action.coefficient * amplitudes

        return y

    def __matmul__(self, rhs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        arr = np.asarray(rhs)
        if arr.ndim == 1:
            return self.matvec(arr)
        if arr.ndim == 2:
            return self.matmat(arr)
        raise ValueError("Right operand must be a vector or matrix.")


@dataclass(frozen=True, slots=True)
class _BitmaskBasisOperator:
    """Matrix-like wrapper for bitmask operators on a BinaryEncodedBasis."""

    basis: BinaryEncodedBasis
    operators: tuple[Any, ...]
    drop_zero_atol: float = 0.0
    dtype: npt.DTypeLike = np.complex128

    @property
    def shape(self) -> tuple[int, int]:
        return (self.basis.n_states, self.basis.n_states)

    def matvec(self, vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        x = np.asarray(vector, dtype=self.dtype)
        if x.ndim != 1:
            raise ValueError("matvec expects a one-dimensional vector.")
        if x.shape[0] != self.basis.n_states:
            raise ValueError(f"Expected vector length {self.basis.n_states}, got {x.shape[0]}.")

        y = np.zeros(self.basis.n_states, dtype=self.dtype)
        index = self.basis.index

        for col, code_obj in enumerate(self.basis.codes):
            amplitude = x[col]
            if amplitude == 0:
                continue

            code = int(code_obj)
            for operator in self.operators:
                for action in operator.apply_code(code):
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue
                    row = index.get(int(action.code))
                    if row is None:
                        continue
                    y[row] += action.coefficient * amplitude

        return y

    def matmat(self, matrix: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        x = np.asarray(matrix, dtype=self.dtype)
        if x.ndim != 2:
            raise ValueError("matmat expects a two-dimensional array.")
        if x.shape[0] != self.basis.n_states:
            raise ValueError(f"Expected matrix shape ({self.basis.n_states}, k), got {x.shape}.")

        y = np.zeros_like(x, dtype=self.dtype)
        index = self.basis.index

        for col, code_obj in enumerate(self.basis.codes):
            amplitudes = x[col, :]
            if np.all(amplitudes == 0):
                continue

            code = int(code_obj)
            for operator in self.operators:
                for action in operator.apply_code(code):
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue
                    row = index.get(int(action.code))
                    if row is None:
                        continue
                    y[row, :] += action.coefficient * amplitudes

        return y

    def __matmul__(self, rhs: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        arr = np.asarray(rhs)
        if arr.ndim == 1:
            return self.matvec(arr)
        if arr.ndim == 2:
            return self.matmat(arr)
        raise ValueError("Right operand must be a vector or matrix.")


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
