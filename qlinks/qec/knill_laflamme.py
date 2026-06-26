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

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact summary of this error image."""
        return {
            "name": self.name,
            "support_variables": self.support_variables,
            "leakage_frobenius_norm": self.leakage_frobenius_norm,
            "leakage_spectral_norm": self.leakage_spectral_norm,
            "image_frobenius_norm": self.image_frobenius_norm,
            "relative_leakage_frobenius_norm": self.relative_leakage_frobenius_norm,
        }

    def to_text(self) -> str:
        from qlinks.qec.reporting import format_float

        return (
            f"{self.name}: leakage={format_float(self.leakage_frobenius_norm)} "
            f"(relative={format_float(self.relative_leakage_frobenius_norm)}), "
            f"support={self.support_variables}"
        )

    def format_summary(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()


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

    def to_summary_dict(self, *, include_matrices: bool = False) -> dict[str, object]:
        """Return a compact summary of this KL pair residual."""
        from qlinks.qec.reporting import complex_to_summary, matrix_to_summary

        summary: dict[str, object] = {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "names": self.names,
            "scalar": complex_to_summary(self.scalar),
            "frobenius_residual": self.frobenius_residual,
            "spectral_residual": self.spectral_residual,
            "relative_frobenius_residual": self.relative_frobenius_residual,
            "max_offdiagonal_abs": self.max_offdiagonal_abs,
            "diagonal_spread": self.diagonal_spread,
            "dominant_failure": self.dominant_failure,
        }
        if include_matrices:
            summary["matrix"] = matrix_to_summary(self.matrix)
            summary["residual_matrix"] = matrix_to_summary(self.residual_matrix)
        return summary

    def to_text(self) -> str:
        from qlinks.qec.reporting import format_complex, format_float

        return (
            f"{self.left_name}^† {self.right_name}: "
            f"rel-KL={format_float(self.relative_frobenius_residual)}, "
            f"||R||_F={format_float(self.frobenius_residual)}, "
            f"scalar={format_complex(self.scalar)}, "
            f"failure={self.dominant_failure}"
        )

    def format_summary(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()


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

    def to_summary_dict(self, *, max_pairs: int = 5, max_images: int = 5) -> dict[str, object]:
        """Return a compact, serialization-friendly KL diagnostic summary."""
        worst_pair = self.worst_pair
        worst_image = self.worst_error_image
        failure_counts: dict[str, int] = {}
        for pair in self.pair_reports:
            failure_counts[pair.dominant_failure] = failure_counts.get(pair.dominant_failure, 0) + 1

        top_pairs = tuple(
            pair.to_summary_dict()
            for pair in sorted(
                self.pair_reports,
                key=lambda pair: pair.relative_frobenius_residual,
                reverse=True,
            )[:max_pairs]
        )
        top_images = tuple(
            image.to_summary_dict()
            for image in sorted(
                self.error_image_reports,
                key=lambda image: image.relative_leakage_frobenius_norm,
                reverse=True,
            )[:max_images]
        )

        return {
            "code_dimension": self.code_dimension,
            "ambient_dimension": self.ambient_dimension,
            "error_set_name": self.error_set_name,
            "n_error_pairs": len(self.pair_reports),
            "n_error_images": len(self.error_image_reports),
            "tolerance": self.tolerance,
            "passes_exact_kl": self.passes_exact_kl,
            "max_frobenius_residual": self.max_frobenius_residual,
            "max_spectral_residual": self.max_spectral_residual,
            "max_relative_frobenius_residual": self.max_relative_frobenius_residual,
            "worst_pair": None if worst_pair is None else worst_pair.to_summary_dict(),
            "worst_error_image": (None if worst_image is None else worst_image.to_summary_dict()),
            "failure_counts": dict(sorted(failure_counts.items())),
            "top_pairs": top_pairs,
            "top_error_images": top_images,
        }

    def to_text(self, *, max_pairs: int = 5, max_images: int = 5) -> str:
        """Return a human-readable KL diagnostic summary."""
        from qlinks.qec.reporting import format_bool, format_float, format_key_value_lines

        summary = self.to_summary_dict(max_pairs=max_pairs, max_images=max_images)
        lines = [
            format_key_value_lines(
                f"Knill-Laflamme report: {self.error_set_name}",
                (
                    ("code dimension", self.code_dimension),
                    ("ambient dimension", self.ambient_dimension),
                    ("error pairs", len(self.pair_reports)),
                    ("passes exact KL", format_bool(self.passes_exact_kl)),
                    (
                        "max relative KL residual",
                        format_float(self.max_relative_frobenius_residual),
                    ),
                    ("max ||R||_F", format_float(self.max_frobenius_residual)),
                    ("max ||R||_2", format_float(self.max_spectral_residual)),
                    ("failure counts", summary["failure_counts"]),
                ),
            )
        ]

        if self.worst_pair is not None:
            lines.append("worst KL pairs")
            for pair in sorted(
                self.pair_reports,
                key=lambda pair: pair.relative_frobenius_residual,
                reverse=True,
            )[:max_pairs]:
                lines.append(f"  - {pair.to_text()}")

        if self.worst_error_image is not None:
            lines.append("largest leakage images")
            for image in sorted(
                self.error_image_reports,
                key=lambda image: image.relative_leakage_frobenius_norm,
                reverse=True,
            )[:max_images]:
                lines.append(f"  - {image.to_text()}")

        return "\n".join(lines)

    def format_summary(self, *, max_pairs: int = 5, max_images: int = 5) -> str:
        return self.to_text(max_pairs=max_pairs, max_images=max_images)

    def __str__(self) -> str:
        return self.to_text(max_pairs=3, max_images=3)

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_pairs: int = 8, max_images: int = 8):
        """Return a rich renderable KL diagnostic summary."""
        from rich.console import Group

        from qlinks.qec.reporting import add_summary_rows, format_bool, format_float, require_rich

        _group, Panel, Table, _text = require_rich("KnillLaflammeReport")
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        add_summary_rows(
            overview,
            (
                ("code dimension", self.code_dimension),
                ("ambient dimension", self.ambient_dimension),
                ("error pairs", len(self.pair_reports)),
                ("passes exact KL", format_bool(self.passes_exact_kl)),
                ("max relative KL residual", format_float(self.max_relative_frobenius_residual)),
                ("max ||R||_F", format_float(self.max_frobenius_residual)),
                ("max ||R||_2", format_float(self.max_spectral_residual)),
            ),
        )

        pair_table = Table(title="Worst KL pairs")
        pair_table.add_column("left")
        pair_table.add_column("right")
        pair_table.add_column("rel KL", justify="right")
        pair_table.add_column("||R||_F", justify="right")
        pair_table.add_column("failure")
        for pair in sorted(
            self.pair_reports,
            key=lambda pair: pair.relative_frobenius_residual,
            reverse=True,
        )[:max_pairs]:
            pair_table.add_row(
                pair.left_name,
                pair.right_name,
                format_float(pair.relative_frobenius_residual),
                format_float(pair.frobenius_residual),
                pair.dominant_failure,
            )

        leakage_table = Table(title="Largest leakage images")
        leakage_table.add_column("error")
        leakage_table.add_column("relative leakage", justify="right")
        leakage_table.add_column("||leakage||_F", justify="right")
        leakage_table.add_column("support")
        for image in sorted(
            self.error_image_reports,
            key=lambda image: image.relative_leakage_frobenius_norm,
            reverse=True,
        )[:max_images]:
            leakage_table.add_row(
                image.name,
                format_float(image.relative_leakage_frobenius_norm),
                format_float(image.leakage_frobenius_norm),
                str(image.support_variables),
            )

        return Panel(
            Group(overview, pair_table, leakage_table),
            title=f"Knill-Laflamme report: {self.error_set_name}",
        )


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
