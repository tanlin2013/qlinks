from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


def _as_csr(operator: Any) -> sp.csr_array:
    if hasattr(operator, "tocsr"):
        return operator.tocsr()
    return sp.csr_array(operator)


def _normalize_state_columns(
    states: npt.ArrayLike,
    *,
    tolerance: float,
) -> tuple[npt.NDArray[np.complex128], float]:
    matrix = np.asarray(states, dtype=np.complex128)

    if matrix.ndim == 1:
        matrix = matrix.reshape(matrix.size, 1)
    elif matrix.ndim != 2:
        raise ValueError("states must be one- or two-dimensional.")

    if matrix.shape[0] < matrix.shape[1]:
        # This is only a convenience heuristic.  Most callers pass columns, but
        # small test/state lists often come as rows.
        row_norms = np.linalg.norm(matrix, axis=1)
        column_norms = np.linalg.norm(matrix, axis=0)
        if np.count_nonzero(row_norms > tolerance) <= np.count_nonzero(column_norms > tolerance):
            matrix = matrix.T

    if matrix.shape[1] == 0:
        raise ValueError("states must contain at least one vector.")

    q, r = np.linalg.qr(matrix)
    diagonal = np.abs(np.diag(r))
    rank = int(np.count_nonzero(diagonal > tolerance))
    if rank == 0:
        raise ValueError("states have numerical rank zero.")

    q = q[:, :rank].astype(np.complex128, copy=False)
    gram_residual = float(np.linalg.norm(q.conj().T @ q - np.eye(rank)))
    return q, gram_residual


@dataclass(frozen=True, slots=True)
class DarkOperatorTerm:
    """One non-negligible coefficient in a dark detector candidate."""

    operator_index: int
    operator_name: str
    coefficient: complex
    weight: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "operator_index": self.operator_index,
            "operator_name": self.operator_name,
            "coefficient": self.coefficient,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class ManifoldDarkOperatorCandidate:
    """Linear-combination detector satisfying ``D P_M ~= 0``."""

    candidate_index: int
    coefficients: npt.NDArray[np.complex128]
    action_residual: float
    relative_action_residual: float
    operator_frobenius_norm: float
    terms: tuple[DarkOperatorTerm, ...]

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def is_dark(self) -> bool:
        return self.relative_action_residual <= 1.0e-10

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "coefficients": tuple(complex(value) for value in self.coefficients),
            "action_residual": self.action_residual,
            "relative_action_residual": self.relative_action_residual,
            "operator_frobenius_norm": self.operator_frobenius_norm,
            "n_terms": self.n_terms,
            "terms": tuple(term.to_summary_dict() for term in self.terms),
        }


@dataclass(frozen=True, slots=True)
class ManifoldDarkOperatorBasisReport:
    """Nullspace report for collective local operators dark on a manifold.

    Given an operator basis ``O_a`` and target manifold basis ``Q``, this report
    solves

        sum_a c_a O_a Q = 0.

    A nonzero solution is a collective dark detector for the supplied manifold.
    This is strictly more general than the local RDM null-space test: each
    individual region may have full local support, while a sum of local terms can
    still annihilate the manifold by cancellation.
    """

    operator_names: tuple[str, ...]
    manifold_dimension: int
    hilbert_dimension: int
    gram_residual: float
    constraint_matrix_shape: tuple[int, int]
    constraint_rank: int
    detector_nullity: int
    singular_values: npt.NDArray[np.float64]
    cutoff: float
    candidates: tuple[ManifoldDarkOperatorCandidate, ...]
    tolerance: float

    @property
    def n_operators(self) -> int:
        return len(self.operator_names)

    @property
    def has_dark_detectors(self) -> bool:
        return self.detector_nullity > 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_operators": self.n_operators,
            "operator_names": self.operator_names,
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "gram_residual": self.gram_residual,
            "constraint_matrix_shape": self.constraint_matrix_shape,
            "constraint_rank": self.constraint_rank,
            "detector_nullity": self.detector_nullity,
            "singular_values": tuple(float(value) for value in self.singular_values),
            "cutoff": self.cutoff,
            "has_dark_detectors": self.has_dark_detectors,
            "candidates": tuple(candidate.to_summary_dict() for candidate in self.candidates),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 8, max_terms: int = 8):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "ManifoldDarkOperatorBasisReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("operators", str(self.n_operators))
        overview.add_row("constraint shape", str(self.constraint_matrix_shape))
        overview.add_row("constraint rank", str(self.constraint_rank))
        overview.add_row("dark-detector nullity", str(self.detector_nullity))
        overview.add_row("cutoff", f"{self.cutoff:.3e}")

        table = Table(title="Collective dark-detector candidates")
        table.add_column("#", justify="right")
        table.add_column("residual", justify="right")
        table.add_column("relative", justify="right")
        table.add_column("||D||_F", justify="right")
        table.add_column("terms")

        for candidate in self.candidates[: max(int(max_candidates), 0)]:
            term_text = ", ".join(
                f"{term.coefficient:.3g}·{term.operator_name}"
                for term in candidate.terms[: max(int(max_terms), 0)]
            )
            if len(candidate.terms) > max_terms:
                term_text += f", … {len(candidate.terms) - max_terms} more"
            table.add_row(
                str(candidate.candidate_index),
                f"{candidate.action_residual:.3e}",
                f"{candidate.relative_action_residual:.3e}",
                f"{candidate.operator_frobenius_norm:.3e}",
                term_text,
            )

        if len(self.candidates) > max_candidates:
            table.add_row(
                "…",
                "",
                "",
                "",
                f"{len(self.candidates) - max_candidates} more candidates",
            )

        return Panel(
            Group(overview, table),
            title=Text("Manifold dark-operator basis report", style="bold magenta"),
            border_style="magenta",
        )


@dataclass(frozen=True, slots=True)
class DressedManifoldDarkDetectorCandidate:
    """One dressed jump candidate ``J = V D`` for a dark manifold.

    ``D`` is a collective detector satisfying ``D P_M ~= 0``.  The left
    multiplier ``V`` is tested as a possible recycler/inflow operator.
    """

    candidate_index: int
    detector_index: int
    detector_name: str
    left_multiplier_index: int
    left_multiplier_name: str
    dark_residual: float
    relative_dark_residual: float
    inflow_norm: float
    jump_frobenius_norm: float
    target_block_norm: float
    detector_action_residual: float
    detector_relative_action_residual: float

    @property
    def is_dark(self) -> bool:
        return self.relative_dark_residual <= 1.0e-10

    @property
    def has_inflow(self) -> bool:
        return self.inflow_norm > 1.0e-12

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "detector_index": self.detector_index,
            "detector_name": self.detector_name,
            "left_multiplier_index": self.left_multiplier_index,
            "left_multiplier_name": self.left_multiplier_name,
            "dark_residual": self.dark_residual,
            "relative_dark_residual": self.relative_dark_residual,
            "inflow_norm": self.inflow_norm,
            "jump_frobenius_norm": self.jump_frobenius_norm,
            "target_block_norm": self.target_block_norm,
            "detector_action_residual": self.detector_action_residual,
            "detector_relative_action_residual": self.detector_relative_action_residual,
            "is_dark": self.is_dark,
            "has_inflow": self.has_inflow,
        }


@dataclass(frozen=True, slots=True)
class DressedManifoldDarkDetectorReport:
    """Report for paper-style dressed jumps ``J = V D``.

    The supplied detector coefficients define operators ``D_alpha`` that are
    expected to annihilate the target manifold.  This report tests whether
    left multipliers ``V_beta`` turn those dark detectors into jump operators
    with direct inflow into the manifold.
    """

    manifold_dimension: int
    hilbert_dimension: int
    gram_residual: float
    detector_names: tuple[str, ...]
    left_multiplier_names: tuple[str, ...]
    dark_tolerance: float
    inflow_tolerance: float
    candidates: tuple[DressedManifoldDarkDetectorCandidate, ...]

    @property
    def n_detectors(self) -> int:
        return len(self.detector_names)

    @property
    def n_left_multipliers(self) -> int:
        return len(self.left_multiplier_names)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_dark_candidates(self) -> int:
        return sum(
            candidate.relative_dark_residual <= self.dark_tolerance for candidate in self.candidates
        )

    @property
    def n_candidates_with_inflow(self) -> int:
        return sum(
            candidate.relative_dark_residual <= self.dark_tolerance
            and candidate.inflow_norm > self.inflow_tolerance
            for candidate in self.candidates
        )

    @property
    def has_attractive_candidates(self) -> bool:
        return self.n_candidates_with_inflow > 0

    @property
    def best_inflow_norm(self) -> float:
        if not self.candidates:
            return 0.0
        return max(candidate.inflow_norm for candidate in self.candidates)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "gram_residual": self.gram_residual,
            "n_detectors": self.n_detectors,
            "detector_names": self.detector_names,
            "n_left_multipliers": self.n_left_multipliers,
            "left_multiplier_names": self.left_multiplier_names,
            "n_candidates": self.n_candidates,
            "n_dark_candidates": self.n_dark_candidates,
            "n_candidates_with_inflow": self.n_candidates_with_inflow,
            "has_attractive_candidates": self.has_attractive_candidates,
            "best_inflow_norm": self.best_inflow_norm,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "candidates": tuple(candidate.to_summary_dict() for candidate in self.candidates),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 24):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DressedManifoldDarkDetectorReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("detectors", str(self.n_detectors))
        overview.add_row("left multipliers", str(self.n_left_multipliers))
        overview.add_row("candidates", str(self.n_candidates))
        overview.add_row("dark candidates", str(self.n_dark_candidates))
        overview.add_row("candidates with inflow", str(self.n_candidates_with_inflow))
        overview.add_row("best inflow", f"{self.best_inflow_norm:.3e}")

        table = Table(title="Best dressed dark-detector candidates")
        table.add_column("#", justify="right")
        table.add_column("detector")
        table.add_column("left multiplier")
        table.add_column("inflow", justify="right")
        table.add_column("dark residual", justify="right")
        table.add_column("relative dark", justify="right")
        table.add_column("||J||_F", justify="right")

        sorted_candidates = sorted(
            self.candidates,
            key=lambda candidate: (
                candidate.relative_dark_residual > self.dark_tolerance,
                -candidate.inflow_norm,
                candidate.relative_dark_residual,
            ),
        )
        for candidate in sorted_candidates[: max(int(max_candidates), 0)]:
            style = "green" if candidate.inflow_norm > self.inflow_tolerance else ""
            table.add_row(
                str(candidate.candidate_index),
                candidate.detector_name,
                candidate.left_multiplier_name,
                f"{candidate.inflow_norm:.3e}",
                f"{candidate.dark_residual:.3e}",
                f"{candidate.relative_dark_residual:.3e}",
                f"{candidate.jump_frobenius_norm:.3e}",
                style=style,
            )

        if len(sorted_candidates) > max_candidates:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                f"{len(sorted_candidates) - max_candidates} more candidates",
            )

        return Panel(
            Group(overview, table),
            title=Text("Dressed manifold dark-detector report", style="bold yellow"),
            border_style="yellow",
        )


@dataclass(frozen=True, slots=True)
class RecycledManifoldDarkDetectorCandidate:
    """One candidate jump ``J = R D`` for a dark manifold.

    ``D`` is a collective detector satisfying ``D P_M ~= 0``.  ``R`` is a
    local recycler/matrix-unit operator embedded on one bounded region.  Unlike
    a standalone RDM recycler, ``R`` does not need to annihilate the target
    manifold because target darkness is supplied by the right detector.
    """

    candidate_index: int
    detector_index: int
    detector_name: str
    region_index: int
    variable_indices: tuple[int, ...]
    local_dim: int
    recycler_index: int
    recycler_name: str
    dark_residual: float
    relative_dark_residual: float
    inflow_norm: float
    jump_frobenius_norm: float
    target_block_norm: float
    detector_action_residual: float
    detector_relative_action_residual: float
    recycler_frobenius_norm: float
    recycler_nnz: int
    jump_nnz: int

    @property
    def n_variables(self) -> int:
        return len(self.variable_indices)

    @property
    def is_dark(self) -> bool:
        return self.relative_dark_residual <= 1.0e-10

    @property
    def has_inflow(self) -> bool:
        return self.inflow_norm > 1.0e-12

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "detector_index": self.detector_index,
            "detector_name": self.detector_name,
            "region_index": self.region_index,
            "variable_indices": self.variable_indices,
            "n_variables": self.n_variables,
            "local_dim": self.local_dim,
            "recycler_index": self.recycler_index,
            "recycler_name": self.recycler_name,
            "dark_residual": self.dark_residual,
            "relative_dark_residual": self.relative_dark_residual,
            "inflow_norm": self.inflow_norm,
            "jump_frobenius_norm": self.jump_frobenius_norm,
            "target_block_norm": self.target_block_norm,
            "detector_action_residual": self.detector_action_residual,
            "detector_relative_action_residual": self.detector_relative_action_residual,
            "recycler_frobenius_norm": self.recycler_frobenius_norm,
            "recycler_nnz": self.recycler_nnz,
            "jump_nnz": self.jump_nnz,
            "is_dark": self.is_dark,
            "has_inflow": self.has_inflow,
        }


@dataclass(frozen=True, slots=True)
class RecycledManifoldDarkDetectorReport:
    """Report for RDM/matrix-unit recycled dark-detector jumps ``J = R D``.

    The report is meant as a necessary-condition scan for attractive manifold
    Lindblad constructions.  Candidates with small dark residual and nonzero
    inflow satisfy ``J P_M ~= 0`` and ``P_M J (I-P_M) != 0``.  A selected set of
    such jumps still needs the full dark-manifold diagnostic to rule out closed
    complement sectors.
    """

    manifold_dimension: int
    hilbert_dimension: int
    gram_residual: float
    detector_names: tuple[str, ...]
    region_variable_indices: tuple[tuple[int, ...], ...]
    local_dims: tuple[int, ...]
    recycler_source: str
    n_tested_candidates: int
    dark_tolerance: float
    inflow_tolerance: float
    candidates: tuple[RecycledManifoldDarkDetectorCandidate, ...]

    @property
    def n_detectors(self) -> int:
        return len(self.detector_names)

    @property
    def n_regions(self) -> int:
        return len(self.region_variable_indices)

    @property
    def max_region_size(self) -> int:
        return max((len(region) for region in self.region_variable_indices), default=0)

    @property
    def max_local_dim(self) -> int:
        return max(self.local_dims, default=0)

    @property
    def total_local_recyclers_per_detector(self) -> int:
        if self.recycler_source == "matrix_units":
            return sum(local_dim * local_dim for local_dim in self.local_dims)
        if self.recycler_source == "rdm_support_matrix_units":
            # The exact number is already represented by n_tested_candidates.
            return self.n_tested_candidates // max(self.n_detectors, 1)
        return self.n_tested_candidates // max(self.n_detectors, 1)

    @property
    def n_reported_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_dark_candidates(self) -> int:
        return sum(
            candidate.relative_dark_residual <= self.dark_tolerance for candidate in self.candidates
        )

    @property
    def n_candidates_with_inflow(self) -> int:
        return sum(
            candidate.relative_dark_residual <= self.dark_tolerance
            and candidate.inflow_norm > self.inflow_tolerance
            for candidate in self.candidates
        )

    @property
    def has_attractive_candidates(self) -> bool:
        return self.n_candidates_with_inflow > 0

    @property
    def best_inflow_norm(self) -> float:
        if not self.candidates:
            return 0.0
        return max(candidate.inflow_norm for candidate in self.candidates)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "gram_residual": self.gram_residual,
            "n_detectors": self.n_detectors,
            "detector_names": self.detector_names,
            "n_regions": self.n_regions,
            "region_variable_indices": self.region_variable_indices,
            "max_region_size": self.max_region_size,
            "local_dims": self.local_dims,
            "max_local_dim": self.max_local_dim,
            "recycler_source": self.recycler_source,
            "total_local_recyclers_per_detector": self.total_local_recyclers_per_detector,
            "n_tested_candidates": self.n_tested_candidates,
            "n_reported_candidates": self.n_reported_candidates,
            "n_dark_candidates": self.n_dark_candidates,
            "n_candidates_with_inflow": self.n_candidates_with_inflow,
            "has_attractive_candidates": self.has_attractive_candidates,
            "best_inflow_norm": self.best_inflow_norm,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "candidates": tuple(candidate.to_summary_dict() for candidate in self.candidates),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 24):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "RecycledManifoldDarkDetectorReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("detectors", str(self.n_detectors))
        overview.add_row("regions", str(self.n_regions))
        overview.add_row("max region size", str(self.max_region_size))
        overview.add_row("max local dim", str(self.max_local_dim))
        overview.add_row("recycler source", self.recycler_source)
        overview.add_row("tested candidates", str(self.n_tested_candidates))
        overview.add_row("reported candidates", str(self.n_reported_candidates))
        overview.add_row("candidates with inflow", str(self.n_candidates_with_inflow))
        overview.add_row("best inflow", f"{self.best_inflow_norm:.3e}")

        table = Table(title="Best recycled dark-detector candidates")
        table.add_column("#", justify="right")
        table.add_column("detector")
        table.add_column("region")
        table.add_column("recycler")
        table.add_column("inflow", justify="right")
        table.add_column("relative dark", justify="right")
        table.add_column("||J||_F", justify="right")
        table.add_column("nnz", justify="right")

        sorted_candidates = sorted(
            self.candidates,
            key=lambda candidate: (
                candidate.relative_dark_residual > self.dark_tolerance,
                -candidate.inflow_norm,
                candidate.relative_dark_residual,
                candidate.jump_nnz,
            ),
        )
        for candidate in sorted_candidates[: max(int(max_candidates), 0)]:
            style = "green" if candidate.inflow_norm > self.inflow_tolerance else ""
            table.add_row(
                str(candidate.candidate_index),
                candidate.detector_name,
                str(candidate.variable_indices),
                candidate.recycler_name,
                f"{candidate.inflow_norm:.3e}",
                f"{candidate.relative_dark_residual:.3e}",
                f"{candidate.jump_frobenius_norm:.3e}",
                str(candidate.jump_nnz),
                style=style,
            )

        if len(sorted_candidates) > max_candidates:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{len(sorted_candidates) - max_candidates} more candidates",
            )

        return Panel(
            Group(overview, table),
            title=Text("Recycled manifold dark-detector report", style="bold green"),
            border_style="green",
        )


def _combined_operator_frobenius_norm(
    *,
    operators: tuple[sp.csr_array, ...],
    coefficients: npt.NDArray[np.complex128],
) -> float:
    if len(operators) == 0:
        return 0.0
    combined = sp.csr_array(operators[0].shape, dtype=np.complex128)
    for coefficient, operator in zip(coefficients, operators, strict=True):
        if abs(coefficient) == 0.0:
            continue
        combined = combined + coefficient * operator
    return float(sp.linalg.norm(combined))


def diagnose_manifold_dark_operator_basis(
    *,
    states: npt.ArrayLike,
    operators: tuple[Any, ...] | list[Any],
    operator_names: tuple[str, ...] | list[str] | None = None,
    tolerance: float = 1.0e-10,
    coefficient_tolerance: float = 1.0e-8,
    max_candidates: int | None = 16,
) -> ManifoldDarkOperatorBasisReport:
    """Find linear combinations of supplied operators annihilating a manifold.

    Args:
        states: Target manifold basis with shape ``(dim, n_states)`` or rows as
            states.  The columns are orthonormalized before the nullspace solve.
        operators: Operator basis matrices with the same Hilbert dimension.
        operator_names: Optional names for the operators.
        tolerance: Absolute/relative SVD tolerance used for the dark-detector
            nullspace.
        coefficient_tolerance: Coefficient magnitude threshold for term readout.
        max_candidates: Maximum number of nullspace candidates to store.  Use
            ``None`` to keep all candidates.

    Returns:
        A report whose candidate coefficient columns define
        ``D=sum_a c_a O_a`` with ``D P_M ~= 0``.
    """
    operator_matrices = tuple(_as_csr(operator) for operator in operators)
    if len(operator_matrices) == 0:
        raise ValueError("operators must contain at least one matrix.")

    state_basis, gram_residual = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    for operator in operator_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(
                "operator has incompatible shape: " f"{operator.shape} != {(dim, dim)}."
            )

    if operator_names is None:
        names = tuple(f"O_{index}" for index in range(len(operator_matrices)))
    else:
        names = tuple(str(name) for name in operator_names)
        if len(names) != len(operator_matrices):
            raise ValueError("operator_names length must match operators length.")

    action_columns = [
        np.asarray(operator @ state_basis, dtype=np.complex128).reshape(-1)
        for operator in operator_matrices
    ]
    constraint_matrix = np.column_stack(action_columns).astype(np.complex128, copy=False)

    _, singular_values, vh = np.linalg.svd(constraint_matrix, full_matrices=True)
    if singular_values.size == 0:
        cutoff = float(tolerance)
        rank = 0
    else:
        cutoff = float(tolerance * max(float(singular_values[0]), 1.0))
        rank = int(np.count_nonzero(singular_values > cutoff))

    nullspace = vh.conj().T[:, rank:]
    detector_nullity = int(nullspace.shape[1])

    candidate_columns = nullspace
    if max_candidates is not None:
        candidate_columns = candidate_columns[:, : max(int(max_candidates), 0)]

    candidates: list[ManifoldDarkOperatorCandidate] = []
    for candidate_index in range(candidate_columns.shape[1]):
        coefficients = np.asarray(candidate_columns[:, candidate_index], dtype=np.complex128)
        coefficient_norm = float(np.linalg.norm(coefficients))
        if coefficient_norm == 0.0:
            continue
        coefficients = coefficients / coefficient_norm
        residual = float(np.linalg.norm(constraint_matrix @ coefficients))
        operator_norm = _combined_operator_frobenius_norm(
            operators=operator_matrices,
            coefficients=coefficients,
        )
        relative_residual = residual / max(operator_norm, 1.0)

        terms = tuple(
            DarkOperatorTerm(
                operator_index=int(index),
                operator_name=names[index],
                coefficient=complex(coefficient),
                weight=float(abs(coefficient)),
            )
            for index, coefficient in sorted(
                enumerate(coefficients),
                key=lambda item: -abs(item[1]),
            )
            if abs(coefficient) > coefficient_tolerance
        )

        candidates.append(
            ManifoldDarkOperatorCandidate(
                candidate_index=int(candidate_index),
                coefficients=coefficients,
                action_residual=residual,
                relative_action_residual=float(relative_residual),
                operator_frobenius_norm=operator_norm,
                terms=terms,
            )
        )

    return ManifoldDarkOperatorBasisReport(
        operator_names=names,
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        gram_residual=gram_residual,
        constraint_matrix_shape=tuple(int(value) for value in constraint_matrix.shape),
        constraint_rank=rank,
        detector_nullity=detector_nullity,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        cutoff=cutoff,
        candidates=tuple(candidates),
        tolerance=float(tolerance),
    )


def _combined_operator(
    *,
    operators: tuple[sp.csr_array, ...],
    coefficients: npt.NDArray[np.complex128],
) -> sp.csr_array:
    if len(operators) == 0:
        raise ValueError("operators must contain at least one matrix.")
    combined = sp.csr_array(operators[0].shape, dtype=np.complex128)
    for coefficient, operator in zip(coefficients, operators, strict=True):
        if abs(coefficient) == 0.0:
            continue
        combined = combined + coefficient * operator
    return combined.tocsr()


def _projected_inflow_norm(
    *,
    jump: sp.csr_array,
    state_basis: npt.NDArray[np.complex128],
) -> tuple[float, float]:
    """Return ``||P J (I-P)||_F`` and ``||P J P||_F`` for ``P=QQ^dag``."""
    adjoint_action = np.asarray(jump.conj().T @ state_basis, dtype=np.complex128)
    left_projected_norm_sq = float(np.linalg.norm(adjoint_action) ** 2)
    target_block = np.asarray(state_basis.conj().T @ (jump @ state_basis), dtype=np.complex128)
    target_block_norm_sq = float(np.linalg.norm(target_block) ** 2)
    inflow_sq = max(left_projected_norm_sq - target_block_norm_sq, 0.0)
    return float(np.sqrt(inflow_sq)), float(np.sqrt(target_block_norm_sq))


def _normalize_detector_coefficients(
    detector_coefficients: npt.ArrayLike,
    *,
    n_operators: int,
) -> npt.NDArray[np.complex128]:
    coefficients = np.asarray(detector_coefficients, dtype=np.complex128)
    if coefficients.ndim == 1:
        if coefficients.shape[0] != n_operators:
            raise ValueError(
                "detector_coefficients has incompatible length: "
                f"{coefficients.shape[0]} != {n_operators}."
            )
        coefficients = coefficients.reshape(n_operators, 1)
    elif coefficients.ndim == 2:
        if coefficients.shape[0] == n_operators:
            pass
        elif coefficients.shape[1] == n_operators:
            coefficients = coefficients.T
        else:
            raise ValueError(
                "detector_coefficients must have shape "
                "(n_operators, n_detectors) or (n_detectors, n_operators)."
            )
    else:
        raise ValueError("detector_coefficients must be one- or two-dimensional.")

    if coefficients.shape[1] == 0:
        raise ValueError("detector_coefficients must contain at least one detector.")

    normalized = coefficients.copy()
    for column_index in range(normalized.shape[1]):
        norm = float(np.linalg.norm(normalized[:, column_index]))
        if norm == 0.0:
            raise ValueError("detector_coefficients contains a zero detector column.")
        normalized[:, column_index] /= norm
    return normalized


def diagnose_dressed_manifold_dark_detectors(
    *,
    states: npt.ArrayLike,
    detector_operators: tuple[Any, ...] | list[Any],
    left_multipliers: tuple[Any, ...] | list[Any],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    left_multiplier_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    max_detectors: int | None = None,
    sort_by_inflow: bool = True,
) -> DressedManifoldDarkDetectorReport:
    """Test paper-style dressed jumps ``J = V D`` for a dark manifold.

    Args:
        states: Target manifold basis.  Columns are orthonormalized.
        detector_operators: Operator basis ``O_a`` used to assemble
            ``D=sum_a c_a O_a``.
        left_multipliers: Candidate left multipliers ``V_beta``.
        detector_coefficients: Optional coefficient matrix for the detectors.
            If omitted, coefficients are taken from ``dark_operator_report``.
        dark_operator_report: Optional report from
            :func:`diagnose_manifold_dark_operator_basis`.
        detector_operator_names: Names for ``detector_operators``.  Only used
            to build default detector names.
        left_multiplier_names: Names for the left multipliers.
        detector_names: Optional explicit detector names.
        tolerance: Orthonormalization and shape-check tolerance.
        dark_tolerance: Relative dark residual threshold.
        inflow_tolerance: Direct-inflow threshold.
        max_detectors: Optional maximum number of detectors to test.
        sort_by_inflow: If true, store candidates with largest inflow first.

    Returns:
        A report of dressed candidates.  A candidate with small dark residual
        and positive inflow satisfies the necessary direct-inflow condition for
        manifold attraction, but does not by itself rule out invariant sectors
        in the complement.
    """
    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    multiplier_matrices = tuple(_as_csr(operator) for operator in left_multipliers)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")
    if len(multiplier_matrices) == 0:
        raise ValueError("left_multipliers must contain at least one matrix.")

    state_basis, gram_residual = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    for operator in detector_matrices + multiplier_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(
                "operator has incompatible shape: " f"{operator.shape} != {(dim, dim)}."
            )

    if detector_coefficients is None:
        if dark_operator_report is None:
            raise ValueError(
                "Pass detector_coefficients or dark_operator_report to define detectors."
            )
        detector_coefficients = np.column_stack(
            [candidate.coefficients for candidate in dark_operator_report.candidates]
        )

    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=len(detector_matrices),
    )
    if max_detectors is not None:
        coefficients = coefficients[:, : max(int(max_detectors), 0)]

    if detector_operator_names is None:
        operator_names = tuple(f"O_{index}" for index in range(len(detector_matrices)))
    else:
        operator_names = tuple(str(name) for name in detector_operator_names)
        if len(operator_names) != len(detector_matrices):
            raise ValueError("detector_operator_names length must match detector_operators.")

    if detector_names is None:
        names = tuple(
            _default_detector_name(
                coefficients=coefficients[:, detector_index],
                operator_names=operator_names,
            )
            for detector_index in range(coefficients.shape[1])
        )
    else:
        names = tuple(str(name) for name in detector_names)
        if len(names) != coefficients.shape[1]:
            raise ValueError("detector_names length must match detector count.")

    if left_multiplier_names is None:
        multiplier_names = tuple(f"V_{index}" for index in range(len(multiplier_matrices)))
    else:
        multiplier_names = tuple(str(name) for name in left_multiplier_names)
        if len(multiplier_names) != len(multiplier_matrices):
            raise ValueError("left_multiplier_names length must match left_multipliers.")

    candidates: list[DressedManifoldDarkDetectorCandidate] = []
    for detector_index in range(coefficients.shape[1]):
        detector = _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        detector_action_residual = float(np.linalg.norm(detector @ state_basis))
        detector_norm = float(sp.linalg.norm(detector))
        detector_relative_residual = detector_action_residual / max(detector_norm, 1.0)
        for multiplier_index, multiplier in enumerate(multiplier_matrices):
            jump = (multiplier @ detector).tocsr()
            dark_residual = float(np.linalg.norm(jump @ state_basis))
            jump_norm = float(sp.linalg.norm(jump))
            relative_dark_residual = dark_residual / max(jump_norm, 1.0)
            inflow_norm, target_block_norm = _projected_inflow_norm(
                jump=jump,
                state_basis=state_basis,
            )
            candidates.append(
                DressedManifoldDarkDetectorCandidate(
                    candidate_index=len(candidates),
                    detector_index=int(detector_index),
                    detector_name=names[detector_index],
                    left_multiplier_index=int(multiplier_index),
                    left_multiplier_name=multiplier_names[multiplier_index],
                    dark_residual=dark_residual,
                    relative_dark_residual=float(relative_dark_residual),
                    inflow_norm=inflow_norm,
                    jump_frobenius_norm=jump_norm,
                    target_block_norm=target_block_norm,
                    detector_action_residual=detector_action_residual,
                    detector_relative_action_residual=float(detector_relative_residual),
                )
            )

    if sort_by_inflow:
        candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate.relative_dark_residual > dark_tolerance,
                -candidate.inflow_norm,
                candidate.relative_dark_residual,
            ),
        )
        candidates = [
            DressedManifoldDarkDetectorCandidate(
                candidate_index=index,
                detector_index=candidate.detector_index,
                detector_name=candidate.detector_name,
                left_multiplier_index=candidate.left_multiplier_index,
                left_multiplier_name=candidate.left_multiplier_name,
                dark_residual=candidate.dark_residual,
                relative_dark_residual=candidate.relative_dark_residual,
                inflow_norm=candidate.inflow_norm,
                jump_frobenius_norm=candidate.jump_frobenius_norm,
                target_block_norm=candidate.target_block_norm,
                detector_action_residual=candidate.detector_action_residual,
                detector_relative_action_residual=candidate.detector_relative_action_residual,
            )
            for index, candidate in enumerate(candidates)
        ]

    return DressedManifoldDarkDetectorReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        gram_residual=gram_residual,
        detector_names=names,
        left_multiplier_names=multiplier_names,
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        candidates=tuple(candidates),
    )


def _normalize_local_regions(
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
) -> tuple[tuple[int, ...], ...]:
    regions: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for region_like in local_regions:
        region = tuple(sorted(int(index) for index in region_like))
        if len(region) == 0:
            raise ValueError("local_regions must not contain empty regions.")
        if region in seen:
            continue
        seen.add(region)
        regions.append(region)
    if len(regions) == 0:
        raise ValueError("local_regions must contain at least one region.")
    return tuple(regions)


def _pattern_name(pattern: tuple[int, ...]) -> str:
    return "(" + ",".join(str(int(value)) for value in pattern) + ")"


def _local_recycler_specs(
    *,
    local_patterns: tuple[tuple[int, ...], ...],
    support_basis: npt.NDArray[np.complex128],
    recycler_source: Literal["matrix_units", "rdm_support_matrix_units"],
) -> tuple[tuple[str, npt.NDArray[np.complex128]], ...]:
    local_dim = len(local_patterns)
    specs: list[tuple[str, npt.NDArray[np.complex128]]] = []

    if recycler_source == "matrix_units":
        for target_index, target_pattern in enumerate(local_patterns):
            for source_index, source_pattern in enumerate(local_patterns):
                local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
                local_operator[target_index, source_index] = 1.0
                specs.append(
                    (
                        f"{_pattern_name(target_pattern)}<-{_pattern_name(source_pattern)}",
                        local_operator,
                    )
                )
        return tuple(specs)

    if recycler_source == "rdm_support_matrix_units":
        if support_basis.shape[1] == 0:
            return ()
        for target_index in range(support_basis.shape[1]):
            target_vector = np.asarray(support_basis[:, target_index], dtype=np.complex128)
            for source_index, source_pattern in enumerate(local_patterns):
                source_vector = np.zeros(local_dim, dtype=np.complex128)
                source_vector[source_index] = 1.0
                local_operator = np.outer(target_vector, source_vector.conj())
                specs.append(
                    (
                        f"support_{target_index}<-{_pattern_name(source_pattern)}",
                        local_operator,
                    )
                )
        return tuple(specs)

    raise ValueError("recycler_source must be 'matrix_units' or 'rdm_support_matrix_units'.")


def diagnose_recycled_manifold_dark_detectors(
    *,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    max_detectors: int | None = None,
    max_report_candidates: int | None = 256,
    sort_by_inflow: bool = True,
) -> RecycledManifoldDarkDetectorReport:
    """Test local RDM/matrix-unit recyclers after dark detectors.

    This scans jumps of the form ``J = R D``.  The right detector ``D`` is a
    collective operator satisfying ``D P_M ~= 0``.  The left operator ``R`` is a
    local recycler on one bounded region.  Since target darkness comes from
    ``D``, the recycler can be a general local matrix unit and need not be dark
    on the target manifold by itself.

    Args:
        states: Target manifold basis; columns are orthonormalized.
        basis_configs: Constrained/product basis configurations aligned with
            the Hilbert-space basis used by ``detector_operators``.
        detector_operators: Operator basis used to assemble the detectors.
        local_regions: Variable-index regions where local recyclers are
            embedded.
        detector_coefficients: Optional coefficient matrix defining detectors.
            If omitted, coefficients are read from ``dark_operator_report``.
        dark_operator_report: Report from
            :func:`diagnose_manifold_dark_operator_basis`.
        detector_operator_names: Names for the detector operator basis.
        detector_names: Optional explicit detector names.
        recycler_source: ``"matrix_units"`` scans canonical local matrix units;
            ``"rdm_support_matrix_units"`` maps canonical source patterns into
            the local support eigenvectors of the target manifold RDM.
        tolerance: Orthonormalization and shape-check tolerance.
        rdm_tolerance: Local RDM support threshold.
        dark_tolerance: Relative dark residual threshold for ``J P_M``.
        inflow_tolerance: Threshold for ``||P_M J (I-P_M)||_F``.
        max_detectors: Optional maximum detector columns to scan.
        max_report_candidates: Optional maximum number of best candidates to
            retain in the report.  All candidates are still counted in
            ``n_tested_candidates``.
        sort_by_inflow: If true, report candidates with largest inflow first.

    Returns:
        A report of local-recycler dressed candidates.  Nonzero inflow is a
        necessary, not sufficient, condition for attractive dark-manifold
        dynamics.
    """
    from qlinks.open_system.local_recycling import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")

    state_basis, gram_residual = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")

    for operator in detector_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(
                "operator has incompatible shape: " f"{operator.shape} != {(dim, dim)}."
            )

    if detector_coefficients is None:
        if dark_operator_report is None:
            raise ValueError(
                "Pass detector_coefficients or dark_operator_report to define detectors."
            )
        detector_coefficients = np.column_stack(
            [candidate.coefficients for candidate in dark_operator_report.candidates]
        )

    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=len(detector_matrices),
    )
    if max_detectors is not None:
        coefficients = coefficients[:, : max(int(max_detectors), 0)]

    if detector_operator_names is None:
        operator_names = tuple(f"O_{index}" for index in range(len(detector_matrices)))
    else:
        operator_names = tuple(str(name) for name in detector_operator_names)
        if len(operator_names) != len(detector_matrices):
            raise ValueError("detector_operator_names length must match detector_operators.")

    if detector_names is None:
        names = tuple(
            _default_detector_name(
                coefficients=coefficients[:, detector_index],
                operator_names=operator_names,
            )
            for detector_index in range(coefficients.shape[1])
        )
    else:
        names = tuple(str(name) for name in detector_names)
        if len(names) != coefficients.shape[1]:
            raise ValueError("detector_names length must match detector count.")

    regions = _normalize_local_regions(local_regions)
    contexts = tuple(
        _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        for region in regions
    )
    embedding_contexts = tuple(
        _embedding_context_from_basis_context(context) for context in contexts
    )
    rdms = tuple(
        _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=state_basis,
            tolerance=rdm_tolerance,
        )
        for context in contexts
    )
    local_dims = tuple(int(rdm.local_dim) for rdm in rdms)

    detectors: list[tuple[sp.csr_array, float, float]] = []
    for detector_index in range(coefficients.shape[1]):
        detector = _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        detector_action_residual = float(np.linalg.norm(detector @ state_basis))
        detector_norm = float(sp.linalg.norm(detector))
        detector_relative_residual = detector_action_residual / max(detector_norm, 1.0)
        detectors.append((detector, detector_action_residual, detector_relative_residual))

    candidate_buffer: list[RecycledManifoldDarkDetectorCandidate] = []
    n_tested_candidates = 0

    for detector_index, (
        detector,
        detector_action_residual,
        detector_relative_residual,
    ) in enumerate(detectors):
        for region_index, (embedding_context, rdm) in enumerate(
            zip(embedding_contexts, rdms, strict=True)
        ):
            recycler_specs = _local_recycler_specs(
                local_patterns=rdm.local_patterns,
                support_basis=rdm.support_basis,
                recycler_source=recycler_source,
            )
            for recycler_index, (recycler_name, local_operator) in enumerate(recycler_specs):
                n_tested_candidates += 1
                recycler = _embed_local_pattern_operator_from_context(
                    context=embedding_context,
                    local_operator=local_operator,
                )
                if recycler.nnz == 0:
                    continue

                jump = (recycler @ detector).tocsr()
                dark_residual = float(np.linalg.norm(jump @ state_basis))
                jump_norm = float(sp.linalg.norm(jump))
                relative_dark_residual = dark_residual / max(jump_norm, 1.0)
                inflow_norm, target_block_norm = _projected_inflow_norm(
                    jump=jump,
                    state_basis=state_basis,
                )
                candidate = RecycledManifoldDarkDetectorCandidate(
                    candidate_index=len(candidate_buffer),
                    detector_index=int(detector_index),
                    detector_name=names[detector_index],
                    region_index=int(region_index),
                    variable_indices=rdm.variable_indices,
                    local_dim=rdm.local_dim,
                    recycler_index=int(recycler_index),
                    recycler_name=recycler_name,
                    dark_residual=dark_residual,
                    relative_dark_residual=float(relative_dark_residual),
                    inflow_norm=inflow_norm,
                    jump_frobenius_norm=jump_norm,
                    target_block_norm=target_block_norm,
                    detector_action_residual=detector_action_residual,
                    detector_relative_action_residual=float(detector_relative_residual),
                    recycler_frobenius_norm=float(sp.linalg.norm(recycler)),
                    recycler_nnz=int(recycler.nnz),
                    jump_nnz=int(jump.nnz),
                )
                candidate_buffer.append(candidate)

    if sort_by_inflow:
        candidate_buffer = sorted(
            candidate_buffer,
            key=lambda candidate: (
                candidate.relative_dark_residual > dark_tolerance,
                -candidate.inflow_norm,
                candidate.relative_dark_residual,
                candidate.jump_nnz,
            ),
        )

    if max_report_candidates is not None:
        candidate_buffer = candidate_buffer[: max(int(max_report_candidates), 0)]

    candidate_buffer = [
        RecycledManifoldDarkDetectorCandidate(
            candidate_index=index,
            detector_index=candidate.detector_index,
            detector_name=candidate.detector_name,
            region_index=candidate.region_index,
            variable_indices=candidate.variable_indices,
            local_dim=candidate.local_dim,
            recycler_index=candidate.recycler_index,
            recycler_name=candidate.recycler_name,
            dark_residual=candidate.dark_residual,
            relative_dark_residual=candidate.relative_dark_residual,
            inflow_norm=candidate.inflow_norm,
            jump_frobenius_norm=candidate.jump_frobenius_norm,
            target_block_norm=candidate.target_block_norm,
            detector_action_residual=candidate.detector_action_residual,
            detector_relative_action_residual=candidate.detector_relative_action_residual,
            recycler_frobenius_norm=candidate.recycler_frobenius_norm,
            recycler_nnz=candidate.recycler_nnz,
            jump_nnz=candidate.jump_nnz,
        )
        for index, candidate in enumerate(candidate_buffer)
    ]

    return RecycledManifoldDarkDetectorReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        gram_residual=gram_residual,
        detector_names=names,
        region_variable_indices=regions,
        local_dims=local_dims,
        recycler_source=recycler_source,
        n_tested_candidates=int(n_tested_candidates),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        candidates=tuple(candidate_buffer),
    )


def _default_detector_name(
    *,
    coefficients: npt.NDArray[np.complex128],
    operator_names: tuple[str, ...],
    max_terms: int = 4,
) -> str:
    terms = []
    for index, coefficient in sorted(
        enumerate(coefficients),
        key=lambda item: -abs(item[1]),
    )[:max_terms]:
        if abs(coefficient) <= 1.0e-8:
            continue
        terms.append(f"{coefficient:.3g}·{operator_names[index]}")
    if len(terms) == 0:
        return "0"
    if np.count_nonzero(np.abs(coefficients) > 1.0e-8) > max_terms:
        terms.append("…")
    return " + ".join(terms)
