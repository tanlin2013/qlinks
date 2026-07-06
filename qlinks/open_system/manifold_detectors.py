from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
    coefficient_ipr: float
    effective_operator_count: float
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
            "coefficient_ipr": self.coefficient_ipr,
            "effective_operator_count": self.effective_operator_count,
            "n_terms": self.n_terms,
            "terms": tuple(term.to_summary_dict() for term in self.terms),
        }


@dataclass(frozen=True, slots=True)
class DarkDetectorMatrixReadout:
    """Coefficient readout for a collective dark detector.

    This intentionally does not store a dense global matrix.  The detector is a
    linear combination of the supplied detector operator family; the readout is
    meant for inspecting the algebraic structure of that combination.
    """

    detector_index: int
    label: str
    coefficients: npt.NDArray[np.complex128]
    operator_names: tuple[str, ...]
    terms: tuple[DarkOperatorTerm, ...]
    action_residual: float
    relative_action_residual: float
    operator_frobenius_norm: float
    coefficient_ipr: float
    effective_operator_count: float

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def is_local_matrix_readout(self) -> bool:
        """Whether this readout can be drawn by ``LocalBasisGridVisualizer``."""
        return False

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "detector_index": self.detector_index,
            "label": self.label,
            "coefficients": tuple(complex(value) for value in self.coefficients),
            "operator_names": self.operator_names,
            "n_terms": self.n_terms,
            "terms": tuple(term.to_summary_dict() for term in self.terms),
            "action_residual": self.action_residual,
            "relative_action_residual": self.relative_action_residual,
            "operator_frobenius_norm": self.operator_frobenius_norm,
            "coefficient_ipr": self.coefficient_ipr,
            "effective_operator_count": self.effective_operator_count,
        }


@dataclass(frozen=True, slots=True)
class LocalOperatorMatrixReadout:
    """Local matrix readout compatible with ``LocalBasisGridVisualizer``.

    The visualizer only requires ``variable_indices``, ``local_patterns``, and a
    ``local_operator``/``density_matrix`` attribute.  This readout carries the
    extra candidate metadata needed to interpret selected Lindblad recyclers and
    targeted completion operators in notebooks.
    """

    label: str
    source: str
    variable_indices: tuple[int, ...]
    local_patterns: tuple[tuple[int, ...], ...]
    local_operator: npt.NDArray[np.complex128]
    metadata: tuple[tuple[str, object], ...] = ()

    @property
    def local_dim(self) -> int:
        return len(self.local_patterns)

    @property
    def is_local_matrix_readout(self) -> bool:
        """Whether this readout can be drawn by ``LocalBasisGridVisualizer``."""
        return True

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.local_operator.shape)

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(np.abs(self.local_operator) > 0.0))

    def nonzero_matrix_elements(
        self,
        *,
        tolerance: float = 0.0,
    ) -> tuple[tuple[int, int, complex], ...]:
        """Return ``(target_index, source_index, value)`` nonzero local entries."""
        operator = np.asarray(self.local_operator, dtype=np.complex128)
        entries: list[tuple[int, int, complex]] = []
        for target_index, source_index in zip(*np.nonzero(np.abs(operator) > tolerance)):
            entries.append(
                (
                    int(target_index),
                    int(source_index),
                    complex(operator[int(target_index), int(source_index)]),
                )
            )
        return tuple(entries)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "source": self.source,
            "variable_indices": self.variable_indices,
            "local_patterns": self.local_patterns,
            "local_dim": self.local_dim,
            "shape": self.shape,
            "nnz": self.nnz,
            "nonzero_matrix_elements": self.nonzero_matrix_elements(),
            "metadata": self.metadata,
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
    candidate_strategy: str = "svd_basis"

    @property
    def n_operators(self) -> int:
        return len(self.operator_names)

    @property
    def has_dark_detectors(self) -> bool:
        return self.detector_nullity > 0

    def detector_readout(self, detector_index: int = 0) -> DarkDetectorMatrixReadout:
        """Return a coefficient readout for one dark detector candidate."""
        candidate = self.candidates[int(detector_index)]
        return DarkDetectorMatrixReadout(
            detector_index=int(candidate.candidate_index),
            label=f"detector_{candidate.candidate_index}",
            coefficients=np.asarray(candidate.coefficients, dtype=np.complex128),
            operator_names=self.operator_names,
            terms=candidate.terms,
            action_residual=float(candidate.action_residual),
            relative_action_residual=float(candidate.relative_action_residual),
            operator_frobenius_norm=float(candidate.operator_frobenius_norm),
            coefficient_ipr=float(candidate.coefficient_ipr),
            effective_operator_count=float(candidate.effective_operator_count),
        )

    def detector_readouts(
        self,
        *,
        max_readouts: int | None = None,
    ) -> tuple[DarkDetectorMatrixReadout, ...]:
        """Return coefficient readouts for reported dark detector candidates."""
        candidates = self.candidates
        if max_readouts is not None:
            candidates = candidates[: max(int(max_readouts), 0)]
        return tuple(
            DarkDetectorMatrixReadout(
                detector_index=int(candidate.candidate_index),
                label=f"detector_{candidate.candidate_index}",
                coefficients=np.asarray(candidate.coefficients, dtype=np.complex128),
                operator_names=self.operator_names,
                terms=candidate.terms,
                action_residual=float(candidate.action_residual),
                relative_action_residual=float(candidate.relative_action_residual),
                operator_frobenius_norm=float(candidate.operator_frobenius_norm),
                coefficient_ipr=float(candidate.coefficient_ipr),
                effective_operator_count=float(candidate.effective_operator_count),
            )
            for candidate in candidates
        )

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
    n_nonzero_candidates: int
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
    def candidate_report_is_truncated(self) -> bool:
        return self.n_reported_candidates < self.n_nonzero_candidates

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
            "n_nonzero_candidates": self.n_nonzero_candidates,
            "n_reported_candidates": self.n_reported_candidates,
            "candidate_report_is_truncated": self.candidate_report_is_truncated,
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
        overview.add_row("nonzero candidates", str(self.n_nonzero_candidates))
        overview.add_row("reported candidates", str(self.n_reported_candidates))
        overview.add_row("truncated report", str(self.candidate_report_is_truncated))
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


@dataclass(frozen=True, slots=True)
class RecycledManifoldCollectiveRecyclerGroup:
    """One collective local recycler replacing selected microscopic recyclers.

    The bundled jump has the form ``J = R_bundle D`` where ``D`` is the
    selected dark detector and ``R_bundle`` is a local matrix supported on one
    region. Bundling only within a fixed ``(detector_index, region_index)``
    preserves the same real-space support as the selected microscopic
    recyclers while reducing the number of Lindblad channels.
    """

    group_index: int
    detector_index: int
    detector_name: str
    region_index: int
    variable_indices: tuple[int, ...]
    local_dim: int
    candidate_indices: tuple[int, ...]
    recycler_indices: tuple[int, ...]
    recycler_names: tuple[str, ...]
    weights: tuple[complex, ...]
    local_operator: npt.NDArray[np.complex128]
    jump_frobenius_norm: float
    recycler_frobenius_norm: float
    recycler_nnz: int
    jump_nnz: int

    @property
    def n_variables(self) -> int:
        return len(self.variable_indices)

    @property
    def n_bundled_recyclers(self) -> int:
        return len(self.candidate_indices)

    @property
    def recycler_name(self) -> str:
        return f"collective[{self.n_bundled_recyclers}]"

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "group_index": self.group_index,
            "detector_index": self.detector_index,
            "detector_name": self.detector_name,
            "region_index": self.region_index,
            "variable_indices": self.variable_indices,
            "n_variables": self.n_variables,
            "local_dim": self.local_dim,
            "n_bundled_recyclers": self.n_bundled_recyclers,
            "candidate_indices": self.candidate_indices,
            "recycler_indices": self.recycler_indices,
            "recycler_names": self.recycler_names,
            "weights": self.weights,
            "recycler_frobenius_norm": self.recycler_frobenius_norm,
            "recycler_nnz": self.recycler_nnz,
            "jump_frobenius_norm": self.jump_frobenius_norm,
            "jump_nnz": self.jump_nnz,
        }


@dataclass(frozen=True, slots=True)
class RecycledManifoldJumpSelectionStep:
    """One greedy selection step for recycled dark-detector jumps."""

    step_index: int
    candidate: RecycledManifoldDarkDetectorCandidate
    bad_common_jump_kernel_dimension: int
    inflow_norm: float
    max_target_jump_residual: float
    n_selected_jumps: int

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "candidate": self.candidate.to_summary_dict(),
            "bad_common_jump_kernel_dimension": self.bad_common_jump_kernel_dimension,
            "inflow_norm": self.inflow_norm,
            "max_target_jump_residual": self.max_target_jump_residual,
            "n_selected_jumps": self.n_selected_jumps,
        }


@dataclass(frozen=True, slots=True)
class RecycledManifoldJumpSelectionReport:
    """Greedy small-subset selection report for recycled dark-detector jumps.

    The report owns the selected jump operators.  The summary intentionally omits
    raw sparse matrices, but ``report.jumps`` can be passed directly to
    :func:`qlinks.open_system.diagnose_dark_manifold` or a Lindblad solver.
    The stopping criterion uses the common jump kernel in the complement of the
    target manifold.  Reaching zero is a strong sufficient condition that no
    complement vector is dark under all selected jumps.
    """

    manifold_dimension: int
    hilbert_dimension: int
    candidate_pool_size: int
    max_selected_jumps: int
    target_bad_kernel_dimension: int
    dark_tolerance: float
    inflow_tolerance: float
    jumps: tuple[sp.csr_array, ...]
    steps: tuple[RecycledManifoldJumpSelectionStep, ...]
    final_diagnostics: Any | None
    candidate_report: RecycledManifoldDarkDetectorReport
    candidate_report_was_expanded: bool = False
    candidate_pool_was_limited: bool = False
    compression_strategy: str = "none"
    n_compression_passes: int = 0
    n_compressed_jumps_removed: int = 0
    collective_recycler_strategy: str = "none"
    unbundled_n_jumps: int | None = None
    collective_groups: tuple[RecycledManifoldCollectiveRecyclerGroup, ...] = ()

    @property
    def n_selected_jumps(self) -> int:
        return len(self.jumps)

    @property
    def n_unbundled_jumps(self) -> int:
        if self.unbundled_n_jumps is not None:
            return int(self.unbundled_n_jumps)
        return len(self.steps)

    @property
    def n_collective_groups(self) -> int:
        return len(self.collective_groups)

    @property
    def n_bundled_recyclers(self) -> int:
        return int(sum(group.n_bundled_recyclers for group in self.collective_groups))

    @property
    def collective_jump_reduction(self) -> int:
        return max(self.n_unbundled_jumps - self.n_selected_jumps, 0)

    @property
    def uses_collective_recyclers(self) -> bool:
        return bool(self.collective_groups)

    @property
    def selected_candidates(self) -> tuple[RecycledManifoldDarkDetectorCandidate, ...]:
        return tuple(step.candidate for step in self.steps)

    @property
    def n_reported_candidates(self) -> int:
        return len(self.candidate_report.candidates)

    @property
    def n_tested_candidates(self) -> int:
        return int(self.candidate_report.n_tested_candidates)

    @property
    def n_nonzero_candidates(self) -> int:
        return int(self.candidate_report.n_nonzero_candidates)

    @property
    def candidate_report_is_truncated(self) -> bool:
        return self.candidate_report.candidate_report_is_truncated

    @property
    def candidate_pool_is_truncated(self) -> bool:
        return bool(self.candidate_pool_was_limited)

    @property
    def stopped_with_available_candidates(self) -> bool:
        return (
            self.final_bad_common_jump_kernel_dimension is not None
            and self.final_bad_common_jump_kernel_dimension > self.target_bad_kernel_dimension
            and self.candidate_pool_is_truncated
        )

    @property
    def final_bad_kernel_iprs(self) -> tuple[float, ...]:
        if self.final_diagnostics is None:
            return ()
        return tuple(float(value) for value in self.final_diagnostics.bad_common_jump_kernel_iprs)

    @property
    def final_bad_kernel_ipr_min(self) -> float | None:
        values = self.final_bad_kernel_iprs
        if len(values) == 0:
            return None
        return float(min(values))

    @property
    def final_bad_kernel_ipr_max(self) -> float | None:
        values = self.final_bad_kernel_iprs
        if len(values) == 0:
            return None
        return float(max(values))

    @property
    def final_bad_common_jump_kernel_dimension(self) -> int | None:
        if self.final_diagnostics is None:
            return None
        return int(self.final_diagnostics.bad_common_jump_kernel_dimension)

    @property
    def final_inflow_norm(self) -> float | None:
        if self.final_diagnostics is None:
            return None
        return float(self.final_diagnostics.inflow_norm)

    @property
    def complement_common_kernel_removed(self) -> bool | None:
        final_bad = self.final_bad_common_jump_kernel_dimension
        if final_bad is None:
            return None
        return final_bad <= self.target_bad_kernel_dimension

    @property
    def total_jump_nnz(self) -> int:
        return int(sum(jump.nnz for jump in self.jumps))

    @property
    def max_jump_nnz(self) -> int:
        return int(max((jump.nnz for jump in self.jumps), default=0))

    @property
    def selected_region_indices(self) -> tuple[int, ...]:
        if self.collective_groups:
            return tuple(group.region_index for group in self.collective_groups)
        return tuple(step.candidate.region_index for step in self.steps)

    @property
    def selected_detector_indices(self) -> tuple[int, ...]:
        if self.collective_groups:
            return tuple(group.detector_index for group in self.collective_groups)
        return tuple(step.candidate.detector_index for step in self.steps)

    def selected_recycler_readouts(
        self,
        *,
        basis_configs: npt.NDArray[np.integer],
        states: npt.ArrayLike | None = None,
        max_readouts: int | None = None,
        tolerance: float = 1.0e-10,
        rdm_tolerance: float = 1.0e-10,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local-matrix readouts for selected recycled jump recyclers.

        The returned objects can be passed directly to
        ``LocalBasisGridVisualizer.plot_readout``.  For
        ``rdm_support_matrix_units`` recyclers, pass the target state/manifold
        through ``states`` so the local RDM support basis can be reconstructed.
        """
        if self.collective_groups:
            groups = self.collective_groups
            if max_readouts is not None:
                groups = groups[: max(int(max_readouts), 0)]
            return tuple(
                _local_operator_from_collective_recycler_group(
                    group=group,
                    basis_configs=basis_configs,
                )
                for group in groups
            )

        candidates = self.selected_candidates
        if max_readouts is not None:
            candidates = candidates[: max(int(max_readouts), 0)]
        return tuple(
            _local_operator_from_recycler_candidate(
                candidate=candidate,
                basis_configs=basis_configs,
                states=states,
                recycler_source=self.candidate_report.recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
            )
            for candidate in candidates
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "candidate_pool_size": self.candidate_pool_size,
            "n_reported_candidates": self.n_reported_candidates,
            "n_nonzero_candidates": self.n_nonzero_candidates,
            "n_tested_candidates": self.n_tested_candidates,
            "candidate_report_was_expanded": self.candidate_report_was_expanded,
            "candidate_pool_was_limited": self.candidate_pool_was_limited,
            "compression_strategy": self.compression_strategy,
            "n_compression_passes": self.n_compression_passes,
            "n_compressed_jumps_removed": self.n_compressed_jumps_removed,
            "candidate_report_is_truncated": self.candidate_report_is_truncated,
            "candidate_pool_is_truncated": self.candidate_pool_is_truncated,
            "stopped_with_available_candidates": self.stopped_with_available_candidates,
            "max_selected_jumps": self.max_selected_jumps,
            "target_bad_kernel_dimension": self.target_bad_kernel_dimension,
            "n_selected_jumps": self.n_selected_jumps,
            "n_unbundled_jumps": self.n_unbundled_jumps,
            "collective_recycler_strategy": self.collective_recycler_strategy,
            "uses_collective_recyclers": self.uses_collective_recyclers,
            "n_collective_groups": self.n_collective_groups,
            "n_bundled_recyclers": self.n_bundled_recyclers,
            "collective_jump_reduction": self.collective_jump_reduction,
            "collective_groups": tuple(group.to_summary_dict() for group in self.collective_groups),
            "selected_region_indices": self.selected_region_indices,
            "selected_detector_indices": self.selected_detector_indices,
            "total_jump_nnz": self.total_jump_nnz,
            "max_jump_nnz": self.max_jump_nnz,
            "final_bad_common_jump_kernel_dimension": (self.final_bad_common_jump_kernel_dimension),
            "final_inflow_norm": self.final_inflow_norm,
            "complement_common_kernel_removed": self.complement_common_kernel_removed,
            "final_bad_kernel_ipr_min": self.final_bad_kernel_ipr_min,
            "final_bad_kernel_ipr_max": self.final_bad_kernel_ipr_max,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "steps": tuple(step.to_summary_dict() for step in self.steps),
            "final_diagnostics": (
                None if self.final_diagnostics is None else self.final_diagnostics.to_summary_dict()
            ),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_steps: int = 24):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "RecycledManifoldJumpSelectionReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("candidate pool", str(self.candidate_pool_size))
        overview.add_row(
            "reported/nonzero/tested candidates",
            f"{self.n_reported_candidates}/{self.n_nonzero_candidates}/{self.n_tested_candidates}",
        )
        overview.add_row("expanded report", str(self.candidate_report_was_expanded))
        overview.add_row("pool truncated", str(self.candidate_pool_is_truncated))
        overview.add_row("selected jumps", str(self.n_selected_jumps))
        overview.add_row(
            "compression", f"{self.compression_strategy}, removed={self.n_compressed_jumps_removed}"
        )
        overview.add_row("target bad-kernel dim", str(self.target_bad_kernel_dimension))
        overview.add_row(
            "final bad-kernel dim",
            (
                "not checked"
                if self.final_bad_common_jump_kernel_dimension is None
                else str(self.final_bad_common_jump_kernel_dimension)
            ),
        )
        overview.add_row("complement kernel removed", str(self.complement_common_kernel_removed))
        overview.add_row(
            "final inflow",
            "not checked" if self.final_inflow_norm is None else f"{self.final_inflow_norm:.3e}",
        )
        overview.add_row("total jump nnz", str(self.total_jump_nnz))
        overview.add_row(
            "stopped with available candidates",
            str(self.stopped_with_available_candidates),
        )

        table = Table(title="Greedy selected recycled dark-detector jumps")
        table.add_column("step", justify="right")
        table.add_column("detector")
        table.add_column("region")
        table.add_column("recycler")
        table.add_column("candidate inflow", justify="right")
        table.add_column("bad kernel", justify="right")
        table.add_column("selected", justify="right")
        table.add_column("jump nnz", justify="right")

        for step in self.steps[: max(int(max_steps), 0)]:
            candidate = step.candidate
            table.add_row(
                str(step.step_index),
                candidate.detector_name,
                str(candidate.variable_indices),
                candidate.recycler_name,
                f"{candidate.inflow_norm:.3e}",
                str(step.bad_common_jump_kernel_dimension),
                str(step.n_selected_jumps),
                str(candidate.jump_nnz),
                style=(
                    "green"
                    if step.bad_common_jump_kernel_dimension <= self.target_bad_kernel_dimension
                    else ""
                ),
            )

        if len(self.steps) > max_steps:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{len(self.steps) - max_steps} more steps",
            )

        return Panel(
            Group(overview, table),
            title=Text("Recycled manifold jump-selection report", style="bold green"),
            border_style="green",
        )


@dataclass(frozen=True, slots=True)
class RecycledFamilyKernelDiagnostics:
    """Lightweight dark-manifold kernel diagnostics for a streamed jump family.

    This mirrors the common-kernel fields of :class:`DarkManifoldDiagnostics`
    without requiring every jump operator to be materialized at once.  It is
    intended for large recycled-detector families where the decisive question is
    whether the whole family removes the complement common jump kernel.
    """

    dim: int
    n_jumps: int
    manifold_dimension: int
    hamiltonian_closure_residual: float
    max_target_jump_residual: float
    target_density_liouvillian_residual: float
    inflow_norm: float
    common_jump_kernel_dimension: int
    target_projection_onto_common_kernel: float
    target_distance_from_common_kernel: float
    target_in_common_jump_kernel: bool
    bad_common_jump_kernel_dimension: int
    bad_common_jump_kernel_iprs: tuple[float, ...]
    internal_hamiltonian_eigenvalues: tuple[complex, ...]
    expected_internal_zero_mode_count: int
    expected_internal_peripheral_mode_count: int
    liouvillian_zero_tolerance: float

    @property
    def target_jump_residuals(self) -> tuple[float, ...]:
        # The streamed diagnostic keeps only the maximum residual to avoid
        # storing one value per candidate jump.
        return ()

    @property
    def expected_internal_liouvillian_eigenvalues(self) -> tuple[complex, ...]:
        return _internal_liouvillian_eigenvalues_from_energies(
            self.internal_hamiltonian_eigenvalues
        )

    @property
    def liouvillian_zero_mode_count(self) -> None:
        return None

    @property
    def liouvillian_zero_mode_count_is_lower_bound(self) -> bool:
        return False

    @property
    def liouvillian_spectral_gap(self) -> None:
        return None

    @property
    def liouvillian_decay_gap(self) -> None:
        return None

    @property
    def liouvillian_peripheral_mode_count(self) -> None:
        return None

    @property
    def liouvillian_spectrum_method(self) -> str:
        return "streamed_kernel"

    @property
    def liouvillian_eigenvalues(self) -> tuple[complex, ...]:
        return ()

    @property
    def matched_internal_nondecaying_mode_count(self) -> None:
        return None

    @property
    def missing_internal_nondecaying_mode_count(self) -> None:
        return None

    @property
    def extra_nondecaying_mode_count(self) -> None:
        return None

    @property
    def extra_zero_mode_count(self) -> None:
        return None

    @property
    def external_decay_gap(self) -> None:
        return None

    @property
    def likely_attractive_dark_manifold(self) -> None:
        return None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "manifold_dimension": self.manifold_dimension,
            "h_closure_residual": self.hamiltonian_closure_residual,
            "max_target_jump_residual": self.max_target_jump_residual,
            "target_density_liouvillian_residual": self.target_density_liouvillian_residual,
            "inflow_norm": self.inflow_norm,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "target_projection_onto_common_kernel": self.target_projection_onto_common_kernel,
            "target_distance_from_common_kernel": self.target_distance_from_common_kernel,
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "bad_common_jump_kernel_dimension": self.bad_common_jump_kernel_dimension,
            "bad_common_jump_kernel_iprs": self.bad_common_jump_kernel_iprs,
            "internal_hamiltonian_eigenvalues": [
                complex(value) for value in self.internal_hamiltonian_eigenvalues
            ],
            "expected_internal_zero_mode_count": self.expected_internal_zero_mode_count,
            "expected_internal_peripheral_mode_count": (
                self.expected_internal_peripheral_mode_count
            ),
            "liouvillian_zero_mode_count": None,
            "liouvillian_zero_mode_count_is_lower_bound": False,
            "liouvillian_spectral_gap": None,
            "liouvillian_decay_gap": None,
            "liouvillian_peripheral_mode_count": None,
            "liouvillian_spectrum_method": self.liouvillian_spectrum_method,
            "matched_internal_nondecaying_mode_count": None,
            "missing_internal_nondecaying_mode_count": None,
            "extra_nondecaying_mode_count": None,
            "extra_zero_mode_count": None,
            "external_decay_gap": None,
            "likely_attractive_dark_manifold": None,
        }


@dataclass(frozen=True, slots=True)
class RecycledManifoldCandidateFamilyKernelReport:
    """Common-kernel diagnostic for an entire recycled-detector family.

    This report answers a different question from greedy subset selection: if
    *all* eligible local candidates are used as jumps, does the family itself
    remove the complement common jump kernel?  If the bad kernel remains
    nonzero for the full family, no subset of that candidate family can remove
    it.
    """

    manifold_dimension: int
    hilbert_dimension: int
    candidate_report: RecycledManifoldDarkDetectorReport
    candidate_report_was_expanded: bool
    dark_tolerance: float
    inflow_tolerance: float
    candidate_jumps: tuple[sp.csr_array, ...]
    diagnostics: Any
    candidate_jump_count: int | None = None
    candidate_total_jump_nnz: int | None = None
    candidate_max_jump_nnz: int | None = None

    @property
    def n_candidate_jumps(self) -> int:
        if self.candidate_jump_count is not None:
            return int(self.candidate_jump_count)
        return len(self.candidate_jumps)

    @property
    def n_reported_candidates(self) -> int:
        return self.candidate_report.n_reported_candidates

    @property
    def n_nonzero_candidates(self) -> int:
        return self.candidate_report.n_nonzero_candidates

    @property
    def n_tested_candidates(self) -> int:
        return self.candidate_report.n_tested_candidates

    @property
    def candidate_report_is_truncated(self) -> bool:
        return self.candidate_report.candidate_report_is_truncated

    @property
    def total_jump_nnz(self) -> int:
        if self.candidate_total_jump_nnz is not None:
            return int(self.candidate_total_jump_nnz)
        return int(sum(jump.nnz for jump in self.candidate_jumps))

    @property
    def max_jump_nnz(self) -> int:
        if self.candidate_max_jump_nnz is not None:
            return int(self.candidate_max_jump_nnz)
        return int(max((jump.nnz for jump in self.candidate_jumps), default=0))

    @property
    def family_bad_common_jump_kernel_dimension(self) -> int:
        return int(self.diagnostics.bad_common_jump_kernel_dimension)

    @property
    def family_common_jump_kernel_dimension(self) -> int:
        return int(self.diagnostics.common_jump_kernel_dimension)

    @property
    def family_inflow_norm(self) -> float:
        return float(self.diagnostics.inflow_norm)

    @property
    def family_kernel_method(self) -> str:
        return str(getattr(self.diagnostics, "liouvillian_spectrum_method", "unknown"))

    @property
    def complement_common_kernel_removed(self) -> bool:
        return self.family_bad_common_jump_kernel_dimension == 0

    @property
    def bad_kernel_iprs(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.diagnostics.bad_common_jump_kernel_iprs)

    @property
    def bad_kernel_ipr_min(self) -> float | None:
        values = self.bad_kernel_iprs
        if len(values) == 0:
            return None
        return float(min(values))

    @property
    def bad_kernel_ipr_max(self) -> float | None:
        values = self.bad_kernel_iprs
        if len(values) == 0:
            return None
        return float(max(values))

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "n_candidate_jumps": self.n_candidate_jumps,
            "n_reported_candidates": self.n_reported_candidates,
            "n_nonzero_candidates": self.n_nonzero_candidates,
            "n_tested_candidates": self.n_tested_candidates,
            "candidate_report_was_expanded": self.candidate_report_was_expanded,
            "candidate_report_is_truncated": self.candidate_report_is_truncated,
            "total_jump_nnz": self.total_jump_nnz,
            "max_jump_nnz": self.max_jump_nnz,
            "family_common_jump_kernel_dimension": self.family_common_jump_kernel_dimension,
            "family_bad_common_jump_kernel_dimension": (
                self.family_bad_common_jump_kernel_dimension
            ),
            "family_inflow_norm": self.family_inflow_norm,
            "family_kernel_method": self.family_kernel_method,
            "complement_common_kernel_removed": self.complement_common_kernel_removed,
            "bad_kernel_ipr_min": self.bad_kernel_ipr_min,
            "bad_kernel_ipr_max": self.bad_kernel_ipr_max,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "diagnostics": self.diagnostics.to_summary_dict(),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        try:
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "RecycledManifoldCandidateFamilyKernelReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Hilbert dimension", str(self.hilbert_dimension))
        table.add_row("manifold dimension", str(self.manifold_dimension))
        table.add_row("candidate jumps", str(self.n_candidate_jumps))
        table.add_row(
            "reported/nonzero/tested candidates",
            f"{self.n_reported_candidates}/{self.n_nonzero_candidates}/{self.n_tested_candidates}",
        )
        table.add_row("expanded report", str(self.candidate_report_was_expanded))
        table.add_row("report truncated", str(self.candidate_report_is_truncated))
        table.add_row("kernel method", self.family_kernel_method)
        table.add_row("common jump kernel", str(self.family_common_jump_kernel_dimension))
        table.add_row("bad complement kernel", str(self.family_bad_common_jump_kernel_dimension))
        table.add_row("complement kernel removed", str(self.complement_common_kernel_removed))
        table.add_row("family inflow", f"{self.family_inflow_norm:.3e}")
        table.add_row("total jump nnz", str(self.total_jump_nnz))
        table.add_row("max jump nnz", str(self.max_jump_nnz))
        table.add_row(
            "bad-kernel IPR range",
            (
                "none"
                if self.bad_kernel_ipr_min is None
                else f"{self.bad_kernel_ipr_min:.3e} .. {self.bad_kernel_ipr_max:.3e}"
            ),
        )

        style = "green" if self.complement_common_kernel_removed else "red"
        return Panel(
            table,
            title=Text("Recycled candidate-family common-kernel report", style=f"bold {style}"),
            border_style=style,
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


def _coefficient_ipr(coefficients: npt.ArrayLike) -> float:
    values = np.asarray(coefficients, dtype=np.complex128)
    norm_squared = float(np.vdot(values, values).real)
    if norm_squared <= 0.0:
        return 0.0
    return float(np.sum(np.abs(values) ** 4) / (norm_squared * norm_squared))


def _effective_coefficient_count(coefficients: npt.ArrayLike) -> float:
    ipr = _coefficient_ipr(coefficients)
    if ipr <= 0.0:
        return float("inf")
    return float(1.0 / ipr)


def _phase_fixed_normalized_vector(
    vector: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128] | None:
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        return None
    normalized = np.asarray(vector / norm, dtype=np.complex128)
    pivot = int(np.argmax(np.abs(normalized)))
    pivot_value = normalized[pivot]
    if abs(pivot_value) > tolerance:
        normalized = normalized * np.exp(-1j * np.angle(pivot_value))
    return normalized


def _deduplicate_coefficient_vectors(
    vectors: list[npt.NDArray[np.complex128]],
    *,
    overlap_tolerance: float,
) -> list[npt.NDArray[np.complex128]]:
    unique: list[npt.NDArray[np.complex128]] = []
    for vector in vectors:
        if any(abs(np.vdot(existing, vector)) >= 1.0 - overlap_tolerance for existing in unique):
            continue
        unique.append(vector)
    return unique


def _sparse_ipr_dark_detector_columns(
    *,
    nullspace: npt.NDArray[np.complex128],
    max_candidates: int | None,
    tolerance: float,
    overlap_tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return nullspace vectors biased toward small operator support.

    The ordinary SVD basis is arbitrary inside a degenerate dark-detector
    nullspace.  To get more interpretable detector readouts, project each
    coordinate unit vector onto the dark nullspace and rank the resulting
    vectors by coefficient IPR.  This is a cheap deterministic proxy for a
    sparse/IPR-optimized basis: a high score means the detector is concentrated
    on fewer supplied local operators.
    """
    if nullspace.ndim != 2:
        raise ValueError("nullspace must be two-dimensional.")
    n_operators, nullity = nullspace.shape
    if n_operators == 0 or nullity == 0:
        return np.zeros((n_operators, 0), dtype=np.complex128)

    projected: list[npt.NDArray[np.complex128]] = []
    for operator_index in range(n_operators):
        row = np.asarray(nullspace[operator_index, :], dtype=np.complex128)
        # Projection of the coordinate vector e_i onto span(nullspace).
        vector = nullspace @ row.conj()
        normalized = _phase_fixed_normalized_vector(vector, tolerance=tolerance)
        if normalized is not None:
            projected.append(normalized)

    projected.sort(
        key=lambda vector: (
            -_coefficient_ipr(vector),
            int(np.count_nonzero(np.abs(vector) > tolerance)),
            int(np.argmax(np.abs(vector))),
        )
    )
    unique = _deduplicate_coefficient_vectors(
        projected,
        overlap_tolerance=overlap_tolerance,
    )

    # If coordinate projections produced fewer vectors than requested, append
    # the orthonormal SVD basis as a robust fallback.
    for column_index in range(nullity):
        normalized = _phase_fixed_normalized_vector(
            np.asarray(nullspace[:, column_index], dtype=np.complex128),
            tolerance=tolerance,
        )
        if normalized is not None:
            unique = _deduplicate_coefficient_vectors(
                unique + [normalized],
                overlap_tolerance=overlap_tolerance,
            )

    if max_candidates is not None:
        unique = unique[: max(int(max_candidates), 0)]
    if len(unique) == 0:
        return np.zeros((n_operators, 0), dtype=np.complex128)
    return np.column_stack(unique).astype(np.complex128, copy=False)


def diagnose_manifold_dark_operator_basis(
    *,
    states: npt.ArrayLike,
    operators: tuple[Any, ...] | list[Any],
    operator_names: tuple[str, ...] | list[str] | None = None,
    tolerance: float = 1.0e-10,
    coefficient_tolerance: float = 1.0e-8,
    max_candidates: int | None = 16,
    candidate_strategy: Literal["svd_basis", "coordinate_ipr"] = "svd_basis",
    candidate_overlap_tolerance: float = 1.0e-7,
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
        candidate_strategy: ``"svd_basis"`` keeps the numerical nullspace basis.
            ``"coordinate_ipr"`` projects individual supplied operators onto the
            dark nullspace and ranks the results by coefficient IPR, producing
            more localized/interpretable detector combinations when the dark
            solution space is degenerate.
        candidate_overlap_tolerance: Deduplication tolerance for
            ``candidate_strategy="coordinate_ipr"``.

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

    # The constraint matrix is usually tall in production QDM runs:
    # ``(hilbert_dimension * manifold_dimension) x n_operators``.  A full SVD
    # would materialize the huge left-unitary matrix and can dominate or even
    # time out before any jump selection starts.  We only need right singular
    # vectors in operator-coefficient space, so the economy SVD is complete
    # whenever rows >= columns.  Keep full_matrices=True only for genuinely
    # underdetermined systems, where the economy Vh would omit the extra
    # nullspace directions.
    full_matrices = constraint_matrix.shape[0] < constraint_matrix.shape[1]
    _, singular_values, vh = np.linalg.svd(
        constraint_matrix,
        full_matrices=full_matrices,
    )
    if singular_values.size == 0:
        cutoff = float(tolerance)
        rank = 0
    else:
        cutoff = float(tolerance * max(float(singular_values[0]), 1.0))
        rank = int(np.count_nonzero(singular_values > cutoff))

    nullspace = vh.conj().T[:, rank:]
    detector_nullity = int(nullspace.shape[1])

    if candidate_strategy not in {"svd_basis", "coordinate_ipr"}:
        raise ValueError('candidate_strategy must be "svd_basis" or "coordinate_ipr".')
    if candidate_strategy == "svd_basis":
        candidate_columns = nullspace
        if max_candidates is not None:
            candidate_columns = candidate_columns[:, : max(int(max_candidates), 0)]
    else:
        candidate_columns = _sparse_ipr_dark_detector_columns(
            nullspace=nullspace,
            max_candidates=max_candidates,
            tolerance=max(float(tolerance), float(coefficient_tolerance)),
            overlap_tolerance=float(candidate_overlap_tolerance),
        )

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
        coefficient_ipr = _coefficient_ipr(coefficients)
        effective_operator_count = _effective_coefficient_count(coefficients)

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
                coefficient_ipr=coefficient_ipr,
                effective_operator_count=effective_operator_count,
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
        candidate_strategy=candidate_strategy,
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


def _diagonal_vector_if_diagonal(
    operator: sp.csr_array,
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128] | None:
    """Return the diagonal when a sparse operator has no off-diagonal support."""
    coo = operator.tocoo()
    off_diagonal_mask = coo.row != coo.col
    if np.any(np.abs(coo.data[off_diagonal_mask]) > tolerance):
        return None
    return np.asarray(operator.diagonal(), dtype=np.complex128)


def _embedded_matrix_unit_metrics_with_diagonal_right_factor(
    *,
    embedding_context: Any,
    target_local_index: int,
    source_local_index: int,
    right_diagonal: npt.NDArray[np.complex128],
    state_basis: npt.NDArray[np.complex128],
    zero_tolerance: float,
) -> tuple[float, float, float, int, float, int] | None:
    """Fast score ``J = |target><source|_R D`` for diagonal ``D``."""
    transition_mask = (embedding_context.target_local_indices == int(target_local_index)) & (
        embedding_context.source_local_indices == int(source_local_index)
    )
    if not np.any(transition_mask):
        return None

    source_indices = embedding_context.source_full_indices[transition_mask]
    target_indices = embedding_context.target_full_indices[transition_mask]
    jump_values = right_diagonal[source_indices]
    jump_mask = np.abs(jump_values) > zero_tolerance
    jump_nnz = int(np.count_nonzero(jump_mask))
    if jump_nnz == 0:
        return None

    source_indices = source_indices[jump_mask]
    target_indices = target_indices[jump_mask]
    jump_values = jump_values[jump_mask]

    # Matrix units have unit entries on each constrained-basis transition.
    recycler_nnz = int(np.count_nonzero(transition_mask))
    recycler_frobenius_norm = float(np.sqrt(recycler_nnz))

    adjoint_action = np.zeros_like(state_basis, dtype=np.complex128)
    conjugated_values = np.conj(jump_values)
    for state_index in range(state_basis.shape[1]):
        np.add.at(
            adjoint_action[:, state_index],
            source_indices,
            conjugated_values * state_basis[target_indices, state_index],
        )

    target_block = adjoint_action.conj().T @ state_basis
    target_block_norm_sq = float(np.linalg.norm(target_block) ** 2)
    adjoint_norm_sq = float(np.linalg.norm(adjoint_action) ** 2)
    inflow_norm = float(np.sqrt(max(adjoint_norm_sq - target_block_norm_sq, 0.0)))
    jump_frobenius_norm = float(np.linalg.norm(jump_values))
    return (
        inflow_norm,
        float(np.sqrt(max(target_block_norm_sq, 0.0))),
        jump_frobenius_norm,
        jump_nnz,
        recycler_frobenius_norm,
        recycler_nnz,
    )


def _embedded_local_operator_metrics_with_diagonal_right_factor(
    *,
    embedding_context: Any,
    local_operator: npt.NDArray[np.complex128],
    right_diagonal: npt.NDArray[np.complex128],
    state_basis: npt.NDArray[np.complex128],
    zero_tolerance: float,
) -> tuple[float, float, float, int, float, int] | None:
    """Score ``J = R D`` without materializing sparse matrices when ``D`` is diagonal.

    The generic recycled-detector scan used to build every embedded local
    recycler ``R``, multiply it by the detector ``D``, and then multiply the
    resulting sparse matrix by the target-manifold basis.  In QDM production
    runs the detector basis is normally diagonal plaquette-projector data.  For
    that common case, the nonzero entries of ``J`` are just the embedded local
    entries of ``R`` scaled by the source-basis diagonal of ``D``.  Computing the
    projected inflow directly from these arrays avoids hundreds of thousands of
    tiny CSR constructions and sparse products.
    """
    if local_operator.shape != (embedding_context.local_dim, embedding_context.local_dim):
        raise ValueError(
            "local_operator has incompatible shape: "
            f"{local_operator.shape} != "
            f"{(embedding_context.local_dim, embedding_context.local_dim)}."
        )
    if embedding_context.source_full_indices.size == 0:
        return None

    local_values = np.asarray(
        local_operator[
            embedding_context.target_local_indices,
            embedding_context.source_local_indices,
        ],
        dtype=np.complex128,
    )
    recycler_mask = np.abs(local_values) > zero_tolerance
    recycler_nnz = int(np.count_nonzero(recycler_mask))
    if recycler_nnz == 0:
        return None

    source_indices = embedding_context.source_full_indices
    target_indices = embedding_context.target_full_indices
    jump_values = local_values * right_diagonal[source_indices]
    jump_mask = np.abs(jump_values) > zero_tolerance
    jump_nnz = int(np.count_nonzero(jump_mask))
    if jump_nnz == 0:
        return None

    jump_values = jump_values[jump_mask]
    source_indices = source_indices[jump_mask]
    target_indices = target_indices[jump_mask]

    adjoint_action = np.zeros_like(state_basis, dtype=np.complex128)
    conjugated_values = np.conj(jump_values)
    for state_index in range(state_basis.shape[1]):
        np.add.at(
            adjoint_action[:, state_index],
            source_indices,
            conjugated_values * state_basis[target_indices, state_index],
        )

    target_block = adjoint_action.conj().T @ state_basis
    target_block_norm_sq = float(np.linalg.norm(target_block) ** 2)
    adjoint_norm_sq = float(np.linalg.norm(adjoint_action) ** 2)
    inflow_norm = float(np.sqrt(max(adjoint_norm_sq - target_block_norm_sq, 0.0)))

    jump_frobenius_norm = float(np.linalg.norm(jump_values))
    recycler_frobenius_norm = float(np.linalg.norm(local_values[recycler_mask]))
    return (
        inflow_norm,
        float(np.sqrt(max(target_block_norm_sq, 0.0))),
        jump_frobenius_norm,
        jump_nnz,
        recycler_frobenius_norm,
        recycler_nnz,
    )


def _embedded_matrix_unit_times_diagonal_as_csr(
    *,
    embedding_context: Any,
    target_local_index: int,
    source_local_index: int,
    right_diagonal: npt.NDArray[np.complex128],
    dim: int,
    zero_tolerance: float,
) -> sp.csr_array:
    """Build ``|target><source|_R D`` directly for diagonal ``D``."""
    transition_mask = (embedding_context.target_local_indices == int(target_local_index)) & (
        embedding_context.source_local_indices == int(source_local_index)
    )
    if not np.any(transition_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    source_indices = embedding_context.source_full_indices[transition_mask]
    target_indices = embedding_context.target_full_indices[transition_mask]
    jump_values = np.asarray(right_diagonal[source_indices], dtype=np.complex128)
    jump_mask = np.abs(jump_values) > zero_tolerance
    if not np.any(jump_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    return sp.csr_array(
        (
            jump_values[jump_mask],
            (target_indices[jump_mask], source_indices[jump_mask]),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )


def _embedded_local_operator_times_diagonal_as_csr(
    *,
    embedding_context: Any,
    local_operator: npt.NDArray[np.complex128],
    right_diagonal: npt.NDArray[np.complex128],
    dim: int,
    zero_tolerance: float,
) -> sp.csr_array:
    """Build embedded ``R D`` directly when ``D`` is diagonal."""
    if local_operator.shape != (embedding_context.local_dim, embedding_context.local_dim):
        raise ValueError(
            "local_operator has incompatible shape: "
            f"{local_operator.shape} != "
            f"{(embedding_context.local_dim, embedding_context.local_dim)}."
        )
    if embedding_context.source_full_indices.size == 0:
        return sp.csr_array((dim, dim), dtype=np.complex128)

    local_values = np.asarray(
        local_operator[
            embedding_context.target_local_indices,
            embedding_context.source_local_indices,
        ],
        dtype=np.complex128,
    )
    recycler_mask = np.abs(local_values) > zero_tolerance
    if not np.any(recycler_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    source_indices = embedding_context.source_full_indices
    target_indices = embedding_context.target_full_indices
    jump_values = local_values * right_diagonal[source_indices]
    jump_mask = np.abs(jump_values) > zero_tolerance
    if not np.any(jump_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    return sp.csr_array(
        (
            jump_values[jump_mask],
            (target_indices[jump_mask], source_indices[jump_mask]),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )


def _recycled_candidate_sort_key(
    candidate: RecycledManifoldDarkDetectorCandidate,
    *,
    dark_tolerance: float,
) -> tuple[bool, float, float, int, int, int, int]:
    return (
        candidate.relative_dark_residual > dark_tolerance,
        -candidate.inflow_norm,
        candidate.relative_dark_residual,
        candidate.jump_nnz,
        candidate.detector_index,
        candidate.region_index,
        candidate.recycler_index,
    )


def _append_ranked_recycled_candidate(
    candidates: list[RecycledManifoldDarkDetectorCandidate],
    candidate: RecycledManifoldDarkDetectorCandidate,
    *,
    max_report_candidates: int | None,
    dark_tolerance: float,
) -> None:
    candidates.append(candidate)
    if max_report_candidates is None:
        return
    limit = max(int(max_report_candidates), 0)
    if limit == 0:
        candidates.clear()
        return
    if len(candidates) <= limit:
        return
    candidates.sort(
        key=lambda item: _recycled_candidate_sort_key(
            item,
            dark_tolerance=dark_tolerance,
        )
    )
    del candidates[limit:]


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


def expand_local_regions_to_pair_unions(
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    *,
    pair_mode: Literal["overlap", "all"] = "overlap",
    min_overlap: int = 1,
    max_region_size: int | None = None,
    include_single_regions: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Return unions of pairs of local regions, useful for two-block recyclers.

    The single-plaquette recycled-detector family can leave residual bad kernel
    sectors when the target manifold is built from two-plaquette singlet-like
    structures.  This helper creates bounded two-region supports that can be
    supplied to ``diagnose_recycled_manifold_dark_detectors`` or
    ``diagnose_recycled_manifold_candidate_family_kernel`` as ``local_regions``.

    Args:
        local_regions: Base local regions, usually plaquette supports.
        pair_mode: ``"overlap"`` keeps only pairs sharing at least
            ``min_overlap`` variables; ``"all"`` keeps all unordered pairs.
        min_overlap: Minimum number of shared variables when
            ``pair_mode="overlap"``.
        max_region_size: Optional upper bound on the size of the union.
        include_single_regions: Whether to prepend the original regions.

    Returns:
        Deduplicated sorted variable-index unions.
    """
    if pair_mode not in {"overlap", "all"}:
        raise ValueError('pair_mode must be "overlap" or "all".')
    if min_overlap < 0:
        raise ValueError("min_overlap must be non-negative.")
    if max_region_size is not None and max_region_size <= 0:
        raise ValueError("max_region_size must be positive when provided.")

    base_regions = _normalize_local_regions(local_regions)
    expanded: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def maybe_add(region: tuple[int, ...]) -> None:
        if max_region_size is not None and len(region) > max_region_size:
            return
        if region in seen:
            return
        seen.add(region)
        expanded.append(region)

    if include_single_regions:
        for region in base_regions:
            maybe_add(region)

    region_sets = [set(region) for region in base_regions]
    for left_index, left in enumerate(base_regions):
        left_set = region_sets[left_index]
        for right_index in range(left_index + 1, len(base_regions)):
            right = base_regions[right_index]
            overlap = len(left_set.intersection(region_sets[right_index]))
            if pair_mode == "overlap" and overlap < min_overlap:
                continue
            maybe_add(tuple(sorted(set(left).union(right))))

    if len(expanded) == 0:
        raise ValueError(
            "No pair-union regions were generated. Relax pair_mode/min_overlap "
            "or max_region_size, or pass include_single_regions=True."
        )

    return tuple(expanded)


def expand_local_regions_to_cluster_unions(
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    *,
    cluster_size: int = 3,
    cluster_mode: Literal["overlap_connected", "all"] = "overlap_connected",
    min_overlap: int = 1,
    max_region_size: int | None = None,
    include_single_regions: bool = False,
    include_smaller_clusters: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Return bounded unions of multiple local regions.

    This generalizes :func:`expand_local_regions_to_pair_unions` to three- or
    four-region recycler/patch supports.  ``cluster_mode="overlap_connected"``
    keeps only clusters that are connected in the overlap graph of the base
    regions; this is the natural setting for connected multi-plaquette QDM
    patches.

    Args:
        local_regions: Base local regions, usually single-plaquette supports.
        cluster_size: Number of base regions to union in the largest clusters.
        cluster_mode: ``"overlap_connected"`` keeps clusters connected through
            overlaps of at least ``min_overlap`` variables; ``"all"`` keeps all
            unordered clusters.
        min_overlap: Minimum overlap used to define adjacency for
            ``cluster_mode="overlap_connected"``.
        max_region_size: Optional upper bound on the size of the variable union.
        include_single_regions: Whether to include the original base regions.
        include_smaller_clusters: Whether to include all cluster sizes from two
            through ``cluster_size`` instead of only the requested size.

    Returns:
        Deduplicated sorted variable-index unions.
    """
    if cluster_mode not in {"overlap_connected", "all"}:
        raise ValueError('cluster_mode must be "overlap_connected" or "all".')
    if cluster_size < 2:
        raise ValueError("cluster_size must be at least two.")
    if min_overlap < 0:
        raise ValueError("min_overlap must be non-negative.")
    if max_region_size is not None and max_region_size <= 0:
        raise ValueError("max_region_size must be positive when provided.")

    base_regions = _normalize_local_regions(local_regions)
    expanded: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def maybe_add(region: tuple[int, ...]) -> None:
        if max_region_size is not None and len(region) > max_region_size:
            return
        if region in seen:
            return
        seen.add(region)
        expanded.append(region)

    if include_single_regions:
        for region in base_regions:
            maybe_add(region)

    region_sets = [set(region) for region in base_regions]

    def is_overlap_connected(indices: tuple[int, ...]) -> bool:
        if len(indices) <= 1:
            return True
        remaining = set(indices[1:])
        frontier = [indices[0]]
        visited = {indices[0]}
        while frontier:
            left_index = frontier.pop()
            left_set = region_sets[left_index]
            for right_index in tuple(remaining):
                overlap = len(left_set.intersection(region_sets[right_index]))
                if overlap < min_overlap:
                    continue
                remaining.remove(right_index)
                visited.add(right_index)
                frontier.append(right_index)
        return len(visited) == len(indices)

    cluster_sizes = range(2, cluster_size + 1) if include_smaller_clusters else (cluster_size,)
    for size in cluster_sizes:
        for indices in combinations(range(len(base_regions)), size):
            if cluster_mode == "overlap_connected" and not is_overlap_connected(indices):
                continue
            union: set[int] = set()
            for index in indices:
                union.update(base_regions[index])
            maybe_add(tuple(sorted(union)))

    if len(expanded) == 0:
        raise ValueError(
            "No cluster-union regions were generated. Relax cluster_mode/min_overlap "
            "or max_region_size, reduce cluster_size, or pass include_single_regions=True."
        )

    return tuple(expanded)


def _pattern_name(pattern: tuple[int, ...]) -> str:
    return "(" + ",".join(str(int(value)) for value in pattern) + ")"


def _local_patterns_from_basis_configs(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")
    if len(variable_indices) == 0:
        raise ValueError("variable_indices must be nonempty.")
    if any(index < 0 or index >= basis_array.shape[1] for index in variable_indices):
        raise ValueError("variable_indices contains out-of-range entries.")
    variable_index_array = np.asarray(variable_indices, dtype=np.int64)
    return tuple(
        sorted(
            {tuple(int(value) for value in config[variable_index_array]) for config in basis_array}
        )
    )


def _matrix_unit_local_operator(
    *,
    local_dim: int,
    recycler_index: int,
) -> npt.NDArray[np.complex128]:
    if local_dim <= 0:
        raise ValueError("local_dim must be positive.")
    n_matrix_units = int(local_dim) * int(local_dim)
    if recycler_index < 0 or recycler_index >= n_matrix_units:
        raise ValueError("recycler_index is out of range for matrix-unit recyclers.")
    target_index, source_index = divmod(int(recycler_index), int(local_dim))
    local_operator = np.zeros((int(local_dim), int(local_dim)), dtype=np.complex128)
    local_operator[target_index, source_index] = 1.0
    return local_operator


def _local_operator_from_matrix_unit_terms(
    *,
    local_patterns: tuple[tuple[int, ...], ...],
    terms: tuple[Any, ...],
) -> npt.NDArray[np.complex128]:
    pattern_to_index = {pattern: index for index, pattern in enumerate(local_patterns)}
    local_dim = len(local_patterns)
    local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
    for term in terms:
        name = str(term.operator_name)
        coefficient = complex(term.coefficient)
        if "<-" not in name:
            raise ValueError(
                "Cannot build a matrix readout from non-matrix-unit term name " f"{name!r}."
            )
        target_text, source_text = name.split("<-", 1)
        target_pattern = tuple(
            int(value) for value in target_text.strip()[1:-1].split(",") if value
        )
        source_pattern = tuple(
            int(value) for value in source_text.strip()[1:-1].split(",") if value
        )
        try:
            target_index = pattern_to_index[target_pattern]
            source_index = pattern_to_index[source_pattern]
        except KeyError as exc:
            raise ValueError(
                "matrix-unit term contains a pattern that is absent from local_patterns."
            ) from exc
        local_operator[target_index, source_index] += coefficient
    return local_operator


def _local_operator_from_recycler_candidate(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike | None,
    recycler_source: Literal["matrix_units", "rdm_support_matrix_units"],
    tolerance: float,
    rdm_tolerance: float,
) -> LocalOperatorMatrixReadout:
    variable_indices = tuple(int(value) for value in candidate.variable_indices)
    local_patterns = _local_patterns_from_basis_configs(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    if len(local_patterns) != int(candidate.local_dim):
        raise ValueError(
            "basis_configs/local_patterns are incompatible with the selected candidate."
        )

    if recycler_source == "matrix_units":
        local_operator = _matrix_unit_local_operator(
            local_dim=candidate.local_dim,
            recycler_index=candidate.recycler_index,
        )
    elif recycler_source == "rdm_support_matrix_units":
        if states is None:
            raise ValueError("states are required to read rdm_support_matrix_units recyclers.")
        from qlinks.open_system.local_recycling import (
            _local_pattern_basis_context_from_basis,
            _local_reduced_density_matrix_from_basis_context_and_states,
        )

        state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
        context = _local_pattern_basis_context_from_basis(
            basis_configs=np.asarray(basis_configs),
            variable_indices=variable_indices,
            local_patterns=local_patterns,
        )
        rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=state_basis,
            tolerance=rdm_tolerance,
        )
        specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")
        _name, local_operator = specs[int(candidate.recycler_index)]
    else:
        raise ValueError("recycler_source must be 'matrix_units' or 'rdm_support_matrix_units'.")

    return LocalOperatorMatrixReadout(
        label=f"recycled_{candidate.candidate_index}_{candidate.recycler_name}",
        source="recycled_recycler",
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        local_operator=np.asarray(local_operator, dtype=np.complex128),
        metadata=(
            ("candidate_index", int(candidate.candidate_index)),
            ("detector_index", int(candidate.detector_index)),
            ("detector_name", candidate.detector_name),
            ("region_index", int(candidate.region_index)),
            ("recycler_index", int(candidate.recycler_index)),
            ("recycler_name", candidate.recycler_name),
            ("inflow_norm", float(candidate.inflow_norm)),
            ("jump_nnz", int(candidate.jump_nnz)),
        ),
    )


def _local_operator_from_collective_recycler_group(
    *,
    group: RecycledManifoldCollectiveRecyclerGroup,
    basis_configs: npt.NDArray[np.integer],
) -> LocalOperatorMatrixReadout:
    variable_indices = tuple(int(value) for value in group.variable_indices)
    local_patterns = _local_patterns_from_basis_configs(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    if len(local_patterns) != int(group.local_dim):
        raise ValueError("basis_configs/local_patterns are incompatible with the collective group.")
    return LocalOperatorMatrixReadout(
        label=f"collective_recycled_{group.group_index}_{group.detector_name}",
        source="collective_recycled_recycler",
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        local_operator=np.asarray(group.local_operator, dtype=np.complex128),
        metadata=(
            ("group_index", int(group.group_index)),
            ("detector_index", int(group.detector_index)),
            ("detector_name", group.detector_name),
            ("region_index", int(group.region_index)),
            ("n_bundled_recyclers", int(group.n_bundled_recyclers)),
            ("candidate_indices", group.candidate_indices),
            ("recycler_indices", group.recycler_indices),
            ("recycler_names", group.recycler_names),
            ("jump_nnz", int(group.jump_nnz)),
        ),
    )


def _local_operator_from_targeted_candidate(
    *,
    candidate: TargetedResidualKernelLinearCandidate,
    basis_configs: npt.NDArray[np.integer],
) -> LocalOperatorMatrixReadout:
    variable_indices = tuple(int(value) for value in candidate.variable_indices)
    local_patterns = _local_patterns_from_basis_configs(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    if len(local_patterns) != int(candidate.local_dim):
        raise ValueError(
            "basis_configs/local_patterns are incompatible with the selected candidate."
        )
    if candidate.operator_source != "matrix_units":
        raise ValueError(
            "targeted matrix readouts currently support operator_source='matrix_units'."
        )
    local_operator = _local_operator_from_matrix_unit_terms(
        local_patterns=local_patterns,
        terms=candidate.terms,
    )
    return LocalOperatorMatrixReadout(
        label=f"targeted_{candidate.candidate_index}",
        source="targeted_operator",
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        local_operator=local_operator,
        metadata=(
            ("candidate_index", int(candidate.candidate_index)),
            ("region_index", int(candidate.region_index)),
            ("operator_source", candidate.operator_source),
            ("residual_objective", candidate.residual_objective),
            ("residual_score_norm", float(candidate.residual_score_norm)),
            ("total_inflow_norm", float(candidate.total_inflow_norm)),
            ("jump_nnz", int(candidate.jump_nnz)),
        ),
    )


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

    detectors: list[tuple[sp.csr_array, float, float, npt.NDArray[np.complex128] | None]] = []
    for detector_index in range(coefficients.shape[1]):
        detector = _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        detector_action_residual = float(np.linalg.norm(detector @ state_basis))
        detector_norm = float(sp.linalg.norm(detector))
        detector_relative_residual = detector_action_residual / max(detector_norm, 1.0)
        detector_diagonal = _diagonal_vector_if_diagonal(
            detector,
            tolerance=tolerance,
        )
        detectors.append(
            (
                detector,
                detector_action_residual,
                detector_relative_residual,
                detector_diagonal,
            )
        )

    candidate_buffer: list[RecycledManifoldDarkDetectorCandidate] = []
    n_tested_candidates = 0
    n_nonzero_candidates = 0

    for detector_index, (
        detector,
        detector_action_residual,
        detector_relative_residual,
        detector_diagonal,
    ) in enumerate(detectors):
        for region_index, (embedding_context, rdm) in enumerate(
            zip(embedding_contexts, rdms, strict=True)
        ):
            if recycler_source == "matrix_units" and detector_diagonal is not None:
                for target_index, target_pattern in enumerate(rdm.local_patterns):
                    for source_index, source_pattern in enumerate(rdm.local_patterns):
                        recycler_index = target_index * rdm.local_dim + source_index
                        recycler_name = (
                            f"{_pattern_name(target_pattern)}<-" f"{_pattern_name(source_pattern)}"
                        )
                        n_tested_candidates += 1
                        fast_metrics = _embedded_matrix_unit_metrics_with_diagonal_right_factor(
                            embedding_context=embedding_context,
                            target_local_index=target_index,
                            source_local_index=source_index,
                            right_diagonal=detector_diagonal,
                            state_basis=state_basis,
                            zero_tolerance=0.0,
                        )
                        if fast_metrics is None:
                            continue

                        (
                            inflow_norm,
                            target_block_norm,
                            jump_norm,
                            jump_nnz,
                            recycler_frobenius_norm,
                            recycler_nnz,
                        ) = fast_metrics
                        n_nonzero_candidates += 1
                        dark_residual = float(detector_action_residual * recycler_frobenius_norm)
                        relative_dark_residual = dark_residual / max(jump_norm, 1.0)
                        candidate = RecycledManifoldDarkDetectorCandidate(
                            candidate_index=n_nonzero_candidates - 1,
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
                            recycler_frobenius_norm=recycler_frobenius_norm,
                            recycler_nnz=recycler_nnz,
                            jump_nnz=jump_nnz,
                        )
                        if sort_by_inflow:
                            _append_ranked_recycled_candidate(
                                candidate_buffer,
                                candidate,
                                max_report_candidates=max_report_candidates,
                                dark_tolerance=dark_tolerance,
                            )
                        elif max_report_candidates is None or len(candidate_buffer) < max(
                            int(max_report_candidates),
                            0,
                        ):
                            candidate_buffer.append(candidate)
                continue

            recycler_specs = _local_recycler_specs(
                local_patterns=rdm.local_patterns,
                support_basis=rdm.support_basis,
                recycler_source=recycler_source,
            )
            for recycler_index, (recycler_name, local_operator) in enumerate(recycler_specs):
                n_tested_candidates += 1

                fast_metrics = None
                if detector_diagonal is not None:
                    fast_metrics = _embedded_local_operator_metrics_with_diagonal_right_factor(
                        embedding_context=embedding_context,
                        local_operator=local_operator,
                        right_diagonal=detector_diagonal,
                        state_basis=state_basis,
                        zero_tolerance=0.0,
                    )

                if fast_metrics is None:
                    recycler = _embed_local_pattern_operator_from_context(
                        context=embedding_context,
                        local_operator=local_operator,
                    )
                    if recycler.nnz == 0:
                        continue

                    jump = (recycler @ detector).tocsr()
                    if jump.nnz == 0:
                        continue
                    n_nonzero_candidates += 1

                    dark_residual = float(np.linalg.norm(jump @ state_basis))
                    jump_norm = float(sp.linalg.norm(jump))
                    relative_dark_residual = dark_residual / max(jump_norm, 1.0)
                    inflow_norm, target_block_norm = _projected_inflow_norm(
                        jump=jump,
                        state_basis=state_basis,
                    )
                    recycler_frobenius_norm = float(sp.linalg.norm(recycler))
                    recycler_nnz = int(recycler.nnz)
                    jump_nnz = int(jump.nnz)
                else:
                    (
                        inflow_norm,
                        target_block_norm,
                        jump_norm,
                        jump_nnz,
                        recycler_frobenius_norm,
                        recycler_nnz,
                    ) = fast_metrics
                    n_nonzero_candidates += 1
                    # Darkness comes from the right detector: J Q = R (D Q).
                    # This inexpensive bound is exact when the detector is
                    # exactly dark, which is the intended use of this scan.
                    dark_residual = float(detector_action_residual * recycler_frobenius_norm)
                    relative_dark_residual = dark_residual / max(jump_norm, 1.0)

                candidate = RecycledManifoldDarkDetectorCandidate(
                    candidate_index=n_nonzero_candidates - 1,
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
                    recycler_frobenius_norm=recycler_frobenius_norm,
                    recycler_nnz=recycler_nnz,
                    jump_nnz=jump_nnz,
                )
                if sort_by_inflow:
                    _append_ranked_recycled_candidate(
                        candidate_buffer,
                        candidate,
                        max_report_candidates=max_report_candidates,
                        dark_tolerance=dark_tolerance,
                    )
                elif max_report_candidates is None or len(candidate_buffer) < max(
                    int(max_report_candidates),
                    0,
                ):
                    candidate_buffer.append(candidate)

    if sort_by_inflow:
        candidate_buffer = sorted(
            candidate_buffer,
            key=lambda candidate: _recycled_candidate_sort_key(
                candidate,
                dark_tolerance=dark_tolerance,
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
        n_nonzero_candidates=int(n_nonzero_candidates),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        candidates=tuple(candidate_buffer),
    )


def _recycled_jump_for_candidate_from_cache(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    dim: int,
    detector_matrices: tuple[sp.csr_array, ...],
    detector_coefficients: npt.NDArray[np.complex128],
    detector_diagonals: dict[int, npt.NDArray[np.complex128] | None],
    embedding_contexts: dict[int, Any],
    rdms: dict[int, Any],
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ],
    zero_tolerance: float,
) -> sp.csr_array:
    """Rebuild a candidate jump using cached local contexts and detectors."""
    from qlinks.open_system.local_recycling import _embed_local_pattern_operator_from_context

    detector_diagonal = detector_diagonals.get(candidate.detector_index)
    embedding_context = embedding_contexts[candidate.region_index]

    if detector_diagonal is not None:
        if recycler_source == "matrix_units":
            target_index, source_index = divmod(
                int(candidate.recycler_index),
                int(candidate.local_dim),
            )
            return _embedded_matrix_unit_times_diagonal_as_csr(
                embedding_context=embedding_context,
                target_local_index=target_index,
                source_local_index=source_index,
                right_diagonal=detector_diagonal,
                dim=dim,
                zero_tolerance=zero_tolerance,
            )

        rdm = rdms[candidate.region_index]
        recycler_specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        _, local_operator = recycler_specs[candidate.recycler_index]
        return _embedded_local_operator_times_diagonal_as_csr(
            embedding_context=embedding_context,
            local_operator=local_operator,
            right_diagonal=detector_diagonal,
            dim=dim,
            zero_tolerance=zero_tolerance,
        )

    detector = _combined_operator(
        operators=detector_matrices,
        coefficients=detector_coefficients[:, candidate.detector_index],
    )

    if recycler_source == "matrix_units":
        local_dim = int(candidate.local_dim)
        if local_dim != int(embedding_context.local_dim):
            raise ValueError(
                "candidate.local_dim is incompatible with the cached embedding context."
            )
        n_matrix_units = local_dim * local_dim
        if candidate.recycler_index < 0 or candidate.recycler_index >= n_matrix_units:
            raise ValueError("candidate.recycler_index is out of range for matrix-unit recyclers.")
        target_index, source_index = divmod(int(candidate.recycler_index), local_dim)
        local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
        local_operator[target_index, source_index] = 1.0
    else:
        rdm = rdms[candidate.region_index]
        recycler_specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")
        _, local_operator = recycler_specs[candidate.recycler_index]

    recycler = _embed_local_pattern_operator_from_context(
        context=embedding_context,
        local_operator=local_operator,
    )
    return (recycler @ detector).tocsr()


def _local_recycler_operator_from_candidate_cache(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    embedding_contexts: dict[int, Any],
    rdms: dict[int, Any],
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ],
) -> tuple[str, npt.NDArray[np.complex128]]:
    """Return a candidate's local recycler matrix from cached local data."""
    embedding_context = embedding_contexts[candidate.region_index]
    if recycler_source == "matrix_units":
        local_dim = int(embedding_context.local_dim)
        if int(candidate.local_dim) != local_dim:
            raise ValueError(
                "candidate.local_dim is incompatible with the cached embedding context."
            )
        n_matrix_units = local_dim * local_dim
        if candidate.recycler_index < 0 or candidate.recycler_index >= n_matrix_units:
            raise ValueError("candidate.recycler_index is out of range for matrix-unit recyclers.")
        target_index, source_index = divmod(int(candidate.recycler_index), local_dim)
        local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
        local_operator[target_index, source_index] = 1.0
        return candidate.recycler_name, local_operator

    rdm = rdms[candidate.region_index]
    recycler_specs = _local_recycler_specs(
        local_patterns=rdm.local_patterns,
        support_basis=rdm.support_basis,
        recycler_source=recycler_source,
    )
    if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
        raise ValueError("candidate.recycler_index is out of range for recycler specs.")
    recycler_name, local_operator = recycler_specs[candidate.recycler_index]
    return recycler_name, np.asarray(local_operator, dtype=np.complex128)


def _collective_recycler_weight(
    candidate: RecycledManifoldDarkDetectorCandidate,
    *,
    weighting: Literal["unit", "inflow", "normalized_inflow"],
) -> complex:
    if weighting == "unit":
        return 1.0 + 0.0j
    if weighting == "inflow":
        return complex(float(candidate.inflow_norm))
    if weighting == "normalized_inflow":
        return complex(
            float(candidate.inflow_norm) / max(float(candidate.jump_frobenius_norm), 1.0e-300)
        )
    raise ValueError(
        'collective_recycler_weighting must be "unit", "inflow", or "normalized_inflow".'
    )


def _bundle_recycled_jumps_by_region_detector(
    *,
    selected_candidates: tuple[RecycledManifoldDarkDetectorCandidate, ...],
    dim: int,
    detector_matrices: tuple[sp.csr_array, ...],
    detector_coefficients: npt.NDArray[np.complex128],
    embedding_contexts: dict[int, Any],
    rdms: dict[int, Any],
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ],
    weighting: Literal["unit", "inflow", "normalized_inflow"],
    normalize_recyclers: bool,
    tolerance: float,
) -> tuple[tuple[sp.csr_array, ...], tuple[RecycledManifoldCollectiveRecyclerGroup, ...]]:
    """Bundle selected microscopic recyclers into collective local recyclers.

    Candidates are grouped only by ``(detector_index, region_index)``. This is
    locality-safe: every bundled recycler acts on the same local region as the
    selected microscopic recyclers, and the same dark detector remains on the
    right.
    """
    from qlinks.open_system.local_recycling import _embed_local_pattern_operator_from_context

    grouped: dict[tuple[int, int], list[RecycledManifoldDarkDetectorCandidate]] = {}
    group_order: list[tuple[int, int]] = []
    for candidate in selected_candidates:
        key = (int(candidate.detector_index), int(candidate.region_index))
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(candidate)

    detector_cache: dict[int, sp.csr_array] = {}
    jumps: list[sp.csr_array] = []
    groups: list[RecycledManifoldCollectiveRecyclerGroup] = []

    for group_index, key in enumerate(group_order):
        detector_index, region_index = key
        candidates = tuple(grouped[key])
        embedding_context = embedding_contexts[region_index]
        local_dim = int(embedding_context.local_dim)
        local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
        raw_weights: list[complex] = []
        recycler_names: list[str] = []

        for candidate in candidates:
            recycler_name, candidate_operator = _local_recycler_operator_from_candidate_cache(
                candidate=candidate,
                embedding_contexts=embedding_contexts,
                rdms=rdms,
                recycler_source=recycler_source,
            )
            weight = _collective_recycler_weight(candidate, weighting=weighting)
            raw_weights.append(weight)
            recycler_names.append(recycler_name)
            local_operator += weight * candidate_operator

        frobenius_norm = float(np.linalg.norm(local_operator))
        if normalize_recyclers and frobenius_norm > tolerance:
            local_operator = local_operator / frobenius_norm
            weights = tuple(complex(weight / frobenius_norm) for weight in raw_weights)
        else:
            weights = tuple(complex(weight) for weight in raw_weights)

        recycler = _embed_local_pattern_operator_from_context(
            context=embedding_context,
            local_operator=local_operator,
        ).tocsr()
        detector = detector_cache.get(detector_index)
        if detector is None:
            detector = _combined_operator(
                operators=detector_matrices,
                coefficients=detector_coefficients[:, detector_index],
            )
            detector_cache[detector_index] = detector
        jump = (recycler @ detector).tocsr()
        if jump.shape != (dim, dim):
            raise ValueError("bundled jump has incompatible shape.")
        jumps.append(jump)
        groups.append(
            RecycledManifoldCollectiveRecyclerGroup(
                group_index=int(group_index),
                detector_index=int(detector_index),
                detector_name=candidates[0].detector_name,
                region_index=int(region_index),
                variable_indices=tuple(int(value) for value in candidates[0].variable_indices),
                local_dim=local_dim,
                candidate_indices=tuple(int(candidate.candidate_index) for candidate in candidates),
                recycler_indices=tuple(int(candidate.recycler_index) for candidate in candidates),
                recycler_names=tuple(recycler_names),
                weights=weights,
                local_operator=np.asarray(local_operator, dtype=np.complex128),
                jump_frobenius_norm=float(sp.linalg.norm(jump)),
                recycler_frobenius_norm=float(sp.linalg.norm(recycler)),
                recycler_nnz=int(recycler.nnz),
                jump_nnz=int(jump.nnz),
            )
        )

    return tuple(jumps), tuple(groups)


def _recycled_jump_for_candidate(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
) -> sp.csr_array:
    """Rebuild the sparse jump operator for a reported recycled candidate."""
    from qlinks.open_system.local_recycling import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")

    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
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
            [report_candidate.coefficients for report_candidate in dark_operator_report.candidates]
        )

    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=len(detector_matrices),
    )
    if candidate.detector_index < 0 or candidate.detector_index >= coefficients.shape[1]:
        raise ValueError("candidate.detector_index is out of range for detector coefficients.")

    detector = _combined_operator(
        operators=detector_matrices,
        coefficients=coefficients[:, candidate.detector_index],
    )

    regions = _normalize_local_regions(local_regions)
    if candidate.region_index < 0 or candidate.region_index >= len(regions):
        raise ValueError("candidate.region_index is out of range for local_regions.")

    context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_array,
        variable_indices=regions[candidate.region_index],
    )
    embedding_context = _embedding_context_from_basis_context(context)
    rdm = _local_reduced_density_matrix_from_basis_context_and_states(
        context=context,
        states=state_basis,
        tolerance=rdm_tolerance,
    )
    recycler_specs = _local_recycler_specs(
        local_patterns=rdm.local_patterns,
        support_basis=rdm.support_basis,
        recycler_source=recycler_source,
    )
    if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
        raise ValueError("candidate.recycler_index is out of range for recycler specs.")

    _, local_operator = recycler_specs[candidate.recycler_index]
    recycler = _embed_local_pattern_operator_from_context(
        context=embedding_context,
        local_operator=local_operator,
    )
    return (recycler @ detector).tocsr()


def _internal_liouvillian_eigenvalues_from_energies(
    energies: tuple[complex, ...],
) -> tuple[complex, ...]:
    return tuple(-1j * (left - right) for left in energies for right in energies)


def _state_ipr(state: npt.NDArray[np.complex128]) -> float:
    norm_sq = float(np.vdot(state, state).real)
    if norm_sq <= 0.0:
        return 0.0
    probabilities = np.abs(state) ** 2 / norm_sq
    return float(np.sum(probabilities * probabilities))


def _hamiltonian_projector_commutator_residual(
    *,
    hamiltonian: sp.csr_array,
    state_basis: npt.NDArray[np.complex128],
) -> float:
    manifold_dimension = int(state_basis.shape[1])
    density = state_basis @ state_basis.conj().T / float(manifold_dimension)
    action_left = np.asarray(hamiltonian @ density, dtype=np.complex128)
    action_right = np.asarray((density @ hamiltonian).astype(np.complex128))
    return float(np.linalg.norm(-1j * (action_left - action_right)))


def _detector_coefficients_from_report(
    *,
    detector_coefficients: npt.ArrayLike | None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None,
    n_operators: int,
    max_detectors: int | None,
) -> npt.NDArray[np.complex128]:
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
        n_operators=n_operators,
    )
    if max_detectors is not None:
        coefficients = coefficients[:, : max(int(max_detectors), 0)]
    return coefficients


def _stream_recycled_family_kernel_diagnostics(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    candidates: tuple[RecycledManifoldDarkDetectorCandidate, ...],
    detector_coefficients: npt.ArrayLike | None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None,
    recycler_source: Literal["matrix_units", "rdm_support_matrix_units"],
    tolerance: float,
    rdm_tolerance: float,
    kernel_tolerance: float,
    liouvillian_zero_tolerance: float,
    max_detectors: int | None,
) -> tuple[RecycledFamilyKernelDiagnostics, int, int, int]:
    from qlinks.open_system.local_recycling import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    hamiltonian_matrix = _as_csr(hamiltonian)
    if hamiltonian_matrix.shape != (dim, dim):
        raise ValueError("hamiltonian must have shape (hilbert_dimension, hilbert_dimension).")

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")
    for operator in detector_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(
                "operator has incompatible shape: " f"{operator.shape} != {(dim, dim)}."
            )

    coefficients = _detector_coefficients_from_report(
        detector_coefficients=detector_coefficients,
        dark_operator_report=dark_operator_report,
        n_operators=len(detector_matrices),
        max_detectors=max_detectors,
    )
    detectors = tuple(
        _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        for detector_index in range(coefficients.shape[1])
    )

    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")

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
    recycler_specs_by_region = tuple(
        _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        for rdm in rdms
    )
    recycler_cache: dict[tuple[int, int], sp.csr_array] = {}

    complement_basis = _orthogonal_complement_basis(state_basis, tolerance=kernel_tolerance)
    complement_dimension = int(complement_basis.shape[1])
    family_gram = np.zeros(
        (complement_dimension, complement_dimension),
        dtype=np.complex128,
    )
    total_jump_nnz = 0
    max_jump_nnz = 0
    max_target_jump_residual = 0.0
    inflow_squared = 0.0
    n_jumps = 0

    for candidate in candidates:
        if candidate.detector_index < 0 or candidate.detector_index >= len(detectors):
            raise ValueError("candidate.detector_index is out of range for detector coefficients.")
        if candidate.region_index < 0 or candidate.region_index >= len(regions):
            raise ValueError("candidate.region_index is out of range for local_regions.")
        recycler_specs = recycler_specs_by_region[candidate.region_index]
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")

        cache_key = (int(candidate.region_index), int(candidate.recycler_index))
        recycler = recycler_cache.get(cache_key)
        if recycler is None:
            _, local_operator = recycler_specs[candidate.recycler_index]
            recycler = _embed_local_pattern_operator_from_context(
                context=embedding_contexts[candidate.region_index],
                local_operator=local_operator,
            ).tocsr()
            recycler_cache[cache_key] = recycler

        jump = (recycler @ detectors[candidate.detector_index]).tocsr()
        if jump.nnz == 0:
            continue
        n_jumps += 1
        total_jump_nnz += int(jump.nnz)
        max_jump_nnz = max(max_jump_nnz, int(jump.nnz))
        max_target_jump_residual = max(
            max_target_jump_residual,
            float(np.linalg.norm(jump @ state_basis)),
        )
        inflow_squared += float(candidate.inflow_norm) ** 2

        if complement_dimension > 0:
            image = np.asarray(jump @ complement_basis, dtype=np.complex128)
            family_gram += image.conj().T @ image

    if complement_dimension == 0:
        bad_basis = np.zeros((dim, 0), dtype=np.complex128)
    else:
        family_gram = 0.5 * (family_gram + family_gram.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(family_gram)
        largest = float(np.max(np.maximum(eigenvalues.real, 0.0))) if eigenvalues.size else 0.0
        cutoff = max(float(kernel_tolerance), float(kernel_tolerance) * max(largest, 1.0))
        kernel_mask = np.asarray(eigenvalues.real <= cutoff, dtype=bool)
        bad_basis = complement_basis @ eigenvectors[:, kernel_mask]

    bad_dimension = int(bad_basis.shape[1])
    common_dimension = int(manifold_dimension + bad_dimension)
    bad_iprs = tuple(_state_ipr(bad_basis[:, index]) for index in range(bad_dimension))

    hamiltonian_action = np.asarray(hamiltonian_matrix @ state_basis, dtype=np.complex128)
    internal_hamiltonian = state_basis.conj().T @ hamiltonian_action
    projected_hamiltonian_action = state_basis @ internal_hamiltonian
    hamiltonian_closure_residual = float(
        np.linalg.norm(hamiltonian_action - projected_hamiltonian_action)
    )
    internal_hamiltonian = 0.5 * (internal_hamiltonian + internal_hamiltonian.conj().T)
    internal_hamiltonian_eigenvalues = tuple(
        complex(value) for value in np.linalg.eigvalsh(internal_hamiltonian)
    )
    expected_internal_liouvillian_eigenvalues = _internal_liouvillian_eigenvalues_from_energies(
        internal_hamiltonian_eigenvalues
    )
    expected_internal_zero_mode_count = int(
        sum(
            abs(value) <= liouvillian_zero_tolerance
            for value in expected_internal_liouvillian_eigenvalues
        )
    )
    expected_internal_peripheral_mode_count = int(
        len(expected_internal_liouvillian_eigenvalues) - expected_internal_zero_mode_count
    )

    target_density_liouvillian_residual = _hamiltonian_projector_commutator_residual(
        hamiltonian=hamiltonian_matrix,
        state_basis=state_basis,
    )
    target_in_common_jump_kernel = max_target_jump_residual <= kernel_tolerance
    target_distance = 0.0 if target_in_common_jump_kernel else float("nan")
    target_projection = float(np.sqrt(manifold_dimension)) if target_in_common_jump_kernel else 0.0

    diagnostics = RecycledFamilyKernelDiagnostics(
        dim=dim,
        n_jumps=n_jumps,
        manifold_dimension=manifold_dimension,
        hamiltonian_closure_residual=hamiltonian_closure_residual,
        max_target_jump_residual=max_target_jump_residual,
        target_density_liouvillian_residual=target_density_liouvillian_residual,
        inflow_norm=float(np.sqrt(max(inflow_squared, 0.0))),
        common_jump_kernel_dimension=common_dimension,
        target_projection_onto_common_kernel=target_projection,
        target_distance_from_common_kernel=target_distance,
        target_in_common_jump_kernel=bool(target_in_common_jump_kernel),
        bad_common_jump_kernel_dimension=bad_dimension,
        bad_common_jump_kernel_iprs=bad_iprs,
        internal_hamiltonian_eigenvalues=internal_hamiltonian_eigenvalues,
        expected_internal_zero_mode_count=expected_internal_zero_mode_count,
        expected_internal_peripheral_mode_count=expected_internal_peripheral_mode_count,
        liouvillian_zero_tolerance=float(liouvillian_zero_tolerance),
    )
    return diagnostics, n_jumps, total_jump_nnz, max_jump_nnz


def diagnose_recycled_manifold_candidate_family_kernel(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
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
    kernel_tolerance: float = 1.0e-10,
    liouvillian_zero_tolerance: float = 1.0e-9,
    max_detectors: int | None = None,
    expand_candidate_report: bool = True,
    kernel_method: Literal["streamed", "diagnostics"] = "streamed",
    store_candidate_jumps: bool = False,
) -> RecycledManifoldCandidateFamilyKernelReport:
    """Diagnose the common jump kernel of the full recycled-detector family.

    This is the decisive follow-up when greedy selection saturates at a
    nonzero complement kernel.  If the family of all eligible local candidates
    still has a bad common jump kernel, no subset selected from that family can
    remove it.  If the family removes the kernel but the greedy subset does not,
    the problem is the subset-selection heuristic rather than the operator
    family.
    """
    from qlinks.open_system.diagnostics import diagnose_dark_manifold

    regions = _normalize_local_regions(local_regions)
    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])
    candidate_report_was_expanded = False
    if candidate_report is None:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=None,
            sort_by_inflow=True,
        )
    elif expand_candidate_report and candidate_report.candidate_report_is_truncated:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=None,
            sort_by_inflow=True,
        )
        candidate_report_was_expanded = True

    eligible_candidates = tuple(
        candidate
        for candidate in candidate_report.candidates
        if candidate.relative_dark_residual <= dark_tolerance
        and candidate.inflow_norm > inflow_tolerance
    )
    if kernel_method not in {"streamed", "diagnostics"}:
        raise ValueError('kernel_method must be "streamed" or "diagnostics".')

    candidate_jumps: tuple[sp.csr_array, ...] = ()
    candidate_jump_count: int | None = None
    candidate_total_jump_nnz: int | None = None
    candidate_max_jump_nnz: int | None = None

    if kernel_method == "streamed":
        diagnostics, jump_count, total_jump_nnz, max_jump_nnz = (
            _stream_recycled_family_kernel_diagnostics(
                hamiltonian=hamiltonian,
                states=state_basis,
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=regions,
                candidates=eligible_candidates,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                recycler_source=recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                kernel_tolerance=kernel_tolerance,
                liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                max_detectors=max_detectors,
            )
        )
        candidate_jump_count = jump_count
        candidate_total_jump_nnz = total_jump_nnz
        candidate_max_jump_nnz = max_jump_nnz
        if store_candidate_jumps:
            candidate_jumps = tuple(
                _recycled_jump_for_candidate(
                    candidate=candidate,
                    states=state_basis,
                    basis_configs=basis_configs,
                    detector_operators=detector_operators,
                    local_regions=regions,
                    detector_coefficients=detector_coefficients,
                    dark_operator_report=dark_operator_report,
                    recycler_source=recycler_source,
                    tolerance=tolerance,
                    rdm_tolerance=rdm_tolerance,
                )
                for candidate in eligible_candidates
            )
    else:
        candidate_jumps = tuple(
            _recycled_jump_for_candidate(
                candidate=candidate,
                states=state_basis,
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                recycler_source=recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
            )
            for candidate in eligible_candidates
        )

        diagnostics = diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=candidate_jumps,
            target_states=state_basis,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=False,
            liouvillian_spectrum_method="none",
        )

    return RecycledManifoldCandidateFamilyKernelReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        candidate_report=candidate_report,
        candidate_report_was_expanded=candidate_report_was_expanded,
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        candidate_jumps=candidate_jumps,
        diagnostics=diagnostics,
        candidate_jump_count=candidate_jump_count,
        candidate_total_jump_nnz=candidate_total_jump_nnz,
        candidate_max_jump_nnz=candidate_max_jump_nnz,
    )


@dataclass(frozen=True, slots=True)
class ResidualKernelOperatorActionEntry:
    """Action of one probe operator on the residual bad-kernel subspace."""

    operator_index: int
    operator_name: str
    action_norm: float
    target_component_norm: float
    residual_component_norm: float
    outside_component_norm: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "operator_index": self.operator_index,
            "operator_name": self.operator_name,
            "action_norm": self.action_norm,
            "target_component_norm": self.target_component_norm,
            "residual_component_norm": self.residual_component_norm,
            "outside_component_norm": self.outside_component_norm,
        }


@dataclass(frozen=True, slots=True)
class ResidualKernelOperatorActionReport:
    """Probe-operator action on the residual bad-kernel subspace."""

    group_name: str
    n_operators: int
    entries: tuple[ResidualKernelOperatorActionEntry, ...]

    @property
    def total_action_norm(self) -> float:
        return float(np.sqrt(sum(entry.action_norm**2 for entry in self.entries)))

    @property
    def total_target_component_norm(self) -> float:
        return float(np.sqrt(sum(entry.target_component_norm**2 for entry in self.entries)))

    @property
    def total_residual_component_norm(self) -> float:
        return float(np.sqrt(sum(entry.residual_component_norm**2 for entry in self.entries)))

    @property
    def total_outside_component_norm(self) -> float:
        return float(np.sqrt(sum(entry.outside_component_norm**2 for entry in self.entries)))

    @property
    def max_target_component_norm(self) -> float:
        return max((entry.target_component_norm for entry in self.entries), default=0.0)

    @property
    def max_outside_component_norm(self) -> float:
        return max((entry.outside_component_norm for entry in self.entries), default=0.0)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "group_name": self.group_name,
            "n_operators": self.n_operators,
            "total_action_norm": self.total_action_norm,
            "total_target_component_norm": self.total_target_component_norm,
            "total_residual_component_norm": self.total_residual_component_norm,
            "total_outside_component_norm": self.total_outside_component_norm,
            "max_target_component_norm": self.max_target_component_norm,
            "max_outside_component_norm": self.max_outside_component_norm,
            "entries": tuple(entry.to_summary_dict() for entry in self.entries),
        }


@dataclass(frozen=True, slots=True)
class ResidualKernelLocalSupportEntry:
    """Local support comparison between target and residual bad-kernel subspaces."""

    variable_indices: tuple[int, ...]
    local_dim: int
    target_support_rank: int
    target_nullity: int
    residual_support_rank: int
    residual_nullity: int
    combined_support_rank: int
    combined_nullity: int
    residual_support_outside_target_norm: float

    @property
    def n_variables(self) -> int:
        return len(self.variable_indices)

    @property
    def residual_support_inside_target(self) -> bool:
        return self.residual_support_outside_target_norm <= 1.0e-10

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "variable_indices": self.variable_indices,
            "n_variables": self.n_variables,
            "local_dim": self.local_dim,
            "target_support_rank": self.target_support_rank,
            "target_nullity": self.target_nullity,
            "residual_support_rank": self.residual_support_rank,
            "residual_nullity": self.residual_nullity,
            "combined_support_rank": self.combined_support_rank,
            "combined_nullity": self.combined_nullity,
            "residual_support_outside_target_norm": self.residual_support_outside_target_norm,
            "residual_support_inside_target": self.residual_support_inside_target,
        }


@dataclass(frozen=True, slots=True)
class RecycledManifoldResidualKernelReport:
    """Diagnostics for the residual complement kernel left by a recycled family.

    The report focuses on the bad subspace

        B = (cap_mu ker J_mu) cap M^perp,

    where ``J_mu`` ranges over the chosen recycled-detector family.  It is meant
    to distinguish a mere subset-selection failure from a structural residual
    sector that the current local operator family cannot see.
    """

    manifold_dimension: int
    hilbert_dimension: int
    family_report: RecycledManifoldCandidateFamilyKernelReport
    residual_basis: npt.NDArray[np.complex128]
    hamiltonian_target_coupling_norm: float
    hamiltonian_residual_block_norm: float
    hamiltonian_outside_residual_norm: float
    hamiltonian_residual_eigenvalues: tuple[complex, ...]
    operator_action_reports: tuple[ResidualKernelOperatorActionReport, ...]
    local_support_entries: tuple[ResidualKernelLocalSupportEntry, ...]
    kernel_tolerance: float

    @property
    def residual_dimension(self) -> int:
        return int(self.residual_basis.shape[1])

    @property
    def residual_iprs(self) -> tuple[float, ...]:
        return tuple(
            _state_ipr(self.residual_basis[:, index]) for index in range(self.residual_dimension)
        )

    @property
    def residual_ipr_min(self) -> float | None:
        values = self.residual_iprs
        if len(values) == 0:
            return None
        return float(min(values))

    @property
    def residual_ipr_max(self) -> float | None:
        values = self.residual_iprs
        if len(values) == 0:
            return None
        return float(max(values))

    @property
    def hamiltonian_keeps_residual_sector(self) -> bool:
        return (
            self.hamiltonian_target_coupling_norm <= self.kernel_tolerance
            and self.hamiltonian_outside_residual_norm <= self.kernel_tolerance
        )

    @property
    def n_local_support_entries(self) -> int:
        return len(self.local_support_entries)

    @property
    def all_local_residual_support_inside_target(self) -> bool:
        return all(entry.residual_support_inside_target for entry in self.local_support_entries)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "residual_dimension": self.residual_dimension,
            "residual_ipr_min": self.residual_ipr_min,
            "residual_ipr_max": self.residual_ipr_max,
            "hamiltonian_target_coupling_norm": self.hamiltonian_target_coupling_norm,
            "hamiltonian_residual_block_norm": self.hamiltonian_residual_block_norm,
            "hamiltonian_outside_residual_norm": self.hamiltonian_outside_residual_norm,
            "hamiltonian_keeps_residual_sector": self.hamiltonian_keeps_residual_sector,
            "hamiltonian_residual_eigenvalues": tuple(
                complex(value) for value in self.hamiltonian_residual_eigenvalues
            ),
            "operator_action_reports": tuple(
                report.to_summary_dict() for report in self.operator_action_reports
            ),
            "n_local_support_entries": self.n_local_support_entries,
            "all_local_residual_support_inside_target": (
                self.all_local_residual_support_inside_target
            ),
            "local_support_entries": tuple(
                entry.to_summary_dict() for entry in self.local_support_entries
            ),
            "family_report": self.family_report.to_summary_dict(),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_operator_rows: int = 12, max_local_rows: int = 16):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "RecycledManifoldResidualKernelReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("residual bad-kernel dimension", str(self.residual_dimension))
        overview.add_row(
            "residual IPR range",
            (
                "none"
                if self.residual_ipr_min is None
                else f"{self.residual_ipr_min:.3e} .. {self.residual_ipr_max:.3e}"
            ),
        )
        overview.add_row("H target coupling", f"{self.hamiltonian_target_coupling_norm:.3e}")
        overview.add_row("H outside residual", f"{self.hamiltonian_outside_residual_norm:.3e}")
        overview.add_row("H keeps residual sector", str(self.hamiltonian_keeps_residual_sector))
        overview.add_row(
            "local residual support inside target",
            str(self.all_local_residual_support_inside_target),
        )

        operator_table = Table(title="Probe-operator action on residual kernel")
        operator_table.add_column("group")
        operator_table.add_column("operators", justify="right")
        operator_table.add_column("||O B||", justify="right")
        operator_table.add_column("target", justify="right")
        operator_table.add_column("residual", justify="right")
        operator_table.add_column("outside", justify="right")
        for report in self.operator_action_reports[: max(int(max_operator_rows), 0)]:
            operator_table.add_row(
                report.group_name,
                str(report.n_operators),
                f"{report.total_action_norm:.3e}",
                f"{report.total_target_component_norm:.3e}",
                f"{report.total_residual_component_norm:.3e}",
                f"{report.total_outside_component_norm:.3e}",
            )

        local_table = Table(title="Local support comparison")
        local_table.add_column("region")
        local_table.add_column("dim", justify="right")
        local_table.add_column("target rank", justify="right")
        local_table.add_column("residual rank", justify="right")
        local_table.add_column("combined rank", justify="right")
        local_table.add_column("resid outside target", justify="right")
        for entry in self.local_support_entries[: max(int(max_local_rows), 0)]:
            local_table.add_row(
                str(entry.variable_indices),
                str(entry.local_dim),
                str(entry.target_support_rank),
                str(entry.residual_support_rank),
                str(entry.combined_support_rank),
                f"{entry.residual_support_outside_target_norm:.3e}",
            )
        if len(self.local_support_entries) > max_local_rows:
            local_table.add_row(
                "…",
                "",
                "",
                "",
                "",
                f"{len(self.local_support_entries) - max_local_rows} more regions",
            )

        return Panel(
            Group(overview, operator_table, local_table),
            title=Text("Recycled residual-kernel report", style="bold red"),
            border_style="red" if self.residual_dimension else "green",
        )


@dataclass(frozen=True, slots=True)
class TargetedResidualKernelLinearTerm:
    """One local matrix-unit term in a targeted residual-kernel jump."""

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
class TargetedResidualKernelLinearCandidate:
    """A local jump candidate found by constrained residual-kernel search."""

    candidate_index: int
    region_index: int
    variable_indices: tuple[int, ...]
    local_dim: int
    operator_source: str
    dark_constraint_rank: int
    dark_nullity: int
    singular_value: float
    residual_target_inflow_norm: float
    dark_residual: float
    relative_dark_residual: float
    total_inflow_norm: float
    target_block_norm: float
    jump_frobenius_norm: float
    jump_nnz: int
    coefficients: npt.NDArray[np.complex128]
    terms: tuple[TargetedResidualKernelLinearTerm, ...]
    residual_action_norm: float = 0.0
    residual_score_norm: float = 0.0
    residual_objective: str = "target_inflow"

    @property
    def n_variables(self) -> int:
        return len(self.variable_indices)

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def is_dark(self) -> bool:
        return self.relative_dark_residual <= 1.0e-10

    @property
    def hits_residual_kernel(self) -> bool:
        return max(self.residual_score_norm, self.residual_target_inflow_norm) > 1.0e-12

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "region_index": self.region_index,
            "variable_indices": self.variable_indices,
            "n_variables": self.n_variables,
            "local_dim": self.local_dim,
            "operator_source": self.operator_source,
            "dark_constraint_rank": self.dark_constraint_rank,
            "dark_nullity": self.dark_nullity,
            "singular_value": self.singular_value,
            "residual_target_inflow_norm": self.residual_target_inflow_norm,
            "residual_action_norm": self.residual_action_norm,
            "residual_score_norm": max(
                self.residual_score_norm,
                self.residual_target_inflow_norm,
            ),
            "residual_objective": self.residual_objective,
            "dark_residual": self.dark_residual,
            "relative_dark_residual": self.relative_dark_residual,
            "total_inflow_norm": self.total_inflow_norm,
            "target_block_norm": self.target_block_norm,
            "jump_frobenius_norm": self.jump_frobenius_norm,
            "jump_nnz": self.jump_nnz,
            "n_terms": self.n_terms,
            "coefficients": tuple(complex(value) for value in self.coefficients),
            "terms": tuple(term.to_summary_dict() for term in self.terms),
            "is_dark": self.is_dark,
            "hits_residual_kernel": self.hits_residual_kernel,
        }


@dataclass(frozen=True, slots=True)
class TargetedResidualKernelLinearSearchReport:
    """Constrained local search targeting a recycled-family residual kernel.

    For each local region with operator basis ``O_a``, the search solves

        sum_a c_a O_a P_M = 0

    and, inside that dark nullspace, maximizes either

        ||P_M (sum_a c_a O_a) B||_F

    or

        ||(sum_a c_a O_a) B||_F,

    where ``B`` is the residual bad common-kernel basis left by a recycled
    detector family.  The first objective emphasizes direct inflow to the
    target manifold; the second directly attacks a remaining dark kernel sector
    even when it does not couple to the target in one jump.
    """

    manifold_dimension: int
    hilbert_dimension: int
    residual_basis: npt.NDArray[np.complex128]
    region_variable_indices: tuple[tuple[int, ...], ...]
    operator_source: str
    family_report: RecycledManifoldCandidateFamilyKernelReport | None
    candidates: tuple[TargetedResidualKernelLinearCandidate, ...]
    candidate_jumps: tuple[sp.csr_array, ...]
    tolerance: float
    dark_tolerance: float
    inflow_tolerance: float
    residual_objective: str = "target_inflow"
    n_regions_evaluated: int = 0
    n_regions_skipped_by_local_dim: int = 0
    n_regions_with_no_recycler_specs: int = 0
    n_regions_with_no_nonzero_local_operators: int = 0
    n_regions_with_zero_dark_nullity: int = 0
    n_regions_with_dark_nullity_detected: int = -1
    n_regions_with_zero_residual_inflow: int = 0
    n_candidate_modes_generated: int = -1
    max_encountered_local_dim: int = 0

    @property
    def residual_dimension(self) -> int:
        return int(self.residual_basis.shape[1])

    @property
    def n_regions(self) -> int:
        return len(self.region_variable_indices)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def max_region_size(self) -> int:
        return max((len(region) for region in self.region_variable_indices), default=0)

    @property
    def max_local_dim(self) -> int:
        return max(
            self.max_encountered_local_dim,
            max((candidate.local_dim for candidate in self.candidates), default=0),
        )

    @property
    def n_regions_with_dark_nullity(self) -> int:
        if self.n_regions_with_dark_nullity_detected >= 0:
            return self.n_regions_with_dark_nullity_detected
        return len(
            {candidate.region_index for candidate in self.candidates if candidate.dark_nullity > 0}
        )

    @property
    def n_reported_candidate_modes(self) -> int:
        return len(self.candidates)

    @property
    def n_generated_candidate_modes(self) -> int:
        if self.n_candidate_modes_generated >= 0:
            return self.n_candidate_modes_generated
        return len(self.candidates)

    @property
    def targeted_search_failure_counts(self) -> dict[str, int]:
        return {
            "n_regions_evaluated": self.n_regions_evaluated,
            "n_regions_skipped_by_local_dim": self.n_regions_skipped_by_local_dim,
            "n_regions_with_no_recycler_specs": self.n_regions_with_no_recycler_specs,
            "n_regions_with_no_nonzero_local_operators": (
                self.n_regions_with_no_nonzero_local_operators
            ),
            "n_regions_with_zero_dark_nullity": self.n_regions_with_zero_dark_nullity,
            "n_regions_with_dark_nullity": self.n_regions_with_dark_nullity,
            "n_regions_with_zero_residual_inflow": self.n_regions_with_zero_residual_inflow,
        }

    @property
    def n_candidates_hitting_residual(self) -> int:
        return sum(
            candidate.relative_dark_residual <= self.dark_tolerance
            and max(candidate.residual_score_norm, candidate.residual_target_inflow_norm)
            > self.inflow_tolerance
            for candidate in self.candidates
        )

    @property
    def best_residual_target_inflow_norm(self) -> float:
        return max(
            (candidate.residual_target_inflow_norm for candidate in self.candidates), default=0.0
        )

    @property
    def best_residual_action_norm(self) -> float:
        return max((candidate.residual_action_norm for candidate in self.candidates), default=0.0)

    @property
    def best_residual_score_norm(self) -> float:
        return max(
            (
                max(candidate.residual_score_norm, candidate.residual_target_inflow_norm)
                for candidate in self.candidates
            ),
            default=0.0,
        )

    @property
    def best_total_inflow_norm(self) -> float:
        return max((candidate.total_inflow_norm for candidate in self.candidates), default=0.0)

    @property
    def has_targeted_solution(self) -> bool:
        return self.n_candidates_hitting_residual > 0

    def residual_kernel_dimension_after_candidate_prefix(
        self,
        n_candidates: int | None = None,
        *,
        tolerance: float | None = None,
    ) -> int:
        """Return residual-kernel dimension after the first reported jumps."""
        residual_dimension = int(self.residual_basis.shape[1])
        if residual_dimension == 0:
            return 0
        n_used = len(self.candidate_jumps) if n_candidates is None else max(int(n_candidates), 0)
        if n_used == 0:
            return residual_dimension
        gram = np.zeros((residual_dimension, residual_dimension), dtype=np.complex128)
        for jump in self.candidate_jumps[:n_used]:
            image = np.asarray(jump @ self.residual_basis, dtype=np.complex128)
            gram += image.conj().T @ image
        gram = 0.5 * (gram + gram.conj().T)
        eigenvalues = np.linalg.eigvalsh(gram).real
        largest = float(np.max(np.maximum(eigenvalues, 0.0))) if eigenvalues.size else 0.0
        cutoff = self.tolerance if tolerance is None else float(tolerance)
        cutoff = max(cutoff, cutoff * max(largest, 1.0))
        return int(np.count_nonzero(eigenvalues <= cutoff))

    @property
    def reported_candidate_residual_kernel_dimension(self) -> int:
        """Residual-family kernel dimension after all reported targeted candidates.

        This is the dimension of the residual bad kernel supplied to this
        targeted search after applying the reported candidate jumps. In the
        end-to-end workflow this is the residual left by the full recycled
        detector family, not necessarily the bad common kernel left by a
        compact selected recycled subset.
        """
        return self.residual_kernel_dimension_after_candidate_prefix(None)

    @property
    def reported_candidate_family_residual_kernel_dimension(self) -> int:
        """Alias with explicit workflow terminology."""
        return self.reported_candidate_residual_kernel_dimension

    @property
    def reported_candidates_remove_residual_kernel(self) -> bool:
        return self.reported_candidate_residual_kernel_dimension == 0

    @property
    def reported_candidates_remove_family_residual_kernel(self) -> bool:
        """Whether reported candidates remove the full-family residual kernel."""
        return self.reported_candidates_remove_residual_kernel

    def candidate_readouts(
        self,
        *,
        basis_configs: npt.NDArray[np.integer],
        max_readouts: int | None = None,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local-matrix readouts for reported targeted candidates."""
        candidates = self.candidates
        if max_readouts is not None:
            candidates = candidates[: max(int(max_readouts), 0)]
        return tuple(
            _local_operator_from_targeted_candidate(
                candidate=candidate,
                basis_configs=basis_configs,
            )
            for candidate in candidates
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "residual_dimension": self.residual_dimension,
            "n_regions": self.n_regions,
            "region_variable_indices": self.region_variable_indices,
            "max_region_size": self.max_region_size,
            "max_local_dim": self.max_local_dim,
            "operator_source": self.operator_source,
            "residual_objective": self.residual_objective,
            "n_candidates": self.n_candidates,
            "n_generated_candidate_modes": self.n_generated_candidate_modes,
            "n_reported_candidate_modes": self.n_reported_candidate_modes,
            "n_regions_evaluated": self.n_regions_evaluated,
            "n_regions_skipped_by_local_dim": self.n_regions_skipped_by_local_dim,
            "n_regions_with_no_recycler_specs": self.n_regions_with_no_recycler_specs,
            "n_regions_with_no_nonzero_local_operators": (
                self.n_regions_with_no_nonzero_local_operators
            ),
            "n_regions_with_zero_dark_nullity": self.n_regions_with_zero_dark_nullity,
            "n_regions_with_dark_nullity": self.n_regions_with_dark_nullity,
            "n_regions_with_zero_residual_inflow": self.n_regions_with_zero_residual_inflow,
            "targeted_search_failure_counts": self.targeted_search_failure_counts,
            "n_candidates_hitting_residual": self.n_candidates_hitting_residual,
            "has_targeted_solution": self.has_targeted_solution,
            "reported_candidate_residual_kernel_dimension": (
                self.reported_candidate_residual_kernel_dimension
            ),
            "reported_candidate_family_residual_kernel_dimension": (
                self.reported_candidate_family_residual_kernel_dimension
            ),
            "reported_candidates_remove_residual_kernel": (
                self.reported_candidates_remove_residual_kernel
            ),
            "reported_candidates_remove_family_residual_kernel": (
                self.reported_candidates_remove_family_residual_kernel
            ),
            "best_residual_target_inflow_norm": self.best_residual_target_inflow_norm,
            "best_residual_action_norm": self.best_residual_action_norm,
            "best_residual_score_norm": self.best_residual_score_norm,
            "best_total_inflow_norm": self.best_total_inflow_norm,
            "family_bad_common_jump_kernel_dimension": (
                None
                if self.family_report is None
                else self.family_report.family_bad_common_jump_kernel_dimension
            ),
            "tolerance": self.tolerance,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "candidates": tuple(candidate.to_summary_dict() for candidate in self.candidates),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 16, max_terms: int = 6):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "TargetedResidualKernelLinearSearchReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("residual dimension", str(self.residual_dimension))
        overview.add_row("regions", str(self.n_regions))
        overview.add_row("max region size", str(self.max_region_size))
        overview.add_row("operator source", self.operator_source)
        overview.add_row("residual objective", self.residual_objective)
        overview.add_row("candidates", str(self.n_candidates))
        overview.add_row("hits residual", str(self.n_candidates_hitting_residual))
        overview.add_row(
            "reported family residual kernel",
            str(self.reported_candidate_family_residual_kernel_dimension),
        )
        overview.add_row("best target inflow", f"{self.best_residual_target_inflow_norm:.3e}")
        overview.add_row("best residual action", f"{self.best_residual_action_norm:.3e}")
        overview.add_row("best residual score", f"{self.best_residual_score_norm:.3e}")
        overview.add_row("best total inflow", f"{self.best_total_inflow_norm:.3e}")

        table = Table(title="Targeted residual-kernel local dark jumps")
        table.add_column("#", justify="right")
        table.add_column("region")
        table.add_column("dim", justify="right")
        table.add_column("dark null", justify="right")
        table.add_column("resid score", justify="right")
        table.add_column("target inflow", justify="right")
        table.add_column("dark resid", justify="right")
        table.add_column("||J||", justify="right")
        table.add_column("terms")

        for candidate in self.candidates[: max(int(max_candidates), 0)]:
            term_text = ", ".join(
                f"{term.coefficient:.3g}·{term.operator_name}"
                for term in candidate.terms[: max(int(max_terms), 0)]
            )
            if candidate.n_terms > max_terms:
                term_text += f", … {candidate.n_terms - max_terms} more"
            style = "green" if candidate.hits_residual_kernel else ""
            table.add_row(
                str(candidate.candidate_index),
                str(candidate.variable_indices),
                str(candidate.local_dim),
                str(candidate.dark_nullity),
                f"{max(candidate.residual_score_norm, candidate.residual_target_inflow_norm):.3e}",
                f"{candidate.residual_target_inflow_norm:.3e}",
                f"{candidate.relative_dark_residual:.3e}",
                f"{candidate.jump_frobenius_norm:.3e}",
                term_text,
                style=style,
            )

        if len(self.candidates) > max_candidates:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{len(self.candidates) - max_candidates} more candidates",
            )

        return Panel(
            Group(overview, table),
            title=Text("Targeted residual-kernel linear search", style="bold blue"),
            border_style="green" if self.has_targeted_solution else "red",
        )


@dataclass(frozen=True, slots=True)
class TargetedResidualKernelJumpSelectionStep:
    """One greedy step selecting a targeted residual-kernel jump."""

    step_index: int
    candidate: TargetedResidualKernelLinearCandidate
    residual_kernel_dimension: int
    n_selected_jumps: int

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "candidate": self.candidate.to_summary_dict(),
            "residual_kernel_dimension": self.residual_kernel_dimension,
            "n_selected_jumps": self.n_selected_jumps,
        }


@dataclass(frozen=True, slots=True)
class TargetedResidualKernelJumpSelectionReport:
    """Greedy subset of targeted local jumps that removes a residual kernel."""

    manifold_dimension: int
    hilbert_dimension: int
    residual_dimension: int
    max_selected_jumps: int
    target_residual_kernel_dimension: int
    targeted_report: TargetedResidualKernelLinearSearchReport
    base_jumps: tuple[sp.csr_array, ...]
    jumps: tuple[sp.csr_array, ...]
    steps: tuple[TargetedResidualKernelJumpSelectionStep, ...]
    final_diagnostics: Any | None
    selection_target: Literal["reported_residual_kernel", "combined_common_kernel"]
    initial_selection_kernel_dimension: int
    target_selection_kernel_dimension: int
    kernel_tolerance: float
    dark_tolerance: float
    inflow_tolerance: float

    @property
    def n_selected_jumps(self) -> int:
        return len(self.jumps)

    @property
    def n_base_jumps(self) -> int:
        return len(self.base_jumps)

    @property
    def n_combined_jumps(self) -> int:
        return self.n_base_jumps + self.n_selected_jumps

    @property
    def all_jumps(self) -> tuple[sp.csr_array, ...]:
        return self.base_jumps + self.jumps

    @property
    def selected_candidates(self) -> tuple[TargetedResidualKernelLinearCandidate, ...]:
        return tuple(step.candidate for step in self.steps)

    @property
    def selected_region_indices(self) -> tuple[int, ...]:
        return tuple(step.candidate.region_index for step in self.steps)

    def selected_operator_readouts(
        self,
        *,
        basis_configs: npt.NDArray[np.integer],
        max_readouts: int | None = None,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local-matrix readouts for selected targeted completion operators."""
        candidates = self.selected_candidates
        if max_readouts is not None:
            candidates = candidates[: max(int(max_readouts), 0)]
        return tuple(
            _local_operator_from_targeted_candidate(
                candidate=candidate,
                basis_configs=basis_configs,
            )
            for candidate in candidates
        )

    @property
    def final_residual_kernel_dimension(self) -> int:
        current_basis = np.asarray(
            self.targeted_report.residual_basis,
            dtype=np.complex128,
        )
        for jump in self.jumps:
            image = np.asarray(jump @ current_basis, dtype=np.complex128)
            kernel_basis = _right_kernel_basis(image, tolerance=self.kernel_tolerance)
            current_basis = current_basis @ kernel_basis
        return int(current_basis.shape[1])

    @property
    def residual_kernel_removed(self) -> bool:
        return self.final_residual_kernel_dimension <= self.target_residual_kernel_dimension

    @property
    def final_selection_kernel_dimension(self) -> int:
        if len(self.steps) == 0:
            return self.initial_selection_kernel_dimension
        return int(self.steps[-1].residual_kernel_dimension)

    @property
    def selection_kernel_removed(self) -> bool:
        return self.final_selection_kernel_dimension <= self.target_selection_kernel_dimension

    @property
    def total_selected_jump_nnz(self) -> int:
        return int(sum(jump.nnz for jump in self.jumps))

    @property
    def total_combined_jump_nnz(self) -> int:
        return int(sum(jump.nnz for jump in self.all_jumps))

    @property
    def max_selected_jump_nnz(self) -> int:
        return int(max((jump.nnz for jump in self.jumps), default=0))

    @property
    def combined_bad_common_jump_kernel_dimension(self) -> int | None:
        if self.final_diagnostics is None:
            return None
        return int(self.final_diagnostics.bad_common_jump_kernel_dimension)

    @property
    def combined_inflow_norm(self) -> float | None:
        if self.final_diagnostics is None:
            return None
        return float(self.final_diagnostics.inflow_norm)

    @property
    def combined_complement_common_kernel_removed(self) -> bool | None:
        value = self.combined_bad_common_jump_kernel_dimension
        if value is None:
            return None
        return value == 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "residual_dimension": self.residual_dimension,
            "target_residual_kernel_dimension": self.target_residual_kernel_dimension,
            "selection_target": self.selection_target,
            "initial_selection_kernel_dimension": self.initial_selection_kernel_dimension,
            "target_selection_kernel_dimension": self.target_selection_kernel_dimension,
            "max_selected_jumps": self.max_selected_jumps,
            "n_base_jumps": self.n_base_jumps,
            "n_selected_jumps": self.n_selected_jumps,
            "n_combined_jumps": self.n_combined_jumps,
            "selected_region_indices": self.selected_region_indices,
            "total_selected_jump_nnz": self.total_selected_jump_nnz,
            "total_combined_jump_nnz": self.total_combined_jump_nnz,
            "max_selected_jump_nnz": self.max_selected_jump_nnz,
            "final_residual_kernel_dimension": self.final_residual_kernel_dimension,
            "residual_kernel_removed": self.residual_kernel_removed,
            "final_selection_kernel_dimension": self.final_selection_kernel_dimension,
            "selection_kernel_removed": self.selection_kernel_removed,
            "combined_bad_common_jump_kernel_dimension": (
                self.combined_bad_common_jump_kernel_dimension
            ),
            "combined_inflow_norm": self.combined_inflow_norm,
            "combined_complement_common_kernel_removed": (
                self.combined_complement_common_kernel_removed
            ),
            "kernel_tolerance": self.kernel_tolerance,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "steps": tuple(step.to_summary_dict() for step in self.steps),
            "final_diagnostics": (
                None if self.final_diagnostics is None else self.final_diagnostics.to_summary_dict()
            ),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_steps: int = 24):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "TargetedResidualKernelJumpSelectionReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("residual dimension", str(self.residual_dimension))
        overview.add_row("selection target", self.selection_target)
        overview.add_row("initial selection kernel", str(self.initial_selection_kernel_dimension))
        overview.add_row("base jumps", str(self.n_base_jumps))
        overview.add_row("selected targeted jumps", str(self.n_selected_jumps))
        overview.add_row("combined jumps", str(self.n_combined_jumps))
        overview.add_row("final residual kernel", str(self.final_residual_kernel_dimension))
        overview.add_row("selection kernel removed", str(self.selection_kernel_removed))
        overview.add_row(
            "combined bad kernel",
            (
                "not checked"
                if self.combined_bad_common_jump_kernel_dimension is None
                else str(self.combined_bad_common_jump_kernel_dimension)
            ),
        )
        overview.add_row(
            "combined inflow",
            (
                "not checked"
                if self.combined_inflow_norm is None
                else f"{self.combined_inflow_norm:.3e}"
            ),
        )

        table = Table(title="Greedy selected targeted residual-kernel jumps")
        table.add_column("step", justify="right")
        table.add_column("region")
        table.add_column("selection kernel", justify="right")
        table.add_column("resid inflow", justify="right")
        table.add_column("total inflow", justify="right")
        table.add_column("jump nnz", justify="right")
        table.add_column("terms", justify="right")
        for step in self.steps[: max(int(max_steps), 0)]:
            candidate = step.candidate
            table.add_row(
                str(step.step_index),
                str(candidate.variable_indices),
                str(step.residual_kernel_dimension),
                f"{candidate.residual_target_inflow_norm:.3e}",
                f"{candidate.total_inflow_norm:.3e}",
                str(candidate.jump_nnz),
                str(candidate.n_terms),
                style=(
                    "green"
                    if step.residual_kernel_dimension <= self.target_residual_kernel_dimension
                    else ""
                ),
            )
        if len(self.steps) > max_steps:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                f"{len(self.steps) - max_steps} more steps",
            )
        return Panel(
            Group(overview, table),
            title=Text("Targeted residual-kernel jump-selection report", style="bold blue"),
            border_style="green" if self.selection_kernel_removed else "red",
        )


def _bad_kernel_basis_from_recycled_family(
    *,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    candidates: tuple[RecycledManifoldDarkDetectorCandidate, ...],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    kernel_tolerance: float = 1.0e-10,
    max_detectors: int | None = None,
) -> npt.NDArray[np.complex128]:
    """Return the bad complement common kernel for a recycled family."""
    from qlinks.open_system.local_recycling import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    coefficients = _detector_coefficients_from_report(
        detector_coefficients=detector_coefficients,
        dark_operator_report=dark_operator_report,
        n_operators=len(detector_matrices),
        max_detectors=max_detectors,
    )
    detectors = tuple(
        _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        for detector_index in range(coefficients.shape[1])
    )

    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")

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
    recycler_specs_by_region = tuple(
        _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        for rdm in rdms
    )
    recycler_cache: dict[tuple[int, int], sp.csr_array] = {}

    complement_basis = _orthogonal_complement_basis(state_basis, tolerance=kernel_tolerance)
    complement_dimension = int(complement_basis.shape[1])
    if complement_dimension == 0:
        return np.zeros((dim, 0), dtype=np.complex128)

    family_gram = np.zeros(
        (complement_dimension, complement_dimension),
        dtype=np.complex128,
    )

    for candidate in candidates:
        if candidate.detector_index < 0 or candidate.detector_index >= len(detectors):
            raise ValueError("candidate.detector_index is out of range for detector coefficients.")
        if candidate.region_index < 0 or candidate.region_index >= len(regions):
            raise ValueError("candidate.region_index is out of range for local_regions.")
        recycler_specs = recycler_specs_by_region[candidate.region_index]
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")

        cache_key = (int(candidate.region_index), int(candidate.recycler_index))
        recycler = recycler_cache.get(cache_key)
        if recycler is None:
            _, local_operator = recycler_specs[candidate.recycler_index]
            recycler = _embed_local_pattern_operator_from_context(
                context=embedding_contexts[candidate.region_index],
                local_operator=local_operator,
            ).tocsr()
            recycler_cache[cache_key] = recycler

        jump = (recycler @ detectors[candidate.detector_index]).tocsr()
        if jump.nnz == 0:
            continue
        image = np.asarray(jump @ complement_basis, dtype=np.complex128)
        family_gram += image.conj().T @ image

    family_gram = 0.5 * (family_gram + family_gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(family_gram)
    largest = float(np.max(np.maximum(eigenvalues.real, 0.0))) if eigenvalues.size else 0.0
    cutoff = max(float(kernel_tolerance), float(kernel_tolerance) * max(largest, 1.0))
    kernel_mask = np.asarray(eigenvalues.real <= cutoff, dtype=bool)
    return (complement_basis @ eigenvectors[:, kernel_mask]).astype(np.complex128, copy=False)


def _operator_action_report_on_residual_kernel(
    *,
    group_name: str,
    operators: tuple[Any, ...] | list[Any],
    operator_names: tuple[str, ...] | list[str] | None,
    target_basis: npt.NDArray[np.complex128],
    residual_basis: npt.NDArray[np.complex128],
    max_entries: int | None,
) -> ResidualKernelOperatorActionReport:
    matrices = tuple(_as_csr(operator) for operator in operators)
    if operator_names is None:
        names = tuple(f"O_{index}" for index in range(len(matrices)))
    else:
        names = tuple(str(name) for name in operator_names)
        if len(names) != len(matrices):
            raise ValueError("operator_names length must match operators.")

    dim = int(target_basis.shape[0])
    if residual_basis.shape[0] != dim:
        raise ValueError("residual_basis has incompatible dimension.")
    for matrix in matrices:
        if matrix.shape != (dim, dim):
            raise ValueError("operator has incompatible shape for residual-kernel report.")

    entries: list[ResidualKernelOperatorActionEntry] = []
    for operator_index, matrix in enumerate(matrices):
        action = np.asarray(matrix @ residual_basis, dtype=np.complex128)
        target_component = target_basis.conj().T @ action
        residual_component = residual_basis.conj().T @ action
        projected = target_basis @ target_component + residual_basis @ residual_component
        outside = action - projected
        entries.append(
            ResidualKernelOperatorActionEntry(
                operator_index=operator_index,
                operator_name=names[operator_index],
                action_norm=float(np.linalg.norm(action)),
                target_component_norm=float(np.linalg.norm(target_component)),
                residual_component_norm=float(np.linalg.norm(residual_component)),
                outside_component_norm=float(np.linalg.norm(outside)),
            )
        )

    sorted_entries = tuple(
        sorted(
            entries,
            key=lambda entry: (
                -entry.action_norm,
                -entry.target_component_norm,
                entry.operator_index,
            ),
        )
    )
    if max_entries is not None:
        sorted_entries = sorted_entries[: max(int(max_entries), 0)]
    return ResidualKernelOperatorActionReport(
        group_name=str(group_name),
        n_operators=len(matrices),
        entries=sorted_entries,
    )


def _residual_local_support_entries(
    *,
    target_basis: npt.NDArray[np.complex128],
    residual_basis: npt.NDArray[np.complex128],
    basis_configs: npt.NDArray[np.integer],
    local_regions: tuple[tuple[int, ...], ...],
    tolerance: float,
) -> tuple[ResidualKernelLocalSupportEntry, ...]:
    from qlinks.open_system.local_recycling import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    if residual_basis.shape[1] == 0:
        return ()

    basis_array = np.asarray(basis_configs)
    entries: list[ResidualKernelLocalSupportEntry] = []
    for region in local_regions:
        context = _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        target_rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=target_basis,
            tolerance=tolerance,
        )
        residual_rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=residual_basis,
            tolerance=tolerance,
        )
        combined_rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=np.column_stack([target_basis, residual_basis]),
            tolerance=tolerance,
        )

        target_support = target_rdm.support_basis
        residual_support = residual_rdm.support_basis
        if residual_support.shape[1] == 0:
            outside_norm = 0.0
        elif target_support.shape[1] == 0:
            outside_norm = float(np.linalg.norm(residual_support))
        else:
            projected = target_support @ (target_support.conj().T @ residual_support)
            outside_norm = float(np.linalg.norm(residual_support - projected))

        entries.append(
            ResidualKernelLocalSupportEntry(
                variable_indices=tuple(int(index) for index in region),
                local_dim=context.local_dim,
                target_support_rank=target_rdm.support_rank,
                target_nullity=target_rdm.nullity,
                residual_support_rank=residual_rdm.support_rank,
                residual_nullity=residual_rdm.nullity,
                combined_support_rank=combined_rdm.support_rank,
                combined_nullity=combined_rdm.nullity,
                residual_support_outside_target_norm=outside_norm,
            )
        )
    return tuple(entries)


def diagnose_recycled_manifold_residual_kernel(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
    family_report: RecycledManifoldCandidateFamilyKernelReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    operator_groups: (
        tuple[
            tuple[str, tuple[Any, ...] | list[Any], tuple[str, ...] | list[str] | None],
            ...,
        ]
        | None
    ) = None,
    local_support_regions: (
        tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]] | None
    ) = None,
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    kernel_tolerance: float = 1.0e-10,
    liouvillian_zero_tolerance: float = 1.0e-9,
    max_detectors: int | None = None,
    expand_candidate_report: bool = True,
    max_operator_entries: int | None = 64,
) -> RecycledManifoldResidualKernelReport:
    """Diagnose the residual bad kernel left by a recycled-detector family.

    This function first computes the full-family common jump kernel for the
    specified recycled-detector candidates.  It then extracts the bad complement
    subspace and reports whether the Hamiltonian and optional probe-operator
    groups couple that subspace to the target manifold, leave it invariant, or
    push it outside the target-plus-residual sector.
    """
    regions = _normalize_local_regions(local_regions)
    target_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(target_basis.shape[0])
    manifold_dimension = int(target_basis.shape[1])

    if family_report is None:
        family_report = diagnose_recycled_manifold_candidate_family_kernel(
            hamiltonian=hamiltonian,
            states=target_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            candidate_report=candidate_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            max_detectors=max_detectors,
            expand_candidate_report=expand_candidate_report,
            kernel_method="streamed",
            store_candidate_jumps=False,
        )

    if family_report.candidate_report_is_truncated and expand_candidate_report:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=target_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=None,
            sort_by_inflow=True,
        )
    else:
        candidate_report = family_report.candidate_report

    eligible_candidates = tuple(
        candidate
        for candidate in candidate_report.candidates
        if candidate.relative_dark_residual <= dark_tolerance
        and candidate.inflow_norm > inflow_tolerance
    )
    residual_basis = _bad_kernel_basis_from_recycled_family(
        states=target_basis,
        basis_configs=basis_configs,
        detector_operators=detector_operators,
        local_regions=regions,
        candidates=eligible_candidates,
        detector_coefficients=detector_coefficients,
        dark_operator_report=dark_operator_report,
        recycler_source=recycler_source,
        tolerance=tolerance,
        rdm_tolerance=rdm_tolerance,
        kernel_tolerance=kernel_tolerance,
        max_detectors=max_detectors,
    )

    hamiltonian_matrix = _as_csr(hamiltonian)
    if hamiltonian_matrix.shape != (dim, dim):
        raise ValueError("hamiltonian must have shape (hilbert_dimension, hilbert_dimension).")

    if residual_basis.shape[1] == 0:
        hamiltonian_target_coupling_norm = 0.0
        hamiltonian_residual_block_norm = 0.0
        hamiltonian_outside_residual_norm = 0.0
        hamiltonian_residual_eigenvalues: tuple[complex, ...] = ()
    else:
        h_residual = np.asarray(hamiltonian_matrix @ residual_basis, dtype=np.complex128)
        h_target_block = target_basis.conj().T @ h_residual
        h_residual_block = residual_basis.conj().T @ h_residual
        projected = target_basis @ h_target_block + residual_basis @ h_residual_block
        h_outside = h_residual - projected
        hamiltonian_target_coupling_norm = float(np.linalg.norm(h_target_block))
        hamiltonian_residual_block_norm = float(np.linalg.norm(h_residual_block))
        hamiltonian_outside_residual_norm = float(np.linalg.norm(h_outside))
        h_residual_block = 0.5 * (h_residual_block + h_residual_block.conj().T)
        hamiltonian_residual_eigenvalues = tuple(
            complex(value) for value in np.linalg.eigvalsh(h_residual_block)
        )

    action_reports: list[ResidualKernelOperatorActionReport] = []
    if operator_groups is not None:
        for group_name, operators, operator_names in operator_groups:
            action_reports.append(
                _operator_action_report_on_residual_kernel(
                    group_name=group_name,
                    operators=operators,
                    operator_names=operator_names,
                    target_basis=target_basis,
                    residual_basis=residual_basis,
                    max_entries=max_operator_entries,
                )
            )

    support_regions = (
        regions
        if local_support_regions is None
        else _normalize_local_regions(local_support_regions)
    )
    local_entries = _residual_local_support_entries(
        target_basis=target_basis,
        residual_basis=residual_basis,
        basis_configs=basis_configs,
        local_regions=support_regions,
        tolerance=rdm_tolerance,
    )

    return RecycledManifoldResidualKernelReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        family_report=family_report,
        residual_basis=residual_basis,
        hamiltonian_target_coupling_norm=hamiltonian_target_coupling_norm,
        hamiltonian_residual_block_norm=hamiltonian_residual_block_norm,
        hamiltonian_outside_residual_norm=hamiltonian_outside_residual_norm,
        hamiltonian_residual_eigenvalues=hamiltonian_residual_eigenvalues,
        operator_action_reports=tuple(action_reports),
        local_support_entries=local_entries,
        kernel_tolerance=float(kernel_tolerance),
    )


def _right_kernel_basis_from_gram_matrix(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return a right-kernel basis via the smaller column Gram matrix.

    This is useful when ``matrix`` is tall and has at most a few hundred
    columns.  It avoids the much more expensive full SVD used by the generic
    helper while preserving the same relative singular-value cutoff semantics.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    n_columns = int(matrix.shape[1])
    if n_columns == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    if matrix.shape[0] == 0:
        return np.eye(n_columns, dtype=np.complex128)

    gram = matrix.conj().T @ matrix
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues.real, 0.0)
    largest_singular = float(np.sqrt(np.max(eigenvalues))) if eigenvalues.size else 0.0
    cutoff = max(float(tolerance), float(tolerance) * max(largest_singular, 1.0))
    kernel_mask = eigenvalues <= cutoff * cutoff
    return eigenvectors[:, kernel_mask].astype(np.complex128, copy=False)


def diagnose_targeted_residual_kernel_linear_search(
    *,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    residual_basis: npt.ArrayLike | None = None,
    residual_report: RecycledManifoldResidualKernelReport | None = None,
    detector_operators: tuple[Any, ...] | list[Any] | None = None,
    residual_family_local_regions: (
        tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]] | None
    ) = None,
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
    family_report: RecycledManifoldCandidateFamilyKernelReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    operator_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "matrix_units",
    residual_objective: Literal["target_inflow", "action_norm"] = "target_inflow",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    kernel_tolerance: float = 1.0e-10,
    max_detectors: int | None = None,
    max_modes_per_region: int = 1,
    max_report_candidates: int | None = 32,
    max_local_dim: int | None = None,
    coefficient_tolerance: float = 1.0e-8,
) -> TargetedResidualKernelLinearSearchReport:
    """Search local dark jumps that directly target a residual bad kernel.

    This is stronger than optimizing linear combinations of the factorized
    recycled-detector family ``R D``.  For each supplied local region, it builds
    a local operator basis ``O_a`` and solves the constrained problem

        sum_a c_a O_a P_M = 0,

    then ranks dark combinations by their action

        ||P_M (sum_a c_a O_a) B||_F,

    where ``B`` is the residual complement common kernel left by a recycled
    detector family.
    """
    from qlinks.open_system.local_recycling import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    target_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(target_basis.shape[0])
    manifold_dimension = int(target_basis.shape[1])
    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")
    if residual_objective not in {"target_inflow", "action_norm"}:
        raise ValueError('residual_objective must be "target_inflow" or "action_norm".')

    report_family = family_report
    if residual_report is not None:
        bad_basis = np.asarray(residual_report.residual_basis, dtype=np.complex128)
        report_family = residual_report.family_report
    elif residual_basis is not None:
        bad_basis, _ = _normalize_state_columns(residual_basis, tolerance=tolerance)
    else:
        if detector_operators is None:
            raise ValueError(
                "Pass residual_basis, residual_report, or detector_operators to infer "
                "the recycled-family residual kernel."
            )
        family_regions = _normalize_local_regions(
            local_regions
            if residual_family_local_regions is None
            else residual_family_local_regions
        )
        if family_report is not None:
            candidate_report = family_report.candidate_report
        if candidate_report is None:
            candidate_report = diagnose_recycled_manifold_dark_detectors(
                states=target_basis,
                basis_configs=basis_array,
                detector_operators=detector_operators,
                local_regions=family_regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                detector_operator_names=detector_operator_names,
                detector_names=detector_names,
                recycler_source=recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                dark_tolerance=dark_tolerance,
                inflow_tolerance=inflow_tolerance,
                max_detectors=max_detectors,
                max_report_candidates=None,
                sort_by_inflow=True,
            )
        eligible_candidates = tuple(
            candidate
            for candidate in candidate_report.candidates
            if candidate.relative_dark_residual <= dark_tolerance
            and candidate.inflow_norm > inflow_tolerance
        )
        bad_basis = _bad_kernel_basis_from_recycled_family(
            states=target_basis,
            basis_configs=basis_array,
            detector_operators=detector_operators,
            local_regions=family_regions,
            candidates=eligible_candidates,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            kernel_tolerance=kernel_tolerance,
            max_detectors=max_detectors,
        )

    if bad_basis.ndim == 1:
        bad_basis = bad_basis.reshape(dim, 1)
    if bad_basis.ndim != 2 or bad_basis.shape[0] != dim:
        raise ValueError("residual_basis has incompatible shape.")
    if bad_basis.shape[1] > 0:
        # Re-orthonormalize and explicitly remove any numerical target component.
        bad_basis = bad_basis - target_basis @ (target_basis.conj().T @ bad_basis)
        bad_basis, _ = _normalize_state_columns(bad_basis, tolerance=tolerance)
    else:
        bad_basis = np.zeros((dim, 0), dtype=np.complex128)

    regions = _normalize_local_regions(local_regions)
    candidates: list[TargetedResidualKernelLinearCandidate] = []
    candidate_jumps: list[sp.csr_array] = []
    n_regions_evaluated = 0
    n_regions_skipped_by_local_dim = 0
    n_regions_with_no_recycler_specs = 0
    n_regions_with_no_nonzero_local_operators = 0
    n_regions_with_zero_dark_nullity = 0
    n_regions_with_dark_nullity_detected = 0
    n_regions_with_zero_residual_inflow = 0
    max_encountered_local_dim = 0

    for region_index, region in enumerate(regions):
        n_regions_evaluated += 1
        context = _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=target_basis,
            tolerance=rdm_tolerance,
        )
        max_encountered_local_dim = max(max_encountered_local_dim, int(rdm.local_dim))
        if max_local_dim is not None and rdm.local_dim > max_local_dim:
            n_regions_skipped_by_local_dim += 1
            continue

        specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=operator_source,
        )
        if len(specs) == 0:
            n_regions_with_no_recycler_specs += 1
            continue

        embedding_context = _embedding_context_from_basis_context(context)
        local_ops = tuple(
            _embed_local_pattern_operator_from_context(
                context=embedding_context,
                local_operator=local_operator,
            ).tocsr()
            for _name, local_operator in specs
        )
        nonzero_indices = [index for index, operator in enumerate(local_ops) if operator.nnz > 0]
        if len(nonzero_indices) == 0:
            n_regions_with_no_nonzero_local_operators += 1
            continue
        local_ops = tuple(local_ops[index] for index in nonzero_indices)
        spec_names = tuple(specs[index][0] for index in nonzero_indices)

        dark_matrix = np.column_stack(
            [
                np.asarray(operator @ target_basis, dtype=np.complex128).reshape(-1)
                for operator in local_ops
            ]
        )
        dark_kernel = _right_kernel_basis_from_gram_matrix(dark_matrix, tolerance=dark_tolerance)
        dark_nullity = int(dark_kernel.shape[1])
        dark_rank = int(dark_kernel.shape[0] - dark_nullity)
        if dark_nullity == 0:
            n_regions_with_zero_dark_nullity += 1
            continue
        n_regions_with_dark_nullity_detected += 1
        region_has_residual_inflow = False

        if bad_basis.shape[1] == 0:
            score_matrix = np.zeros((0, len(local_ops)), dtype=np.complex128)
        elif residual_objective == "target_inflow":
            score_matrix = np.column_stack(
                [
                    (
                        target_basis.conj().T
                        @ np.asarray(operator @ bad_basis, dtype=np.complex128)
                    ).reshape(-1)
                    for operator in local_ops
                ]
            )
        else:
            score_matrix = np.column_stack(
                [
                    np.asarray(operator @ bad_basis, dtype=np.complex128).reshape(-1)
                    for operator in local_ops
                ]
            )
        restricted_inflow = score_matrix @ dark_kernel
        if restricted_inflow.size == 0:
            singular_values = np.zeros((0,), dtype=np.float64)
            right_vectors = np.zeros((dark_nullity, 0), dtype=np.complex128)
        else:
            _u, singular_values, vh = np.linalg.svd(restricted_inflow, full_matrices=False)
            right_vectors = vh.conj().T

        n_modes = min(max(int(max_modes_per_region), 0), int(right_vectors.shape[1]))
        for mode_index in range(n_modes):
            coefficients = np.asarray(
                dark_kernel @ right_vectors[:, mode_index], dtype=np.complex128
            )
            coefficient_norm = float(np.linalg.norm(coefficients))
            if coefficient_norm == 0.0:
                continue
            coefficients = coefficients / coefficient_norm
            jump = sp.csr_array((dim, dim), dtype=np.complex128)
            for coefficient, operator in zip(coefficients, local_ops, strict=True):
                if abs(coefficient) <= 0.0:
                    continue
                jump = jump + coefficient * operator
            jump = jump.tocsr()
            if jump.nnz == 0:
                continue

            dark_residual = float(np.linalg.norm(jump @ target_basis))
            jump_norm = float(sp.linalg.norm(jump))
            relative_dark_residual = dark_residual / max(jump_norm, 1.0)
            bad_action = np.asarray(jump @ bad_basis, dtype=np.complex128)
            residual_action_norm = float(np.linalg.norm(bad_action))
            residual_target_action = target_basis.conj().T @ bad_action
            residual_target_inflow_norm = float(np.linalg.norm(residual_target_action))
            residual_score_norm = (
                residual_target_inflow_norm
                if residual_objective == "target_inflow"
                else residual_action_norm
            )
            if residual_score_norm > inflow_tolerance:
                region_has_residual_inflow = True
            total_inflow_norm, target_block_norm = _projected_inflow_norm(
                jump=jump,
                state_basis=target_basis,
            )
            terms = tuple(
                TargetedResidualKernelLinearTerm(
                    operator_index=int(index),
                    operator_name=spec_names[index],
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
                TargetedResidualKernelLinearCandidate(
                    candidate_index=len(candidates),
                    region_index=int(region_index),
                    variable_indices=tuple(int(value) for value in region),
                    local_dim=int(rdm.local_dim),
                    operator_source=operator_source,
                    dark_constraint_rank=dark_rank,
                    dark_nullity=dark_nullity,
                    singular_value=float(singular_values[mode_index]),
                    residual_target_inflow_norm=residual_target_inflow_norm,
                    residual_action_norm=residual_action_norm,
                    residual_score_norm=residual_score_norm,
                    residual_objective=residual_objective,
                    dark_residual=dark_residual,
                    relative_dark_residual=float(relative_dark_residual),
                    total_inflow_norm=total_inflow_norm,
                    target_block_norm=target_block_norm,
                    jump_frobenius_norm=jump_norm,
                    jump_nnz=int(jump.nnz),
                    coefficients=coefficients,
                    terms=terms,
                )
            )
            candidate_jumps.append(jump)

        if not region_has_residual_inflow:
            n_regions_with_zero_residual_inflow += 1

    n_candidate_modes_generated = len(candidates)

    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index].relative_dark_residual > dark_tolerance,
            -max(
                candidates[index].residual_score_norm,
                candidates[index].residual_target_inflow_norm,
            ),
            -candidates[index].residual_target_inflow_norm,
            -candidates[index].total_inflow_norm,
            candidates[index].jump_nnz,
            candidates[index].region_index,
        ),
    )
    if max_report_candidates is not None:
        order = order[: max(int(max_report_candidates), 0)]

    sorted_candidates: list[TargetedResidualKernelLinearCandidate] = []
    sorted_jumps: list[sp.csr_array] = []
    for new_index, old_index in enumerate(order):
        candidate = candidates[old_index]
        sorted_candidates.append(
            TargetedResidualKernelLinearCandidate(
                candidate_index=new_index,
                region_index=candidate.region_index,
                variable_indices=candidate.variable_indices,
                local_dim=candidate.local_dim,
                operator_source=candidate.operator_source,
                dark_constraint_rank=candidate.dark_constraint_rank,
                dark_nullity=candidate.dark_nullity,
                singular_value=candidate.singular_value,
                residual_target_inflow_norm=candidate.residual_target_inflow_norm,
                residual_action_norm=candidate.residual_action_norm,
                residual_score_norm=candidate.residual_score_norm,
                residual_objective=candidate.residual_objective,
                dark_residual=candidate.dark_residual,
                relative_dark_residual=candidate.relative_dark_residual,
                total_inflow_norm=candidate.total_inflow_norm,
                target_block_norm=candidate.target_block_norm,
                jump_frobenius_norm=candidate.jump_frobenius_norm,
                jump_nnz=candidate.jump_nnz,
                coefficients=candidate.coefficients,
                terms=candidate.terms,
            )
        )
        sorted_jumps.append(candidate_jumps[old_index])

    return TargetedResidualKernelLinearSearchReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        residual_basis=bad_basis,
        region_variable_indices=regions,
        operator_source=operator_source,
        residual_objective=residual_objective,
        family_report=report_family,
        candidates=tuple(sorted_candidates),
        candidate_jumps=tuple(sorted_jumps),
        tolerance=float(tolerance),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        n_regions_evaluated=int(n_regions_evaluated),
        n_regions_skipped_by_local_dim=int(n_regions_skipped_by_local_dim),
        n_regions_with_no_recycler_specs=int(n_regions_with_no_recycler_specs),
        n_regions_with_no_nonzero_local_operators=int(n_regions_with_no_nonzero_local_operators),
        n_regions_with_zero_dark_nullity=int(n_regions_with_zero_dark_nullity),
        n_regions_with_dark_nullity_detected=int(n_regions_with_dark_nullity_detected),
        n_regions_with_zero_residual_inflow=int(n_regions_with_zero_residual_inflow),
        n_candidate_modes_generated=int(n_candidate_modes_generated),
        max_encountered_local_dim=int(max_encountered_local_dim),
    )


def _orthogonal_complement_basis(
    basis: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return an orthonormal basis of the complement of ``span(basis)``."""
    q, _ = _normalize_state_columns(basis, tolerance=tolerance)
    _u, singular_values, vh = np.linalg.svd(q.conj().T, full_matrices=True)
    singular_scale = float(singular_values[0]) if singular_values.size else 1.0
    cutoff = max(float(tolerance), float(tolerance) * singular_scale)
    rank = int(np.count_nonzero(singular_values > cutoff))
    return vh.conj().T[:, rank:].astype(np.complex128, copy=False)


def _bad_common_kernel_basis_for_jumps(
    *,
    jumps: tuple[sp.csr_array, ...],
    target_basis: npt.NDArray[np.complex128],
    dim: int,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return the complement common kernel for a concrete jump list."""
    from qlinks.open_system.diagnostics import (
        _common_kernel_basis_from_sparse_operators,
        _kernel_basis_orthogonal_to_manifold,
    )

    if len(jumps) == 0:
        common_kernel_basis = np.eye(int(dim), dtype=np.complex128)
    else:
        common_kernel_basis = _common_kernel_basis_from_sparse_operators(
            operators=jumps,
            dim=int(dim),
            tolerance=float(tolerance),
        )
    return _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=target_basis,
        tolerance=float(tolerance),
    )


def _right_kernel_basis(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return an orthonormal basis for the right kernel of ``matrix``."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    n_columns = int(matrix.shape[1])
    if n_columns == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    full_matrices = matrix.shape[0] < matrix.shape[1]
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    if singular_values.size == 0:
        rank = 0
    else:
        cutoff = max(float(tolerance), float(np.sqrt(tolerance)) * float(singular_values[0]))
        rank = int(np.count_nonzero(singular_values > cutoff))
    if rank >= n_columns:
        return np.zeros((n_columns, 0), dtype=np.complex128)
    return vh.conj().T[:, rank:].astype(np.complex128, copy=False)


def select_targeted_residual_kernel_jumps(
    *,
    targeted_report: TargetedResidualKernelLinearSearchReport,
    hamiltonian: Any | None = None,
    states: npt.ArrayLike | None = None,
    base_jumps: tuple[Any, ...] | list[Any] = (),
    max_selected_jumps: int = 16,
    target_residual_kernel_dimension: int = 0,
    selection_target: Literal[
        "reported_residual_kernel",
        "combined_common_kernel",
    ] = "reported_residual_kernel",
    allow_non_improving: bool = False,
    kernel_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    liouvillian_zero_tolerance: float = 1.0e-9,
    check_manifold_diagnostics: bool = True,
    liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "none",
    sparse_liouvillian_eigenvalue_count: int = 32,
) -> TargetedResidualKernelJumpSelectionReport:
    """Greedily select targeted local jumps removing a residual bad kernel.

    The input report is produced by
    :func:`diagnose_targeted_residual_kernel_linear_search`.

    With ``selection_target="reported_residual_kernel"`` this preserves the
    original behavior: each step minimizes the remaining kernel inside the
    residual basis stored in ``targeted_report``.  With
    ``selection_target="combined_common_kernel"`` the selector instead starts
    from the complement common jump-kernel of the supplied ``base_jumps`` and
    greedily minimizes the combined bad kernel of ``base_jumps`` plus the
    targeted candidates.  The latter is the right mode after a compressed
    recycled-detector subset leaves a small complement kernel that is not
    identical to the full-family residual basis used to create the targeted
    report.
    """
    from qlinks.open_system.diagnostics import diagnose_dark_manifold

    residual_basis = np.asarray(targeted_report.residual_basis, dtype=np.complex128)
    if residual_basis.ndim != 2:
        raise ValueError("targeted_report.residual_basis must be two-dimensional.")
    residual_dimension = int(residual_basis.shape[1])
    dim = int(targeted_report.hilbert_dimension)
    if residual_basis.shape[0] != dim:
        raise ValueError("targeted_report has inconsistent residual-basis dimension.")

    if selection_target not in {"reported_residual_kernel", "combined_common_kernel"}:
        raise ValueError(
            'selection_target must be "reported_residual_kernel" or "combined_common_kernel".'
        )

    base_jump_tuple = tuple(_as_csr(jump) for jump in base_jumps)
    for jump in base_jump_tuple:
        if jump.shape != (dim, dim):
            raise ValueError("base_jumps must have shape (hilbert_dimension, hilbert_dimension).")

    if selection_target == "combined_common_kernel":
        if states is None:
            raise ValueError(
                "states must be supplied when selection_target='combined_common_kernel'."
            )
        target_basis, _ = _normalize_state_columns(states, tolerance=kernel_tolerance)
        if target_basis.shape[0] != dim:
            raise ValueError("states has incompatible Hilbert-space dimension.")
        current_basis = _bad_common_kernel_basis_for_jumps(
            jumps=base_jump_tuple,
            target_basis=target_basis,
            dim=dim,
            tolerance=kernel_tolerance,
        )
        eligible = [
            (candidate, _as_csr(jump))
            for candidate, jump in zip(
                targeted_report.candidates,
                targeted_report.candidate_jumps,
                strict=True,
            )
            if candidate.relative_dark_residual <= dark_tolerance
            and max(
                candidate.residual_score_norm,
                candidate.residual_target_inflow_norm,
                candidate.total_inflow_norm,
            )
            > inflow_tolerance
        ]
    else:
        current_basis = residual_basis
        eligible = [
            (candidate, _as_csr(jump))
            for candidate, jump in zip(
                targeted_report.candidates,
                targeted_report.candidate_jumps,
                strict=True,
            )
            if candidate.relative_dark_residual <= dark_tolerance
            and max(candidate.residual_score_norm, candidate.residual_target_inflow_norm)
            > inflow_tolerance
        ]

    selected_candidates: list[TargetedResidualKernelLinearCandidate] = []
    selected_jumps: list[sp.csr_array] = []
    selected_ids: set[int] = set()
    steps: list[TargetedResidualKernelJumpSelectionStep] = []
    initial_selection_kernel_dimension = int(current_basis.shape[1])
    current_dimension = initial_selection_kernel_dimension
    target_selection_kernel_dimension = int(target_residual_kernel_dimension)

    for _step_index in range(max(int(max_selected_jumps), 0)):
        if current_dimension <= target_selection_kernel_dimension:
            break
        best_entry = None
        for candidate, jump in eligible:
            candidate_id = id(candidate)
            if candidate_id in selected_ids:
                continue
            image = np.asarray(jump @ current_basis, dtype=np.complex128)
            kernel_basis = _right_kernel_basis(image, tolerance=kernel_tolerance)
            next_dimension = int(kernel_basis.shape[1])
            score = (
                next_dimension,
                -max(candidate.residual_score_norm, candidate.residual_target_inflow_norm),
                -candidate.residual_target_inflow_norm,
                -candidate.total_inflow_norm,
                candidate.jump_nnz,
                candidate.n_terms,
                candidate.region_index,
                candidate.candidate_index,
            )
            if best_entry is None or score < best_entry[0]:
                best_entry = (score, candidate, jump, kernel_basis)

        if best_entry is None:
            break

        _score, candidate, jump, kernel_basis = best_entry
        next_dimension = int(kernel_basis.shape[1])
        if not allow_non_improving and next_dimension >= current_dimension:
            break

        selected_candidates.append(candidate)
        selected_jumps.append(jump)
        selected_ids.add(id(candidate))
        current_basis = current_basis @ kernel_basis
        current_dimension = next_dimension
        steps.append(
            TargetedResidualKernelJumpSelectionStep(
                step_index=len(steps),
                candidate=candidate,
                residual_kernel_dimension=current_dimension,
                n_selected_jumps=len(selected_jumps),
            )
        )

    final_diagnostics = None
    if check_manifold_diagnostics and hamiltonian is not None and states is not None:
        final_diagnostics = diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=base_jump_tuple + tuple(selected_jumps),
            target_states=states,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=liouvillian_spectrum_method != "none",
            liouvillian_spectrum_method=liouvillian_spectrum_method,
            sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
        )

    return TargetedResidualKernelJumpSelectionReport(
        manifold_dimension=targeted_report.manifold_dimension,
        hilbert_dimension=targeted_report.hilbert_dimension,
        residual_dimension=residual_dimension,
        max_selected_jumps=max(int(max_selected_jumps), 0),
        target_residual_kernel_dimension=int(target_residual_kernel_dimension),
        targeted_report=targeted_report,
        selection_target=selection_target,
        initial_selection_kernel_dimension=initial_selection_kernel_dimension,
        target_selection_kernel_dimension=target_selection_kernel_dimension,
        base_jumps=base_jump_tuple,
        jumps=tuple(selected_jumps),
        steps=tuple(steps),
        final_diagnostics=final_diagnostics,
        kernel_tolerance=float(kernel_tolerance),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
    )


def select_recycled_manifold_dark_detector_jumps(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
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
    kernel_tolerance: float = 1.0e-10,
    liouvillian_zero_tolerance: float = 1.0e-9,
    max_detectors: int | None = None,
    max_candidate_pool: int | None = 128,
    max_selected_jumps: int = 16,
    target_bad_kernel_dimension: int = 0,
    allow_non_improving: bool = False,
    expand_candidate_report: bool = False,
    selection_strategy: Literal[
        "diagnostics", "kernel_projection", "ranked_inflow"
    ] = "diagnostics",
    compression_strategy: Literal["none", "h_invariant"] = "none",
    max_compression_passes: int = 1,
    collective_recycler_strategy: Literal["none", "bundle_by_region_detector"] = "none",
    collective_recycler_weighting: Literal["unit", "inflow", "normalized_inflow"] = "unit",
    normalize_collective_recyclers: bool = True,
    check_final_diagnostics: bool | None = None,
) -> RecycledManifoldJumpSelectionReport:
    """Greedily select a small recycled-detector jump subset.

    The candidate family is ``J=R D``: ``D`` is a collective detector dark on
    the target manifold, and ``R`` is a local matrix-unit/RDM-support recycler.
    The selector first keeps the best direct-inflow candidates, then adds jumps
    one at a time to minimize the complement common jump-kernel dimension

        dim( intersection_mu ker J_mu  ∩  P_M^perp ).

    Reaching ``target_bad_kernel_dimension=0`` is a strong, jump-only sufficient
    condition that no complement vector remains dark under all selected jumps.
    It is stronger than the true invariant-subspace condition including ``H``,
    but cheaper and useful before expensive Liouvillian spectrum checks.

    If ``candidate_report`` is a truncated diagnostic report, set
    ``expand_candidate_report=True`` and ``max_candidate_pool=None`` to rescan
    and use the full local recycler candidate family.  This is often the right
    follow-up when a small inflow-ranked pool leaves a low-dimensional bad
    complement kernel.

    ``selection_strategy="kernel_projection"`` is faster for large two-region
    recycler pools: it updates the current complement common kernel directly by
    applying each candidate to the current bad subspace, and runs the full
    diagnostics only once at the end.  ``"ranked_inflow"`` is the production
    preselection mode for large scans: it trusts the inflow-ranked candidate
    report and selects the top candidates directly.  By default this production
    mode skips the expensive final common-kernel diagnostic; pass
    ``check_final_diagnostics=True`` when you want the full certificate.
    """
    from qlinks.open_system.diagnostics import diagnose_dark_manifold

    regions = _normalize_local_regions(local_regions)
    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])
    if selection_strategy not in {"diagnostics", "kernel_projection", "ranked_inflow"}:
        raise ValueError(
            'selection_strategy must be "diagnostics", "kernel_projection", or ' '"ranked_inflow".'
        )
    if compression_strategy not in {"none", "h_invariant"}:
        raise ValueError('compression_strategy must be "none" or "h_invariant".')
    if collective_recycler_strategy not in {"none", "bundle_by_region_detector"}:
        raise ValueError(
            'collective_recycler_strategy must be "none" or "bundle_by_region_detector".'
        )
    if collective_recycler_weighting not in {"unit", "inflow", "normalized_inflow"}:
        raise ValueError(
            'collective_recycler_weighting must be "unit", "inflow", or "normalized_inflow".'
        )
    if check_final_diagnostics is None:
        check_final_diagnostics = selection_strategy != "ranked_inflow"

    candidate_report_was_expanded = False
    if candidate_report is None:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=max_candidate_pool,
            sort_by_inflow=True,
        )
    elif expand_candidate_report and candidate_report.candidate_report_is_truncated:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=max_candidate_pool,
            sort_by_inflow=True,
        )
        candidate_report_was_expanded = True

    eligible_pool = [
        candidate
        for candidate in candidate_report.candidates
        if candidate.relative_dark_residual <= dark_tolerance
        and candidate.inflow_norm > inflow_tolerance
    ]
    candidate_pool_was_limited = False
    if max_candidate_pool is None:
        pool = eligible_pool
    else:
        pool_limit = max(int(max_candidate_pool), 0)
        candidate_pool_was_limited = len(eligible_pool) > pool_limit
        pool = eligible_pool[:pool_limit]

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
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

    from qlinks.open_system.local_recycling import (
        _embedding_context_from_basis_context,
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    basis_array = np.asarray(basis_configs)
    used_region_indices = sorted({int(candidate.region_index) for candidate in pool})
    contexts = {
        region_index: _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=regions[region_index],
        )
        for region_index in used_region_indices
    }
    embedding_contexts = {
        region_index: _embedding_context_from_basis_context(context)
        for region_index, context in contexts.items()
    }
    rdms = (
        {}
        if recycler_source == "matrix_units"
        else {
            region_index: _local_reduced_density_matrix_from_basis_context_and_states(
                context=context,
                states=state_basis,
                tolerance=rdm_tolerance,
            )
            for region_index, context in contexts.items()
        }
    )

    used_detector_indices = {candidate.detector_index for candidate in pool}
    detector_diagonals: dict[int, npt.NDArray[np.complex128] | None] = {}
    for detector_index in used_detector_indices:
        detector = _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        detector_diagonals[detector_index] = _diagonal_vector_if_diagonal(
            detector,
            tolerance=tolerance,
        )

    candidate_jumps = {
        id(candidate): _recycled_jump_for_candidate_from_cache(
            candidate=candidate,
            dim=dim,
            detector_matrices=detector_matrices,
            detector_coefficients=coefficients,
            detector_diagonals=detector_diagonals,
            embedding_contexts=embedding_contexts,
            rdms=rdms,
            recycler_source=recycler_source,
            zero_tolerance=0.0,
        )
        for candidate in pool
    }

    selected_candidates: list[RecycledManifoldDarkDetectorCandidate] = []
    selected_jumps: list[sp.csr_array] = []
    selected_ids: set[int] = set()
    steps: list[RecycledManifoldJumpSelectionStep] = []
    current_bad_dimension = dim - manifold_dimension
    final_diagnostics = None

    if selection_strategy == "ranked_inflow":
        selected_candidates = list(pool[: max(int(max_selected_jumps), 0)])
        selected_jumps = [candidate_jumps[id(candidate)] for candidate in selected_candidates]
        selected_ids = {id(candidate) for candidate in selected_candidates}

        cumulative_inflow_squared = 0.0
        max_target_jump_residual = 0.0
        for selected_candidate in selected_candidates:
            cumulative_inflow_squared += float(selected_candidate.inflow_norm) ** 2
            max_target_jump_residual = max(
                max_target_jump_residual,
                float(selected_candidate.dark_residual),
            )
            steps.append(
                RecycledManifoldJumpSelectionStep(
                    step_index=len(steps),
                    candidate=selected_candidate,
                    bad_common_jump_kernel_dimension=current_bad_dimension,
                    inflow_norm=float(np.sqrt(cumulative_inflow_squared)),
                    max_target_jump_residual=max_target_jump_residual,
                    n_selected_jumps=len(steps) + 1,
                )
            )

    elif selection_strategy == "kernel_projection":
        current_bad_basis = _orthogonal_complement_basis(
            state_basis,
            tolerance=kernel_tolerance,
        )
        current_bad_dimension = int(current_bad_basis.shape[1])
        cumulative_inflow_squared = 0.0
        max_target_jump_residual = 0.0

        for _step_index in range(max(int(max_selected_jumps), 0)):
            best_entry = None
            for candidate in pool:
                candidate_id = id(candidate)
                if candidate_id in selected_ids:
                    continue
                jump = candidate_jumps[candidate_id]
                image = np.asarray(jump @ current_bad_basis, dtype=np.complex128)
                kernel_basis = _right_kernel_basis(image, tolerance=kernel_tolerance)
                next_bad_dimension = int(kernel_basis.shape[1])
                score = (
                    next_bad_dimension,
                    -candidate.inflow_norm,
                    candidate.jump_nnz,
                    candidate.detector_index,
                    candidate.region_index,
                    candidate.recycler_index,
                )
                if best_entry is None or score < best_entry[0]:
                    best_entry = (score, candidate, jump, kernel_basis)

            if best_entry is None:
                break

            _, best_candidate, best_jump, best_kernel_basis = best_entry
            next_bad_dimension = int(best_kernel_basis.shape[1])
            if not allow_non_improving and next_bad_dimension >= current_bad_dimension:
                break

            selected_candidates.append(best_candidate)
            selected_jumps.append(best_jump)
            selected_ids.add(id(best_candidate))
            current_bad_basis = current_bad_basis @ best_kernel_basis
            current_bad_dimension = next_bad_dimension
            cumulative_inflow_squared += float(best_candidate.inflow_norm) ** 2
            max_target_jump_residual = max(
                max_target_jump_residual,
                float(best_candidate.dark_residual),
            )
            steps.append(
                RecycledManifoldJumpSelectionStep(
                    step_index=len(steps),
                    candidate=best_candidate,
                    bad_common_jump_kernel_dimension=current_bad_dimension,
                    inflow_norm=float(np.sqrt(cumulative_inflow_squared)),
                    max_target_jump_residual=max_target_jump_residual,
                    n_selected_jumps=len(selected_jumps),
                )
            )

            if current_bad_dimension <= target_bad_kernel_dimension:
                break
    else:
        for _step_index in range(max(int(max_selected_jumps), 0)):
            best_entry = None
            for candidate in pool:
                candidate_id = id(candidate)
                if candidate_id in selected_ids:
                    continue
                trial_jumps = tuple(selected_jumps + [candidate_jumps[candidate_id]])
                diagnostics = diagnose_dark_manifold(
                    hamiltonian=hamiltonian,
                    jumps=trial_jumps,
                    target_states=state_basis,
                    kernel_tolerance=kernel_tolerance,
                    liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                    check_liouvillian_spectrum=False,
                    liouvillian_spectrum_method="none",
                )
                score = (
                    diagnostics.bad_common_jump_kernel_dimension,
                    diagnostics.max_target_jump_residual,
                    -diagnostics.inflow_norm,
                    -candidate.inflow_norm,
                    candidate.jump_nnz,
                    candidate.detector_index,
                    candidate.region_index,
                    candidate.recycler_index,
                )
                if best_entry is None or score < best_entry[0]:
                    best_entry = (score, candidate, candidate_jumps[candidate_id], diagnostics)

            if best_entry is None:
                break

            _, best_candidate, best_jump, best_diagnostics = best_entry
            if (
                not allow_non_improving
                and best_diagnostics.bad_common_jump_kernel_dimension >= current_bad_dimension
            ):
                break

            selected_candidates.append(best_candidate)
            selected_jumps.append(best_jump)
            selected_ids.add(id(best_candidate))
            current_bad_dimension = int(best_diagnostics.bad_common_jump_kernel_dimension)
            final_diagnostics = best_diagnostics
            steps.append(
                RecycledManifoldJumpSelectionStep(
                    step_index=len(steps),
                    candidate=best_candidate,
                    bad_common_jump_kernel_dimension=current_bad_dimension,
                    inflow_norm=float(best_diagnostics.inflow_norm),
                    max_target_jump_residual=float(best_diagnostics.max_target_jump_residual),
                    n_selected_jumps=len(selected_jumps),
                )
            )

            if current_bad_dimension <= target_bad_kernel_dimension:
                break

    n_compression_passes = 0
    n_compressed_jumps_removed = 0
    if compression_strategy == "h_invariant" and len(selected_jumps) > 0:
        from qlinks.open_system.diagnostics import diagnose_common_kernel_h_invariant_sector

        current_h_report = diagnose_common_kernel_h_invariant_sector(
            hamiltonian=hamiltonian,
            jumps=tuple(selected_jumps),
            target_states=state_basis,
            kernel_tolerance=kernel_tolerance,
        )
        if current_h_report.likely_attractive_by_h_invariant_kernel:
            max_passes = max(int(max_compression_passes), 0)
            for _pass_index in range(max_passes):
                removed_this_pass = False
                n_compression_passes += 1
                # Try weak/inexpensive jumps first.  A successful removal keeps
                # the physical H-invariant certificate true while reducing the
                # number of implemented recyclers.
                order = sorted(
                    range(len(selected_jumps)),
                    key=lambda index: (
                        float(selected_candidates[index].inflow_norm),
                        int(selected_candidates[index].jump_nnz),
                        int(selected_candidates[index].region_index),
                        int(selected_candidates[index].recycler_index),
                    ),
                )
                for remove_index in order:
                    if len(selected_jumps) <= 1:
                        break
                    trial_jumps = tuple(
                        jump for index, jump in enumerate(selected_jumps) if index != remove_index
                    )
                    trial_report = diagnose_common_kernel_h_invariant_sector(
                        hamiltonian=hamiltonian,
                        jumps=trial_jumps,
                        target_states=state_basis,
                        kernel_tolerance=kernel_tolerance,
                    )
                    if not trial_report.likely_attractive_by_h_invariant_kernel:
                        continue
                    selected_jumps.pop(remove_index)
                    selected_candidates.pop(remove_index)
                    selected_ids = {id(candidate) for candidate in selected_candidates}
                    current_h_report = trial_report
                    n_compressed_jumps_removed += 1
                    removed_this_pass = True
                    break
                if not removed_this_pass:
                    break

            # Keep the selected-candidate readouts aligned with the compressed
            # jump list.  The per-step kernel metadata comes from the original
            # selection trajectory, but the final diagnostics below is recomputed
            # on the compressed list when requested.
            old_step_by_candidate_id = {id(step.candidate): step for step in steps}
            compressed_steps: list[RecycledManifoldJumpSelectionStep] = []
            cumulative_inflow_squared = 0.0
            max_target_jump_residual = 0.0
            for candidate in selected_candidates:
                old_step = old_step_by_candidate_id.get(id(candidate))
                cumulative_inflow_squared += float(candidate.inflow_norm) ** 2
                max_target_jump_residual = max(
                    max_target_jump_residual, float(candidate.dark_residual)
                )
                compressed_steps.append(
                    RecycledManifoldJumpSelectionStep(
                        step_index=len(compressed_steps),
                        candidate=candidate,
                        bad_common_jump_kernel_dimension=(
                            current_bad_dimension
                            if old_step is None
                            else old_step.bad_common_jump_kernel_dimension
                        ),
                        inflow_norm=float(np.sqrt(cumulative_inflow_squared)),
                        max_target_jump_residual=max_target_jump_residual,
                        n_selected_jumps=len(compressed_steps) + 1,
                    )
                )
            steps = compressed_steps

    unbundled_n_jumps = len(selected_jumps)
    collective_groups: tuple[RecycledManifoldCollectiveRecyclerGroup, ...] = ()
    if selected_jumps and collective_recycler_strategy == "bundle_by_region_detector":
        selected_jumps, collective_groups = _bundle_recycled_jumps_by_region_detector(
            selected_candidates=tuple(selected_candidates),
            dim=dim,
            detector_matrices=detector_matrices,
            detector_coefficients=coefficients,
            embedding_contexts=embedding_contexts,
            rdms=rdms,
            recycler_source=recycler_source,
            weighting=collective_recycler_weighting,
            normalize_recyclers=normalize_collective_recyclers,
            tolerance=tolerance,
        )
        final_diagnostics = None

    if selected_jumps and check_final_diagnostics:
        final_diagnostics = diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=tuple(selected_jumps),
            target_states=state_basis,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=False,
            liouvillian_spectrum_method="none",
        )

    return RecycledManifoldJumpSelectionReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        candidate_pool_size=len(pool),
        max_selected_jumps=int(max_selected_jumps),
        target_bad_kernel_dimension=int(target_bad_kernel_dimension),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        jumps=tuple(selected_jumps),
        steps=tuple(steps),
        final_diagnostics=final_diagnostics,
        candidate_report=candidate_report,
        candidate_report_was_expanded=candidate_report_was_expanded,
        candidate_pool_was_limited=candidate_pool_was_limited,
        compression_strategy=compression_strategy,
        n_compression_passes=n_compression_passes,
        n_compressed_jumps_removed=n_compressed_jumps_removed,
        collective_recycler_strategy=collective_recycler_strategy,
        unbundled_n_jumps=unbundled_n_jumps,
        collective_groups=collective_groups,
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
