"""Data contracts for dark-manifold detector workflows.

The numerical detector, recycler, and residual-kernel algorithms remain in
the focused manifold-detector modules; this module owns passive result objects whose
role is to carry diagnostics across those stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


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
    unbundled_inflow_norm: float | None = None
    bundled_inflow_norm: float | None = None

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
            "unbundled_inflow_norm": self.unbundled_inflow_norm,
            "bundled_inflow_norm": self.bundled_inflow_norm,
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
