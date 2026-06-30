from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
