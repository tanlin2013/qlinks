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

    def to_summary_dict(self, *, include_matrices: bool = False) -> dict[str, object]:
        """Return a compact summary of this projected logical operator."""
        from qlinks.qec.reporting import complex_to_summary, matrix_to_summary

        summary: dict[str, object] = {
            "name": self.name,
            "support_variables": self.support_variables,
            "weight": self.weight,
            "scalar": complex_to_summary(self.scalar),
            "traceless_frobenius_norm": self.traceless_frobenius_norm,
            "traceless_spectral_norm": self.traceless_spectral_norm,
            "relative_traceless_frobenius_norm": self.relative_traceless_frobenius_norm,
            "leakage_frobenius_norm": self.leakage_frobenius_norm,
            "leakage_spectral_norm": self.leakage_spectral_norm,
        }
        if include_matrices:
            summary["projected_matrix"] = matrix_to_summary(self.projected_matrix)
            summary["traceless_matrix"] = matrix_to_summary(self.traceless_matrix)
        return summary

    def to_text(self) -> str:
        from qlinks.qec.reporting import format_complex, format_float

        return (
            f"{self.name}: logical_strength={format_float(self.traceless_spectral_norm)}, "
            f"relative_traceless={format_float(self.relative_traceless_frobenius_norm)}, "
            f"leakage={format_float(self.leakage_frobenius_norm)}, "
            f"scalar={format_complex(self.scalar)}, support={self.support_variables}"
        )

    def format_summary(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()


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

    def to_summary_dict(self, *, max_candidates: int = 8) -> dict[str, object]:
        """Return a compact projected-logical-operator summary."""
        best = self.best_candidate
        sorted_candidates = self.sorted_by_logical_strength()[:max_candidates]
        return {
            "code_dimension": self.code_dimension,
            "n_candidates": len(self.candidates),
            "best_candidate": None if best is None else best.to_summary_dict(),
            "top_candidates": tuple(candidate.to_summary_dict() for candidate in sorted_candidates),
        }

    def to_text(self, *, max_candidates: int = 8) -> str:
        """Return a human-readable projected-logical-operator summary."""
        from qlinks.qec.reporting import format_float, format_key_value_lines

        best = self.best_candidate
        lines = [
            format_key_value_lines(
                "Projected logical operators",
                (
                    ("code dimension", self.code_dimension),
                    ("candidate operators", len(self.candidates)),
                    (
                        "best logical strength",
                        "none" if best is None else format_float(best.traceless_spectral_norm),
                    ),
                    ("best candidate", "none" if best is None else best.name),
                ),
            )
        ]
        if self.candidates:
            lines.append("top projected operators")
            for candidate in self.sorted_by_logical_strength()[:max_candidates]:
                lines.append(f"  - {candidate.to_text()}")
        return "\n".join(lines)

    def format_summary(self, *, max_candidates: int = 8) -> str:
        return self.to_text(max_candidates=max_candidates)

    def __str__(self) -> str:
        return self.to_text(max_candidates=5)

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 10):
        """Return a rich renderable projected-logical-operator summary."""
        from rich.console import Group

        from qlinks.qec.reporting import add_summary_rows, format_float, require_rich

        _group, panel_cls, table_cls, _text = require_rich("LogicalOperatorReport")
        best = self.best_candidate
        overview = table_cls.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        add_summary_rows(
            overview,
            (
                ("code dimension", self.code_dimension),
                ("candidate operators", len(self.candidates)),
                ("best candidate", "none" if best is None else best.name),
                (
                    "best logical strength",
                    "none" if best is None else format_float(best.traceless_spectral_norm),
                ),
            ),
        )

        table = table_cls(title="Top projected operators")
        table.add_column("name")
        table.add_column("weight", justify="right")
        table.add_column("logical strength", justify="right")
        table.add_column("relative traceless", justify="right")
        table.add_column("leakage", justify="right")
        table.add_column("support")
        for candidate in self.sorted_by_logical_strength()[:max_candidates]:
            table.add_row(
                candidate.name,
                str(candidate.weight),
                format_float(candidate.traceless_spectral_norm),
                format_float(candidate.relative_traceless_frobenius_norm),
                format_float(candidate.leakage_frobenius_norm),
                str(candidate.support_variables),
            )

        return panel_cls(Group(overview, table), title="Projected logical operators")


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
