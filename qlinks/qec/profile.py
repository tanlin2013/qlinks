from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qlinks.basis import Basis
from qlinks.encoded.binary_basis import BinaryEncodedBasis
from qlinks.qec.code_space import CodeSpace
from qlinks.qec.error_algebra import ProjectedErrorAlgebraReport, diagnose_projected_error_algebra
from qlinks.qec.error_sets import LocalErrorSet
from qlinks.qec.knill_laflamme import KnillLaflammeReport, diagnose_knill_laflamme
from qlinks.qec.logical_operators import LogicalOperatorReport, search_projected_logical_operators


@dataclass(frozen=True, slots=True)
class KnillLaflammeWeightSummary:
    """Compact KL summary for one cumulative local-error weight."""

    max_weight: int
    n_errors: int
    max_frobenius_residual: float
    max_spectral_residual: float
    max_relative_frobenius_residual: float
    max_relative_leakage_frobenius_norm: float
    passes_exact_kl: bool
    worst_pair_names: tuple[str, str] | None
    worst_pair_support_variables: tuple[int, ...]
    dominant_failure: str | None

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact summary of this weight scan row."""
        return {
            "max_weight": self.max_weight,
            "n_errors": self.n_errors,
            "max_frobenius_residual": self.max_frobenius_residual,
            "max_spectral_residual": self.max_spectral_residual,
            "max_relative_frobenius_residual": self.max_relative_frobenius_residual,
            "max_relative_leakage_frobenius_norm": (self.max_relative_leakage_frobenius_norm),
            "passes_exact_kl": self.passes_exact_kl,
            "worst_pair_names": self.worst_pair_names,
            "worst_pair_support_variables": self.worst_pair_support_variables,
            "dominant_failure": self.dominant_failure,
        }

    def to_text(self) -> str:
        from qlinks.qec.reporting import format_bool, format_float

        return (
            f"w≤{self.max_weight}: errors={self.n_errors}, "
            f"passes={format_bool(self.passes_exact_kl)}, "
            f"max rel-KL={format_float(self.max_relative_frobenius_residual)}, "
            f"max rel-leakage={format_float(self.max_relative_leakage_frobenius_norm)}, "
            f"worst={self.worst_pair_names}, failure={self.dominant_failure}"
        )

    def format_summary(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()


@dataclass(frozen=True, slots=True)
class LocalIndistinguishabilityReport:
    """KL/code-distance style scan over increasing local error weights."""

    code_dimension: int
    ambient_dimension: int
    error_set_name: str
    weight_summaries: tuple[KnillLaflammeWeightSummary, ...]
    kl_reports_by_weight: Mapping[int, KnillLaflammeReport] = field(repr=False)
    tolerance: float = 1e-10

    @property
    def max_weight_tested(self) -> int:
        return max((summary.max_weight for summary in self.weight_summaries), default=0)

    @property
    def first_violating_weight(self) -> int | None:
        for summary in self.weight_summaries:
            if not summary.passes_exact_kl:
                return summary.max_weight
        return None

    @property
    def local_indistinguishability_weight(self) -> int:
        """Largest cumulative weight that satisfies the exact KL tolerance."""
        passed = [
            summary.max_weight for summary in self.weight_summaries if summary.passes_exact_kl
        ]
        return max(passed, default=0)

    @property
    def passes_all_tested_weights(self) -> bool:
        return self.first_violating_weight is None

    @property
    def worst_summary(self) -> KnillLaflammeWeightSummary | None:
        if len(self.weight_summaries) == 0:
            return None
        return max(
            self.weight_summaries,
            key=lambda summary: summary.max_relative_frobenius_residual,
        )

    def report_for_weight(self, max_weight: int) -> KnillLaflammeReport:
        return self.kl_reports_by_weight[int(max_weight)]

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact local-indistinguishability profile summary."""
        worst = self.worst_summary
        return {
            "code_dimension": self.code_dimension,
            "ambient_dimension": self.ambient_dimension,
            "error_set_name": self.error_set_name,
            "tolerance": self.tolerance,
            "max_weight_tested": self.max_weight_tested,
            "first_violating_weight": self.first_violating_weight,
            "local_indistinguishability_weight": self.local_indistinguishability_weight,
            "passes_all_tested_weights": self.passes_all_tested_weights,
            "worst_summary": None if worst is None else worst.to_summary_dict(),
            "weight_summaries": tuple(
                summary.to_summary_dict() for summary in self.weight_summaries
            ),
        }

    def to_text(self) -> str:
        """Return a human-readable local-indistinguishability profile."""
        from qlinks.qec.reporting import format_bool, format_key_value_lines

        lines = [
            format_key_value_lines(
                f"Local indistinguishability profile: {self.error_set_name}",
                (
                    ("code dimension", self.code_dimension),
                    ("ambient dimension", self.ambient_dimension),
                    ("max weight tested", self.max_weight_tested),
                    ("first violating weight", self.first_violating_weight),
                    (
                        "local indistinguishability weight",
                        self.local_indistinguishability_weight,
                    ),
                    ("passes all tested weights", format_bool(self.passes_all_tested_weights)),
                ),
            )
        ]
        if self.weight_summaries:
            lines.append("weight scan")
            lines.extend(f"  - {summary.to_text()}" for summary in self.weight_summaries)
        return "\n".join(lines)

    def format_summary(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        """Return a rich renderable local-indistinguishability profile."""
        from rich.console import Group

        from qlinks.qec.reporting import add_summary_rows, format_bool, format_float, require_rich

        _group, Panel, Table, _text = require_rich("LocalIndistinguishabilityReport")
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        add_summary_rows(
            overview,
            (
                ("code dimension", self.code_dimension),
                ("ambient dimension", self.ambient_dimension),
                ("max weight tested", self.max_weight_tested),
                ("first violating weight", self.first_violating_weight),
                (
                    "local indistinguishability weight",
                    self.local_indistinguishability_weight,
                ),
                ("passes all tested weights", format_bool(self.passes_all_tested_weights)),
            ),
        )

        table = Table(title="Weight scan")
        table.add_column("max weight", justify="right")
        table.add_column("errors", justify="right")
        table.add_column("passes KL")
        table.add_column("max rel-KL", justify="right")
        table.add_column("max rel-leakage", justify="right")
        table.add_column("worst pair")
        table.add_column("failure")
        for summary in self.weight_summaries:
            table.add_row(
                str(summary.max_weight),
                str(summary.n_errors),
                format_bool(summary.passes_exact_kl),
                format_float(summary.max_relative_frobenius_residual),
                format_float(summary.max_relative_leakage_frobenius_norm),
                str(summary.worst_pair_names),
                str(summary.dominant_failure),
            )

        return Panel(
            Group(overview, table),
            title=f"Local indistinguishability profile: {self.error_set_name}",
        )


@dataclass(frozen=True, slots=True)
class QECCodeCandidateReport:
    """QEC diagnostics for one degenerate cage/code candidate."""

    code_space: CodeSpace = field(repr=False)
    signature: tuple[int, int] | None
    record_count: int
    local_indistinguishability: LocalIndistinguishabilityReport
    logical_operators: LogicalOperatorReport | None = None
    error_algebra: ProjectedErrorAlgebraReport | None = None
    classification_labels: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def code_dimension(self) -> int:
        return self.code_space.dimension

    @property
    def first_violating_weight(self) -> int | None:
        return self.local_indistinguishability.first_violating_weight

    @property
    def local_indistinguishability_weight(self) -> int:
        return self.local_indistinguishability.local_indistinguishability_weight

    @property
    def qec_candidate(self) -> bool:
        return self.code_dimension >= 2 and self.first_violating_weight is None

    def to_summary_dict(self, *, max_logical_candidates: int = 5) -> dict[str, object]:
        """Return a compact summary of this cage-code candidate."""
        return {
            "signature": self.signature,
            "record_count": self.record_count,
            "code_dimension": self.code_dimension,
            "ambient_dimension": self.code_space.ambient_dimension,
            "qec_candidate": self.qec_candidate,
            "first_violating_weight": self.first_violating_weight,
            "local_indistinguishability_weight": self.local_indistinguishability_weight,
            "classification_labels": self.classification_labels,
            "metadata": dict(self.metadata),
            "local_indistinguishability": (self.local_indistinguishability.to_summary_dict()),
            "logical_operators": (
                None
                if self.logical_operators is None
                else self.logical_operators.to_summary_dict(max_candidates=max_logical_candidates)
            ),
            "error_algebra": (
                None if self.error_algebra is None else self.error_algebra.to_summary_dict()
            ),
        }

    def to_text(self, *, max_logical_candidates: int = 5) -> str:
        """Return a human-readable cage-code candidate report."""
        from qlinks.qec.reporting import format_bool, format_key_value_lines

        lines = [
            format_key_value_lines(
                "QEC code candidate",
                (
                    ("signature", self.signature),
                    ("record count", self.record_count),
                    ("code dimension", self.code_dimension),
                    ("qec candidate", format_bool(self.qec_candidate)),
                    ("first violating weight", self.first_violating_weight),
                    (
                        "local indistinguishability weight",
                        self.local_indistinguishability_weight,
                    ),
                    ("classification labels", self.classification_labels),
                ),
            ),
            self.local_indistinguishability.to_text(),
        ]
        if self.logical_operators is not None:
            lines.append(self.logical_operators.to_text(max_candidates=max_logical_candidates))
        if self.error_algebra is not None:
            lines.append(self.error_algebra.to_text())
        return "\n".join(lines)

    def format_summary(self, *, max_logical_candidates: int = 5) -> str:
        return self.to_text(max_logical_candidates=max_logical_candidates)

    def __str__(self) -> str:
        return self.to_text(max_logical_candidates=3)

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_logical_candidates: int = 8):
        """Return a rich renderable cage-code candidate report."""
        from rich.console import Group

        from qlinks.qec.reporting import add_summary_rows, format_bool, require_rich

        _group, Panel, Table, _text = require_rich("QECCodeCandidateReport")
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        add_summary_rows(
            overview,
            (
                ("signature", self.signature),
                ("record count", self.record_count),
                ("code dimension", self.code_dimension),
                ("qec candidate", format_bool(self.qec_candidate)),
                ("first violating weight", self.first_violating_weight),
                (
                    "local indistinguishability weight",
                    self.local_indistinguishability_weight,
                ),
                ("classification labels", self.classification_labels),
            ),
        )
        renderables = [overview, self.local_indistinguishability.to_rich()]
        if self.logical_operators is not None:
            renderables.append(
                self.logical_operators.to_rich(max_candidates=max_logical_candidates)
            )
        if self.error_algebra is not None:
            renderables.append(self.error_algebra.to_rich())
        return Panel(Group(*renderables), title="QEC code candidate")


@dataclass(frozen=True, slots=True)
class CageQECScanReport:
    """Collection of QEC candidate reports from a cage-search result."""

    candidate_reports: tuple[QECCodeCandidateReport, ...]

    def __len__(self) -> int:
        return len(self.candidate_reports)

    def __iter__(self):
        return iter(self.candidate_reports)

    def __getitem__(self, index: int) -> QECCodeCandidateReport:
        return self.candidate_reports[index]

    @property
    def qec_candidates(self) -> tuple[QECCodeCandidateReport, ...]:
        return tuple(report for report in self.candidate_reports if report.qec_candidate)

    @property
    def best_by_indistinguishability_weight(self) -> QECCodeCandidateReport | None:
        if len(self.candidate_reports) == 0:
            return None
        return max(
            self.candidate_reports,
            key=lambda report: (
                report.local_indistinguishability_weight,
                report.code_dimension,
                -_none_as_large(report.first_violating_weight),
            ),
        )

    def by_signature(self, signature: tuple[int, int]) -> QECCodeCandidateReport:
        signature = (int(signature[0]), int(signature[1]))
        for report in self.candidate_reports:
            if report.signature == signature:
                return report
        raise KeyError(f"No QEC candidate report for signature {signature}.")

    def to_summary_dict(self, *, max_candidates: int = 20) -> dict[str, object]:
        """Return a compact summary of a cage-result QEC scan."""
        best = self.best_by_indistinguishability_weight
        reports = self.candidate_reports[:max_candidates]
        return {
            "n_candidate_reports": len(self.candidate_reports),
            "n_qec_candidates": len(self.qec_candidates),
            "best_signature": None if best is None else best.signature,
            "best_local_indistinguishability_weight": (
                None if best is None else best.local_indistinguishability_weight
            ),
            "candidate_reports": tuple(
                report.to_summary_dict(max_logical_candidates=3) for report in reports
            ),
            "n_preview_candidates": len(reports),
        }

    def to_text(self, *, max_candidates: int = 20) -> str:
        """Return a human-readable cage-result QEC scan summary."""
        from qlinks.qec.reporting import format_key_value_lines

        best = self.best_by_indistinguishability_weight
        lines = [
            format_key_value_lines(
                "Cage QEC scan",
                (
                    ("candidate reports", len(self.candidate_reports)),
                    ("qec candidates", len(self.qec_candidates)),
                    ("best signature", None if best is None else best.signature),
                    (
                        "best indistinguishability weight",
                        None if best is None else best.local_indistinguishability_weight,
                    ),
                ),
            )
        ]
        if self.candidate_reports:
            lines.append("candidate overview")
            for report in self.candidate_reports[:max_candidates]:
                lines.append(
                    "  - "
                    f"signature={report.signature}, dim={report.code_dimension}, "
                    f"records={report.record_count}, "
                    f"qec_candidate={report.qec_candidate}, "
                    f"first_violation={report.first_violating_weight}, "
                    f"local_indistinguishability_weight="
                    f"{report.local_indistinguishability_weight}"
                )
            if len(self.candidate_reports) > max_candidates:
                lines.append(
                    f"  ... {len(self.candidate_reports) - max_candidates} more candidates"
                )
        return "\n".join(lines)

    def format_summary(self, *, max_candidates: int = 20) -> str:
        return self.to_text(max_candidates=max_candidates)

    def __str__(self) -> str:
        return self.to_text(max_candidates=10)

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 20):
        """Return a rich renderable cage-result QEC scan summary."""
        from rich.console import Group

        from qlinks.qec.reporting import add_summary_rows, format_bool, require_rich

        _group, Panel, Table, _text = require_rich("CageQECScanReport")
        best = self.best_by_indistinguishability_weight
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        add_summary_rows(
            overview,
            (
                ("candidate reports", len(self.candidate_reports)),
                ("qec candidates", len(self.qec_candidates)),
                ("best signature", None if best is None else best.signature),
                (
                    "best indistinguishability weight",
                    None if best is None else best.local_indistinguishability_weight,
                ),
            ),
        )

        table = Table(title="Candidate overview")
        table.add_column("signature")
        table.add_column("dim", justify="right")
        table.add_column("records", justify="right")
        table.add_column("qec candidate")
        table.add_column("first violation", justify="right")
        table.add_column("LI weight", justify="right")
        for report in self.candidate_reports[:max_candidates]:
            table.add_row(
                str(report.signature),
                str(report.code_dimension),
                str(report.record_count),
                format_bool(report.qec_candidate),
                str(report.first_violating_weight),
                str(report.local_indistinguishability_weight),
            )
        if len(self.candidate_reports) > max_candidates:
            table.caption = f"Showing {max_candidates} of {len(self.candidate_reports)} candidates"

        return Panel(Group(overview, table), title="Cage QEC scan")


def diagnose_local_indistinguishability(
    code_space: CodeSpace,
    errors: LocalErrorSet,
    *,
    max_weight: int | None = None,
    cumulative: bool = True,
    tolerance: float = 1e-10,
) -> LocalIndistinguishabilityReport:
    """Run KL diagnostics while increasing the allowed local error weight.

    By default each step uses all errors with support size ``<= w``.  Set
    ``cumulative=False`` to inspect exact-weight sectors separately.
    """
    if max_weight is None:
        max_weight = errors.max_weight

    max_weight = int(max_weight)
    if max_weight < 1:
        raise ValueError("max_weight must be at least 1.")

    summaries: list[KnillLaflammeWeightSummary] = []
    reports_by_weight: dict[int, KnillLaflammeReport] = {}

    for weight in range(1, max_weight + 1):
        if cumulative:
            error_subset = errors.by_max_weight(weight)
        else:
            error_subset = errors.by_exact_weight(weight)
        kl_report = diagnose_knill_laflamme(
            code_space=code_space,
            errors=error_subset,
            tolerance=tolerance,
        )
        reports_by_weight[weight] = kl_report
        summaries.append(_summarize_weight_report(weight, error_subset, kl_report))

    return LocalIndistinguishabilityReport(
        code_dimension=code_space.dimension,
        ambient_dimension=code_space.ambient_dimension,
        error_set_name=errors.name,
        weight_summaries=tuple(summaries),
        kl_reports_by_weight=reports_by_weight,
        tolerance=tolerance,
    )


def diagnose_cage_code_candidate(
    *,
    basis: Basis | BinaryEncodedBasis,
    records: Sequence[Any],
    errors: LocalErrorSet,
    signature: tuple[int, int] | None = None,
    max_weight: int | None = None,
    tolerance: float = 1e-10,
    include_logical_operators: bool = True,
    include_error_algebra: bool = False,
    allow_rank_deficient: bool = True,
    classification_reports: Sequence[Any] = (),
    metadata: Mapping[str, object] | None = None,
) -> QECCodeCandidateReport:
    """Build a code from cage records and diagnose local indistinguishability."""
    code_space = CodeSpace.from_cage_records(
        basis,
        records,
        allow_rank_deficient=allow_rank_deficient,
    )
    local_report = diagnose_local_indistinguishability(
        code_space,
        errors,
        max_weight=max_weight,
        tolerance=tolerance,
    )
    logical_report = (
        search_projected_logical_operators(code_space, errors)
        if include_logical_operators
        else None
    )
    algebra_report = (
        diagnose_projected_error_algebra(
            code_space,
            errors,
            max_weight=max_weight,
            tolerance=tolerance,
        )
        if include_error_algebra
        else None
    )

    return QECCodeCandidateReport(
        code_space=code_space,
        signature=None if signature is None else (int(signature[0]), int(signature[1])),
        record_count=len(records),
        local_indistinguishability=local_report,
        logical_operators=logical_report,
        error_algebra=algebra_report,
        classification_labels=_classification_label_tuple(classification_reports),
        metadata=dict(metadata or {}),
    )


def diagnose_cage_collection_code_candidate(
    *,
    collection: Any,
    errors: LocalErrorSet,
    signature: tuple[int, int] | None = None,
    max_weight: int | None = None,
    tolerance: float = 1e-10,
    include_logical_operators: bool = True,
    include_error_algebra: bool = False,
    allow_rank_deficient: bool = True,
    classification_reports: Sequence[Any] = (),
    metadata: Mapping[str, object] | None = None,
) -> QECCodeCandidateReport:
    """Diagnose a cross-sector cage collection as one candidate code space."""
    selected_collection = collection if signature is None else collection.by_signature(signature)
    code_space = CodeSpace.from_cage_collection(
        selected_collection,
        allow_rank_deficient=allow_rank_deficient,
    )
    local_report = diagnose_local_indistinguishability(
        code_space,
        errors,
        max_weight=max_weight,
        tolerance=tolerance,
    )
    logical_report = (
        search_projected_logical_operators(code_space, errors)
        if include_logical_operators
        else None
    )
    algebra_report = (
        diagnose_projected_error_algebra(
            code_space,
            errors,
            max_weight=max_weight,
            tolerance=tolerance,
        )
        if include_error_algebra
        else None
    )

    merged_metadata = {"source": "cage_sector_collection"}
    collection_summary = getattr(selected_collection, "to_summary_dict", None)
    if callable(collection_summary):
        merged_metadata["collection"] = collection_summary(max_entries=5)
    merged_metadata.update(dict(metadata or {}))

    return QECCodeCandidateReport(
        code_space=code_space,
        signature=selected_collection.common_signature,
        record_count=len(selected_collection),
        local_indistinguishability=local_report,
        logical_operators=logical_report,
        error_algebra=algebra_report,
        classification_labels=_classification_label_tuple(classification_reports),
        metadata=merged_metadata,
    )


def diagnose_cage_result_code_candidates(
    *,
    cage_result: Any,
    basis: Basis | BinaryEncodedBasis,
    errors: LocalErrorSet,
    signatures: Sequence[tuple[int, int]] | None = None,
    min_record_count: int = 2,
    max_weight: int | None = None,
    tolerance: float = 1e-10,
    include_logical_operators: bool = True,
    include_error_algebra: bool = False,
    allow_rank_deficient: bool = True,
) -> CageQECScanReport:
    """Scan degenerate signature sectors of a ``CageSearchResult`` for QEC behavior."""
    if signatures is None:
        signatures = tuple(cage_result.signatures)

    reports: list[QECCodeCandidateReport] = []
    for signature in signatures:
        signature = (int(signature[0]), int(signature[1]))
        records = tuple(cage_result.records_by_signature(signature))
        if len(records) < min_record_count:
            continue
        reports.append(
            diagnose_cage_code_candidate(
                basis=basis,
                records=records,
                errors=errors,
                signature=signature,
                max_weight=max_weight,
                tolerance=tolerance,
                include_logical_operators=include_logical_operators,
                include_error_algebra=include_error_algebra,
                allow_rank_deficient=allow_rank_deficient,
                metadata={"source": "cage_result"},
            )
        )

    return CageQECScanReport(candidate_reports=tuple(reports))


def _summarize_weight_report(
    weight: int,
    errors: LocalErrorSet,
    kl_report: KnillLaflammeReport,
) -> KnillLaflammeWeightSummary:
    worst_pair = kl_report.worst_pair
    worst_image = kl_report.worst_error_image
    support_variables: tuple[int, ...] = ()
    if worst_pair is not None:
        support_variables = _pair_support_variables(errors, worst_pair.names)

    return KnillLaflammeWeightSummary(
        max_weight=weight,
        n_errors=len(errors),
        max_frobenius_residual=kl_report.max_frobenius_residual,
        max_spectral_residual=kl_report.max_spectral_residual,
        max_relative_frobenius_residual=kl_report.max_relative_frobenius_residual,
        max_relative_leakage_frobenius_norm=(
            0.0 if worst_image is None else worst_image.relative_leakage_frobenius_norm
        ),
        passes_exact_kl=kl_report.passes_exact_kl,
        worst_pair_names=None if worst_pair is None else worst_pair.names,
        worst_pair_support_variables=support_variables,
        dominant_failure=None if worst_pair is None else worst_pair.dominant_failure,
    )


def _pair_support_variables(errors: LocalErrorSet, names: tuple[str, str]) -> tuple[int, ...]:
    support_by_name = {error.name: set(error.support_variables) for error in errors}
    support = set(support_by_name.get(names[0], set())) | set(support_by_name.get(names[1], set()))
    return tuple(sorted(int(index) for index in support))


def _classification_label_tuple(classification_reports: Sequence[Any]) -> tuple[str, ...]:
    labels: list[str] = []
    for report in classification_reports:
        label = getattr(report, "label", None)
        if label is not None:
            labels.append(str(label))
            continue
        closure = getattr(report, "closure_mechanism_label", None)
        fock = getattr(report, "fock_support_morphology_label", None)
        real_space = getattr(report, "real_space_support_morphology_label", None)
        parts = [str(part) for part in (closure, fock, real_space) if part is not None]
        if parts:
            labels.append("/".join(parts))
    return tuple(labels)


def _none_as_large(value: int | None) -> int:
    if value is None:
        return 10**9
    return int(value)
