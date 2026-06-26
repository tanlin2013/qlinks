from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qlinks.basis import Basis
from qlinks.encoded.binary_basis import BinaryEncodedBasis
from qlinks.qec.code_space import CodeSpace
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


@dataclass(frozen=True, slots=True)
class QECCodeCandidateReport:
    """QEC diagnostics for one degenerate cage/code candidate."""

    code_space: CodeSpace = field(repr=False)
    signature: tuple[int, int] | None
    record_count: int
    local_indistinguishability: LocalIndistinguishabilityReport
    logical_operators: LogicalOperatorReport | None = None
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

    return QECCodeCandidateReport(
        code_space=code_space,
        signature=None if signature is None else (int(signature[0]), int(signature[1])),
        record_count=len(records),
        local_indistinguishability=local_report,
        logical_operators=logical_report,
        classification_labels=_classification_label_tuple(classification_reports),
        metadata=dict(metadata or {}),
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
