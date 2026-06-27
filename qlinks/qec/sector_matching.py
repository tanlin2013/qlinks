from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, islice, product
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.encoded.binary_basis import BinaryEncodedBasis
from qlinks.qec.cage_collection import (
    CageSectorCollection,
    CollectedCageRecord,
)
from qlinks.qec.code_space import CodeSpace
from qlinks.qec.error_sets import ErrorOperator, LocalErrorSet
from qlinks.qec.knill_laflamme import apply_error_to_code
from qlinks.qec.profile import QECCodeCandidateReport, diagnose_cage_collection_code_candidate

BasisLike = Basis | BinaryEncodedBasis
FingerprintMode = Literal["expectations", "kl_diagonal"]


@dataclass(frozen=True, slots=True)
class CageRecordFingerprint:
    """Local-error fingerprint for one collected cage record.

    The fingerprint is a real vector built from expectation-like local-error
    data for one state.  It is intended as a cheap prefilter for cross-sector
    matching before running full KL/error-algebra diagnostics on the best
    matched subsets.
    """

    entry: CollectedCageRecord = field(repr=False)
    feature_vector: npt.NDArray[np.float64] = field(repr=False)
    mode: FingerprintMode = "kl_diagonal"
    normalized: bool = False

    @property
    def sector_label(self) -> Any:
        return self.entry.sector_label

    @property
    def signature(self) -> tuple[int, int] | None:
        return self.entry.signature

    @property
    def record_index(self) -> int:
        return self.entry.record_index

    @property
    def dimension(self) -> int:
        return int(self.feature_vector.size)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.feature_vector))

    def distance_to(self, other: CageRecordFingerprint) -> float:
        """Euclidean distance between two fingerprints."""
        if self.feature_vector.shape != other.feature_vector.shape:
            raise ValueError("fingerprints must have matching dimensions.")
        return float(np.linalg.norm(self.feature_vector - other.feature_vector))

    def to_summary_dict(self, *, max_features: int = 8) -> dict[str, object]:
        """Return a compact, notebook-friendly summary."""
        preview = tuple(float(x) for x in self.feature_vector[:max_features])
        return {
            "sector_label": repr(self.sector_label),
            "signature": self.signature,
            "record_index": self.record_index,
            "source_name": self.entry.source_name,
            "mode": self.mode,
            "normalized": self.normalized,
            "dimension": self.dimension,
            "norm": self.norm,
            "feature_preview": preview,
            "n_preview_features": len(preview),
        }


@dataclass(frozen=True, slots=True)
class CageSectorMatchCandidate:
    """One sector-balanced matched subset of collected cage records."""

    entries: tuple[CollectedCageRecord, ...]
    ambient_basis: BasisLike = field(repr=False)
    score: float
    max_pairwise_distance: float
    mean_pairwise_distance: float
    rms_centroid_distance: float
    signature_filter: tuple[tuple[int, int], ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def sector_labels(self) -> tuple[Any, ...]:
        return tuple(entry.sector_label for entry in self.entries)

    @property
    def record_indices(self) -> tuple[int, ...]:
        return tuple(entry.record_index for entry in self.entries)

    @property
    def signatures(self) -> tuple[tuple[int, int] | None, ...]:
        return tuple(entry.signature for entry in self.entries)

    @property
    def labels(self) -> tuple[tuple[Any, tuple[int, int] | None, int], ...]:
        """Default labels used by :class:`CodeSpace` constructors."""
        return tuple(entry.label for entry in self.entries)

    @property
    def common_signature(self) -> tuple[int, int] | None:
        signatures = set(self.signatures)
        if len(signatures) == 1:
            return next(iter(signatures))
        return None

    def to_collection(
        self, *, metadata: Mapping[str, object] | None = None
    ) -> CageSectorCollection:
        """Return this candidate as a small :class:`CageSectorCollection`."""
        merged_metadata = {
            "source": "cage_sector_match_candidate",
            "match_score": self.score,
            "max_pairwise_distance": self.max_pairwise_distance,
            "mean_pairwise_distance": self.mean_pairwise_distance,
            "rms_centroid_distance": self.rms_centroid_distance,
        }
        merged_metadata.update(self.metadata)
        merged_metadata.update(dict(metadata or {}))
        return CageSectorCollection(
            entries=self.entries,
            ambient_basis=self.ambient_basis,
            signature_filter=self.signature_filter,
            metadata=merged_metadata,
        )

    def to_ambient_row_vectors(self) -> npt.NDArray[np.complex128]:
        """Embed matched records into ``ambient_basis`` as row vectors.

        This method makes the match candidate directly acceptable to
        ``CodeSpace.from_cage_collection``.
        """
        return self.to_collection().to_ambient_row_vectors()

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact summary of this matched subset."""
        return {
            "n_records": len(self.entries),
            "sector_labels": tuple(repr(label) for label in self.sector_labels),
            "record_indices": self.record_indices,
            "signatures": self.signatures,
            "score": self.score,
            "max_pairwise_distance": self.max_pairwise_distance,
            "mean_pairwise_distance": self.mean_pairwise_distance,
            "rms_centroid_distance": self.rms_centroid_distance,
            "metadata": dict(self.metadata),
        }

    def to_text(self) -> str:
        from qlinks.qec.reporting import format_float

        records = ", ".join(
            f"{entry.sector_label!r}:rec{entry.record_index}" for entry in self.entries
        )
        return (
            f"score={format_float(self.score)}, "
            f"max pairwise={format_float(self.max_pairwise_distance)}, "
            f"mean pairwise={format_float(self.mean_pairwise_distance)}, records=[{records}]"
        )

    def format_summary(self) -> str:
        return self.to_text()

    def __str__(self) -> str:
        return self.to_text()


@dataclass(frozen=True, slots=True)
class CageSectorMatchingReport:
    """Ranked cross-sector matching report based on local-error fingerprints."""

    collection: CageSectorCollection = field(repr=False)
    error_set_name: str
    fingerprint_mode: FingerprintMode
    max_weight: int | None
    normalized: bool
    sector_labels: tuple[Any, ...]
    records_per_sector: int
    fingerprints: tuple[CageRecordFingerprint, ...] = field(repr=False)
    candidates: tuple[CageSectorMatchCandidate, ...]
    n_sector_combinations: tuple[int, ...]
    n_candidates_evaluated: int
    truncated: bool = False

    @property
    def n_fingerprints(self) -> int:
        return len(self.fingerprints)

    @property
    def fingerprint_dimension(self) -> int:
        if not self.fingerprints:
            return 0
        return self.fingerprints[0].dimension

    @property
    def best_candidate(self) -> CageSectorMatchCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]

    def to_summary_dict(self, *, max_candidates: int = 10) -> dict[str, object]:
        candidates = self.candidates[:max_candidates]
        return {
            "error_set_name": self.error_set_name,
            "fingerprint_mode": self.fingerprint_mode,
            "max_weight": self.max_weight,
            "normalized": self.normalized,
            "n_fingerprints": self.n_fingerprints,
            "fingerprint_dimension": self.fingerprint_dimension,
            "sector_labels": tuple(repr(label) for label in self.sector_labels),
            "records_per_sector": self.records_per_sector,
            "n_sector_combinations": self.n_sector_combinations,
            "n_candidates_evaluated": self.n_candidates_evaluated,
            "truncated": self.truncated,
            "best_candidate": (
                None if self.best_candidate is None else self.best_candidate.to_summary_dict()
            ),
            "top_candidates": tuple(candidate.to_summary_dict() for candidate in candidates),
            "n_preview_candidates": len(candidates),
        }

    def to_text(self, *, max_candidates: int = 10) -> str:
        from qlinks.qec.reporting import format_bool, format_key_value_lines

        lines = [
            format_key_value_lines(
                "Cage sector matching report",
                (
                    ("error set", self.error_set_name),
                    ("fingerprint mode", self.fingerprint_mode),
                    ("max weight", self.max_weight),
                    ("normalized", format_bool(self.normalized)),
                    ("fingerprints", self.n_fingerprints),
                    ("fingerprint dimension", self.fingerprint_dimension),
                    ("sector labels", tuple(repr(label) for label in self.sector_labels)),
                    ("records per sector", self.records_per_sector),
                    ("candidates evaluated", self.n_candidates_evaluated),
                    ("truncated", format_bool(self.truncated)),
                ),
            )
        ]
        if self.candidates:
            lines.append("top matched subsets")
            for candidate in self.candidates[:max_candidates]:
                lines.append(f"  - {candidate.to_text()}")
            if len(self.candidates) > max_candidates:
                lines.append(f"  ... {len(self.candidates) - max_candidates} more candidates")
        return "\n".join(lines)

    def format_summary(self, *, max_candidates: int = 10) -> str:
        return self.to_text(max_candidates=max_candidates)

    def __str__(self) -> str:
        return self.to_text(max_candidates=5)

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_candidates: int = 10):
        from rich.console import Group

        from qlinks.qec.reporting import add_summary_rows, format_bool, format_float, require_rich

        _group, Panel, Table, _text = require_rich("CageSectorMatchingReport")
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        add_summary_rows(
            overview,
            (
                ("error set", self.error_set_name),
                ("fingerprint mode", self.fingerprint_mode),
                ("max weight", self.max_weight),
                ("normalized", format_bool(self.normalized)),
                ("fingerprints", self.n_fingerprints),
                ("fingerprint dimension", self.fingerprint_dimension),
                ("records per sector", self.records_per_sector),
                ("candidates evaluated", self.n_candidates_evaluated),
                ("truncated", format_bool(self.truncated)),
            ),
        )

        table = Table(title="Top matched subsets")
        table.add_column("score", justify="right")
        table.add_column("max pairwise", justify="right")
        table.add_column("mean pairwise", justify="right")
        table.add_column("records")
        for candidate in self.candidates[:max_candidates]:
            records = ", ".join(
                f"{entry.sector_label!r}:rec{entry.record_index}" for entry in candidate.entries
            )
            table.add_row(
                format_float(candidate.score),
                format_float(candidate.max_pairwise_distance),
                format_float(candidate.mean_pairwise_distance),
                records,
            )
        if len(self.candidates) > max_candidates:
            table.caption = f"Showing {max_candidates} of {len(self.candidates)} candidates"

        return Panel(Group(overview, table), title="Cage sector matching report")


@dataclass(frozen=True, slots=True)
class MatchedCageQECScanReport:
    """QEC diagnostics for the top cross-sector matched subsets."""

    matching_report: CageSectorMatchingReport
    candidate_reports: tuple[QECCodeCandidateReport, ...]

    @property
    def qec_candidates(self) -> tuple[QECCodeCandidateReport, ...]:
        return tuple(report for report in self.candidate_reports if report.qec_candidate)

    @property
    def best_report(self) -> QECCodeCandidateReport | None:
        if not self.candidate_reports:
            return None
        return max(
            self.candidate_reports,
            key=lambda report: (
                report.local_indistinguishability_weight,
                1 if report.qec_candidate else 0,
                -1 if report.first_violating_weight is None else -report.first_violating_weight,
            ),
        )

    def to_summary_dict(self, *, max_candidates: int = 10) -> dict[str, object]:
        best = self.best_report
        reports = self.candidate_reports[:max_candidates]
        return {
            "matching": self.matching_report.to_summary_dict(max_candidates=max_candidates),
            "n_candidate_reports": len(self.candidate_reports),
            "n_qec_candidates": len(self.qec_candidates),
            "best_candidate": (
                None if best is None else best.to_summary_dict(max_logical_candidates=3)
            ),
            "candidate_reports": tuple(
                report.to_summary_dict(max_logical_candidates=3) for report in reports
            ),
            "n_preview_candidates": len(reports),
        }

    def to_text(self, *, max_candidates: int = 10) -> str:
        from qlinks.qec.reporting import format_key_value_lines

        best = self.best_report
        lines = [
            self.matching_report.to_text(max_candidates=max_candidates),
            format_key_value_lines(
                "Matched cage QEC scan",
                (
                    ("candidate reports", len(self.candidate_reports)),
                    ("qec candidates", len(self.qec_candidates)),
                    ("best first violation", None if best is None else best.first_violating_weight),
                    (
                        "best LI weight",
                        None if best is None else best.local_indistinguishability_weight,
                    ),
                ),
            ),
        ]
        if self.candidate_reports:
            lines.append("candidate diagnostics")
            for report in self.candidate_reports[:max_candidates]:
                match = report.metadata.get("match", {})
                lines.append(
                    "  - "
                    f"score={match.get('score')}, dim={report.code_dimension}, "
                    f"qec_candidate={report.qec_candidate}, "
                    f"first_violation={report.first_violating_weight}, "
                    f"LI_weight={report.local_indistinguishability_weight}"
                )
        return "\n".join(lines)

    def format_summary(self, *, max_candidates: int = 10) -> str:
        return self.to_text(max_candidates=max_candidates)

    def __str__(self) -> str:
        return self.to_text(max_candidates=5)


def compute_cage_record_fingerprints(
    collection: CageSectorCollection,
    errors: LocalErrorSet,
    *,
    signature: tuple[int, int] | None = None,
    sector_labels: Sequence[Any] | None = None,
    max_weight: int | None = None,
    mode: FingerprintMode = "kl_diagonal",
    normalize: bool = False,
    rank_tolerance: float = 1e-12,
) -> tuple[CageRecordFingerprint, ...]:
    """Compute local-error fingerprints for records in a collection.

    ``mode="expectations"`` uses ``<psi|E_a|psi>`` for each local error.
    ``mode="kl_diagonal"`` uses ``<psi|E_a^dagger E_b|psi>`` for all tested
    error pairs, matching the diagonal part of the KL data.  The latter is more
    expensive but better aligned with local distinguishability diagnostics.
    """
    selected = _select_collection(collection, signature=signature, sector_labels=sector_labels)
    error_subset = _error_subset(errors, max_weight=max_weight)
    if mode not in {"expectations", "kl_diagonal"}:
        raise ValueError("mode must be 'expectations' or 'kl_diagonal'.")

    fingerprints: list[CageRecordFingerprint] = []
    for entry in selected.entries:
        one_record = CageSectorCollection(
            entries=(entry,),
            ambient_basis=selected.ambient_basis,
            signature_filter=selected.signature_filter,
            metadata=dict(selected.metadata),
        )
        code = CodeSpace.from_cage_collection(
            one_record,
            rank_tolerance=rank_tolerance,
            allow_rank_deficient=True,
        )
        vector = _fingerprint_for_one_state(code, error_subset.errors, mode=mode)
        if normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        fingerprints.append(
            CageRecordFingerprint(
                entry=entry,
                feature_vector=np.asarray(vector, dtype=np.float64),
                mode=mode,
                normalized=normalize,
            )
        )

    return tuple(fingerprints)


def match_cage_records_across_sectors(
    collection: CageSectorCollection,
    errors: LocalErrorSet,
    *,
    signature: tuple[int, int] | None = None,
    sector_labels: Sequence[Any] | None = None,
    records_per_sector: int = 1,
    max_weight: int | None = None,
    fingerprint_mode: FingerprintMode = "kl_diagonal",
    normalize_fingerprints: bool = False,
    max_matches: int = 10,
    max_combinations: int | None = None,
    max_sector_combinations: int | None = None,
    rank_tolerance: float = 1e-12,
) -> CageSectorMatchingReport:
    """Rank sector-balanced subsets by local-fingerprint similarity.

    This is a cheap prefilter for topological-sector code candidates.  It picks
    ``records_per_sector`` records from each requested sector and scores the
    flattened subset by the largest pairwise fingerprint distance.  The lowest
    scores are the most locally similar matched subsets.
    """
    if records_per_sector < 1:
        raise ValueError("records_per_sector must be at least 1.")
    if max_matches < 1:
        raise ValueError("max_matches must be at least 1.")

    selected = _select_collection(collection, signature=signature, sector_labels=sector_labels)
    target_sector_labels = (
        tuple(sector_labels) if sector_labels is not None else selected.sector_labels
    )
    if len(target_sector_labels) == 0:
        raise ValueError("At least one sector is required for matching.")

    fingerprints = compute_cage_record_fingerprints(
        selected,
        errors,
        sector_labels=target_sector_labels,
        max_weight=max_weight,
        mode=fingerprint_mode,
        normalize=normalize_fingerprints,
        rank_tolerance=rank_tolerance,
    )
    by_sector: dict[int, list[CageRecordFingerprint]] = defaultdict(list)
    for fingerprint in fingerprints:
        sector_index = _sector_index(target_sector_labels, fingerprint.sector_label)
        if sector_index is not None:
            by_sector[sector_index].append(fingerprint)

    sector_choices: list[tuple[tuple[CageRecordFingerprint, ...], ...]] = []
    n_sector_combinations: list[int] = []
    for sector_index, sector_label in enumerate(target_sector_labels):
        sector_fingerprints = tuple(by_sector.get(sector_index, ()))
        if len(sector_fingerprints) < records_per_sector:
            raise ValueError(
                f"Sector {sector_label!r} has {len(sector_fingerprints)} records, "
                f"fewer than records_per_sector={records_per_sector}."
            )
        choices_iter = combinations(sector_fingerprints, records_per_sector)
        if max_sector_combinations is not None:
            choices = tuple(islice(choices_iter, int(max_sector_combinations)))
        else:
            choices = tuple(choices_iter)
        sector_choices.append(choices)
        n_sector_combinations.append(len(choices))

    candidates: list[CageSectorMatchCandidate] = []
    evaluated = 0
    truncated = False
    for sector_group_tuple in product(*sector_choices):
        if max_combinations is not None and evaluated >= int(max_combinations):
            truncated = True
            break
        evaluated += 1
        flat_fingerprints = tuple(fp for sector_group in sector_group_tuple for fp in sector_group)
        metrics = _subset_distance_metrics(flat_fingerprints)
        entries = tuple(fp.entry for fp in flat_fingerprints)
        candidates.append(
            CageSectorMatchCandidate(
                entries=entries,
                ambient_basis=selected.ambient_basis,
                signature_filter=selected.signature_filter,
                score=metrics["score"],
                max_pairwise_distance=metrics["max_pairwise_distance"],
                mean_pairwise_distance=metrics["mean_pairwise_distance"],
                rms_centroid_distance=metrics["rms_centroid_distance"],
                metadata={
                    "fingerprint_mode": fingerprint_mode,
                    "fingerprint_max_weight": max_weight,
                    "normalized_fingerprints": normalize_fingerprints,
                },
            )
        )

    candidates_sorted = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.mean_pairwise_distance,
                candidate.record_indices,
            ),
        )[:max_matches]
    )

    return CageSectorMatchingReport(
        collection=selected,
        error_set_name=_error_subset(errors, max_weight=max_weight).name,
        fingerprint_mode=fingerprint_mode,
        max_weight=max_weight,
        normalized=normalize_fingerprints,
        sector_labels=target_sector_labels,
        records_per_sector=records_per_sector,
        fingerprints=fingerprints,
        candidates=candidates_sorted,
        n_sector_combinations=tuple(n_sector_combinations),
        n_candidates_evaluated=evaluated,
        truncated=truncated,
    )


def diagnose_matched_cage_collection_code_candidates(
    *,
    collection: CageSectorCollection,
    errors: LocalErrorSet,
    signature: tuple[int, int] | None = None,
    sector_labels: Sequence[Any] | None = None,
    records_per_sector: int = 1,
    match_max_weight: int | None = None,
    diagnostic_max_weight: int | None = None,
    fingerprint_mode: FingerprintMode = "kl_diagonal",
    normalize_fingerprints: bool = False,
    max_matches: int = 10,
    max_combinations: int | None = None,
    max_sector_combinations: int | None = None,
    tolerance: float = 1e-10,
    include_logical_operators: bool = True,
    include_error_algebra: bool = True,
    allow_rank_deficient: bool = True,
) -> MatchedCageQECScanReport:
    """Match cross-sector records, then run QEC diagnostics on top matches."""
    matching = match_cage_records_across_sectors(
        collection,
        errors,
        signature=signature,
        sector_labels=sector_labels,
        records_per_sector=records_per_sector,
        max_weight=match_max_weight,
        fingerprint_mode=fingerprint_mode,
        normalize_fingerprints=normalize_fingerprints,
        max_matches=max_matches,
        max_combinations=max_combinations,
        max_sector_combinations=max_sector_combinations,
    )

    reports: list[QECCodeCandidateReport] = []
    for index, candidate in enumerate(matching.candidates):
        candidate_collection = candidate.to_collection(metadata={"match_rank": index})
        report = diagnose_cage_collection_code_candidate(
            collection=candidate_collection,
            errors=errors,
            max_weight=diagnostic_max_weight,
            tolerance=tolerance,
            include_logical_operators=include_logical_operators,
            include_error_algebra=include_error_algebra,
            allow_rank_deficient=allow_rank_deficient,
            metadata={
                "source": "matched_cage_sector_collection",
                "match_rank": index,
                "match": candidate.to_summary_dict(),
            },
        )
        reports.append(report)

    return MatchedCageQECScanReport(
        matching_report=matching,
        candidate_reports=tuple(reports),
    )


def _select_collection(
    collection: CageSectorCollection,
    *,
    signature: tuple[int, int] | None,
    sector_labels: Sequence[Any] | None,
) -> CageSectorCollection:
    selected = collection
    if signature is not None:
        selected = selected.by_signature(signature)
    if sector_labels is not None:
        selected = selected.select(sector_labels=sector_labels)
    return selected


def _error_subset(errors: LocalErrorSet, *, max_weight: int | None) -> LocalErrorSet:
    if max_weight is None:
        return errors
    return errors.by_max_weight(int(max_weight))


def _fingerprint_for_one_state(
    code: CodeSpace,
    errors: Sequence[ErrorOperator],
    *,
    mode: FingerprintMode,
) -> npt.NDArray[np.float64]:
    images = tuple(apply_error_to_code(code, error.operator) for error in errors)
    complex_features: list[complex] = []
    if mode == "expectations":
        for image in images:
            projected = code.vectors.conj().T @ image
            complex_features.append(complex(projected[0, 0]))
    elif mode == "kl_diagonal":
        for left_image in images:
            for right_image in images:
                complex_features.append(complex((left_image.conj().T @ right_image)[0, 0]))
    else:  # pragma: no cover - validated by caller.
        raise ValueError("unsupported fingerprint mode")

    out = np.empty(2 * len(complex_features), dtype=np.float64)
    out[0::2] = [float(np.real(value)) for value in complex_features]
    out[1::2] = [float(np.imag(value)) for value in complex_features]
    return out


def _sector_index(sector_labels: Sequence[Any], sector_label: Any) -> int | None:
    for index, candidate in enumerate(sector_labels):
        if sector_label == candidate:
            return index
    return None


def _subset_distance_metrics(
    fingerprints: Sequence[CageRecordFingerprint],
) -> dict[str, float]:
    if len(fingerprints) == 0:
        return {
            "score": 0.0,
            "max_pairwise_distance": 0.0,
            "mean_pairwise_distance": 0.0,
            "rms_centroid_distance": 0.0,
        }

    matrix = np.vstack([fp.feature_vector for fp in fingerprints])
    centroid = np.mean(matrix, axis=0)
    deviations = matrix - centroid
    rms_centroid = float(np.sqrt(np.mean(np.sum(np.abs(deviations) ** 2, axis=1))))

    pairwise: list[float] = []
    for left, right in combinations(range(len(fingerprints)), 2):
        pairwise.append(float(np.linalg.norm(matrix[left] - matrix[right])))

    max_pairwise = max(pairwise, default=0.0)
    mean_pairwise = float(np.mean(pairwise)) if pairwise else 0.0
    return {
        "score": max_pairwise,
        "max_pairwise_distance": max_pairwise,
        "mean_pairwise_distance": mean_pairwise,
        "rms_centroid_distance": rms_centroid,
    }
