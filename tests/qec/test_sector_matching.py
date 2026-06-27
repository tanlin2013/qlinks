from dataclasses import dataclass

import numpy as np

from qlinks.basis import Basis, full_basis_from_layout
from qlinks.qec import (
    CageSectorCollection,
    CodeSpace,
    LocalErrorSet,
    compute_cage_record_fingerprints,
    diagnose_matched_cage_collection_code_candidates,
    match_cage_records_across_sectors,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class _Record:
    support: np.ndarray
    local_state: np.ndarray
    signature: tuple[int, int]


@dataclass(frozen=True)
class _CageResult:
    records: list[_Record]


def _basis_from_configs(layout, configs):
    return Basis.from_states(layout, np.asarray(configs, dtype=np.int64), sort=True)


def _matching_inputs():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    ambient = full_basis_from_layout(layout, sort=True)
    sector_a = _basis_from_configs(layout, [[0, 0], [0, 1]])
    sector_b = _basis_from_configs(layout, [[1, 0], [1, 1]])

    idx_a_00 = sector_a.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_a_01 = sector_a.require_index(np.asarray([0, 1], dtype=np.int64))
    idx_b_10 = sector_b.require_index(np.asarray([1, 0], dtype=np.int64))
    idx_b_11 = sector_b.require_index(np.asarray([1, 1], dtype=np.int64))

    result_a = _CageResult(
        records=[
            _Record(
                support=np.asarray([idx_a_00], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 4),
            ),
            _Record(
                support=np.asarray([idx_a_01], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 4),
            ),
        ]
    )
    result_b = _CageResult(
        records=[
            _Record(
                support=np.asarray([idx_b_10], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 4),
            ),
            _Record(
                support=np.asarray([idx_b_11], dtype=np.int64),
                local_state=np.asarray([1.0], dtype=np.complex128),
                signature=(0, 4),
            ),
        ]
    )
    collection = CageSectorCollection.from_sector_results(
        [
            ("a", sector_a, result_a),
            ("b", sector_b, result_b),
        ],
        signature=(0, 4),
        ambient_basis=ambient,
    )
    errors = LocalErrorSet.from_layout(
        layout,
        variable_indices=[1],
        max_weight=1,
        include_value_diagonal=True,
        include_projectors=False,
        include_transitions=False,
    )
    return layout, collection, errors


def test_compute_cage_record_fingerprints_for_collection_records() -> None:
    _layout, collection, errors = _matching_inputs()

    fingerprints = compute_cage_record_fingerprints(
        collection,
        errors,
        mode="expectations",
    )

    assert len(fingerprints) == 4
    assert fingerprints[0].dimension == 2
    assert fingerprints[0].distance_to(fingerprints[2]) == 0.0
    assert fingerprints[1].distance_to(fingerprints[3]) == 0.0


def test_match_cage_records_across_sectors_finds_locally_similar_representatives() -> None:
    _layout, collection, errors = _matching_inputs()

    report = match_cage_records_across_sectors(
        collection,
        errors,
        fingerprint_mode="expectations",
        max_matches=4,
    )

    assert report.n_fingerprints == 4
    assert report.best_candidate is not None
    assert report.best_candidate.score == 0.0
    assert report.best_candidate.sector_labels == ("a", "b")
    assert report.best_candidate.record_indices in {(0, 0), (1, 1)}
    assert "Cage sector matching report" in report.format_summary()

    code = CodeSpace.from_cage_collection(report.best_candidate)
    assert code.dimension == 2


def test_diagnose_matched_cage_collection_code_candidates_runs_qec_on_top_matches() -> None:
    _layout, collection, errors = _matching_inputs()

    scan = diagnose_matched_cage_collection_code_candidates(
        collection=collection,
        errors=errors,
        fingerprint_mode="expectations",
        max_matches=2,
        diagnostic_max_weight=1,
        include_error_algebra=True,
    )

    assert len(scan.candidate_reports) == 2
    assert scan.candidate_reports[0].metadata["source"] == "matched_cage_sector_collection"
    assert scan.candidate_reports[0].metadata["match_rank"] == 0
    assert scan.matching_report.best_candidate is not None
    assert scan.matching_report.best_candidate.score == 0.0
    assert scan.candidate_reports[0].code_dimension == 2
    assert scan.candidate_reports[0].qec_candidate
