import numpy as np

from qlinks.caging import (
    CageRecord,
    CageSearchConfig,
    CageSearchResult,
    CageState,
    CandidateSubgraph,
)


def test_cage_search_result_indexes_all_records():
    records = [
        CageRecord(
            cage_state=_dummy_cage_state(index),
            signature=(0, 4),
            candidate=_dummy_candidate(index),
        )
        for index in range(3)
    ]

    result = CageSearchResult(
        records=records,
        hilbert_size=10,
        config=CageSearchConfig(),
    )

    assert result[0] is records[0]
    assert result[1] is records[1]
    assert result[-1] is records[-1]
    assert result[:2] == records[:2]
    assert len(result) == 3
    assert list(result) == records


def test_cage_search_result_indexes_by_signature():
    records = [
        CageRecord(
            cage_state=_dummy_cage_state(0),
            signature=(0, 4),
            candidate=_dummy_candidate(0),
        ),
        CageRecord(
            cage_state=_dummy_cage_state(1),
            signature=(0, 6),
            candidate=_dummy_candidate(1),
        ),
        CageRecord(
            cage_state=_dummy_cage_state(2),
            signature=(0, 4),
            candidate=_dummy_candidate(2),
        ),
    ]

    result = CageSearchResult(
        records=records,
        hilbert_size=10,
        config=CageSearchConfig(),
    )

    assert result[(0, 4)][0] is records[0]
    assert result[(0, 4)][1] is records[2]
    assert result[(0, 4), 0] is records[0]
    assert result[(0, 4), 1] is records[2]
    assert result[(0, 4), :] == [records[0], records[2]]

    assert result.by_signature((0, 6))[0] is records[1]
    assert result[(0, 6)].first() is records[1]


def _dummy_candidate(index: int) -> CandidateSubgraph:
    return CandidateSubgraph(
        vertices=np.array([index], dtype=np.int64),
        label=f"candidate_{index}",
    )


def _dummy_cage_state(index: int) -> CageState:
    return CageState(
        energy=complex(index),
        local_state=np.array([1.0], dtype=np.complex128),
        support=np.array([index], dtype=np.int64),
        boundary_residual=0.0,
        eigen_residual=0.0,
        full_residual=0.0,
    )
