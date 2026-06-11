import numpy as np

from qlinks.caging import (
    CageRecord,
    CageSearchConfig,
    CageSearcher,
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


def test_cage_searcher_uses_fixed_kappa_solver_for_supplied_candidates() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.full(3, 2.0, dtype=np.complex128)
    hamiltonian = kinetic_matrix + np.diag(self_loop_values)
    candidate = CandidateSubgraph(vertices=np.array([0, 1], dtype=np.int64))

    searcher = CageSearcher(
        hamiltonian_matrix=hamiltonian,
        kinetic_matrix=kinetic_matrix,
        self_loop_values=self_loop_values,
        config=CageSearchConfig(
            search_type="custom",
            include_type1=True,
            include_type2=False,
            type1_kappas=(-1,),
            tolerance=1e-12,
        ),
    )

    result = searcher.run(type1_candidates=[candidate])

    assert len(result.records) == 1
    record = result.records[0]
    assert record.signature == (-1, 2)
    assert record.cage_state.metadata["fixed_kappa_solver"] is True


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
