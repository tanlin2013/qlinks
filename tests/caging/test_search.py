import numpy as np
import pytest

from qlinks.caging import (
    CageRecord,
    CageSearchConfig,
    CageSearcher,
    CageSearchResult,
    CageState,
    CandidateSubgraph,
    signature_from_energy_and_self_loop,
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


def test_cage_search_config_rejects_legacy_model_named_search_types() -> None:
    with pytest.raises(ValueError, match="search_type"):
        CageSearchConfig(search_type="qdm")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="search_type"):
        CageSearchConfig(search_type="qlm")  # type: ignore[arg-type]


def test_cage_search_config_candidate_type_presets() -> None:
    assert CageSearcher(
        hamiltonian_matrix=np.zeros((0, 0), dtype=np.complex128),
        kinetic_matrix=np.zeros((0, 0), dtype=np.complex128),
        self_loop_values=np.array([], dtype=np.complex128),
        config=CageSearchConfig(search_type="type1"),
    )._enabled_candidate_types() == (True, False)

    assert CageSearcher(
        hamiltonian_matrix=np.zeros((0, 0), dtype=np.complex128),
        kinetic_matrix=np.zeros((0, 0), dtype=np.complex128),
        self_loop_values=np.array([], dtype=np.complex128),
        config=CageSearchConfig(search_type="type2"),
    )._enabled_candidate_types() == (False, True)

    assert CageSearcher(
        hamiltonian_matrix=np.zeros((0, 0), dtype=np.complex128),
        kinetic_matrix=np.zeros((0, 0), dtype=np.complex128),
        self_loop_values=np.array([], dtype=np.complex128),
        config=CageSearchConfig(search_type="type1_and_type2"),
    )._enabled_candidate_types() == (True, True)


def test_signature_from_energy_and_self_loop_normalizes_potential_unit() -> None:
    assert signature_from_energy_and_self_loop(
        4.0,
        5.0,
        tolerance=1.0e-12,
        potential_unit=2.5,
    ) == (-1, 2)


def test_signature_from_energy_and_self_loop_rejects_noninteger_normalized_potential() -> None:
    assert (
        signature_from_energy_and_self_loop(
            4.0,
            5.0,
            tolerance=1.0e-12,
            potential_unit=2.0,
        )
        is None
    )


def test_cage_searcher_from_model_build_result_infers_scalar_potential_unit() -> None:
    class DummyModel:
        coup_pot = 2.5

    class DummyBuildResult:
        model = DummyModel()
        kinetic = np.zeros((1, 1), dtype=np.complex128)
        potential = np.array([[5.0]], dtype=np.complex128)
        hamiltonian = np.array([[5.0]], dtype=np.complex128)

    searcher = CageSearcher.from_model_build_result(
        DummyBuildResult(),
        config=CageSearchConfig(search_type="type1"),
    )

    assert searcher.config.potential_signature_unit == 2.5
    np.testing.assert_array_equal(
        searcher.self_loop_values,
        np.array([5.0], dtype=np.complex128),
    )


def test_cage_searcher_uses_potential_signature_unit_for_lazy_index() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.full(3, 5.0, dtype=np.complex128)
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
            potential_signature_unit=2.5,
            tolerance=1e-12,
        ),
    )

    result = searcher.run(type1_candidates=[candidate])

    assert result.counts_by_signature == {(-1, 2): 1}
    assert result[(-1, 2)].first().signature == (-1, 2)


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


def test_cage_searcher_stores_full_states_by_default() -> None:
    result = _search_toy_fixed_kappa_cage()

    assert len(result.records) == 1
    assert result.records[0].full_state is not None
    np.testing.assert_allclose(
        result.records[0].full_state,
        result.full_state_matrix()[0],
    )


def test_cage_searcher_can_keep_records_compact_after_rank_deduplication() -> None:
    result = _search_toy_fixed_kappa_cage(
        config_kwargs={"store_full_states": False},
    )

    assert len(result.records) == 1
    assert result.records[0].full_state is None

    full_state_matrix = result.full_state_matrix()
    assert full_state_matrix.shape == (1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(full_state_matrix[0]),
        1.0,
    )


def _search_toy_fixed_kappa_cage(
    *,
    config_kwargs: dict[str, object] | None = None,
) -> CageSearchResult:
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

    kwargs = {} if config_kwargs is None else dict(config_kwargs)
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
            **kwargs,
        ),
    )

    return searcher.run(type1_candidates=[candidate])


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
