import numpy as np

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.prefilters import (
    BoundaryNullityFilter,
    CandidateFilterContext,
    CombinedBoundaryKineticTargetNullityFilter,
    KineticTargetNullityFilter,
    SameBipartitionSideFilter,
    SupportSizeFilter,
    UniformSelfLoopFilter,
    ZeroInternalKineticFilter,
    filter_candidates,
    make_type1_bipartite_prefilters,
    make_type2_integer_kappa_prefilters,
    run_candidate_filters,
)


def test_support_size_filter_accepts_candidate() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext()

    filter_result = SupportSizeFilter(min_size=2)(context, candidate)

    assert filter_result.accepted


def test_support_size_filter_rejects_small_candidate() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0]))
    context = CandidateFilterContext()

    filter_result = SupportSizeFilter(min_size=2)(context, candidate)

    assert not filter_result.accepted


def test_uniform_self_loop_filter_accepts_scalar_values() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        self_loop_values=np.array([2.0, 2.0, 3.0], dtype=np.complex128),
    )

    filter_result = UniformSelfLoopFilter(tolerance=1e-12)(context, candidate)

    assert filter_result.accepted
    assert filter_result.metadata is not None
    np.testing.assert_allclose(filter_result.metadata["self_loop_value"], 2.0)


def test_uniform_self_loop_filter_rejects_nonuniform_scalar_values() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0, 2]))
    context = CandidateFilterContext(
        self_loop_values=np.array([2.0, 2.0, 3.0], dtype=np.complex128),
    )

    filter_result = UniformSelfLoopFilter(tolerance=1e-12)(context, candidate)

    assert not filter_result.accepted


def test_uniform_self_loop_filter_accepts_parameter_vectors() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        self_loop_values=np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
            ],
            dtype=np.complex128,
        ),
    )

    filter_result = UniformSelfLoopFilter(tolerance=1e-12)(context, candidate)

    assert filter_result.accepted


def test_same_bipartition_side_filter_accepts_one_side_support() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0, 2]))
    context = CandidateFilterContext(
        bipartition_labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )

    filter_result = SameBipartitionSideFilter()(context, candidate)

    assert filter_result.accepted


def test_same_bipartition_side_filter_rejects_mixed_support() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        bipartition_labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )

    filter_result = SameBipartitionSideFilter()(context, candidate)

    assert not filter_result.accepted


def test_zero_internal_kinetic_filter_accepts_bipartite_same_side_support() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 2]))
    context = CandidateFilterContext(kinetic_matrix=kinetic_matrix)

    filter_result = ZeroInternalKineticFilter(tolerance=1e-12)(context, candidate)

    assert filter_result.accepted


def test_zero_internal_kinetic_filter_rejects_nonzero_internal_block() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(kinetic_matrix=kinetic_matrix)

    filter_result = ZeroInternalKineticFilter(tolerance=1e-12)(context, candidate)

    assert not filter_result.accepted


def test_boundary_nullity_filter_accepts_candidate() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(kinetic_matrix=kinetic_matrix)

    filter_result = BoundaryNullityFilter(
        tolerance=1e-12,
        matrix_name="kinetic",
    )(context, candidate)

    assert filter_result.accepted
    assert filter_result.metadata is not None
    assert filter_result.metadata["boundary_nullity"] == 1


def test_kinetic_target_nullity_filter_accepts_target_kappa() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(kinetic_matrix=kinetic_matrix)

    filter_result = KineticTargetNullityFilter(
        target_kappas=(2.0, -2.0),
        tolerance=1e-12,
        require_nonzero_kappa=True,
    )(context, candidate)

    assert filter_result.accepted
    assert filter_result.metadata is not None
    assert 2.0 + 0.0j in filter_result.metadata["accepted_kappas"]
    assert -2.0 + 0.0j in filter_result.metadata["accepted_kappas"]


def test_combined_boundary_kinetic_target_nullity_filter_accepts_cage() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(kinetic_matrix=kinetic_matrix)

    filter_result = CombinedBoundaryKineticTargetNullityFilter(
        target_kappas=(-2.0,),
        tolerance=1e-12,
        require_nonzero_kappa=True,
    )(context, candidate)

    assert filter_result.accepted
    assert filter_result.metadata is not None
    assert -2.0 + 0.0j in filter_result.metadata["accepted_kappas"]


def test_run_candidate_filters_stops_on_rejection() -> None:
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        self_loop_values=np.array([1.0, 2.0], dtype=np.complex128),
    )

    filter_result = run_candidate_filters(
        context,
        candidate,
        [
            SupportSizeFilter(min_size=2),
            UniformSelfLoopFilter(tolerance=1e-12),
        ],
    )

    assert not filter_result.accepted


def test_filter_candidates_returns_only_accepted_candidates() -> None:
    candidates = [
        CandidateSubgraph(vertices=np.array([0, 1])),
        CandidateSubgraph(vertices=np.array([0, 2])),
    ]
    context = CandidateFilterContext(
        self_loop_values=np.array([2.0, 2.0, 3.0], dtype=np.complex128),
    )

    accepted_candidates = filter_candidates(
        context,
        candidates,
        [UniformSelfLoopFilter(tolerance=1e-12)],
    )

    assert len(accepted_candidates) == 1
    np.testing.assert_array_equal(
        accepted_candidates[0].vertices,
        np.array([0, 1]),
    )


# ------------------------------------------------------------------
# The following tests cover the prefilter factory functions, which combine multiple filters.
# ------------------------------------------------------------------


def test_type1_bipartite_prefilters_accept_candidate() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        kinetic_matrix=kinetic_matrix,
        self_loop_values=np.array([3.0, 3.0, 0.0], dtype=np.complex128),
        bipartition_labels=np.array([0, 0, 1], dtype=np.int64),
    )

    filter_result = run_candidate_filters(
        context,
        candidate,
        make_type1_bipartite_prefilters(tolerance=1e-12),
    )

    assert filter_result.accepted


def test_type1_bipartite_prefilters_reject_mixed_bipartition_candidate() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        kinetic_matrix=kinetic_matrix,
        self_loop_values=np.array([3.0, 3.0, 3.0], dtype=np.complex128),
        bipartition_labels=np.array([0, 1, 0], dtype=np.int64),
    )

    filter_result = run_candidate_filters(
        context,
        candidate,
        make_type1_bipartite_prefilters(tolerance=1e-12),
    )

    assert not filter_result.accepted


def test_type2_integer_kappa_prefilters_accept_candidate() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        kinetic_matrix=kinetic_matrix,
        self_loop_values=np.array([4.0, 4.0, 0.0], dtype=np.complex128),
    )

    filter_result = run_candidate_filters(
        context,
        candidate,
        make_type2_integer_kappa_prefilters(
            target_kappas=(-2.0, 2.0),
            tolerance=1e-12,
            use_combined_boundary_kinetic_test=True,
        ),
    )

    assert filter_result.accepted


def test_type2_integer_kappa_prefilters_reject_nonuniform_self_loop() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    context = CandidateFilterContext(
        kinetic_matrix=kinetic_matrix,
        self_loop_values=np.array([4.0, 5.0, 0.0], dtype=np.complex128),
    )

    filter_result = run_candidate_filters(
        context,
        candidate,
        make_type2_integer_kappa_prefilters(
            target_kappas=(-2.0, 2.0),
            tolerance=1e-12,
            use_combined_boundary_kinetic_test=True,
        ),
    )

    assert not filter_result.accepted
