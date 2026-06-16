from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.caging.classification import (
    InterferenceZeroReport,
    LocalTransitionPattern,
)
from qlinks.caging.open_system import (
    CageLindbladConstruction,
    _build_component_decomposition_jump_operators,
    _build_monitor_recycler_jump_operators,
    _build_reduced_iz_monitor_from_reports,
    _build_reduced_iz_operator_matrix,
    _group_local_transitions_by_source,
    _group_reduced_iz_reports_for_monitor,
    _infer_sharp_potential_value,
    _lazy_left_multiply_sparse_csr,
    _LazyReducedIZMonitorOperator,
    _LazySparseSumOperator,
    _left_multiply_sparse_csr,
    _LocalTermMatrixCache,
    _partition_plaquette_terms_by_region,
    _reduced_iz_monitor_state_from_reports,
    _ReducedIZAssemblyCache,
    _resolve_local_term_builder,
    _select_jump_terms,
    _select_monitor_terms,
    _transition_pattern_key,
)
from qlinks.encoded import BinaryEncodedBasis
from qlinks.models import LocalTermDescriptor
from qlinks.open_system import (
    LindbladEvolutionOptions,
    LocalRecyclingBuildResult,
    LocalRecyclingCandidate,
    LocalRecyclingScanResult,
    LocalRecyclingSelection,
    LocalReducedDensityMatrix,
    initial_density_matrix,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class _FakeRegion:
    region_size: int = 0


def _fake_construction() -> CageLindbladConstruction:
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    zero = sp.csr_array((2, 2), dtype=np.complex128)

    return CageLindbladConstruction(
        cage_state=ket0,
        region=_FakeRegion(),  # type: ignore[arg-type]
        z_value=None,
        inside_plaquette_ids=(),
        outside_plaquette_ids=(),
        crossing_plaquette_ids=(),
        monitor=zero,
        jumps=(),
        n_jumps=0,
        n_component_jumps=0,
        n_global_jump_terms=0,
        open_system_backend="scipy",
        monitor_source="reduced_iz_operators",
        reduced_iz_monitor_decomposition="single_sum",
        reduced_iz_monitor_content="offdiagonal_only",
        n_reduced_iz_monitor_terms=0,
        reduced_iz_monitor_zero_indices=(),
        monitor_components=(),
        component_z_values=(),
        jump_operator_design="kinetic_times_monitor",
        monitor_plaquette_policy="strict_inside",
        jump_plaquette_policy="outside_or_crossing",
        monitor_plaquette_ids=(),
        jump_plaquette_ids=(),
        kinetic_terms_monitor=(),
        potential_terms_monitor=(),
        kinetic_terms_jump=(),
        recycling_jump_source="none",
        n_recycling_jumps=0,
        recycling_jump_variable_indices=(),
        recycling_jump_alpha_beta_indices=(),
        recycling_two_pattern_count=0,
        recycling_build_result=None,
        monitor_residual=0.0,
        max_jump_residual=0.0,
        jump_residuals=(),
        liouvillian_residual=0.0,
    )


def _zero_report(
    *,
    zero_index: int,
    local_mask: tuple[bool, ...],
    label: str = "q_empty",
    transitions: tuple[LocalTransitionPattern, ...] = (),
    reduced_action_vector: np.ndarray | None = None,
) -> InterferenceZeroReport:
    return InterferenceZeroReport(
        zero_index=zero_index,
        active_neighbors=np.array([], dtype=np.int64),
        active_matrix_elements=np.array([], dtype=np.complex128),
        active_amplitudes=np.array([], dtype=np.complex128),
        cancellation_residual=0.0,
        common_mask=np.zeros(len(local_mask), dtype=np.bool_),
        local_mask=np.asarray(local_mask, dtype=np.bool_),
        local_transitions=transitions,
        q_sector_weight=1.0,
        reduced_action_norm=0.0,
        complement_action_norm=0.0,
        complement_target_indices=np.array([], dtype=np.int64),
        explained_complement_target_indices=np.array([], dtype=np.int64),
        unexplained_complement_target_indices=np.array([], dtype=np.int64),
        complement_targets_are_known_zeros=True,
        trivial_target_indices=np.array([], dtype=np.int64),
        known_nonprojector_iz_target_indices=np.array([], dtype=np.int64),
        projector_like_iz_target_indices=np.array([], dtype=np.int64),
        unexpected_target_indices=np.array([], dtype=np.int64),
        complement_support_indices=np.array([], dtype=np.int64),
        complement_contributing_input_indices=np.array([], dtype=np.int64),
        projector_like_annihilated_input_indices=np.array([], dtype=np.int64),
        source_projector_like=False,
        has_unexpected_targets=False,
        has_nonzero_complement_action=False,
        unexpected_target_probe_failure_indices=np.array([], dtype=np.int64),
        nonzero_complement_action_target_indices=np.array([], dtype=np.int64),
        probe_mechanism_label=label,  # type: ignore[arg-type]
        reduced_action_vector=(
            np.array([], dtype=np.complex128)
            if reduced_action_vector is None
            else np.asarray(reduced_action_vector, dtype=np.complex128)
        ),
    )


def test_local_term_builder_uses_bitmask_for_encoded_build_result():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    encoded_basis = BinaryEncodedBasis.from_codes(layout, [0, 1, 2, 3])
    build_result = SimpleNamespace(basis=encoded_basis)

    assert _resolve_local_term_builder("sparse", build_result) == "bitmask"
    assert _resolve_local_term_builder("optimized", build_result) == "bitmask"


class _RecordingLocalTermModel:
    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    def build_local_term(self, term, build_result, *, builder, backend):
        del build_result, backend
        self.calls.append((int(term.term_id), str(builder)))
        return sp.csr_array(float(term.term_id) * np.eye(2, dtype=np.complex128))


def test_local_term_matrix_cache_reuses_terms_and_promotes_bitmask_builder():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    encoded_basis = BinaryEncodedBasis.from_codes(layout, [0, 1, 2, 3])
    build_result = SimpleNamespace(
        basis=encoded_basis,
        hamiltonian=sp.csr_array((2, 2), dtype=np.complex128),
    )
    model = _RecordingLocalTermModel()
    term = LocalTermDescriptor(
        term_id=3,
        term_kind="plaquette",
        operator_kind="kinetic",
        support_links=(0, 1),
    )
    cache = _LocalTermMatrixCache(
        model=model,
        build_result=build_result,
        builder="sparse",
        backend="scipy",
    )

    first = cache.get(term)
    second = cache.get(term)

    assert first is second
    assert model.calls == [(3, "bitmask")]
    np.testing.assert_allclose(first.toarray(), 3.0 * np.eye(2))


def test_lazy_left_multiply_materializes_on_demand():
    left = sp.csr_array(np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.complex128))
    right = sp.csr_array(np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.complex128))

    lazy = _lazy_left_multiply_sparse_csr(left, right)

    assert lazy.shape == (2, 2)
    vector = np.array([1.0, -1.0])

    np.testing.assert_allclose(lazy.toarray(), (left @ right).toarray())
    np.testing.assert_allclose(lazy @ vector, (left @ right) @ vector)


def test_left_multiply_sparse_csr_matches_scipy_product():
    left = sp.csr_array(
        (
            np.array([2.0, -1.0, 0.5], dtype=np.complex128),
            (np.array([0, 1, 1]), np.array([1, 0, 2])),
        ),
        shape=(3, 3),
    )
    right = sp.csr_array(
        (
            np.array([1.0, 3.0j, -2.0, 4.0], dtype=np.complex128),
            (np.array([0, 1, 1, 2]), np.array([0, 0, 2, 1])),
        ),
        shape=(3, 4),
    )

    actual = _left_multiply_sparse_csr(left, right)
    expected = left @ right

    np.testing.assert_allclose(actual.toarray(), expected.toarray())


def test_left_multiply_sparse_csr_handles_general_sparse_left_factor():
    left = sp.csr_array(
        (
            np.array([1.0, -0.5j, 2.0], dtype=np.complex128),
            (np.array([0, 2, 1]), np.array([1, 1, 0])),
        ),
        shape=(3, 3),
    )
    right = sp.csr_array(
        (
            np.array([2.0, 3.0, -1.0j], dtype=np.complex128),
            (np.array([0, 1, 1]), np.array([0, 1, 2])),
        ),
        shape=(3, 4),
    )

    actual = _left_multiply_sparse_csr(left, right)
    expected = left @ right

    np.testing.assert_allclose(actual.toarray(), expected.toarray())


def test_build_monitor_recycler_jump_operators_uses_local_rdm_recycler():
    basis_configs = np.asarray([[0], [1]], dtype=np.int64)
    target_state = np.asarray([1.0, 0.0], dtype=np.complex128)
    monitor = sp.csr_array(np.diag([0.0, 1.0]).astype(np.complex128))

    jumps, recycling_result, closure_orders = _build_monitor_recycler_jump_operators(
        basis_configs=basis_configs,
        state=target_state,
        region_specs=(((0,), monitor),),
        source="local_rdm_rank_one",
        max_jumps_per_region=1,
        rdm_tolerance=1e-10,
        dark_tolerance=1e-10,
        inflow_tolerance=1e-12,
        prefer_sparse=True,
        two_pattern_tolerance=1e-8,
    )

    assert recycling_result.n_jumps == 1
    assert closure_orders == (0,)
    assert len(jumps) == 1
    np.testing.assert_allclose(jumps[0] @ target_state, np.zeros(2, dtype=np.complex128))
    np.testing.assert_allclose(
        jumps[0].toarray(),
        np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
    )


def test_build_monitor_recycler_jump_operators_can_add_hamiltonian_closure():
    basis_configs = np.asarray([[0], [1]], dtype=np.int64)
    target_state = np.asarray([1.0, 0.0], dtype=np.complex128)
    monitor = sp.csr_array(np.diag([0.0, 1.0]).astype(np.complex128))
    hamiltonian = sp.csr_array(np.diag([2.0, 3.0]).astype(np.complex128))

    jumps, recycling_result, closure_orders = _build_monitor_recycler_jump_operators(
        basis_configs=basis_configs,
        state=target_state,
        hamiltonian=hamiltonian,
        region_specs=(((0,), monitor),),
        source="local_rdm_rank_one",
        max_jumps_per_region=1,
        rdm_tolerance=1e-10,
        dark_tolerance=1e-10,
        inflow_tolerance=1e-12,
        prefer_sparse=True,
        two_pattern_tolerance=1e-8,
        hamiltonian_closure_order=1,
    )

    assert recycling_result.n_jumps == 1
    assert closure_orders == (0, 1)
    assert len(jumps) == 2
    np.testing.assert_allclose(jumps[0] @ target_state, np.zeros(2, dtype=np.complex128))
    np.testing.assert_allclose(jumps[1] @ target_state, np.zeros(2, dtype=np.complex128))
    np.testing.assert_allclose(
        jumps[1].toarray(),
        np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
    )


def test_left_multiply_sparse_csr_sums_duplicate_entries():
    left = sp.csr_array(
        (
            np.array([1.0, 2.0], dtype=np.complex128),
            (np.array([0, 0]), np.array([1, 1])),
        ),
        shape=(2, 2),
    )
    right = sp.csr_array(
        (
            np.array([5.0], dtype=np.complex128),
            (np.array([1]), np.array([0])),
        ),
        shape=(2, 2),
    )

    actual = _left_multiply_sparse_csr(left, right)

    np.testing.assert_allclose(
        actual.toarray(),
        np.array([[15.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
    )


def test_select_jump_terms_can_include_crossing_terms():
    inside = (
        LocalTermDescriptor(
            term_id=0,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(0, 1),
        ),
    )
    outside = ()
    crossing = (
        LocalTermDescriptor(
            term_id=1,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(1, 2),
        ),
    )

    selected = _select_jump_terms(
        inside_terms=inside,
        outside_terms=outside,
        crossing_terms=crossing,
        policy="outside_or_crossing",
    )

    assert tuple(term.term_id for term in selected) == (1,)


def test_select_monitor_terms_stays_strict_inside_by_default():
    inside = (
        LocalTermDescriptor(
            term_id=0,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(0, 1),
        ),
    )
    crossing = (
        LocalTermDescriptor(
            term_id=1,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(1, 2),
        ),
    )

    selected = _select_monitor_terms(
        inside_terms=inside,
        outside_terms=(),
        crossing_terms=crossing,
        policy="strict_inside",
    )

    assert tuple(term.term_id for term in selected) == (0,)


def test_cage_lindblad_construction_exposes_lindblad_problem():
    construction = _fake_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))

    problem = construction.to_lindblad_problem(hamiltonian=hamiltonian)

    assert problem.dim == 2
    assert len(problem.jumps) == construction.n_jumps


def test_cage_lindblad_construction_verify_final_state():
    construction = _fake_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))
    rho = np.outer(construction.cage_state, construction.cage_state.conj())

    result = construction.verify_final_state(
        rho,
        hamiltonian=hamiltonian,
    )

    assert result.density_matrix.fidelity_with_target == pytest.approx(1.0)
    assert result.lindblad_residual < 1e-12


def test_cage_lindblad_construction_evolve_uses_new_api():
    construction = _fake_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))

    rho0 = initial_density_matrix(2, kind="mixed", rng=0)
    times = np.linspace(0.0, 0.01, 3)

    result = construction.evolve(
        hamiltonian=hamiltonian,
        density_matrix_initial=rho0,
        times=times,
        options=LindbladEvolutionOptions(
            method="rk4_matrix",
            rk4_step_policy="adaptive",
        ),
    )

    assert len(result.density_matrices) == len(times)


def test_infer_sharp_potential_value_accepts_eigenstate():
    state = np.array([1.0, 0.0], dtype=np.complex128)
    potential = sp.csr_array(np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.complex128))

    value = _infer_sharp_potential_value(
        potential_matrix=potential,
        state=state,
        residual_tolerance=1e-12,
        label="test",
    )

    assert value == pytest.approx(2.0)


def test_infer_sharp_potential_value_rejects_non_eigenstate():
    state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    potential = sp.csr_array(np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.complex128))

    with pytest.raises(ValueError, match="Could not infer"):
        _infer_sharp_potential_value(
            potential_matrix=potential,
            state=state,
            residual_tolerance=1e-12,
            label="test",
        )


def test_component_decomposition_kinetic_times_monitor_uses_only_component_jumps():
    component_jump_0 = sp.csr_array(np.eye(2, dtype=np.complex128))
    component_jump_1 = sp.csr_array(2.0 * np.eye(2, dtype=np.complex128))

    jumps = _build_component_decomposition_jump_operators(
        model=None,
        build_result=None,
        component_jumps=(component_jump_0, component_jump_1),
        jump_kinetic_terms=(),
        potential_terms_by_plaquette_id={},
        builder="sparse",
        backend="scipy",
        jump_operator_design="kinetic_times_monitor",
    )

    assert len(jumps) == 2
    np.testing.assert_allclose(jumps[0].toarray(), component_jump_0.toarray())
    np.testing.assert_allclose(jumps[1].toarray(), component_jump_1.toarray())


@dataclass(frozen=True)
class _FakeTerm:
    term_id: int

    @property
    def support_link_set(self):
        return frozenset()


class _FakeModel:
    def build_local_term(self, term, build_result, *, builder, backend):
        del build_result, builder, backend
        return sp.csr_array(
            np.array(
                [[float(term.term_id), 0.0], [0.0, float(term.term_id)]],
                dtype=np.complex128,
            )
        )


def test_component_decomposition_kinetic_outside_appends_bare_kinetic():
    component_jump = sp.csr_array(np.eye(2, dtype=np.complex128))
    model = _FakeModel()

    jumps = _build_component_decomposition_jump_operators(
        model=model,
        build_result=None,
        component_jumps=(component_jump,),
        jump_kinetic_terms=(_FakeTerm(3), _FakeTerm(5)),
        potential_terms_by_plaquette_id={},
        builder="sparse",
        backend="scipy",
        jump_operator_design="kinetic_outside_monitor_inside",
    )

    assert len(jumps) == 3
    np.testing.assert_allclose(jumps[0].toarray(), np.eye(2))
    np.testing.assert_allclose(jumps[1].toarray(), 3.0 * np.eye(2))
    np.testing.assert_allclose(jumps[2].toarray(), 5.0 * np.eye(2))


class _FakeHamiltonianModel:
    def build_local_term(self, term, build_result, *, builder, backend):
        del build_result, builder, backend

        if term.operator_kind == "kinetic":
            value = float(term.term_id)

        elif term.operator_kind == "potential":
            value = 10.0 * float(term.term_id)

        else:
            raise AssertionError(term.operator_kind)

        return sp.csr_array(value * np.eye(2, dtype=np.complex128))


def test_component_decomposition_hamiltonian_outside_appends_kinetic_plus_potential():
    component_jump = sp.csr_array(np.eye(2, dtype=np.complex128))
    model = _FakeHamiltonianModel()

    kinetic_term = LocalTermDescriptor(
        term_id=3,
        term_kind="plaquette",
        operator_kind="kinetic",
        support_links=(0, 1),
    )
    potential_term = LocalTermDescriptor(
        term_id=3,
        term_kind="plaquette",
        operator_kind="potential",
        support_links=(0, 1),
    )

    jumps = _build_component_decomposition_jump_operators(
        model=model,
        build_result=None,
        component_jumps=(component_jump,),
        jump_kinetic_terms=(kinetic_term,),
        potential_terms_by_plaquette_id={3: potential_term},
        builder="sparse",
        backend="scipy",
        jump_operator_design="hamiltonian_outside_monitor_inside",
    )

    assert len(jumps) == 2

    expected_outside = 33.0 * np.eye(2, dtype=np.complex128)
    np.testing.assert_allclose(jumps[1].toarray(), expected_outside)


def test_partition_plaquette_terms_by_region_separates_inside_crossing_outside():
    terms = (
        LocalTermDescriptor(
            term_id=0,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(0, 1),
        ),
        LocalTermDescriptor(
            term_id=1,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(1, 2),
        ),
        LocalTermDescriptor(
            term_id=2,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(3, 4),
        ),
    )

    inside, outside, crossing = _partition_plaquette_terms_by_region(
        terms,
        variable_index_set=frozenset({0, 1}),
    )

    assert tuple(term.term_id for term in inside) == (0,)
    assert tuple(term.term_id for term in outside) == (2,)
    assert tuple(term.term_id for term in crossing) == (1,)


def test_group_reduced_iz_reports_for_monitor_exact_and_connected_supports():
    report_0 = _zero_report(zero_index=0, local_mask=(True, True, False, False))
    report_1 = _zero_report(zero_index=1, local_mask=(False, True, True, False))
    report_2 = _zero_report(zero_index=2, local_mask=(False, False, False, True))
    report_3 = _zero_report(zero_index=3, local_mask=(True, True, False, False))

    exact_groups = _group_reduced_iz_reports_for_monitor(
        (report_0, report_1, report_2, report_3),
        decomposition="exact_support",
    )
    connected_groups = _group_reduced_iz_reports_for_monitor(
        (report_0, report_1, report_2, report_3),
        decomposition="connected_support",
    )

    assert [[report.zero_index for report in group] for group in exact_groups] == [
        [0, 3],
        [1],
        [2],
    ]
    assert [[report.zero_index for report in group] for group in connected_groups] == [
        [0, 1, 3],
        [2],
    ]


def test_reduced_iz_monitor_from_reports_reuses_assembly_cache():
    transitions_a = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0 + 0.0j,
        ),
    )
    transitions_b = (
        LocalTransitionPattern(
            source_local=(1,),
            target_local=(0,),
            matrix_element=3.0 + 0.0j,
        ),
    )
    report_a = _zero_report(
        zero_index=0,
        local_mask=(True,),
        transitions=transitions_a,
    )
    report_b = _zero_report(
        zero_index=1,
        local_mask=(True,),
        transitions=transitions_b,
    )
    basis_configs = np.array([[0], [1]], dtype=np.int64)
    config_to_index = {(0,): 0, (1,): 1}
    assembly_cache = _ReducedIZAssemblyCache(
        basis_configs=basis_configs,
        config_to_index=config_to_index,
    )

    matrix = _build_reduced_iz_monitor_from_reports(
        reports=(report_a, report_b),
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        shape=(2, 2),
        use_collective_coefficients=False,
        assembly_cache=assembly_cache,
    )

    np.testing.assert_allclose(
        matrix.toarray(),
        np.array([[0.0, 3.0], [2.0, 0.0]], dtype=np.complex128),
    )
    assert len(assembly_cache._source_groups_by_mask) == 1
    assert len(assembly_cache._transition_indices_by_pattern) == 2


def test_reduced_iz_monitor_from_reports_aggregates_duplicate_transitions():
    transitions = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0 + 0.0j,
        ),
    )
    report_a = _zero_report(
        zero_index=0,
        local_mask=(True,),
        transitions=transitions,
    )
    report_b = _zero_report(
        zero_index=1,
        local_mask=(True,),
        transitions=transitions,
    )
    basis_configs = np.array([[0], [1]], dtype=np.int64)
    config_to_index = {(0,): 0, (1,): 1}
    assembly_cache = _ReducedIZAssemblyCache(
        basis_configs=basis_configs,
        config_to_index=config_to_index,
    )

    matrix = _build_reduced_iz_monitor_from_reports(
        reports=(report_a, report_b),
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        shape=(2, 2),
        use_collective_coefficients=False,
        assembly_cache=assembly_cache,
    )

    np.testing.assert_allclose(
        matrix.toarray(),
        np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.complex128),
    )
    assert len(assembly_cache._transition_indices_by_pattern) == 1


def test_group_local_transitions_by_source_and_reduced_iz_matrix():
    transitions = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0 + 0.0j,
        ),
        LocalTransitionPattern(
            source_local=(1,),
            target_local=(0,),
            matrix_element=3.0 + 0.0j,
        ),
    )
    grouped = _group_local_transitions_by_source(transitions)

    assert tuple(grouped) == ((0,), (1,))
    assert _transition_pattern_key(np.array([1], dtype=np.int64)) == (1,)

    basis_configs = np.array([[0], [1]], dtype=np.int64)
    zero_report = _zero_report(
        zero_index=0,
        local_mask=(True,),
        transitions=transitions,
    )
    matrix = _build_reduced_iz_operator_matrix(
        zero_report=zero_report,
        basis_configs=basis_configs,
        config_to_index={(0,): 0, (1,): 1},
        shape=(2, 2),
    )

    np.testing.assert_allclose(
        matrix.toarray(),
        np.array([[0.0, 3.0], [2.0, 0.0]], dtype=np.complex128),
    )


def test_reduced_iz_monitor_state_uses_cached_classification_actions():
    state = np.array([1.0, 0.0], dtype=np.complex128)
    report_a = _zero_report(
        zero_index=0,
        local_mask=(True,),
        reduced_action_vector=np.array([0.0, 2.0], dtype=np.complex128),
    )
    report_b = _zero_report(
        zero_index=1,
        local_mask=(True,),
        reduced_action_vector=np.array([3.0, 0.0], dtype=np.complex128),
    )

    action = _reduced_iz_monitor_state_from_reports(
        reports=(report_a, report_b),
        state=state,
        use_collective_coefficients=False,
    )

    np.testing.assert_allclose(action, np.array([3.0, 2.0], dtype=np.complex128))


def test_lazy_reduced_iz_monitor_materializes_only_when_requested():
    transitions = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0 + 0.0j,
        ),
    )
    report = _zero_report(
        zero_index=0,
        local_mask=(True,),
        transitions=transitions,
        reduced_action_vector=np.array([0.0, 2.0], dtype=np.complex128),
    )
    basis_configs = np.array([[0], [1]], dtype=np.int64)
    config_to_index = {(0,): 0, (1,): 1}

    monitor = _LazyReducedIZMonitorOperator(
        reports=(report,),
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        shape=(2, 2),
        use_collective_coefficients=False,
        _state_action=np.array([0.0, 2.0], dtype=np.complex128),
    )
    total = _LazySparseSumOperator(
        terms=(monitor,),
        shape=(2, 2),
        _state_action=np.array([0.0, 2.0], dtype=np.complex128),
    )

    assert monitor._matrix is None
    assert total._matrix is None
    np.testing.assert_allclose(
        _reduced_iz_monitor_state_from_reports(
            reports=(report,),
            state=np.array([1.0, 0.0], dtype=np.complex128),
            use_collective_coefficients=False,
        ),
        np.array([0.0, 2.0], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        total.tocsr().toarray(),
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.complex128),
    )
    assert monitor._matrix is not None
    assert total._matrix is not None


def test_cage_lindblad_construction_recycler_readouts_describe_selected_recycler():
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    zero = sp.csr_array((2, 2), dtype=np.complex128)
    jump = sp.csr_array(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))
    density_matrix = np.diag([1.0, 0.0]).astype(np.complex128)
    reduced_density_matrix = LocalReducedDensityMatrix(
        variable_indices=(0,),
        local_patterns=((0,), (1,)),
        density_matrix=density_matrix,
        eigenvalues=np.asarray([0.0, 1.0], dtype=np.float64),
        support_basis=np.asarray([[1.0], [0.0]], dtype=np.complex128),
        null_basis=np.asarray([[0.0], [1.0]], dtype=np.complex128),
    )
    candidate = LocalRecyclingCandidate(
        variable_indices=(0,),
        alpha_index=0,
        beta_index=0,
        jump=jump,
        target_residual=0.0,
        inflow_norm=1.0,
        outflow_norm=0.0,
        projector_commutator_norm=1.0,
        local_alpha_vector=np.asarray([1.0, 0.0], dtype=np.complex128),
        local_beta_vector=np.asarray([0.0, 1.0], dtype=np.complex128),
    )
    scan_result = LocalRecyclingScanResult(
        reduced_density_matrix=reduced_density_matrix,
        candidates=(candidate,),
    )
    selection = LocalRecyclingSelection(
        candidate=candidate,
        two_pattern_structure=None,
        score=(0.0,),
    )
    recycling_build_result = LocalRecyclingBuildResult(
        scan_results=(scan_result,),
        selections=(selection,),
    )

    construction = CageLindbladConstruction(
        cage_state=ket0,
        region=_FakeRegion(),  # type: ignore[arg-type]
        z_value=None,
        inside_plaquette_ids=(),
        outside_plaquette_ids=(),
        crossing_plaquette_ids=(),
        monitor=zero,
        jumps=(jump,),
        n_jumps=1,
        n_component_jumps=1,
        n_global_jump_terms=0,
        open_system_backend="scipy",
        monitor_source="reduced_iz_operators",
        reduced_iz_monitor_decomposition="single_sum",
        reduced_iz_monitor_content="offdiagonal_only",
        n_reduced_iz_monitor_terms=0,
        reduced_iz_monitor_zero_indices=(),
        monitor_components=(),
        component_z_values=(),
        jump_operator_design="monitor_recycler",
        monitor_plaquette_policy="strict_inside",
        jump_plaquette_policy="outside_or_crossing",
        monitor_plaquette_ids=(),
        jump_plaquette_ids=(),
        kinetic_terms_monitor=(),
        potential_terms_monitor=(),
        kinetic_terms_jump=(),
        recycling_jump_source="local_rdm_rank_one",
        n_recycling_jumps=1,
        recycling_jump_variable_indices=((0,),),
        recycling_jump_alpha_beta_indices=((0, 0),),
        recycling_two_pattern_count=0,
        recycling_build_result=recycling_build_result,
        monitor_residual=0.0,
        max_jump_residual=0.0,
        jump_residuals=(0.0,),
        liouvillian_residual=0.0,
    )

    readout = construction.recycler_readouts()[0]

    assert readout.variable_indices == (0,)
    assert readout.alpha_support_patterns == ((0,),)
    assert readout.beta_support_patterns == ((1,),)
    assert readout.n_matrix_unit_terms == 1
    assert readout.matrix_unit_terms[0].target_pattern == (0,)
    assert readout.matrix_unit_terms[0].source_pattern == (1,)
    assert construction.recycler_summary()[0]["jump_nnz"] == 1
