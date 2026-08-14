import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.caging.analysis.environment import (
    EnvironmentRemovalProbeReport,
    LocalTransitionPattern,
    _active_frontier_zero_indices,
    _annotate_probe_mechanisms,
    _apply_reduced_local_operator,
    _build_config_to_index,
    _build_reduced_local_operator_application_context,
    _complement_support_indices,
    _find_trivial_zero_indices,
    _group_local_transitions_by_source,
    _q_sector_weight,
    support_key_for_zero_report,
)
from tests.caging.analysis._environment_helpers import (
    _minimal_zero_report,
    _zero_indices,
)
from tests.helpers.states import empty_complex_array, empty_int_array


def test_active_frontier_zero_indices_uses_incoming_active_columns() -> None:
    kinetic = sp.csr_array(
        (
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            (
                np.array([2, 0, 3], dtype=np.int64),
                np.array([0, 3, 1], dtype=np.int64),
            ),
        ),
        shape=(4, 4),
    )
    support_mask = np.array([True, False, False, False], dtype=np.bool_)
    domain_mask = np.ones(4, dtype=np.bool_)

    frontier = _active_frontier_zero_indices(
        kinetic,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_state_indices=np.array([0], dtype=np.int64),
    )

    # The frontier is defined by K[h, u] != 0 for active source column u.
    # Row 0 -> column 3 is an outgoing edge from the active vertex and should
    # not make 3 a zero candidate.
    np.testing.assert_array_equal(frontier, np.array([2], dtype=np.int64))


def test_find_trivial_zero_indices_uses_active_frontier_cache() -> None:
    kinetic = sp.csr_array(
        (
            np.array([1.0], dtype=np.float64),
            (np.array([2], dtype=np.int64), np.array([0], dtype=np.int64)),
        ),
        shape=(4, 4),
    )
    support_mask = np.array([True, False, False, False], dtype=np.bool_)
    domain_mask = np.ones(4, dtype=np.bool_)
    frontier = np.array([2], dtype=np.int64)

    cached = _find_trivial_zero_indices(
        np.zeros(4, dtype=np.complex128),
        kinetic,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_frontier_zero_indices=frontier,
    )

    assert cached == {1, 3}


def test_apply_reduced_local_operator_accepts_grouped_transitions() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    full_state = np.array([3.0, 0.0, 5.0, 0.0], dtype=np.complex128)
    local_mask = np.array([False, True], dtype=np.bool_)
    domain_mask = np.ones(basis_configs.shape[0], dtype=np.bool_)
    transitions = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0,
        ),
    )
    transition_lookup = _group_local_transitions_by_source(transitions)

    output, target_indices, input_indices = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=_build_config_to_index(basis_configs),
        local_mask=local_mask,
        local_transitions=transitions,
        local_transition_lookup=transition_lookup,
        domain_mask=domain_mask,
    )

    expected_output = np.array([0.0, 6.0, 0.0, 10.0], dtype=np.complex128)

    np.testing.assert_allclose(output, expected_output)
    np.testing.assert_array_equal(target_indices, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(input_indices, np.array([0, 2], dtype=np.int64))


def test_apply_reduced_local_operator_accepts_source_indices_cache() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    full_state = np.array([3.0, 7.0, 5.0, 11.0], dtype=np.complex128)
    local_mask = np.array([False, True], dtype=np.bool_)
    domain_mask = np.ones(basis_configs.shape[0], dtype=np.bool_)
    transitions = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0,
        ),
    )

    output, target_indices, input_indices = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=_build_config_to_index(basis_configs),
        local_mask=local_mask,
        local_transitions=transitions,
        domain_mask=domain_mask,
        source_indices=np.array([0, 2], dtype=np.int64),
    )

    expected_output = np.array([0.0, 6.0, 0.0, 10.0], dtype=np.complex128)

    np.testing.assert_allclose(output, expected_output)
    np.testing.assert_array_equal(target_indices, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(input_indices, np.array([0, 2], dtype=np.int64))


def test_apply_reduced_local_operator_uses_application_context() -> None:
    basis_configs = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    full_state = np.array([3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 11.0, 0.0], dtype=np.complex128)
    local_mask = np.array([False, False, True], dtype=np.bool_)
    domain_mask = np.ones(basis_configs.shape[0], dtype=np.bool_)
    transitions = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0,
        ),
    )
    context = _build_reduced_local_operator_application_context(
        basis_configs=basis_configs,
        domain_mask=domain_mask,
        local_mask=local_mask,
    )

    uncached = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=_build_config_to_index(basis_configs),
        local_mask=local_mask,
        local_transitions=transitions,
        domain_mask=domain_mask,
        source_indices=np.array([0, 2, 4, 6], dtype=np.int64),
    )
    cached = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=_build_config_to_index(basis_configs),
        local_mask=local_mask,
        local_transitions=transitions,
        domain_mask=domain_mask,
        application_context=context,
        source_indices=np.array([0, 2, 4, 6], dtype=np.int64),
    )

    np.testing.assert_allclose(cached[0], uncached[0])
    np.testing.assert_array_equal(cached[1], uncached[1])
    np.testing.assert_array_equal(cached[2], uncached[2])
    assert context.local_variable_indices == (2,)
    assert context.environment_variable_indices == (0, 1)


def test_apply_reduced_local_operator_context_respects_domain_targets() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    full_state = np.array([3.0, 0.0, 5.0, 0.0], dtype=np.complex128)
    local_mask = np.array([False, True], dtype=np.bool_)
    domain_mask = np.array([True, False, True, True], dtype=np.bool_)
    transitions = (
        LocalTransitionPattern(
            source_local=(0,),
            target_local=(1,),
            matrix_element=2.0,
        ),
    )
    context = _build_reduced_local_operator_application_context(
        basis_configs=basis_configs,
        domain_mask=domain_mask,
        local_mask=local_mask,
    )

    output, target_indices, input_indices = _apply_reduced_local_operator(
        full_state,
        basis_configs=basis_configs,
        config_to_index=_build_config_to_index(basis_configs),
        local_mask=local_mask,
        local_transitions=transitions,
        domain_mask=domain_mask,
        application_context=context,
    )

    np.testing.assert_allclose(output, np.array([0.0, 0.0, 0.0, 10.0], dtype=np.complex128))
    np.testing.assert_array_equal(target_indices, np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(input_indices, np.array([2], dtype=np.int64))


def test_q_sector_weight_uses_active_indices_cache(environment_reduction_config) -> None:
    basis_configs = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    full_state = np.array([0.0, 0.5, 0.0, 0.25j], dtype=np.complex128)
    common_mask = np.array([True, False, True], dtype=np.bool_)
    reference_config = basis_configs[0]

    uncached = _q_sector_weight(
        full_state,
        basis_configs=basis_configs,
        reference_config=reference_config,
        common_mask=common_mask,
        config=environment_reduction_config,
    )
    cached = _q_sector_weight(
        full_state,
        basis_configs=basis_configs,
        reference_config=reference_config,
        common_mask=common_mask,
        active_indices=np.array([1, 3], dtype=np.int64),
        config=environment_reduction_config,
    )

    assert cached == pytest.approx(uncached)
    assert cached == pytest.approx(abs(0.25j) ** 2)


def test_complement_support_indices_uses_active_domain_indices_cache() -> None:
    basis_configs = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    full_state = np.array([0.0, 0.5, 1.0, 0.25j], dtype=np.complex128)
    common_mask = np.array([True, False, True], dtype=np.bool_)
    reference_config = basis_configs[0]
    domain_mask = np.array([True, True, False, True], dtype=np.bool_)

    uncached = _complement_support_indices(
        full_state,
        basis_configs=basis_configs,
        reference_config=reference_config,
        common_mask=common_mask,
        domain_mask=domain_mask,
        amplitude_tolerance=0.0,
    )
    cached = _complement_support_indices(
        full_state,
        basis_configs=basis_configs,
        reference_config=reference_config,
        common_mask=common_mask,
        domain_mask=domain_mask,
        active_domain_indices=np.array([1, 3], dtype=np.int64),
        amplitude_tolerance=0.0,
    )

    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_array_equal(cached, np.array([3], dtype=np.int64))


def test_probe_mechanism_propagates_projector_like_target_dependence(environment_reduction_config):
    """A probe closing onto a domain-blocked IZ target remains safely removable."""
    config = environment_reduction_config

    source_zero = 10
    projector_target_zero = 20

    source_report = _minimal_zero_report(
        zero_index=source_zero,
        q_sector_weight=1.0,
        complement_targets=(projector_target_zero,),
        source_projector_like=False,
    )
    target_report = _minimal_zero_report(
        zero_index=projector_target_zero,
        q_sector_weight=1.0,
        complement_targets=(),
        source_projector_like=True,
    )

    annotated = _annotate_probe_mechanisms(
        [source_report, target_report],
        trivial_zero_indices=set(),
        config=config,
    )

    reports_by_zero = {int(report.zero_index): report for report in annotated}

    source = reports_by_zero[source_zero]
    target = reports_by_zero[projector_target_zero]

    assert target.probe_mechanism_label == "domain_blocked"
    assert target.source_projector_like
    assert not target.has_unexpected_targets
    assert not target.has_nonzero_complement_action

    assert source.probe_mechanism_label == "projector_like"
    assert not source.source_projector_like
    assert not source.has_unexpected_targets
    assert not source.has_nonzero_complement_action
    assert _zero_indices(source.projector_like_iz_target_indices) == {projector_target_zero}
    assert _zero_indices(source.same_pattern_iz_target_indices) == set()
    assert _zero_indices(source.unexpected_target_indices) == set()
    assert _zero_indices(source.nonzero_complement_action_target_indices) == set()

    assert all(report.is_safely_removable for report in annotated)


def test_probe_mechanism_keeps_trivial_and_same_pattern_closure_safe(environment_reduction_config):
    """Trivial targets and same-pattern IZ targets are safely removable."""
    config = environment_reduction_config

    source_zero = 10
    destructive_target_zero = 20
    trivial_target = 99

    source_report = _minimal_zero_report(
        zero_index=source_zero,
        q_sector_weight=1.0,
        complement_targets=(destructive_target_zero,),
        source_projector_like=False,
    )
    target_report = _minimal_zero_report(
        zero_index=destructive_target_zero,
        q_sector_weight=1.0,
        complement_targets=(trivial_target,),
        source_projector_like=False,
    )

    annotated = _annotate_probe_mechanisms(
        [source_report, target_report],
        trivial_zero_indices={trivial_target},
        config=config,
    )

    reports_by_zero = {int(report.zero_index): report for report in annotated}

    source = reports_by_zero[source_zero]
    target = reports_by_zero[destructive_target_zero]

    assert source.probe_mechanism_label == "closed_by_same_pattern_zeros"
    assert target.probe_mechanism_label == "closed_by_same_pattern_zeros"

    assert not source.has_unexpected_targets
    assert not source.has_nonzero_complement_action
    assert not target.has_unexpected_targets
    assert not target.has_nonzero_complement_action

    assert _zero_indices(source.nonzero_complement_action_target_indices) == set()
    assert _zero_indices(source.same_pattern_iz_target_indices) == {destructive_target_zero}
    assert _zero_indices(source.projector_like_iz_target_indices) == set()
    assert _zero_indices(source.unexpected_target_indices) == set()

    assert _zero_indices(target.nonzero_complement_action_target_indices) == set()
    assert _zero_indices(target.trivial_target_indices) == {trivial_target}
    assert _zero_indices(target.projector_like_iz_target_indices) == set()
    assert _zero_indices(target.unexpected_target_indices) == set()

    assert all(report.is_safely_removable for report in annotated)


def test_known_zero_with_different_local_pattern_is_not_safe_environment_removal(
    environment_reduction_config,
):
    """Known IZ status alone does not justify deleting the outer environment."""
    source_zero = 10
    mismatched_target_zero = 20
    local_mask = np.array([True], dtype=np.bool_)

    source_report = _minimal_zero_report(
        zero_index=source_zero,
        q_sector_weight=1.0,
        complement_targets=(mismatched_target_zero,),
        local_mask=local_mask,
        local_transitions=(LocalTransitionPattern((0,), (1,), 1.0 + 0.0j),),
    )
    target_report = _minimal_zero_report(
        zero_index=mismatched_target_zero,
        q_sector_weight=1.0,
        complement_targets=(),
        local_mask=local_mask,
        local_transitions=(LocalTransitionPattern((0,), (1,), -1.0 + 0.0j),),
    )

    annotated = _annotate_probe_mechanisms(
        [source_report, target_report],
        trivial_zero_indices=set(),
        config=environment_reduction_config,
    )
    reports_by_zero = {int(report.zero_index): report for report in annotated}
    source = reports_by_zero[source_zero]

    assert source.probe_mechanism_label == "unexplained_leakage"
    assert source.removal_mechanism == "unsafe"
    assert not source.is_safely_removable
    assert _zero_indices(source.same_pattern_iz_target_indices) == set()
    assert _zero_indices(source.unexpected_target_indices) == {mismatched_target_zero}


def test_probe_mechanism_marks_unexpected_target_invalid(environment_reduction_config):
    config = environment_reduction_config

    source_zero = 10
    unexpected_target = 77

    source_report = _minimal_zero_report(
        zero_index=source_zero,
        q_sector_weight=1.0,
        complement_targets=(unexpected_target,),
        source_projector_like=False,
    )

    annotated = _annotate_probe_mechanisms(
        [source_report],
        trivial_zero_indices=set(),
        config=config,
    )

    report = annotated[0]

    assert report.probe_mechanism_label == "unexplained_leakage"
    assert _zero_indices(report.unexpected_target_indices) == {unexpected_target}

    assert not all(report.is_safely_removable for report in annotated)

    assert report.has_unexpected_targets
    assert not report.has_nonzero_complement_action

    assert _zero_indices(report.unexpected_target_probe_failure_indices) == {unexpected_target}
    assert _zero_indices(report.nonzero_complement_action_target_indices) == set()
    assert _zero_indices(report.unexpected_target_indices) == {unexpected_target}


def test_mixed_projected_and_locally_cancelled_inputs_are_projector_like(
    environment_reduction_config,
):
    report = _minimal_zero_report(
        zero_index=14,
        q_sector_weight=2.0 / 3.0,
        complement_targets=(66,),
        complement_action_norm=0.0,
        complement_support=(24, 56, 60, 72),
        complement_contributing_inputs=(60, 72),
        projector_like_annihilated_inputs=(24, 56),
        source_projector_like=True,
    )

    annotated = _annotate_probe_mechanisms(
        [report],
        trivial_zero_indices={66},
        config=environment_reduction_config,
    )

    assert annotated[0].probe_mechanism_label == "projector_like"
    assert annotated[0].is_projector_like
    assert annotated[0].source_projector_like
    assert not annotated[0].has_unexpected_targets
    assert not annotated[0].has_nonzero_complement_action


def test_interference_zero_report_cached_local_variable_indices() -> None:
    report = EnvironmentRemovalProbeReport(
        zero_index=5,
        active_neighbors=empty_int_array(),
        active_matrix_elements=empty_complex_array(),
        active_amplitudes=empty_complex_array(),
        cancellation_residual=0.0,
        common_mask=np.array([False, True, False, True], dtype=np.bool_),
        local_mask=np.array([True, False, True, False], dtype=np.bool_),
        local_transitions=(),
        q_sector_weight=0.0,
        reduced_action_norm=0.0,
        complement_action_norm=0.0,
        complement_target_indices=empty_int_array(),
        explained_complement_target_indices=empty_int_array(),
        unexplained_complement_target_indices=empty_int_array(),
        complement_targets_are_known_zeros=True,
        trivial_target_indices=empty_int_array(),
        same_pattern_iz_target_indices=empty_int_array(),
        projector_like_iz_target_indices=empty_int_array(),
        unexpected_target_indices=empty_int_array(),
        complement_support_indices=empty_int_array(),
        complement_contributing_input_indices=empty_int_array(),
        projector_like_annihilated_input_indices=empty_int_array(),
        source_projector_like=False,
        has_unexpected_targets=False,
        has_nonzero_complement_action=False,
        unexpected_target_probe_failure_indices=empty_int_array(),
        nonzero_complement_action_target_indices=empty_int_array(),
        probe_mechanism_label="q_empty",
        local_variable_indices=(0, 2),
    )

    assert report.local_region_size == 2
    assert support_key_for_zero_report(report) == (0, 2)
