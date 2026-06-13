from __future__ import annotations

import numpy as np
import pytest

from qlinks.basis import basis_configs_from_basis
from qlinks.caging.classification import (
    CageClassificationConfig,
    InterferenceZeroReport,
    LocalTransitionPattern,
    _annotate_probe_mechanisms,
    _apply_reduced_local_operator,
    _build_config_to_index,
    _classify_from_zero_reports,
    _group_local_transitions_by_source,
    classify_cage_state,
    classify_full_state,
    group_reduced_iz_monitor_reports,
    select_reduced_iz_monitor_reports,
    support_key_for_zero_report,
)
from qlinks.caging.results import CageState
from tests.helpers.states import (
    config_index,
    empty_complex_array,
    empty_int_array,
)


def _zero_indices(indices: np.ndarray) -> set[int]:
    return {int(index) for index in indices}


def _minimal_zero_report(
    *,
    zero_index: int,
    q_sector_weight: float,
    complement_targets: tuple[int, ...],
    complement_action_norm: float = 0.0,
    source_projector_like: bool = False,
    complement_support: tuple[int, ...] = (),
    complement_contributing_inputs: tuple[int, ...] = (),
    projector_like_annihilated_inputs: tuple[int, ...] = (),
) -> InterferenceZeroReport:
    """Build a minimal InterferenceZeroReport for annotation tests."""
    complement_target_indices = np.array(
        complement_targets,
        dtype=np.int64,
    )

    has_nonzero_complement_action = complement_action_norm > 0.0
    nonzero_complement_action_target_indices = (
        complement_target_indices.copy() if has_nonzero_complement_action else empty_int_array()
    )

    return InterferenceZeroReport(
        zero_index=zero_index,
        active_neighbors=empty_int_array(),
        active_matrix_elements=empty_complex_array(),
        active_amplitudes=empty_complex_array(),
        cancellation_residual=0.0,
        common_mask=np.array([True], dtype=np.bool_),
        local_mask=np.array([False], dtype=np.bool_),
        q_sector_weight=q_sector_weight,
        reduced_action_norm=0.0,
        complement_action_norm=complement_action_norm,
        complement_target_indices=complement_target_indices,
        explained_complement_target_indices=empty_int_array(),
        unexplained_complement_target_indices=complement_target_indices,
        complement_targets_are_known_zeros=False,
        trivial_target_indices=empty_int_array(),
        known_nonprojector_iz_target_indices=empty_int_array(),
        projector_like_iz_target_indices=empty_int_array(),
        unexpected_target_indices=empty_int_array(),
        has_unexpected_targets=False,
        has_nonzero_complement_action=has_nonzero_complement_action,
        unexpected_target_probe_failure_indices=empty_int_array(),
        nonzero_complement_action_target_indices=(nonzero_complement_action_target_indices),
        source_projector_like=source_projector_like,
        probe_mechanism_label="unexplained_leakage",
        local_transitions=(),
        complement_support_indices=np.array(
            complement_support,
            dtype=np.int64,
        ),
        complement_contributing_input_indices=np.array(
            complement_contributing_inputs,
            dtype=np.int64,
        ),
        projector_like_annihilated_input_indices=np.array(
            projector_like_annihilated_inputs,
            dtype=np.int64,
        ),
    )


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


def test_classify_full_state_finds_regional_candidate(
    classification_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    config = classification_config

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.label == "regional_candidate"
    assert report.support_size == 2
    assert report.hilbert_size == 8
    assert report.support_fraction == pytest.approx(2.0 / 8.0)

    assert report.n_nontrivial_zeros == 1
    assert report.n_distinct_local_patterns == 1

    zero_report = report.zero_reports[0]

    assert zero_report.zero_index == indices["h"]
    assert set(int(i) for i in zero_report.active_neighbors) == {
        indices["v1"],
        indices["v2"],
    }

    assert zero_report.cancellation_residual <= config.cancellation_tolerance

    # For |000>, |010>, |001>, only the first variable is common.
    assert zero_report.common_mask.tolist() == [True, False, False]
    assert zero_report.local_mask.tolist() == [False, True, True]
    assert zero_report.local_region_size == 2

    assert support_key_for_zero_report(zero_report) == (1, 2)
    assert report.reduced_iz_region_variable_indices == (1, 2)
    assert report.n_reduced_iz_probe_supports == 1
    assert report.reduced_iz_probe_supports[0].zero_index == indices["h"]
    assert report.reduced_iz_probe_supports[0].variable_indices == (1, 2)
    assert report.reduced_iz_component_groups(decomposition="single_sum")[0].zero_indices == (
        indices["h"],
    )
    assert report.reduced_iz_component_groups(decomposition="exact_support")[
        0
    ].support_variables == (1, 2)
    assert report.reduced_iz_component_groups(decomposition="connected_support")[
        0
    ].support_variables == (1, 2)
    assert select_reduced_iz_monitor_reports(report) == report.zero_reports
    assert group_reduced_iz_monitor_reports(
        report.zero_reports,
        decomposition="exact_support",
    ) == (report.zero_reports,)

    # No wavefunction weight lives outside the common beta sector.
    assert zero_report.q_sector_weight <= config.action_tolerance
    assert zero_report.complement_action_norm <= config.action_tolerance
    assert zero_report.reduced_action_norm <= config.action_tolerance

    assert report.n_complement_targets == 0
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_zeros_with_closed_complement_targets == pytest.approx(0.0)

    assert report.n_q_empty_source_probes == 1
    assert report.n_closed_by_known_zero_network_source_probes == 0
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_regional_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.q_empty_source_zero_indices) == {indices["h"]}
    assert _zero_indices(report.closed_by_known_zero_network_source_zero_indices) == set()
    assert _zero_indices(report.projector_like_source_zero_indices) == set()
    assert _zero_indices(report.invalid_source_zero_indices) == set()

    zero_report = report.zero_reports[0]
    assert zero_report.n_complement_targets == 0
    assert zero_report.n_unexplained_complement_targets == 0
    assert not zero_report.complement_targets_are_known_zeros
    assert zero_report.probe_mechanism_label == "q_empty"
    assert zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert not zero_report.is_invalid_probe
    assert not zero_report.source_projector_like

    assert report.n_trivial_targets == 0
    assert report.n_known_nonprojector_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert zero_report.n_trivial_targets == 0
    assert zero_report.n_known_nonprojector_iz_targets == 0
    assert zero_report.n_projector_like_iz_targets == 0
    assert zero_report.n_unexpected_targets == 0

    assert not zero_report.has_unexpected_targets
    assert not zero_report.has_nonzero_complement_action
    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()

    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 0

    rendered = report.to_text()
    assert "Reduced-IZ monitor cache" in rendered
    assert "exact_support groups" in rendered


def test_classify_full_state_regional_when_complement_targets_are_known_zeros(
    classification_config, two_zero_closed_interference_case
):
    """
    Q-sector weight is nonzero, but the complement action of each zero
    lands only on the other known nontrivial interference zero.

    This is the model-free closure criterion for regional_candidate.
    """
    basis_configs, kinetic, indices = two_zero_closed_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)

    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[indices["w1"]] = 0.5
    state[indices["w2"]] = -0.5

    config = classification_config

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.label == "regional_candidate"
    assert report.support_size == 4
    assert report.n_nontrivial_zeros == 2
    assert report.n_complement_targets == 2
    assert report.n_unexplained_complement_targets == 0

    reports_by_zero = {
        int(zero_report.zero_index): zero_report for zero_report in report.zero_reports
    }

    h0_report = reports_by_zero[indices["h0"]]
    h1_report = reports_by_zero[indices["h1"]]

    assert report.reduced_iz_region_variable_indices == (1, 2)
    assert [
        group.zero_indices
        for group in report.reduced_iz_component_groups(decomposition="exact_support")
    ] == [(indices["h0"], indices["h1"])]
    assert [
        group.zero_indices
        for group in report.reduced_iz_component_groups(decomposition="connected_support")
    ] == [(indices["h0"], indices["h1"])]

    assert h0_report.q_sector_weight == pytest.approx(0.5)
    assert h1_report.q_sector_weight == pytest.approx(0.5)

    assert set(int(i) for i in h0_report.complement_target_indices) == {indices["h1"]}
    assert set(int(i) for i in h1_report.complement_target_indices) == {indices["h0"]}

    assert h0_report.n_unexplained_complement_targets == 0
    assert h1_report.n_unexplained_complement_targets == 0

    assert h0_report.complement_action_norm <= config.action_tolerance
    assert h1_report.complement_action_norm <= config.action_tolerance

    assert report.n_q_empty_source_probes == 0
    assert report.n_closed_by_known_zero_network_source_probes == 2
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_regional_source_probes == 2
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.q_empty_source_zero_indices) == set()
    assert _zero_indices(report.closed_by_known_zero_network_source_zero_indices) == {
        indices["h0"],
        indices["h1"],
    }
    assert _zero_indices(report.projector_like_source_zero_indices) == set()
    assert _zero_indices(report.invalid_source_zero_indices) == set()

    assert h0_report.probe_mechanism_label == "closed_by_known_zeros"
    assert h1_report.probe_mechanism_label == "closed_by_known_zeros"

    assert not h0_report.is_q_empty
    assert not h1_report.is_q_empty
    assert not h0_report.is_projector_like
    assert not h1_report.is_projector_like
    assert not h0_report.is_invalid_probe
    assert not h1_report.is_invalid_probe
    assert not h0_report.source_projector_like
    assert not h1_report.source_projector_like

    assert _zero_indices(h0_report.known_nonprojector_iz_target_indices) == {indices["h1"]}
    assert _zero_indices(h1_report.known_nonprojector_iz_target_indices) == {indices["h0"]}

    assert _zero_indices(h0_report.projector_like_iz_target_indices) == set()
    assert _zero_indices(h1_report.projector_like_iz_target_indices) == set()
    assert _zero_indices(h0_report.trivial_target_indices) == set()
    assert _zero_indices(h1_report.trivial_target_indices) == set()
    assert _zero_indices(h0_report.unexpected_target_indices) == set()
    assert _zero_indices(h1_report.unexpected_target_indices) == set()

    assert report.n_trivial_targets == 0
    assert report.n_known_nonprojector_iz_targets == 2
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0


def test_classify_full_state_treats_trivial_target_cancellation_as_regional(
    classification_config, pairwise_interference_case
):
    """
    Add amplitudes in the Q_beta sector.

    The same local Z pattern acts on

        |110> and |101>

    and maps them toward |100>. Their amplitudes cancel there.

    The target |100> is a trivial zero of the parent kinetic graph.
    Because the complement action cancels there, this is regional closure
    through a trivial target.
    """
    basis_configs, kinetic, indices = pairwise_interference_case

    w1 = config_index(basis_configs, (1, 1, 0))
    w2 = config_index(basis_configs, (1, 0, 1))
    trivial_target = config_index(basis_configs, (1, 0, 0))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[w1] = 0.5
    state[w2] = -0.5

    config = classification_config

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.label == "regional_candidate"
    assert report.support_size == 4
    assert report.n_nontrivial_zeros == 1
    assert report.n_complement_targets == 1
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_zeros_with_closed_complement_targets == pytest.approx(1.0)

    zero_report = report.zero_reports[0]

    assert zero_report.q_sector_weight == pytest.approx(0.5)
    assert zero_report.complement_action_norm <= config.action_tolerance

    assert set(int(i) for i in zero_report.complement_target_indices) == {trivial_target}
    assert _zero_indices(zero_report.explained_complement_target_indices) == {trivial_target}
    assert _zero_indices(zero_report.unexplained_complement_target_indices) == set()
    assert zero_report.complement_targets_are_known_zeros

    assert report.n_q_empty_source_probes == 0
    assert report.n_closed_by_known_zero_network_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_regional_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.closed_by_known_zero_network_source_zero_indices) == {indices["h"]}
    assert _zero_indices(report.invalid_source_zero_indices) == set()

    assert zero_report.probe_mechanism_label == "closed_by_known_zeros"
    assert zero_report.is_closed_by_known_zeros
    assert not zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert not zero_report.is_invalid_probe

    assert not zero_report.has_unexpected_targets
    assert not zero_report.has_nonzero_complement_action

    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 0
    assert _zero_indices(report.unexpected_target_probe_failure_indices) == set()
    assert _zero_indices(report.nonzero_complement_action_probe_failure_indices) == set()

    assert report.n_trivial_targets == 1
    assert report.n_known_nonprojector_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert _zero_indices(zero_report.trivial_target_indices) == {trivial_target}
    assert _zero_indices(zero_report.known_nonprojector_iz_target_indices) == set()
    assert _zero_indices(zero_report.projector_like_iz_target_indices) == set()
    assert _zero_indices(zero_report.unexpected_target_indices) == set()
    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()


def test_classify_full_state_marks_nonzero_unexplained_leakage_invalid(
    classification_config, pairwise_interference_case
):
    """
    Add Q-sector weight without the partner needed for cancellation.

    Then the complement action of the same local Z pattern is nonzero,
    so the classification should not be regional or extended.
    """
    basis_configs, kinetic, indices = pairwise_interference_case

    w1 = config_index(basis_configs, (1, 1, 0))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(3.0)
    state[indices["v2"]] = -1.0 / np.sqrt(3.0)
    state[w1] = 1.0 / np.sqrt(3.0)

    config = classification_config

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.label == "invalid_or_inconsistent"
    assert report.n_nontrivial_zeros == 1

    trivial_target = config_index(basis_configs, (1, 0, 0))

    zero_report = report.zero_reports[0]

    assert zero_report.probe_mechanism_label == "unexplained_leakage"
    assert zero_report.is_invalid_probe
    assert zero_report.q_sector_weight > config.action_tolerance
    assert zero_report.complement_action_norm > config.action_tolerance

    assert not zero_report.has_unexpected_targets
    assert zero_report.has_nonzero_complement_action

    assert report.n_invalid_source_probes == 1
    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 1

    assert _zero_indices(report.unexpected_target_probe_failure_indices) == set()
    assert _zero_indices(report.nonzero_complement_action_probe_failure_indices) == {indices["h"]}

    assert _zero_indices(zero_report.trivial_target_indices) == {trivial_target}
    assert set(int(i) for i in zero_report.complement_target_indices) == {trivial_target}

    assert _zero_indices(zero_report.unexpected_target_indices) == set()
    assert _zero_indices(zero_report.unexplained_complement_target_indices) == set()

    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == {trivial_target}

    assert report.n_trivial_targets == 1
    assert report.n_known_nonprojector_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0


def test_classify_cage_state_lifts_compact_state_and_preserves_metadata(
    classification_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    cage_state = CageState(
        energy=0.0 + 0.0j,
        local_state=np.array(
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
            dtype=np.complex128,
        ),
        support=np.array(
            [indices["v1"], indices["v2"]],
            dtype=np.int64,
        ),
        boundary_residual=0.0,
        eigen_residual=0.0,
        full_residual=0.0,
    )

    report = classify_cage_state(
        cage_state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        hilbert_size=basis_configs.shape[0],
        config=classification_config,
    )

    assert report.label == "regional_candidate"
    assert report.support_size == 2
    assert report.n_nontrivial_zeros == 1
    assert report.n_complement_targets == 0
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_zeros_with_closed_complement_targets == pytest.approx(0.0)

    assert report.metadata["energy"] == cage_state.energy
    assert report.metadata["support_size"] == cage_state.support_size
    assert report.metadata["boundary_residual"] == cage_state.boundary_residual
    assert report.metadata["eigen_residual"] == cage_state.eigen_residual
    assert report.metadata["full_residual"] == cage_state.full_residual

    assert report.n_q_empty_source_probes == 1
    assert report.n_closed_by_known_zero_network_source_probes == 0
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_regional_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    zero_report = report.zero_reports[0]
    assert zero_report.probe_mechanism_label == "q_empty"
    assert zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert not zero_report.is_invalid_probe
    assert not zero_report.source_projector_like

    assert report.n_trivial_targets == 0
    assert report.n_known_nonprojector_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert not zero_report.has_unexpected_targets
    assert not zero_report.has_nonzero_complement_action
    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()

    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 0


def test_classify_full_state_ignores_trivial_zeros_without_active_neighbors(
    classification_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=classification_config,
    )

    zero_indices = {int(zero_report.zero_index) for zero_report in report.zero_reports}

    # Only |000> is connected to active support and has nontrivial
    # cancellation. Other zero-amplitude basis states are trivial zeros.
    assert zero_indices == {indices["h"]}

    assert report.n_nontrivial_zeros == 1
    assert report.n_q_empty_source_probes == 1
    assert report.n_regional_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0


def test_classify_full_state_rejects_wrong_basis_shape(pairwise_interference_case):
    basis_configs, kinetic, _indices = pairwise_interference_case
    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)

    bad_basis_configs = np.zeros(8, dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="basis_configs must have shape",
    ):
        classify_full_state(
            state,
            kinetic_matrix=kinetic,
            basis_configs=bad_basis_configs,
        )


def test_classify_full_state_rejects_mismatched_basis_size(pairwise_interference_case):
    basis_configs, kinetic, _indices = pairwise_interference_case
    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)

    bad_basis_configs = basis_configs[:-1]

    with pytest.raises(
        ValueError,
        match="basis_configs.shape\\[0\\] must match full_state.size",
    ):
        classify_full_state(
            state,
            kinetic_matrix=kinetic,
            basis_configs=bad_basis_configs,
        )


def test_classify_full_state_detects_projector_like_extended_mechanism(
    classification_config, pairwise_interference_case
):
    """Finite Q-sector weight with no complement targets is extended-like.

    The active interference zero is still |000>, with active neighbors
    |010> and |001>. We add amplitude on |111>, which lies outside the
    common beta sector but does not match the local source patterns of the
    reduced interference-zero operator. Therefore the complement operator
    has no raw target vertices.
    """
    basis_configs, kinetic, indices = pairwise_interference_case

    q_sector_state = config_index(basis_configs, (1, 1, 1))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(3.0)
    state[indices["v2"]] = -1.0 / np.sqrt(3.0)
    state[q_sector_state] = 1.0 / np.sqrt(3.0)

    config = classification_config

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.label == "extended_candidate"
    assert report.support_size == 3
    assert report.n_nontrivial_zeros == 1

    assert report.n_complement_targets == 0
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_zeros_with_closed_complement_targets == pytest.approx(0.0)

    assert report.n_q_empty_source_probes == 0
    assert report.n_closed_by_known_zero_network_source_probes == 0
    assert report.n_projector_like_source_probes == 1
    assert report.n_invalid_source_probes == 0

    assert report.n_regional_source_probes == 0
    assert report.n_projector_like_source_probes == 1
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.projector_like_source_zero_indices) == {indices["h"]}
    assert _zero_indices(report.invalid_source_zero_indices) == set()

    zero_report = report.zero_reports[0]
    assert zero_report.zero_index == indices["h"]
    assert zero_report.q_sector_weight > config.action_tolerance
    assert zero_report.n_complement_targets == 0
    assert zero_report.n_unexplained_complement_targets == 0
    assert not zero_report.complement_targets_are_known_zeros
    assert zero_report.complement_action_norm <= config.action_tolerance

    assert zero_report.probe_mechanism_label == "projector_like"
    assert not zero_report.is_q_empty
    assert zero_report.is_projector_like
    assert not zero_report.is_invalid_probe
    assert zero_report.source_projector_like

    assert report.n_trivial_targets == 0
    assert report.n_known_nonprojector_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert zero_report.n_trivial_targets == 0
    assert zero_report.n_known_nonprojector_iz_targets == 0
    assert zero_report.n_projector_like_iz_targets == 0
    assert zero_report.n_unexpected_targets == 0

    assert not zero_report.has_unexpected_targets
    assert not zero_report.has_nonzero_complement_action

    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 0

    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()


def test_classification_report_summary_dict_is_stable(
    classification_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=classification_config,
    )

    summary = report.to_summary_dict()

    assert summary["Reduced IZ probe mechanisms"]["q-empty source probes"] == 1
    assert summary["Reduced IZ probe mechanisms"]["closed-by-known-zero-network source probes"] == 0
    assert summary["Invalid probe reasons"]["unexpected-target source probes"] == 0
    assert summary["Invalid probe reasons"]["nonzero-complement-action source probes"] == 0


def test_probe_mechanism_propagates_projector_like_target_dependence(classification_config):
    """A probe closing onto a projector-like IZ target is extended-like."""
    config = classification_config

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

    assert target.probe_mechanism_label == "projector_like"
    assert target.source_projector_like
    assert not target.has_unexpected_targets
    assert not target.has_nonzero_complement_action

    assert source.probe_mechanism_label == "projector_like"
    assert not source.source_projector_like
    assert not source.has_unexpected_targets
    assert not source.has_nonzero_complement_action
    assert _zero_indices(source.projector_like_iz_target_indices) == {projector_target_zero}
    assert _zero_indices(source.known_nonprojector_iz_target_indices) == set()
    assert _zero_indices(source.unexpected_target_indices) == set()
    assert _zero_indices(source.nonzero_complement_action_target_indices) == set()

    label = _classify_from_zero_reports(
        zero_reports=annotated,
        config=config,
    )

    assert label == "extended_candidate"


def test_probe_mechanism_keeps_trivial_and_destructive_closure_regional(classification_config):
    """Closure onto trivial zeros and non-projector IZs is regional."""
    config = classification_config

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

    assert source.probe_mechanism_label == "closed_by_known_zeros"
    assert target.probe_mechanism_label == "closed_by_known_zeros"

    assert not source.has_unexpected_targets
    assert not source.has_nonzero_complement_action
    assert not target.has_unexpected_targets
    assert not target.has_nonzero_complement_action

    assert _zero_indices(source.nonzero_complement_action_target_indices) == set()
    assert _zero_indices(source.known_nonprojector_iz_target_indices) == {destructive_target_zero}
    assert _zero_indices(source.projector_like_iz_target_indices) == set()
    assert _zero_indices(source.unexpected_target_indices) == set()

    assert _zero_indices(target.nonzero_complement_action_target_indices) == set()
    assert _zero_indices(target.trivial_target_indices) == {trivial_target}
    assert _zero_indices(target.projector_like_iz_target_indices) == set()
    assert _zero_indices(target.unexpected_target_indices) == set()

    label = _classify_from_zero_reports(
        zero_reports=annotated,
        config=config,
    )

    assert label == "regional_candidate"


def test_probe_mechanism_marks_unexpected_target_invalid(classification_config):
    config = classification_config

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

    label = _classify_from_zero_reports(
        zero_reports=annotated,
        config=config,
    )

    assert label == "invalid_or_inconsistent"

    assert report.has_unexpected_targets
    assert not report.has_nonzero_complement_action

    assert _zero_indices(report.unexpected_target_probe_failure_indices) == {unexpected_target}
    assert _zero_indices(report.nonzero_complement_action_target_indices) == set()
    assert _zero_indices(report.unexpected_target_indices) == {unexpected_target}


def test_classify_full_state_raises_without_sector_on_disconnected_graph(
    pairwise_interference_case,
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    config = CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
        sector_policy="raise_if_disconnected",
    )

    with pytest.raises(ValueError, match="disconnected"):
        classify_full_state(
            state,
            kinetic_matrix=kinetic,
            basis_configs=basis_configs,
            config=config,
        )


def test_classify_full_state_ignores_complement_targets_outside_sector_mask(
    pairwise_interference_case,
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    sector_mask = np.zeros(basis_configs.shape[0], dtype=np.bool_)
    sector_mask[[indices["h"], indices["v1"], indices["v2"]]] = True

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=CageClassificationConfig(
            amplitude_tolerance=1e-12,
            cancellation_tolerance=1e-12,
            action_tolerance=1e-12,
            sector_policy="raise_if_disconnected",
        ),
    )

    assert report.metadata["classification_domain_size"] == 3
    assert report.label == "regional_candidate"


def test_basis_configs_from_basis_uses_states_attribute():
    class DummyArrayBasis:
        states = np.array([[0, 1], [1, 0]], dtype=np.int8)

    configs = basis_configs_from_basis(DummyArrayBasis())

    np.testing.assert_array_equal(
        configs,
        np.array([[0, 1], [1, 0]], dtype=np.int8),
    )


def test_basis_configs_from_basis_uses_to_array_basis():
    class DummyArrayBasis:
        states = np.array([[0, 1], [1, 0]], dtype=np.int8)

    class DummyEncodedBasis:
        def to_array_basis(self):
            return DummyArrayBasis()

    configs = basis_configs_from_basis(DummyEncodedBasis())

    np.testing.assert_array_equal(
        configs,
        np.array([[0, 1], [1, 0]], dtype=np.int8),
    )


def test_mixed_projected_and_locally_cancelled_inputs_are_projector_like(classification_config):
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
        config=classification_config,
    )

    assert annotated[0].probe_mechanism_label == "projector_like"
    assert annotated[0].is_projector_like
    assert annotated[0].source_projector_like
    assert not annotated[0].has_unexpected_targets
    assert not annotated[0].has_nonzero_complement_action
