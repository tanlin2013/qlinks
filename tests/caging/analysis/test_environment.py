import numpy as np
import pytest

from qlinks.caging.analysis.environment import (
    EnvironmentReductionConfig,
    diagnose_cage_environment_reduction,
    diagnose_environment_reduction,
    group_reduced_iz_monitor_reports,
    select_reduced_iz_monitor_reports,
    support_key_for_zero_report,
)
from qlinks.caging.results import CageState
from tests.caging.analysis._environment_helpers import (
    _zero_indices,
)
from tests.helpers.states import config_index


def test_diagnose_environment_reduction_finds_safe_local_reduction(
    environment_reduction_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    config = environment_reduction_config

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.is_safely_removable
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
    single_group = report.reduced_iz_component_groups(decomposition="single_sum")[0]
    exact_group = report.reduced_iz_component_groups(decomposition="exact_support")[0]
    connected_group = report.reduced_iz_component_groups(decomposition="connected_support")[0]

    assert single_group.zero_indices == (indices["h"],)
    assert exact_group.support_variables == (1, 2)
    assert connected_group.support_variables == (1, 2)
    assert exact_group.has_state_action_vector
    np.testing.assert_allclose(
        exact_group.state_action_vector,
        zero_report.reduced_action_vector,
    )
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
    assert report.fraction_probes_safely_removable == pytest.approx(1.0)

    assert report.n_q_empty_source_probes == 1
    assert report.n_same_pattern_zero_closure_probes == 0
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.q_empty_source_zero_indices) == {indices["h"]}
    assert _zero_indices(report.same_pattern_zero_closure_indices) == set()
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
    assert report.n_same_pattern_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert zero_report.n_trivial_targets == 0
    assert zero_report.n_same_pattern_iz_targets == 0
    assert zero_report.n_projector_like_iz_targets == 0
    assert zero_report.n_unexpected_targets == 0

    assert not zero_report.has_unexpected_targets
    assert not zero_report.has_nonzero_complement_action
    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()

    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 0

    rendered = report.to_text()
    assert "Exterior-environment reduction" in rendered
    assert "no environment weight" in rendered


def test_diagnose_environment_reduction_safe_when_targets_share_local_pattern(
    environment_reduction_config, two_zero_closed_interference_case
):
    """
    Q-sector weight is nonzero, but the complement action of each zero
    lands only on the other known nontrivial interference zero.

    This is the model-free criterion for safe exterior-environment removal.
    """
    basis_configs, kinetic, indices = two_zero_closed_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)

    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[indices["w1"]] = 0.5
    state[indices["w2"]] = -0.5

    config = environment_reduction_config

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.is_safely_removable
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
    exact_groups = report.reduced_iz_component_groups(decomposition="exact_support")
    connected_groups = report.reduced_iz_component_groups(decomposition="connected_support")
    assert [group.zero_indices for group in exact_groups] == [(indices["h0"], indices["h1"])]
    assert [group.zero_indices for group in connected_groups] == [(indices["h0"], indices["h1"])]
    assert exact_groups[0].has_state_action_vector
    np.testing.assert_allclose(
        exact_groups[0].state_action_vector,
        h0_report.reduced_action_vector + h1_report.reduced_action_vector,
    )

    assert h0_report.q_sector_weight == pytest.approx(0.5)
    assert h1_report.q_sector_weight == pytest.approx(0.5)

    assert set(int(i) for i in h0_report.complement_target_indices) == {indices["h1"]}
    assert set(int(i) for i in h1_report.complement_target_indices) == {indices["h0"]}

    assert h0_report.n_unexplained_complement_targets == 0
    assert h1_report.n_unexplained_complement_targets == 0

    assert h0_report.complement_action_norm <= config.action_tolerance
    assert h1_report.complement_action_norm <= config.action_tolerance

    assert report.n_q_empty_source_probes == 0
    assert report.n_same_pattern_zero_closure_probes == 2
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.q_empty_source_zero_indices) == set()
    assert _zero_indices(report.same_pattern_zero_closure_indices) == {
        indices["h0"],
        indices["h1"],
    }
    assert _zero_indices(report.projector_like_source_zero_indices) == set()
    assert _zero_indices(report.invalid_source_zero_indices) == set()

    assert h0_report.probe_mechanism_label == "closed_by_same_pattern_zeros"
    assert h1_report.probe_mechanism_label == "closed_by_same_pattern_zeros"

    assert not h0_report.is_q_empty
    assert not h1_report.is_q_empty
    assert not h0_report.is_projector_like
    assert not h1_report.is_projector_like
    assert not h0_report.is_invalid_probe
    assert not h1_report.is_invalid_probe
    assert not h0_report.source_projector_like
    assert not h1_report.source_projector_like

    assert _zero_indices(h0_report.same_pattern_iz_target_indices) == {indices["h1"]}
    assert _zero_indices(h1_report.same_pattern_iz_target_indices) == {indices["h0"]}

    assert _zero_indices(h0_report.projector_like_iz_target_indices) == set()
    assert _zero_indices(h1_report.projector_like_iz_target_indices) == set()
    assert _zero_indices(h0_report.trivial_target_indices) == set()
    assert _zero_indices(h1_report.trivial_target_indices) == set()
    assert _zero_indices(h0_report.unexpected_target_indices) == set()
    assert _zero_indices(h1_report.unexpected_target_indices) == set()

    assert report.n_trivial_targets == 0
    assert report.n_same_pattern_iz_targets == 2
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0


def test_diagnose_environment_reduction_accepts_trivial_target_cancellation(
    environment_reduction_config, pairwise_interference_case
):
    """
    Add amplitudes in the Q_beta sector.

    The same local Z pattern acts on

        |110> and |101>

    and maps them toward |100>. Their amplitudes cancel there.

    The target |100> is a trivial zero of the parent kinetic graph.
    Because the complement action cancels there, the exterior target is safely
    removable through the same local cancellation pattern and a trivial target.
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

    config = environment_reduction_config

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.is_safely_removable
    assert report.support_size == 4
    assert report.n_nontrivial_zeros == 1
    assert report.n_complement_targets == 1
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_probes_safely_removable == pytest.approx(1.0)

    zero_report = report.zero_reports[0]

    assert zero_report.q_sector_weight == pytest.approx(0.5)
    assert zero_report.complement_action_norm <= config.action_tolerance

    assert set(int(i) for i in zero_report.complement_target_indices) == {trivial_target}
    assert _zero_indices(zero_report.explained_complement_target_indices) == {trivial_target}
    assert _zero_indices(zero_report.unexplained_complement_target_indices) == set()
    assert zero_report.complement_targets_are_known_zeros

    assert report.n_q_empty_source_probes == 0
    assert report.n_same_pattern_zero_closure_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.same_pattern_zero_closure_indices) == {indices["h"]}
    assert _zero_indices(report.invalid_source_zero_indices) == set()

    assert zero_report.probe_mechanism_label == "closed_by_same_pattern_zeros"
    assert zero_report.is_closed_by_same_pattern_zeros
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
    assert report.n_same_pattern_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert _zero_indices(zero_report.trivial_target_indices) == {trivial_target}
    assert _zero_indices(zero_report.same_pattern_iz_target_indices) == set()
    assert _zero_indices(zero_report.projector_like_iz_target_indices) == set()
    assert _zero_indices(zero_report.unexpected_target_indices) == set()
    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()


def test_diagnose_environment_reduction_marks_nonzero_unexplained_leakage_invalid(
    environment_reduction_config, pairwise_interference_case
):
    """
    Add Q-sector weight without the partner needed for cancellation.

    Then the complement action of the same local Z pattern is nonzero,
    so exterior-environment removal must be rejected as unsafe.
    """
    basis_configs, kinetic, indices = pairwise_interference_case

    w1 = config_index(basis_configs, (1, 1, 0))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(3.0)
    state[indices["v2"]] = -1.0 / np.sqrt(3.0)
    state[w1] = 1.0 / np.sqrt(3.0)

    config = environment_reduction_config

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert not report.is_safely_removable
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
    assert report.n_same_pattern_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0


def test_diagnose_cage_environment_reduction_lifts_compact_state_and_preserves_metadata(
    environment_reduction_config, pairwise_interference_case
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

    report = diagnose_cage_environment_reduction(
        cage_state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        hilbert_size=basis_configs.shape[0],
        config=environment_reduction_config,
    )

    assert report.is_safely_removable
    assert report.support_size == 2
    assert report.n_nontrivial_zeros == 1
    assert report.n_complement_targets == 0
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_probes_safely_removable == pytest.approx(1.0)

    assert report.metadata["energy"] == cage_state.energy
    assert report.metadata["support_size"] == cage_state.support_size
    assert report.metadata["boundary_residual"] == cage_state.boundary_residual
    assert report.metadata["eigen_residual"] == cage_state.eigen_residual
    assert report.metadata["full_residual"] == cage_state.full_residual

    assert report.n_q_empty_source_probes == 1
    assert report.n_same_pattern_zero_closure_probes == 0
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    zero_report = report.zero_reports[0]
    assert zero_report.probe_mechanism_label == "q_empty"
    assert zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert not zero_report.is_invalid_probe
    assert not zero_report.source_projector_like

    assert report.n_trivial_targets == 0
    assert report.n_same_pattern_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert not zero_report.has_unexpected_targets
    assert not zero_report.has_nonzero_complement_action
    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()

    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 0


def test_diagnose_environment_reduction_ignores_trivial_zeros_without_active_neighbors(
    environment_reduction_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=environment_reduction_config,
    )

    zero_indices = {int(zero_report.zero_index) for zero_report in report.zero_reports}

    # Only |000> is connected to active support and has nontrivial
    # cancellation. Other zero-amplitude basis states are trivial zeros.
    assert zero_indices == {indices["h"]}

    assert report.n_nontrivial_zeros == 1
    assert report.n_q_empty_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0


def test_diagnose_environment_reduction_rejects_wrong_basis_shape(pairwise_interference_case):
    basis_configs, kinetic, _indices = pairwise_interference_case
    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)

    bad_basis_configs = np.zeros(8, dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="basis_configs must have shape",
    ):
        diagnose_environment_reduction(
            state,
            kinetic_matrix=kinetic,
            basis_configs=bad_basis_configs,
        )


def test_diagnose_environment_reduction_rejects_mismatched_basis_size(pairwise_interference_case):
    basis_configs, kinetic, _indices = pairwise_interference_case
    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)

    bad_basis_configs = basis_configs[:-1]

    with pytest.raises(
        ValueError,
        match="basis_configs.shape\\[0\\] must match full_state.size",
    ):
        diagnose_environment_reduction(
            state,
            kinetic_matrix=kinetic,
            basis_configs=bad_basis_configs,
        )


def test_diagnose_environment_reduction_detects_projective_domain_blocking(
    environment_reduction_config, pairwise_interference_case
):
    """Finite Q-sector weight with no complement targets is domain-blocked.

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

    config = environment_reduction_config

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.is_safely_removable
    assert report.support_size == 3
    assert report.n_nontrivial_zeros == 1

    assert report.n_complement_targets == 0
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_probes_safely_removable == pytest.approx(1.0)

    assert report.n_q_empty_source_probes == 0
    assert report.n_same_pattern_zero_closure_probes == 0
    assert report.n_domain_blocked_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert report.n_domain_blocked_source_probes == 1
    assert report.n_projector_like_source_probes == 0
    assert report.n_invalid_source_probes == 0

    assert _zero_indices(report.domain_blocked_source_zero_indices) == {indices["h"]}
    assert _zero_indices(report.projector_like_source_zero_indices) == set()
    assert _zero_indices(report.invalid_source_zero_indices) == set()

    zero_report = report.zero_reports[0]
    assert zero_report.zero_index == indices["h"]
    assert zero_report.q_sector_weight > config.action_tolerance
    assert zero_report.n_complement_targets == 0
    assert zero_report.n_unexplained_complement_targets == 0
    assert not zero_report.complement_targets_are_known_zeros
    assert zero_report.complement_action_norm <= config.action_tolerance

    assert zero_report.probe_mechanism_label == "domain_blocked"
    assert not zero_report.is_q_empty
    assert zero_report.is_domain_blocked
    assert not zero_report.is_projector_like
    assert not zero_report.is_invalid_probe
    assert zero_report.source_projector_like

    assert report.n_trivial_targets == 0
    assert report.n_same_pattern_iz_targets == 0
    assert report.n_projector_like_iz_targets == 0
    assert report.n_unexpected_targets == 0

    assert zero_report.n_trivial_targets == 0
    assert zero_report.n_same_pattern_iz_targets == 0
    assert zero_report.n_projector_like_iz_targets == 0
    assert zero_report.n_unexpected_targets == 0

    assert not zero_report.has_unexpected_targets
    assert not zero_report.has_nonzero_complement_action

    assert report.n_unexpected_target_probe_failures == 0
    assert report.n_nonzero_complement_action_probe_failures == 0

    assert _zero_indices(zero_report.nonzero_complement_action_target_indices) == set()


def test_environment_reduction_summary_dict_is_stable(
    environment_reduction_config, pairwise_interference_case
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=environment_reduction_config,
    )

    summary = report.to_summary_dict()

    assert summary["Environment reduction"]["is safely removable"] is True
    assert summary["Mechanism counts"]["no environment weight"] == 1
    assert summary["Mechanism counts"]["same local cancellation pattern"] == 0
    assert summary["Mechanism counts"]["unsafe"] == 0


def test_diagnose_environment_reduction_raises_without_sector_on_disconnected_graph(
    pairwise_interference_case,
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    config = EnvironmentReductionConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
        sector_policy="raise_if_disconnected",
    )

    with pytest.raises(ValueError, match="disconnected"):
        diagnose_environment_reduction(
            state,
            kinetic_matrix=kinetic,
            basis_configs=basis_configs,
            config=config,
        )


def test_diagnose_environment_reduction_ignores_complement_targets_outside_sector_mask(
    pairwise_interference_case,
):
    basis_configs, kinetic, indices = pairwise_interference_case

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    sector_mask = np.zeros(basis_configs.shape[0], dtype=np.bool_)
    sector_mask[[indices["h"], indices["v1"], indices["v2"]]] = True

    report = diagnose_environment_reduction(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=EnvironmentReductionConfig(
            amplitude_tolerance=1e-12,
            cancellation_tolerance=1e-12,
            action_tolerance=1e-12,
            sector_policy="raise_if_disconnected",
        ),
    )

    assert report.metadata["environment_domain_size"] == 3
    assert report.is_safely_removable
