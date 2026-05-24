from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.caging.classification import (
    CageClassificationConfig,
    classify_cage_state,
    classify_full_state,
)
from qlinks.caging.results import CageState


def _binary_basis(n_variables: int) -> np.ndarray:
    """Return all binary product-state configurations."""
    return np.array(
        list(itertools.product([0, 1], repeat=n_variables)),
        dtype=np.int64,
    )


def _basis_index(
    basis_configs: np.ndarray,
    config: tuple[int, ...],
) -> int:
    matches = np.all(
        basis_configs == np.array(config, dtype=np.int64)[None, :],
        axis=1,
    )
    indices = np.flatnonzero(matches)

    if indices.size != 1:
        raise ValueError(f"Configuration {config} not found uniquely.")

    return int(indices[0])


def _base_classification_config() -> CageClassificationConfig:
    return CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
    )


def _zero_indices(indices: np.ndarray) -> set[int]:
    return {int(index) for index in indices}


def _make_pairwise_interference_kinetic(
    basis_configs: np.ndarray,
) -> tuple[scipy_sparse.csr_array, dict[str, int]]:
    """
    Build a kinetic matrix with one interference zero.

    The local structure is

        |h>  = |000>
        |v1> = |010>
        |v2> = |001>

    with H0[h, v1] = H0[h, v2] = 1.

    Therefore a state with amplitudes

        c[v1] = +1/sqrt(2)
        c[v2] = -1/sqrt(2)

    has destructive interference at |h>.
    """
    h = _basis_index(basis_configs, (0, 0, 0))
    v1 = _basis_index(basis_configs, (0, 1, 0))
    v2 = _basis_index(basis_configs, (0, 0, 1))

    n_basis = basis_configs.shape[0]

    rows = [h, h, v1, v2]
    cols = [v1, v2, h, h]
    data = [1.0, 1.0, 1.0, 1.0]

    kinetic = scipy_sparse.csr_array(
        (data, (rows, cols)),
        shape=(n_basis, n_basis),
        dtype=np.complex128,
    )

    return kinetic, {"h": h, "v1": v1, "v2": v2}


def _make_two_zero_closed_interference_kinetic(
    basis_configs: np.ndarray,
) -> tuple[scipy_sparse.csr_array, dict[str, int]]:
    """
    Build a kinetic matrix with two related nontrivial zeros.

    First zero:

        h0 = |000>
        v1 = |010>
        v2 = |001>

    Second zero:

        h1 = |100>
        w1 = |110>
        w2 = |101>

    The same local transition pattern on the last two variables
    generates both interference zeros.
    """
    h0 = _basis_index(basis_configs, (0, 0, 0))
    v1 = _basis_index(basis_configs, (0, 1, 0))
    v2 = _basis_index(basis_configs, (0, 0, 1))

    h1 = _basis_index(basis_configs, (1, 0, 0))
    w1 = _basis_index(basis_configs, (1, 1, 0))
    w2 = _basis_index(basis_configs, (1, 0, 1))

    n_basis = basis_configs.shape[0]

    rows = [
        h0,
        h0,
        v1,
        v2,
        h1,
        h1,
        w1,
        w2,
    ]
    cols = [
        v1,
        v2,
        h0,
        h0,
        w1,
        w2,
        h1,
        h1,
    ]
    data = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    kinetic = scipy_sparse.csr_array(
        (data, (rows, cols)),
        shape=(n_basis, n_basis),
        dtype=np.complex128,
    )

    return kinetic, {
        "h0": h0,
        "v1": v1,
        "v2": v2,
        "h1": h1,
        "w1": w1,
        "w2": w2,
    }


def test_classify_full_state_finds_regional_candidate():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    config = _base_classification_config()

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

    # No wavefunction weight lives outside the common beta sector.
    assert zero_report.q_sector_weight <= config.action_tolerance
    assert zero_report.complement_action_norm <= config.action_tolerance
    assert zero_report.reduced_action_norm <= config.action_tolerance

    assert report.n_complement_targets == 0
    assert report.n_unexplained_complement_targets == 0
    assert report.fraction_zeros_with_closed_complement_targets == pytest.approx(0.0)

    assert report.n_q_empty_zeros == 1
    assert report.n_closed_by_known_zero_zeros == 0
    assert report.n_projector_like_zeros == 0
    assert report.n_unexplained_leakage_zeros == 0

    assert report.n_regional_mechanism_zeros == 1
    assert report.n_extended_mechanism_zeros == 0
    assert report.n_failure_mechanism_zeros == 0

    assert _zero_indices(report.q_empty_zero_indices) == {indices["h"]}
    assert _zero_indices(report.closed_by_known_zero_indices) == set()
    assert _zero_indices(report.projector_like_zero_indices) == set()
    assert _zero_indices(report.unexplained_leakage_zero_indices) == set()

    zero_report = report.zero_reports[0]
    assert zero_report.n_complement_targets == 0
    assert zero_report.n_unexplained_complement_targets == 0
    assert not zero_report.complement_targets_are_known_zeros
    assert zero_report.mechanism_label == "q_empty"
    assert zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert not zero_report.is_unexplained_leakage


def test_classify_full_state_regional_when_complement_targets_are_known_zeros():
    """
    Q-sector weight is nonzero, but the complement action of each zero
    lands only on the other known nontrivial interference zero.

    This is the model-free closure criterion for regional_candidate.
    """
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_two_zero_closed_interference_kinetic(basis_configs)

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)

    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[indices["w1"]] = 0.5
    state[indices["w2"]] = -0.5

    config = _base_classification_config()

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

    assert h0_report.q_sector_weight == pytest.approx(0.5)
    assert h1_report.q_sector_weight == pytest.approx(0.5)

    assert set(int(i) for i in h0_report.complement_target_indices) == {indices["h1"]}
    assert set(int(i) for i in h1_report.complement_target_indices) == {indices["h0"]}

    assert h0_report.n_unexplained_complement_targets == 0
    assert h1_report.n_unexplained_complement_targets == 0

    assert h0_report.complement_action_norm <= config.action_tolerance
    assert h1_report.complement_action_norm <= config.action_tolerance

    assert report.n_q_empty_zeros == 0
    assert report.n_closed_by_known_zero_zeros == 2
    assert report.n_projector_like_zeros == 0
    assert report.n_unexplained_leakage_zeros == 0

    assert report.n_regional_mechanism_zeros == 0
    assert report.n_extended_mechanism_zeros == 0
    assert report.n_failure_mechanism_zeros == 0

    assert _zero_indices(report.q_empty_zero_indices) == set()
    assert _zero_indices(report.closed_by_known_zero_indices) == {
        indices["h0"],
        indices["h1"],
    }
    assert _zero_indices(report.projector_like_zero_indices) == set()
    assert _zero_indices(report.unexplained_leakage_zero_indices) == set()

    assert h0_report.mechanism_label == "closed_by_known_zeros"
    assert h1_report.mechanism_label == "closed_by_known_zeros"

    assert not h0_report.is_q_empty
    assert not h1_report.is_q_empty
    assert not h0_report.is_projector_like
    assert not h1_report.is_projector_like
    assert not h0_report.is_unexplained_leakage
    assert not h1_report.is_unexplained_leakage


def test_classify_full_state_marks_unexplained_complement_cancellation_invalid():
    """
    Add amplitudes in the Q_beta sector.

    The same local Z pattern acts on

        |110> and |101>

    and maps them toward |100>. Their amplitudes cancel there.

    However, |100> is not itself a known nontrivial interference zero,
    because the kinetic matrix does not include a corresponding row
    cancellation around |100>. Therefore, this is not regional closure.
    """
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    w1 = _basis_index(basis_configs, (1, 1, 0))
    w2 = _basis_index(basis_configs, (1, 0, 1))
    unexplained_target = _basis_index(basis_configs, (1, 0, 0))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[w1] = 0.5
    state[w2] = -0.5

    config = _base_classification_config()

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.label == "invalid_or_inconsistent"
    assert report.support_size == 4
    assert report.n_nontrivial_zeros == 1
    assert report.n_complement_targets == 1
    assert report.n_unexplained_complement_targets == 1
    assert report.fraction_zeros_with_closed_complement_targets == pytest.approx(0.0)

    zero_report = report.zero_reports[0]

    assert zero_report.q_sector_weight == pytest.approx(0.5)
    assert zero_report.complement_action_norm <= config.action_tolerance

    assert set(int(i) for i in zero_report.complement_target_indices) == {unexplained_target}
    assert set(int(i) for i in zero_report.explained_complement_target_indices) == set()
    assert set(int(i) for i in zero_report.unexplained_complement_target_indices) == {
        unexplained_target
    }
    assert not zero_report.complement_targets_are_known_zeros

    assert report.n_q_empty_zeros == 0
    assert report.n_closed_by_known_zero_zeros == 0
    assert report.n_projector_like_zeros == 0
    assert report.n_unexplained_leakage_zeros == 1

    assert report.n_regional_mechanism_zeros == 0
    assert report.n_extended_mechanism_zeros == 0
    assert report.n_failure_mechanism_zeros == 1

    assert _zero_indices(report.unexplained_leakage_zero_indices) == {indices["h"]}
    assert _zero_indices(report.failure_mechanism_zero_indices) == {indices["h"]}

    assert zero_report.mechanism_label == "unexplained_leakage"
    assert not zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert zero_report.is_unexplained_leakage


def test_classify_full_state_marks_nonzero_unexplained_leakage_invalid():
    """
    Add Q-sector weight without the partner needed for cancellation.

    Then the complement action of the same local Z pattern is nonzero,
    so the classification should not be regional or extended.
    """
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    w1 = _basis_index(basis_configs, (1, 1, 0))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(3.0)
    state[indices["v2"]] = -1.0 / np.sqrt(3.0)
    state[w1] = 1.0 / np.sqrt(3.0)

    config = _base_classification_config()

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    assert report.label == "invalid_or_inconsistent"
    assert report.n_nontrivial_zeros == 1

    unexplained_target = _basis_index(basis_configs, (1, 0, 0))

    zero_report = report.zero_reports[0]

    assert zero_report.q_sector_weight > config.action_tolerance
    assert zero_report.complement_action_norm > config.action_tolerance

    assert set(int(i) for i in zero_report.complement_target_indices) == {unexplained_target}
    assert set(int(i) for i in zero_report.unexplained_complement_target_indices) == {
        unexplained_target
    }
    assert not zero_report.complement_targets_are_known_zeros

    assert report.n_complement_targets == 1
    assert report.n_unexplained_complement_targets == 1

    assert report.n_q_empty_zeros == 0
    assert report.n_closed_by_known_zero_zeros == 0
    assert report.n_projector_like_zeros == 0
    assert report.n_unexplained_leakage_zeros == 1

    assert report.n_regional_mechanism_zeros == 0
    assert report.n_extended_mechanism_zeros == 0
    assert report.n_failure_mechanism_zeros == 1

    assert zero_report.mechanism_label == "unexplained_leakage"
    assert not zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert zero_report.is_unexplained_leakage


def test_classify_cage_state_lifts_compact_state_and_preserves_metadata():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

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

    config = _base_classification_config()

    report = classify_cage_state(
        cage_state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        hilbert_size=basis_configs.shape[0],
        config=config,
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

    assert report.n_q_empty_zeros == 1
    assert report.n_closed_by_known_zero_zeros == 0
    assert report.n_projector_like_zeros == 0
    assert report.n_unexplained_leakage_zeros == 0

    assert report.n_regional_mechanism_zeros == 1
    assert report.n_extended_mechanism_zeros == 0
    assert report.n_failure_mechanism_zeros == 0

    zero_report = report.zero_reports[0]
    assert zero_report.mechanism_label == "q_empty"
    assert zero_report.is_q_empty
    assert not zero_report.is_projector_like
    assert not zero_report.is_unexplained_leakage


def test_classify_full_state_ignores_trivial_zeros_without_active_neighbors():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    config = _base_classification_config()

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=config,
    )

    zero_indices = {int(zero_report.zero_index) for zero_report in report.zero_reports}

    # Only |000> is connected to active support and has nontrivial
    # cancellation. Other zero-amplitude basis states are trivial zeros.
    assert zero_indices == {indices["h"]}

    assert report.n_nontrivial_zeros == 1
    assert report.n_q_empty_zeros == 1
    assert report.n_regional_mechanism_zeros == 1
    assert report.n_extended_mechanism_zeros == 0
    assert report.n_failure_mechanism_zeros == 0


def test_classify_full_state_rejects_wrong_basis_shape():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, _indices = _make_pairwise_interference_kinetic(basis_configs)
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


def test_classify_full_state_rejects_mismatched_basis_size():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, _indices = _make_pairwise_interference_kinetic(basis_configs)
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


def test_classify_full_state_detects_projector_like_extended_mechanism():
    """Finite Q-sector weight with no complement targets is extended-like.

    The active interference zero is still |000>, with active neighbors
    |010> and |001>. We add amplitude on |111>, which lies outside the
    common beta sector but does not match the local source patterns of the
    reduced interference-zero operator. Therefore the complement operator
    has no raw target vertices.
    """
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    q_sector_state = _basis_index(basis_configs, (1, 1, 1))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(3.0)
    state[indices["v2"]] = -1.0 / np.sqrt(3.0)
    state[q_sector_state] = 1.0 / np.sqrt(3.0)

    config = _base_classification_config()

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

    assert report.n_q_empty_zeros == 0
    assert report.n_closed_by_known_zero_zeros == 0
    assert report.n_projector_like_zeros == 1
    assert report.n_unexplained_leakage_zeros == 0

    assert report.n_regional_mechanism_zeros == 0
    assert report.n_extended_mechanism_zeros == 1
    assert report.n_failure_mechanism_zeros == 0

    assert _zero_indices(report.projector_like_zero_indices) == {indices["h"]}
    assert _zero_indices(report.extended_mechanism_zero_indices) == {indices["h"]}
    assert _zero_indices(report.failure_mechanism_zero_indices) == set()

    zero_report = report.zero_reports[0]
    assert zero_report.zero_index == indices["h"]
    assert zero_report.q_sector_weight > config.action_tolerance
    assert zero_report.n_complement_targets == 0
    assert zero_report.n_unexplained_complement_targets == 0
    assert not zero_report.complement_targets_are_known_zeros
    assert zero_report.complement_action_norm <= config.action_tolerance

    assert zero_report.mechanism_label == "projector_like"
    assert not zero_report.is_q_empty
    assert zero_report.is_projector_like
    assert not zero_report.is_unexplained_leakage


def test_classification_report_text_includes_mechanism_sections():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        config=_base_classification_config(),
    )

    text = str(report)

    assert "Zero mechanisms" in text
    assert "q-empty zeros: 1" in text
    assert "closed-by-known-zero zeros: 0" in text
    assert "projector-like zeros: 0" in text
    assert "unexplained-leakage zeros: 0" in text
    assert "State-level interpretation" in text
