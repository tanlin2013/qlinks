from __future__ import annotations

import itertools
from dataclasses import fields

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.caging.classification import (
    CageClassificationConfig,
    CageClassificationReport,
    classify_full_state,
)
from qlinks.caging.support import (
    distinct_reduced_iz_pattern_supports,
    extract_cage_region_support,
)

from .test_collective_cancellation import _fake_zero_report


def _binary_basis(n_variables: int) -> np.ndarray:
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


def _base_config() -> CageClassificationConfig:
    return CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
    )


def _make_pairwise_interference_kinetic(
    basis_configs: np.ndarray,
) -> tuple[scipy_sparse.csr_array, dict[str, int]]:
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


def test_extract_cage_region_support_from_q_empty_probe():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 1.0 / np.sqrt(2.0)
    state[indices["v2"]] = -1.0 / np.sqrt(2.0)

    sector_mask = np.ones(basis_configs.shape[0], dtype=bool)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=_base_config(),
    )

    region = extract_cage_region_support(report)

    assert region.variable_indices == (1, 2)
    assert region.region_size == 2
    assert region.n_total_probes == 1
    assert region.n_used_probes == 1
    assert region.n_ignored_probes == 0
    assert region.n_unexplained_leakage_probes == 0
    assert not region.has_unexplained_leakage

    probe = region.probe_supports[0]
    assert probe.zero_index == indices["h"]
    assert probe.variable_indices == (1, 2)
    assert probe.mechanism_label == "q_empty"
    assert probe.complement_action_norm <= 1e-12


def _make_two_zero_closed_interference_kinetic(
    basis_configs: np.ndarray,
) -> tuple[scipy_sparse.csr_array, dict[str, int]]:
    h0 = _basis_index(basis_configs, (0, 0, 0))
    v1 = _basis_index(basis_configs, (0, 1, 0))
    v2 = _basis_index(basis_configs, (0, 0, 1))

    h1 = _basis_index(basis_configs, (1, 0, 0))
    w1 = _basis_index(basis_configs, (1, 1, 0))
    w2 = _basis_index(basis_configs, (1, 0, 1))

    n_basis = basis_configs.shape[0]
    rows = [h0, h0, v1, v2, h1, h1, w1, w2]
    cols = [v1, v2, h0, h0, w1, w2, h1, h1]
    data = [1.0] * len(rows)

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


def test_extract_cage_region_support_from_closed_zero_network():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_two_zero_closed_interference_kinetic(basis_configs)

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[indices["w1"]] = 0.5
    state[indices["w2"]] = -0.5

    sector_mask = np.ones(basis_configs.shape[0], dtype=bool)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=_base_config(),
    )

    region = extract_cage_region_support(report)

    assert region.variable_indices == (1, 2)
    assert region.n_total_probes == 2
    assert region.n_used_probes == 2
    assert region.n_unexplained_leakage_probes == 0

    assert {probe.mechanism_label for probe in region.probe_supports} == {"closed_by_known_zeros"}


def test_distinct_reduced_iz_pattern_supports_deduplicates_translated_pattern():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_two_zero_closed_interference_kinetic(basis_configs)

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[indices["w1"]] = 0.5
    state[indices["w2"]] = -0.5

    sector_mask = np.ones(basis_configs.shape[0], dtype=bool)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=_base_config(),
    )

    patterns = distinct_reduced_iz_pattern_supports(report)

    assert len(patterns) == 1
    assert patterns[0].variable_indices == (1, 2)
    assert set(patterns[0].source_zero_indices) == {
        indices["h0"],
        indices["h1"],
    }


def test_extract_cage_region_support_raises_on_unexplained_leakage():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    # Same local pair as q-empty, but add complement-sector amplitude without
    # its cancelling partner.  This makes the complement action nonzero.
    w1 = _basis_index(basis_configs, (1, 1, 0))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[w1] = 1.0 / np.sqrt(2.0)

    state /= np.linalg.norm(state)
    sector_mask = np.ones(basis_configs.shape[0], dtype=bool)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=_base_config(),
    )

    assert report.n_invalid_source_probes > 0

    with pytest.raises(ValueError, match="unexplained leakage"):
        extract_cage_region_support(report)


def test_extract_cage_region_support_can_ignore_unexplained_leakage():
    basis_configs = _binary_basis(n_variables=3)
    kinetic, indices = _make_pairwise_interference_kinetic(basis_configs)

    w1 = _basis_index(basis_configs, (1, 1, 0))

    state = np.zeros(basis_configs.shape[0], dtype=np.complex128)
    state[indices["v1"]] = 0.5
    state[indices["v2"]] = -0.5
    state[w1] = 1.0 / np.sqrt(2.0)
    state /= np.linalg.norm(state)
    sector_mask = np.ones(basis_configs.shape[0], dtype=bool)

    report = classify_full_state(
        state,
        kinetic_matrix=kinetic,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=_base_config(),
    )

    region = extract_cage_region_support(
        report,
        policy="ignore_unexplained",
    )

    assert region.has_unexplained_leakage
    assert region.n_unexplained_leakage_probes > 0


def _fake_classification_report(
    zero_reports,
) -> CageClassificationReport:
    zero_reports = tuple(zero_reports)

    empty_i64 = np.array([], dtype=np.int64)

    collective_indices = np.array(
        [
            int(report.zero_index)
            for report in zero_reports
            if report.probe_mechanism_label == "collective_cancellation"
        ],
        dtype=np.int64,
    )

    unexplained_indices = np.array(
        [
            int(report.zero_index)
            for report in zero_reports
            if report.probe_mechanism_label == "unexplained_leakage"
        ],
        dtype=np.int64,
    )

    q_empty_indices = np.array(
        [
            int(report.zero_index)
            for report in zero_reports
            if report.probe_mechanism_label == "q_empty"
        ],
        dtype=np.int64,
    )

    closed_indices = np.array(
        [
            int(report.zero_index)
            for report in zero_reports
            if report.probe_mechanism_label == "closed_by_known_zeros"
        ],
        dtype=np.int64,
    )

    projector_indices = np.array(
        [
            int(report.zero_index)
            for report in zero_reports
            if report.probe_mechanism_label == "projector_like"
        ],
        dtype=np.int64,
    )

    regional_indices = np.concatenate([q_empty_indices, closed_indices]).astype(
        np.int64, copy=False
    )

    extended_indices = np.concatenate([projector_indices, collective_indices]).astype(
        np.int64, copy=False
    )

    n_complement_targets = sum(
        int(report.complement_target_indices.size) for report in zero_reports
    )
    n_unexplained_complement_targets = sum(
        int(report.unexplained_complement_target_indices.size) for report in zero_reports
    )

    kwargs = {
        # High-level label and dimensions.
        "label": "extended_candidate",
        "hilbert_size": 8,
        "support_size": 2,
        "support_fraction": 2.0 / 8.0,
        # Main zero-report payload.
        "zero_reports": zero_reports,
        "n_nontrivial_zeros": len(zero_reports),
        "n_distinct_local_patterns": len(zero_reports),
        # Complement target summaries.
        "n_complement_targets": int(n_complement_targets),
        "n_unexplained_complement_targets": int(n_unexplained_complement_targets),
        "fraction_zeros_with_closed_complement_targets": 0.0,
        # Mechanism counts / index arrays.
        "n_q_empty_source_probes": int(q_empty_indices.size),
        "q_empty_source_zero_indices": q_empty_indices,
        "n_closed_by_known_zero_network_source_probes": int(closed_indices.size),
        "closed_by_known_zero_network_source_zero_indices": closed_indices,
        "n_projector_like_source_probes": int(projector_indices.size),
        "projector_like_source_zero_indices": projector_indices,
        "n_collective_cancellation_source_probes": int(collective_indices.size),
        "collective_cancellation_source_zero_indices": collective_indices,
        "n_invalid_source_probes": int(unexplained_indices.size),
        "invalid_source_zero_indices": unexplained_indices,
        # Regional / extended source summaries.
        "n_regional_source_probes": int(regional_indices.size),
        "regional_source_zero_indices": regional_indices,
        "extended_source_zero_indices": extended_indices,
        # Target-explanation summaries.
        "n_trivial_targets": 0,
        "n_known_nonprojector_iz_targets": 0,
        "n_projector_like_iz_targets": 0,
        "n_unexpected_targets": 0,
        # Probe-failure summaries.
        "n_unexpected_target_probe_failures": 0,
        "n_nonzero_complement_action_probe_failures": int(unexplained_indices.size),
        "unexpected_target_probe_failure_indices": empty_i64,
        "nonzero_complement_action_probe_failure_indices": (unexplained_indices),
        # Projector-like summaries.
        "n_source_projector_like_probes": int(projector_indices.size),
        "n_indirect_projector_like_probes": 0,
        "n_projector_like_annihilated_inputs": 0,
        "source_projector_like_probe_indices": projector_indices,
        "indirect_projector_like_probe_indices": empty_i64,
        "projector_like_annihilated_input_indices": empty_i64,
        # Collective reports can be empty here. The support extractor only
        # needs zero_reports and their mechanism labels/local masks.
        "collective_cancellation_reports": tuple(),
        # Norm summaries.
        "mean_q_sector_weight": 0.0,
        "min_q_sector_weight": 0.0,
        "max_q_sector_weight": 0.0,
        "mean_reduced_action_norm": 0.0,
        "max_reduced_action_norm": 0.0,
        "mean_complement_action_norm": 1.0,
        "max_complement_action_norm": 1.0,
    }

    valid_fields = {field.name for field in fields(CageClassificationReport)}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_fields}

    return CageClassificationReport(**filtered_kwargs)


def test_extract_region_support_includes_collective_cancellation_probes():
    local_mask_0 = np.array([False, True, True, False], dtype=np.bool_)
    local_mask_1 = np.array([False, False, True, True], dtype=np.bool_)

    report = _fake_classification_report(
        [
            _fake_zero_report(
                zero_index=0,
                local_mask=local_mask_0,
                mechanism_label="collective_cancellation",
                complement_action_norm=1.0,
            ),
            _fake_zero_report(
                zero_index=4,
                local_mask=local_mask_1,
                mechanism_label="collective_cancellation",
                complement_action_norm=1.0,
            ),
        ]
    )

    region = extract_cage_region_support(report)

    assert region.variable_indices == (1, 2, 3)
    assert region.region_size == 3
    assert region.n_total_probes == 2
    assert region.n_used_probes == 2
    assert region.n_ignored_probes == 0
    assert region.n_unexplained_leakage_probes == 0
    assert not region.has_unexplained_leakage

    assert {probe.mechanism_label for probe in region.probe_supports} == {"collective_cancellation"}


def test_extract_region_support_can_exclude_collective_cancellation_probes():
    local_mask = np.array([False, True, True, False], dtype=np.bool_)

    report = _fake_classification_report(
        [
            _fake_zero_report(
                zero_index=0,
                local_mask=local_mask,
                mechanism_label="collective_cancellation",
                complement_action_norm=1.0,
            )
        ]
    )

    region = extract_cage_region_support(
        report,
        include_collective_cancellation=False,
    )

    assert region.variable_indices == ()
    assert region.region_size == 0
    assert region.n_total_probes == 1
    assert region.n_used_probes == 0
    assert region.n_ignored_probes == 1
    assert region.n_unexplained_leakage_probes == 0
