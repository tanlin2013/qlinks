from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from qlinks.caging.classification import (
    CageClassificationConfig,
    classify_full_state,
)
from qlinks.caging.support import (
    distinct_reduced_iz_pattern_supports,
    extract_cage_region_support,
)


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
