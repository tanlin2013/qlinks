from __future__ import annotations

import itertools
from dataclasses import fields

import numpy as np

from qlinks.caging.classification import (
    CageClassificationConfig,
    InterferenceZeroReport,
    _find_nullspace_collective_cancellation_from_actions,
    _find_unit_sum_collective_cancellation_from_actions,
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


def _classification_config() -> CageClassificationConfig:
    return CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
        collective_cancellation_mode="same_local_support_sum",
    )


def _fake_zero_report(
    *,
    zero_index: int,
    local_mask: np.ndarray,
    mechanism_label: str = "unexplained_leakage",
    complement_action_norm: float = 1.0,
) -> InterferenceZeroReport:
    """Create the minimal InterferenceZeroReport needed for collective tests.

    The kwargs are filtered against the actual dataclass fields so this helper
    remains compatible while the report schema is evolving.
    """
    local_mask = np.asarray(local_mask, dtype=np.bool_)
    common_mask = ~local_mask

    empty_i64 = np.array([], dtype=np.int64)
    empty_c128 = np.array([], dtype=np.complex128)

    kwargs = {
        # Source zero and parent-Hamiltonian cancellation data.
        "zero_index": int(zero_index),
        "active_neighbors": empty_i64,
        "active_matrix_elements": empty_c128,
        "active_amplitudes": empty_c128,
        "cancellation_residual": 0.0,
        # Local reduced-operator geometry.
        "common_mask": common_mask,
        "local_mask": local_mask,
        "local_transitions": tuple(),
        # Operator-action diagnostics.
        "q_sector_weight": 0.0,
        "reduced_action_norm": 0.0,
        "complement_action_norm": float(complement_action_norm),
        # Complement target structure.
        "complement_target_indices": empty_i64,
        "explained_complement_target_indices": empty_i64,
        "unexplained_complement_target_indices": empty_i64,
        "complement_targets_are_known_zeros": False,
        # Complement target explanations.
        "trivial_target_indices": empty_i64,
        "known_nonprojector_iz_target_indices": empty_i64,
        "projector_like_iz_target_indices": empty_i64,
        "unexpected_target_indices": empty_i64,
        # Projector-like input diagnostics.
        "complement_support_indices": empty_i64,
        "complement_contributing_input_indices": empty_i64,
        "projector_like_annihilated_input_indices": empty_i64,
        "source_projector_like": False,
        # Invalid-probe diagnostics.
        "has_unexpected_targets": False,
        "has_nonzero_complement_action": (float(complement_action_norm) > 0.0),
        "unexpected_target_probe_failure_indices": empty_i64,
        "nonzero_complement_action_target_indices": empty_i64,
        # Final source-probe label.
        "probe_mechanism_label": mechanism_label,
        # New collective-cancellation fields, if you added them.
        "collective_cancellation_group_id": None,
        "collective_cancellation_partner_zero_indices": empty_i64,
        "collective_cancellation_coefficient": 0.0 + 0.0j,
        "collective_cancellation_norm": np.inf,
    }

    valid_fields = {field.name for field in fields(InterferenceZeroReport)}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_fields}

    return InterferenceZeroReport(**filtered_kwargs)


def test_unit_sum_collective_cancellation_from_actions():
    config = CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
        collective_cancellation_mode="same_local_support_sum",
    )

    local_mask = np.array([False, True, True], dtype=np.bool_)

    reports = [
        _fake_zero_report(
            zero_index=0,
            local_mask=local_mask,
            complement_action_norm=1.0,
        ),
        _fake_zero_report(
            zero_index=4,
            local_mask=local_mask,
            complement_action_norm=1.0,
        ),
    ]

    action_0 = np.array([1.0, 0.0, -2.0], dtype=np.complex128)
    action_1 = -action_0

    collective = _find_unit_sum_collective_cancellation_from_actions(
        reports,
        [action_0, action_1],
        [
            np.array([7, 8], dtype=np.int64),
            np.array([7, 8], dtype=np.int64),
        ],
        group_id=0,
        config=config,
    )

    assert collective is not None
    assert collective.group_size == 2
    assert collective.relation_kind == "unit_sum"
    assert collective.collective_action_norm <= 1e-12
    assert collective.source_zero_indices.tolist() == [0, 4]
    assert collective.collective_target_indices.tolist() == [7, 8]

    np.testing.assert_allclose(
        collective.coefficients,
        np.ones(2, dtype=np.complex128),
    )
    np.testing.assert_allclose(
        collective.individual_complement_action_norms,
        [np.linalg.norm(action_0), np.linalg.norm(action_1)],
    )


def test_unit_sum_collective_cancellation_from_actions_rejects_nonzero_sum():
    config = CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
        collective_cancellation_mode="same_local_support_sum",
    )

    local_mask = np.array([False, True, True], dtype=np.bool_)

    reports = [
        _fake_zero_report(
            zero_index=0,
            local_mask=local_mask,
            complement_action_norm=1.0,
        ),
        _fake_zero_report(
            zero_index=4,
            local_mask=local_mask,
            complement_action_norm=1.0,
        ),
    ]

    action_0 = np.array([1.0, 0.0, -2.0], dtype=np.complex128)
    action_1 = np.array([0.5, 0.0, 0.0], dtype=np.complex128)

    collective = _find_unit_sum_collective_cancellation_from_actions(
        reports,
        [action_0, action_1],
        [
            np.array([7, 8], dtype=np.int64),
            np.array([7, 8], dtype=np.int64),
        ],
        group_id=0,
        config=config,
        grouping_kind="same_local_support",
    )

    assert collective is None


def test_all_problematic_unit_sum_collective_cancellation_from_actions():
    config = CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
        collective_cancellation_mode="all_problematic_sum",
    )

    reports = [
        _fake_zero_report(
            zero_index=0,
            local_mask=np.array([True, False, False], dtype=np.bool_),
            complement_action_norm=1.0,
        ),
        _fake_zero_report(
            zero_index=4,
            local_mask=np.array([False, True, False], dtype=np.bool_),
            complement_action_norm=1.0,
        ),
        _fake_zero_report(
            zero_index=6,
            local_mask=np.array([False, False, True], dtype=np.bool_),
            complement_action_norm=2.0,
        ),
    ]

    action_0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    action_1 = np.array([0.0, 2.0, 0.0], dtype=np.complex128)
    action_2 = np.array([-1.0, -2.0, 0.0], dtype=np.complex128)

    collective = _find_unit_sum_collective_cancellation_from_actions(
        reports,
        [action_0, action_1, action_2],
        [
            np.array([7], dtype=np.int64),
            np.array([8], dtype=np.int64),
            np.array([7, 8], dtype=np.int64),
        ],
        group_id=0,
        config=config,
        grouping_kind="all_problematic",
    )

    assert collective is not None
    assert collective.group_size == 3
    assert collective.relation_kind == "unit_sum"
    assert collective.grouping_kind == "all_problematic"
    assert collective.collective_action_norm <= 1e-12
    assert collective.source_zero_indices.tolist() == [0, 4, 6]
    assert collective.collective_target_indices.tolist() == [7, 8]
    assert collective.local_mask.tolist() == [True, True, True]


def test_all_problematic_nullspace_collective_cancellation_from_actions():
    config = CageClassificationConfig(
        amplitude_tolerance=1e-12,
        cancellation_tolerance=1e-12,
        action_tolerance=1e-12,
        collective_cancellation_mode="all_problematic_nullspace",
    )

    reports = [
        _fake_zero_report(
            zero_index=0,
            local_mask=np.array([True, False, False], dtype=np.bool_),
            complement_action_norm=1.0,
        ),
        _fake_zero_report(
            zero_index=4,
            local_mask=np.array([False, True, False], dtype=np.bool_),
            complement_action_norm=1.0,
        ),
        _fake_zero_report(
            zero_index=6,
            local_mask=np.array([False, False, True], dtype=np.bool_),
            complement_action_norm=1.0,
        ),
    ]

    # Columns are linearly dependent, but not equal-weight cancelling.
    action_0 = np.array([1.0, 0.0], dtype=np.complex128)
    action_1 = np.array([0.0, 1.0], dtype=np.complex128)
    action_2 = np.array([1.0, 1.0], dtype=np.complex128)

    collective = _find_nullspace_collective_cancellation_from_actions(
        reports,
        [action_0, action_1, action_2],
        [
            np.array([7], dtype=np.int64),
            np.array([8], dtype=np.int64),
            np.array([7, 8], dtype=np.int64),
        ],
        group_id=0,
        config=config,
        grouping_kind="all_problematic",
    )

    assert collective is not None
    assert collective.group_size == 3
    assert collective.relation_kind == "nullspace"
    assert collective.grouping_kind == "all_problematic"
    assert collective.collective_action_norm <= 1e-12
    assert collective.local_mask.tolist() == [True, True, True]

    residual = (
        action_0 * collective.coefficients[0]
        + action_1 * collective.coefficients[1]
        + action_2 * collective.coefficients[2]
    )

    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
