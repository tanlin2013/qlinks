from __future__ import annotations

import numpy as np

from qlinks.caging.analysis.environment import (
    EnvironmentRemovalProbeReport,
    LocalTransitionPattern,
)
from tests.helpers.states import empty_complex_array, empty_int_array


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
    local_transitions: tuple[LocalTransitionPattern, ...] = (),
    local_mask: np.ndarray | None = None,
) -> EnvironmentRemovalProbeReport:
    """Build a minimal EnvironmentRemovalProbeReport for annotation tests."""
    complement_target_indices = np.array(
        complement_targets,
        dtype=np.int64,
    )

    has_nonzero_complement_action = complement_action_norm > 0.0
    nonzero_complement_action_target_indices = (
        complement_target_indices.copy() if has_nonzero_complement_action else empty_int_array()
    )

    if local_mask is None:
        local_mask = np.array([False], dtype=np.bool_)
    else:
        local_mask = np.asarray(local_mask, dtype=np.bool_)

    return EnvironmentRemovalProbeReport(
        zero_index=zero_index,
        active_neighbors=empty_int_array(),
        active_matrix_elements=empty_complex_array(),
        active_amplitudes=empty_complex_array(),
        cancellation_residual=0.0,
        common_mask=~local_mask,
        local_mask=local_mask,
        q_sector_weight=q_sector_weight,
        reduced_action_norm=0.0,
        complement_action_norm=complement_action_norm,
        complement_target_indices=complement_target_indices,
        explained_complement_target_indices=empty_int_array(),
        unexplained_complement_target_indices=complement_target_indices,
        complement_targets_are_known_zeros=False,
        trivial_target_indices=empty_int_array(),
        same_pattern_iz_target_indices=empty_int_array(),
        projector_like_iz_target_indices=empty_int_array(),
        unexpected_target_indices=empty_int_array(),
        has_unexpected_targets=False,
        has_nonzero_complement_action=has_nonzero_complement_action,
        unexpected_target_probe_failure_indices=empty_int_array(),
        nonzero_complement_action_target_indices=(nonzero_complement_action_target_indices),
        source_projector_like=source_projector_like,
        probe_mechanism_label="unexplained_leakage",
        local_transitions=local_transitions,
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
