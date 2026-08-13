"""Global QDM action and embedding primitives for local cage workflows.

These helpers operate on explicit global configurations or structural cage blocks but do not
perform padding search or certification.  They are a lower layer shared by padding and
residual-certification code.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis
from qlinks.caging.local_search_qdm import _backward_coefficient, _forward_coefficient
from qlinks.caging.local_search_types import LocalQDMCageBlock, _QDMGlobalPlaquetteAction
from qlinks.operators.plaquette import alternating_binary_patterns


def _constant_qdm_block_site_counts(
    model: object,
    link_ids: npt.ArrayLike,
    support_configs: npt.ArrayLike,
) -> npt.NDArray[np.int64] | None:
    local_link_ids = np.asarray(link_ids, dtype=np.int64)
    support_arr = np.asarray(support_configs, dtype=np.int64)
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(local_link_ids)}
    site_counts = np.zeros(int(model.lattice.num_sites), dtype=np.int64)

    for site_id in range(int(model.lattice.num_sites)):
        local_incident = [
            local_index_by_link[int(link_id)]
            for link_id in model.lattice.incident_links(int(site_id))
            if int(link_id) in local_index_by_link
        ]
        if local_incident:
            counts = np.sum(support_arr[:, local_incident], axis=1).astype(np.int64)
        else:
            counts = np.zeros(support_arr.shape[0], dtype=np.int64)
        unique_counts = np.unique(counts)
        if unique_counts.size != 1:
            return None
        site_counts[int(site_id)] = int(unique_counts[0])

    return site_counts


def _qdm_blocks_are_pairwise_link_disjoint(blocks: Sequence[LocalQDMCageBlock]) -> bool:
    used: set[int] = set()
    for block in blocks:
        block_links = set(int(link_id) for link_id in block.link_ids)
        if used.intersection(block_links):
            return False
        used.update(block_links)
    return True


def _qdm_block_is_kinetically_separated(
    model: object,
    existing_blocks: Sequence[LocalQDMCageBlock],
    new_block: LocalQDMCageBlock,
) -> bool:
    return _qdm_blocks_are_kinetically_separated(model, tuple(existing_blocks) + (new_block,))


def _qdm_blocks_are_kinetically_separated(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
) -> bool:
    link_owner: dict[int, int] = {}
    for block_position, block in enumerate(blocks):
        for link_id in block.link_ids:
            link_owner[int(link_id)] = int(block_position)

    for plaquette_id in model.plaquette_ids():
        owners = {
            link_owner[int(link_id)]
            for link_id in model.lattice.plaquette_links(int(plaquette_id))
            if int(link_id) in link_owner
        }
        if len(owners) > 1:
            return False
    return True


def _global_configs_satisfy_qdm_constraints(
    model: object,
    configs: npt.ArrayLike,
) -> bool:
    required_count = int(getattr(model, "required_count", 1))
    arr = np.asarray(configs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for config_row in arr:
        for site_id in range(int(model.lattice.num_sites)):
            incident = np.asarray(model.lattice.incident_links(int(site_id)), dtype=np.int64)
            if int(np.sum(config_row[incident])) != required_count:
                return False
    return True


def _global_configs_satisfy_model_sectors(
    model: object,
    configs: npt.ArrayLike,
) -> bool:
    sectors = tuple(model.make_sectors())
    if not sectors:
        return True
    arr = np.asarray(configs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for config_row in arr:
        for sector in sectors:
            if not sector.is_satisfied(config_row):
                return False
    return True


def build_qdm_global_limited_kinetic_matrix(
    model: object,
    basis: Basis,
) -> scipy_sparse.csr_array:
    """Build QDM kinetic transitions restricted to an explicitly supplied basis."""
    n = int(basis.n_states)
    if n == 0:
        return scipy_sparse.csr_array((0, 0), dtype=np.complex128)

    config_to_index = {_config_key(config): i for i, config in enumerate(basis.states)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    actions = _qdm_global_plaquette_actions(model)
    for col, config_row in enumerate(basis.states):
        for action in actions:
            transition = _qdm_flip_transition_from_action(config_row, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            row = config_to_index.get(_config_key(final_config))
            if row is None:
                continue
            rows.append(int(row))
            cols.append(int(col))
            data.append(complex(coefficient))

    return scipy_sparse.coo_array(
        (np.asarray(data, dtype=np.complex128), (rows, cols)),
        shape=(n, n),
        dtype=np.complex128,
    ).tocsr()


def qdm_global_self_loop_values(
    model: object,
    configs: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Compute full QDM potential/self-loop values for explicit configs."""
    return _qdm_global_self_loop_values_from_actions(
        configs,
        _qdm_global_plaquette_actions(model),
    )


def _qdm_global_plaquette_actions(
    model: object,
    plaquette_ids: Sequence[int] | None = None,
) -> tuple[_QDMGlobalPlaquetteAction, ...]:
    source_ids = model.plaquette_ids() if plaquette_ids is None else plaquette_ids
    ids = tuple(int(pid) for pid in source_ids)
    actions: list[_QDMGlobalPlaquetteAction] = []
    for plaquette_id in ids:
        links = np.asarray(model.lattice.plaquette_links(int(plaquette_id)), dtype=np.int64)
        pattern0, pattern1 = alternating_binary_patterns(int(links.size))
        coupling = model._coup_kin_at(int(plaquette_id))
        actions.append(
            _QDMGlobalPlaquetteAction(
                plaquette_id=int(plaquette_id),
                links=links,
                pattern0=np.asarray(pattern0, dtype=np.int64),
                pattern1=np.asarray(pattern1, dtype=np.int64),
                forward=complex(_forward_coefficient(coupling)),
                backward=complex(_backward_coefficient(coupling)),
                potential=complex(model._coup_pot_at(int(plaquette_id))),
            )
        )
    return tuple(actions)


def _qdm_flip_transition_from_action(
    config_row: npt.ArrayLike,
    action: _QDMGlobalPlaquetteAction,
) -> tuple[npt.NDArray[np.int64], complex] | None:
    config_arr = np.asarray(config_row, dtype=np.int64)
    values = config_arr[action.links]
    if np.array_equal(values, action.pattern0):
        final = config_arr.copy()
        final[action.links] = action.pattern1
        return final, action.forward
    if np.array_equal(values, action.pattern1):
        final = config_arr.copy()
        final[action.links] = action.pattern0
        return final, action.backward
    return None


def _qdm_plaquette_is_flippable_from_action(
    config_row: npt.ArrayLike,
    action: _QDMGlobalPlaquetteAction,
) -> bool:
    config_arr = np.asarray(config_row, dtype=np.int64)
    values = config_arr[action.links]
    return bool(np.array_equal(values, action.pattern0) or np.array_equal(values, action.pattern1))


def _qdm_flip_transition(
    model: object,
    config_row: npt.ArrayLike,
    plaquette_id: int,
) -> tuple[npt.NDArray[np.int64], complex] | None:
    action = _qdm_global_plaquette_actions(model, (int(plaquette_id),))[0]
    return _qdm_flip_transition_from_action(config_row, action)


def _qdm_global_self_loop_values_from_actions(
    configs: npt.ArrayLike,
    actions: Sequence[_QDMGlobalPlaquetteAction],
) -> npt.NDArray[np.complex128]:
    arr = np.asarray(configs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    values = np.zeros(arr.shape[0], dtype=np.complex128)
    for action in actions:
        local_values = arr[:, action.links]
        flippable = np.all(local_values == action.pattern0, axis=1) | np.all(
            local_values == action.pattern1,
            axis=1,
        )
        if np.any(flippable):
            values[flippable] += action.potential
    return values


def _qdm_global_self_loop_value(model: object, config_row: npt.ArrayLike) -> complex:
    return complex(
        _qdm_global_self_loop_values_from_actions(
            config_row,
            _qdm_global_plaquette_actions(model),
        )[0]
    )


def _config_key(config_row: npt.ArrayLike) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(config_row, dtype=np.int64))
