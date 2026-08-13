"""QDM exterior-padding search and structural validation.

This layer turns local cage records into independent blocks and searches shared exterior
assignments.  It depends only on local-search contracts, geometry helpers, and global QDM
primitives; residual-based certification remains in ``local_search_certification``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.caging.local_search_geometry import _unique_int_array, _validate_plaquette_ids
from qlinks.caging.local_search_global import (
    _constant_qdm_block_site_counts,
    _global_configs_satisfy_model_sectors,
    _global_configs_satisfy_qdm_constraints,
    _qdm_block_is_kinetically_separated,
    _qdm_blocks_are_kinetically_separated,
    _qdm_blocks_are_pairwise_link_disjoint,
    _qdm_global_plaquette_actions,
    _qdm_plaquette_is_flippable_from_action,
)
from qlinks.caging.local_search_types import (
    FactorizedLocalQDMPadding,
    LocalQDMCageBlock,
    LocalQDMCageRecord,
    LocalQDMMultiPaddingConfig,
    LocalQDMPadding,
    LocalQDMPaddingConfig,
    MultiLocalQDMPadding,
    _QDMExteriorFlippabilityPreference,
    _QDMExteriorStaticPlaquette,
    _QDMGlobalPlaquetteAction,
)


def make_qdm_cage_block(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    block_id: int = 0,
    guard_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
) -> LocalQDMCageBlock:
    """Create a constant-boundary Lego block from a local QDM cage record.

    Independent product padding requires the number of dimers contributed by
    the block at every global site to be independent of the local support
    configuration.  If this fails, one shared exterior cannot tensor with the
    entire block support, so this function raises ``ValueError``.
    """
    link_ids = np.asarray(local_record.local_link_ids, dtype=np.int64)
    support_configs = np.asarray(local_record.support_configs, dtype=np.int64)
    if support_configs.ndim != 2:
        raise ValueError("local_record.support_configs must have shape (support, n_local_links).")
    if support_configs.shape[1] != link_ids.size:
        raise ValueError("local_record support width must match local_link_ids size.")

    site_counts = _constant_qdm_block_site_counts(model, link_ids, support_configs)
    if site_counts is None:
        raise ValueError(
            "Local cage record is not an independent padding block: "
            "its site occupation contribution changes across support configs."
        )

    if guard_plaquette_ids is None:
        guard = np.unique(
            np.concatenate(
                [
                    np.asarray(local_record.active_plaquette_ids, dtype=np.int64),
                    np.asarray(local_record.unresolved_boundary_plaquette_ids, dtype=np.int64),
                ]
            )
        ).astype(np.int64)
    else:
        guard = _unique_int_array(guard_plaquette_ids, name="guard_plaquette_ids")
        _validate_plaquette_ids(model, guard)

    return LocalQDMCageBlock(
        block_id=int(block_id),
        record=local_record,
        link_ids=link_ids.copy(),
        active_plaquette_ids=np.asarray(local_record.active_plaquette_ids, dtype=np.int64).copy(),
        guard_plaquette_ids=guard,
        support_configs=support_configs.copy(),
        amplitudes=np.asarray(local_record.local_state, dtype=np.complex128).copy(),
        site_counts=site_counts,
    )


def iter_multi_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
    max_yielded: int | None = None,
) -> Iterator[MultiLocalQDMPadding]:
    """Yield shared-exterior paddings built from a pool of QDM blocks.

    This is the streaming counterpart of :func:`find_multi_qdm_block_paddings`.
    It is intended for certification-in-the-loop workflows, where a caller may
    want to keep trying raw exterior completions until enough *certified* cages
    are found.  ``max_yielded`` limits the number of raw candidate paddings
    yielded by this iterator; if omitted, ``config.max_padding_attempts`` is
    used.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    yielded_limit = multi_config.max_padding_attempts if max_yielded is None else max_yielded
    if yielded_limit is not None and yielded_limit <= 0:
        return

    blocks = tuple(block_pool)
    if not blocks:
        return

    block_ids = [int(block.block_id) for block in blocks]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("block_pool contains duplicate block_id values.")

    required_count = int(getattr(model, "required_count", 1))
    max_blocks = multi_config.max_blocks if multi_config.max_blocks is not None else len(blocks)
    max_blocks = min(int(max_blocks), len(blocks))

    selected: list[LocalQDMCageBlock] = []
    used_links: set[int] = set()
    site_counts = np.zeros(int(model.lattice.num_sites), dtype=np.int64)
    product_support_size = 1
    yielded_count = 0

    def can_yield_more() -> bool:
        return yielded_limit is None or yielded_count < yielded_limit

    def can_add(block: LocalQDMCageBlock) -> bool:
        block_link_set = set(int(link_id) for link_id in block.link_ids)
        if used_links.intersection(block_link_set):
            return False
        if np.any(site_counts + block.site_counts > required_count):
            return False
        if multi_config.max_product_support_size is not None:
            next_size = int(product_support_size) * int(block.support_size)
            if next_size > multi_config.max_product_support_size:
                return False
        if multi_config.require_kinetic_separation and not _qdm_block_is_kinetically_separated(
            model,
            tuple(selected),
            block,
        ):
            return False
        return True

    def dfs(start: int) -> Iterator[MultiLocalQDMPadding]:
        nonlocal product_support_size, site_counts, yielded_count
        if not can_yield_more():
            return
        if len(selected) >= multi_config.min_blocks:
            fixed_blocks = tuple(selected)
            for padding in _iter_qdm_exterior_paddings_for_blocks(
                model,
                fixed_blocks,
                config=multi_config,
            ):
                if not can_yield_more():
                    return
                yielded_count += 1
                yield padding
            if not can_yield_more():
                return
        if len(selected) >= max_blocks:
            return

        for block_index in range(start, len(blocks)):
            block = blocks[block_index]
            if not can_add(block):
                continue
            block_link_set = set(int(link_id) for link_id in block.link_ids)
            selected.append(block)
            used_links.update(block_link_set)
            old_site_counts = site_counts.copy()
            site_counts = site_counts + block.site_counts
            old_product_support_size = product_support_size
            product_support_size *= int(block.support_size)
            try:
                yield from dfs(block_index + 1)
            finally:
                product_support_size = old_product_support_size
                site_counts = old_site_counts
                used_links.difference_update(block_link_set)
                selected.pop()
            if not can_yield_more():
                return

    yield from dfs(0)


def find_multi_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> list[MultiLocalQDMPadding]:
    """Find shared-exterior paddings built from a pool of local QDM blocks.

    This materialized API keeps the original raw-padding semantics:
    ``config.max_paddings`` is the maximum number of candidate paddings returned.
    Certification helpers use :func:`iter_multi_qdm_block_paddings` directly so
    they can keep trying candidates until enough certified cages are found.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    if multi_config.max_paddings == 0:
        return []
    return list(
        iter_multi_qdm_block_paddings(
            model,
            block_pool,
            config=multi_config,
            max_yielded=multi_config.max_paddings,
        )
    )


def _qdm_action_plaquette_class(
    action: _QDMGlobalPlaquetteAction,
    blocks: Sequence[LocalQDMCageBlock],
) -> str:
    """Classify a plaquette action relative to selected local blocks."""
    action_link_set = {int(link_id) for link_id in action.links}
    owner_link_sets = [set(int(link_id) for link_id in block.link_ids) for block in blocks]
    owners = {
        owner
        for owner, link_set in enumerate(owner_link_sets)
        if action_link_set.intersection(link_set)
    }
    if len(owners) > 1:
        return "multi_block_spacer"
    if not owners:
        return "pure_exterior"

    owner = next(iter(owners))
    if action_link_set.issubset(owner_link_sets[owner]):
        active_ids = {int(pid) for pid in blocks[owner].active_plaquette_ids}
        if int(action.plaquette_id) in active_ids:
            return "single_block_active"
        return "single_block_internal"
    return "single_block_boundary"


def _qdm_pattern_compatible_with_block_support(
    block: LocalQDMCageBlock,
    action: _QDMGlobalPlaquetteAction,
    pattern: npt.NDArray[np.int64],
) -> bool:
    """Return whether a plaquette pattern can occur on one block support."""
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(block.link_ids)}
    local_indices: list[int] = []
    required_values: list[int] = []
    for position, link_id in enumerate(action.links):
        local_index = local_index_by_link.get(int(link_id))
        if local_index is None:
            continue
        local_indices.append(int(local_index))
        required_values.append(int(pattern[int(position)]))

    if not local_indices:
        return True

    support_values = np.asarray(block.support_configs, dtype=np.int64)[:, local_indices]
    required = np.asarray(required_values, dtype=np.int64)
    return bool(np.any(np.all(support_values == required, axis=1)))


def _qdm_exterior_flippability_preferences_by_variable(
    model: object,
    exterior_link_ids: npt.NDArray[np.int64],
    blocks: Sequence[LocalQDMCageBlock],
    *,
    include_exterior_only: bool,
) -> list[list[_QDMExteriorFlippabilityPreference]]:
    """Return plaquette-flippability preferences touched by each exterior variable.

    A preference stores exterior-link patterns that would allow a plaquette to be
    flippable for at least one product-support configuration of the selected
    blocks.  The DFS value ordering can then prefer assignments that destroy
    these dangerous patterns early, especially on spacer/boundary plaquettes.
    """
    n_exterior = int(exterior_link_ids.size)
    exterior_index_by_link = {
        int(link_id): int(exterior_index)
        for exterior_index, link_id in enumerate(exterior_link_ids)
    }
    preferences_by_variable: list[list[_QDMExteriorFlippabilityPreference]] = [
        [] for _ in range(n_exterior)
    ]

    weight_by_class = {
        "multi_block_spacer": 256,
        "single_block_boundary": 96,
        "pure_exterior": 16,
        "single_block_active": 8,
        "single_block_internal": 4,
    }

    for action in _qdm_global_plaquette_actions(model):
        exterior_positions: list[int] = []
        exterior_indices: list[int] = []
        for position, link_id in enumerate(action.links):
            exterior_index = exterior_index_by_link.get(int(link_id))
            if exterior_index is None:
                continue
            exterior_positions.append(int(position))
            exterior_indices.append(int(exterior_index))
        if not exterior_indices:
            continue

        plaquette_class = _qdm_action_plaquette_class(action, blocks)
        if plaquette_class == "pure_exterior" and not include_exterior_only:
            continue

        dangerous_patterns: list[tuple[int, ...]] = []
        for pattern in (action.pattern0, action.pattern1):
            if not all(
                _qdm_pattern_compatible_with_block_support(block, action, pattern)
                for block in blocks
            ):
                continue
            dangerous_patterns.append(
                tuple(int(pattern[position]) for position in exterior_positions)
            )

        if not dangerous_patterns:
            continue
        unique_patterns = tuple(
            np.asarray(pattern, dtype=np.int64) for pattern in sorted(set(dangerous_patterns))
        )
        preference = _QDMExteriorFlippabilityPreference(
            plaquette_id=int(action.plaquette_id),
            plaquette_class=plaquette_class,
            exterior_indices=np.asarray(exterior_indices, dtype=np.int64),
            dangerous_patterns=unique_patterns,
            weight=int(weight_by_class.get(plaquette_class, 1)),
        )
        for exterior_index in exterior_indices:
            preferences_by_variable[int(exterior_index)].append(preference)

    return preferences_by_variable


def _qdm_count_compatible_dangerous_patterns(
    preference: _QDMExteriorFlippabilityPreference,
    *,
    exterior_config: npt.NDArray[np.int64],
    assigned: npt.NDArray[np.bool_],
    trial_variable: int | None = None,
    trial_value: int | None = None,
) -> int:
    """Count dangerous patterns still compatible with the current partial branch."""
    count = 0
    for pattern in preference.dangerous_patterns:
        compatible = True
        for exterior_index, required_value in zip(
            preference.exterior_indices,
            pattern,
            strict=True,
        ):
            index = int(exterior_index)
            if trial_variable is not None and index == int(trial_variable):
                value = int(trial_value)  # type: ignore[arg-type]
            elif bool(assigned[index]):
                value = int(exterior_config[index])
            else:
                continue
            if value != int(required_value):
                compatible = False
                break
        if compatible:
            count += 1
    return count


def _qdm_exterior_variable_order(
    model: object,
    exterior_link_ids: npt.NDArray[np.int64],
    site_exterior_links: dict[int, npt.NDArray[np.int64]],
    site_targets: dict[int, int],
    *,
    fixed_link_sets: Sequence[set[int]],
    require_static_exterior: bool,
) -> npt.NDArray[np.int64]:
    """Return a deterministic DFS order for exterior QDM padding links.

    The first padding implementation used only local site-constraint scores.
    That is correct, but it may enumerate many globally legal exterior
    completions before touching the boundary/spacer links that decide whether a
    candidate certifies.  This order prioritizes links on plaquettes touching
    selected blocks, then links on exterior-only plaquettes when a static
    exterior is requested, while preserving the old site-constraint preference
    as a secondary signal.
    """
    n_exterior = int(exterior_link_ids.size)
    exterior_index_by_link = {
        int(link_id): int(exterior_index)
        for exterior_index, link_id in enumerate(exterior_link_ids)
    }
    link_owner: dict[int, int] = {}
    for owner, link_set in enumerate(fixed_link_sets):
        for link_id in link_set:
            link_owner[int(link_id)] = int(owner)

    scores = np.zeros(n_exterior, dtype=np.int64)

    for site_id, exterior_indices in site_exterior_links.items():
        n_site_exterior = int(exterior_indices.size)
        target = int(site_targets[int(site_id)])
        if n_site_exterior == 0:
            continue
        if target in {0, n_site_exterior}:
            weight = 256
        elif target in {1, n_site_exterior - 1}:
            weight = 96
        else:
            weight = 32
        for exterior_index in exterior_indices:
            scores[int(exterior_index)] += weight

    for action in _qdm_global_plaquette_actions(model):
        exterior_indices = [
            exterior_index_by_link[int(link_id)]
            for link_id in action.links
            if int(link_id) in exterior_index_by_link
        ]
        if not exterior_indices:
            continue

        owners = {
            link_owner[int(link_id)] for link_id in action.links if int(link_id) in link_owner
        }
        if len(owners) > 1:
            # Spacer plaquettes between independent blocks are the most useful
            # early decisions when kinetic separation is relaxed.
            plaquette_weight = 4096
        elif owners:
            # Boundary plaquettes touching one selected block determine the
            # one-hop leakage/certification pattern.
            plaquette_weight = 2048
        elif require_static_exterior:
            # Exterior-only plaquettes must be frozen; decide their links before
            # unrelated bulk variables so static branches are pruned earlier.
            plaquette_weight = 512
        else:
            plaquette_weight = 16

        for exterior_index in exterior_indices:
            scores[int(exterior_index)] += plaquette_weight

    # Use the physical link id, not the exterior-array position, as the final
    # tie-breaker so the order is stable under equivalent array construction.
    return np.lexsort((exterior_link_ids, -scores)).astype(np.int64)


def _qdm_static_exterior_plaquettes_by_variable(
    model: object,
    exterior_link_ids: npt.NDArray[np.int64],
    *,
    fixed_link_set: set[int],
) -> list[list[_QDMExteriorStaticPlaquette]]:
    """Return exterior-only static plaquette checks touched by each variable."""
    n_exterior = int(exterior_link_ids.size)
    exterior_index_by_link = {
        int(link_id): int(exterior_index)
        for exterior_index, link_id in enumerate(exterior_link_ids)
    }
    by_variable: list[list[_QDMExteriorStaticPlaquette]] = [[] for _ in range(n_exterior)]

    for action in _qdm_global_plaquette_actions(model):
        action_links = [int(link_id) for link_id in action.links]
        if any(link_id in fixed_link_set for link_id in action_links):
            continue
        if any(link_id not in exterior_index_by_link for link_id in action_links):
            continue
        exterior_indices = np.asarray(
            [exterior_index_by_link[link_id] for link_id in action_links],
            dtype=np.int64,
        )
        static_plaquette = _QDMExteriorStaticPlaquette(
            plaquette_id=int(action.plaquette_id),
            exterior_indices=exterior_indices,
            pattern0=action.pattern0,
            pattern1=action.pattern1,
        )
        for exterior_index in exterior_indices:
            by_variable[int(exterior_index)].append(static_plaquette)

    return by_variable


def _qdm_static_exterior_checks_pass(
    static_plaquettes: Sequence[_QDMExteriorStaticPlaquette],
    *,
    exterior_config: npt.NDArray[np.int64],
    assigned: npt.NDArray[np.bool_],
) -> bool:
    """Reject a branch once a required-static exterior plaquette is flippable."""
    for static_plaquette in static_plaquettes:
        exterior_indices = static_plaquette.exterior_indices
        if not bool(np.all(assigned[exterior_indices])):
            continue
        values = exterior_config[exterior_indices]
        if np.array_equal(values, static_plaquette.pattern0) or np.array_equal(
            values,
            static_plaquette.pattern1,
        ):
            return False
    return True


def _qdm_exterior_value_order(
    exterior_variable: int,
    *,
    exterior_config: npt.NDArray[np.int64],
    assigned: npt.NDArray[np.bool_],
    sites_by_exterior_variable: Sequence[Sequence[int]],
    site_exterior_links: dict[int, npt.NDArray[np.int64]],
    site_targets: dict[int, int],
    flippability_preferences_by_variable: (
        Sequence[Sequence[_QDMExteriorFlippabilityPreference]] | None
    ) = None,
) -> tuple[int, ...]:
    """Order binary choices by site constraints and spacer flippability risk."""
    scored_values: list[tuple[int, int]] = []
    preferences = (
        ()
        if flippability_preferences_by_variable is None
        else flippability_preferences_by_variable[int(exterior_variable)]
    )

    for value in (0, 1):
        score = 0
        feasible = True
        for site_id in sites_by_exterior_variable[int(exterior_variable)]:
            exterior_indices = site_exterior_links[int(site_id)]
            assigned_local = assigned[exterior_indices]
            occupied = int(np.sum(exterior_config[exterior_indices[assigned_local]]))
            unassigned = int(exterior_indices.size - np.count_nonzero(assigned_local))
            remaining_need = int(site_targets[int(site_id)]) - occupied
            remaining_after = unassigned - 1
            next_need = remaining_need - int(value)
            if next_need < 0 or next_need > remaining_after:
                feasible = False
                break
            if next_need in {0, remaining_after}:
                score += 4
            if remaining_after == 0:
                score += 8
        if not feasible:
            continue

        for preference in preferences:
            before = _qdm_count_compatible_dangerous_patterns(
                preference,
                exterior_config=exterior_config,
                assigned=assigned,
            )
            if before == 0:
                continue
            after = _qdm_count_compatible_dangerous_patterns(
                preference,
                exterior_config=exterior_config,
                assigned=assigned,
                trial_variable=int(exterior_variable),
                trial_value=int(value),
            )
            killed = before - after
            score += int(preference.weight) * int(killed)
            if after == 0:
                score += 2 * int(preference.weight)

        scored_values.append((score, int(value)))

    if not scored_values:
        return (0, 1)
    scored_values.sort(key=lambda item: (-item[0], item[1]))
    return tuple(value for _, value in scored_values)


def _iter_qdm_exterior_paddings_for_blocks(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig,
    factorized: bool = False,
) -> Iterator[MultiLocalQDMPadding | FactorizedLocalQDMPadding]:
    fixed_blocks = tuple(blocks)
    if not fixed_blocks:
        return
    if config.max_paddings_per_packing == 0:
        return
    if not _qdm_blocks_are_pairwise_link_disjoint(fixed_blocks):
        return
    if config.require_kinetic_separation and not _qdm_blocks_are_kinetically_separated(
        model,
        fixed_blocks,
    ):
        return

    required_count = int(getattr(model, "required_count", 1))
    total_site_counts = np.zeros(int(model.lattice.num_sites), dtype=np.int64)
    block_link_set: set[int] = set()
    for block in fixed_blocks:
        total_site_counts += np.asarray(block.site_counts, dtype=np.int64)
        block_link_set.update(int(link_id) for link_id in block.link_ids)
    if np.any(total_site_counts > required_count):
        return

    n_global_links = int(model.lattice.num_links)
    exterior_link_ids = np.asarray(
        [link_id for link_id in range(n_global_links) if link_id not in block_link_set],
        dtype=np.int64,
    )
    exterior_index_by_link = {int(link_id): i for i, link_id in enumerate(exterior_link_ids)}
    n_exterior = int(exterior_link_ids.size)

    site_targets: dict[int, int] = {}
    site_exterior_links: dict[int, npt.NDArray[np.int64]] = {}
    for site_id in range(int(model.lattice.num_sites)):
        incident = [int(link_id) for link_id in model.lattice.incident_links(int(site_id))]
        exterior_incident = [
            exterior_index_by_link[link_id]
            for link_id in incident
            if link_id in exterior_index_by_link
        ]
        target = required_count - int(total_site_counts[int(site_id)])
        if target < 0 or target > len(exterior_incident):
            return
        site_targets[int(site_id)] = int(target)
        site_exterior_links[int(site_id)] = np.asarray(exterior_incident, dtype=np.int64)

    if n_exterior == 0:
        exterior_config = np.zeros(0, dtype=np.int64)
        if factorized:
            padding = FactorizedLocalQDMPadding(
                block_ids=tuple(int(block.block_id) for block in fixed_blocks),
                exterior_link_ids=exterior_link_ids,
                exterior_config=exterior_config,
            )
            reason, _sector_validation, _max_touched = _factorized_padding_validation_reason(
                model,
                fixed_blocks,
                padding,
                config,
            )
            if reason is None:
                yield padding
        else:
            padding = _make_qdm_multi_padding_from_exterior(
                model,
                fixed_blocks,
                exterior_link_ids=exterior_link_ids,
                exterior_config=exterior_config,
            )
            if _multi_padding_passes_global_filters(model, padding, fixed_blocks, config):
                yield padding
        return

    variable_order = _qdm_exterior_variable_order(
        model,
        exterior_link_ids,
        site_exterior_links,
        site_targets,
        fixed_link_sets=[set(int(link_id) for link_id in block.link_ids) for block in fixed_blocks],
        require_static_exterior=config.require_static_exterior,
    )

    exterior_config = np.zeros(n_exterior, dtype=np.int64)
    assigned = np.zeros(n_exterior, dtype=bool)
    sites_by_exterior_variable: list[list[int]] = [[] for _ in range(n_exterior)]
    for site_id, exterior_indices in site_exterior_links.items():
        for exterior_index in exterior_indices:
            sites_by_exterior_variable[int(exterior_index)].append(int(site_id))

    static_exterior_plaquettes_by_variable = (
        _qdm_static_exterior_plaquettes_by_variable(
            model,
            exterior_link_ids,
            fixed_link_set=block_link_set,
        )
        if config.require_static_exterior
        else [[] for _ in range(n_exterior)]
    )
    flippability_preferences_by_variable = _qdm_exterior_flippability_preferences_by_variable(
        model,
        exterior_link_ids,
        fixed_blocks,
        include_exterior_only=config.require_static_exterior,
    )

    nodes_visited = 0
    yielded_count = 0

    def partial_site_check(site_id: int) -> bool:
        exterior_indices = site_exterior_links[site_id]
        target = site_targets[site_id]
        if exterior_indices.size == 0:
            return target == 0
        assigned_local = assigned[exterior_indices]
        occupied = int(np.sum(exterior_config[exterior_indices[assigned_local]]))
        unassigned = int(exterior_indices.size - np.count_nonzero(assigned_local))
        if occupied > target:
            return False
        if occupied + unassigned < target:
            return False
        if unassigned == 0 and occupied != target:
            return False
        return True

    def full_check() -> bool:
        for site_id in range(int(model.lattice.num_sites)):
            if not partial_site_check(int(site_id)):
                return False
        return True

    def dfs(depth: int) -> Iterator[MultiLocalQDMPadding]:
        nonlocal nodes_visited, yielded_count
        if yielded_count >= config.max_paddings_per_packing:
            return
        if config.max_dfs_nodes is not None and nodes_visited >= config.max_dfs_nodes:
            return
        nodes_visited += 1

        if depth == n_exterior:
            if full_check():
                if factorized:
                    padding = FactorizedLocalQDMPadding(
                        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
                        exterior_link_ids=exterior_link_ids,
                        exterior_config=exterior_config.copy(),
                    )
                    reason, _sector_validation, _max_touched = (
                        _factorized_padding_validation_reason(
                            model,
                            fixed_blocks,
                            padding,
                            config,
                        )
                    )
                    passes_filters = reason is None
                else:
                    padding = _make_qdm_multi_padding_from_exterior(
                        model,
                        fixed_blocks,
                        exterior_link_ids=exterior_link_ids,
                        exterior_config=exterior_config.copy(),
                    )
                    passes_filters = _multi_padding_passes_global_filters(
                        model,
                        padding,
                        fixed_blocks,
                        config,
                    )
                if passes_filters:
                    yielded_count += 1
                    yield padding
            return

        exterior_variable = int(variable_order[depth])
        for value in _qdm_exterior_value_order(
            exterior_variable,
            exterior_config=exterior_config,
            assigned=assigned,
            sites_by_exterior_variable=sites_by_exterior_variable,
            site_exterior_links=site_exterior_links,
            site_targets=site_targets,
            flippability_preferences_by_variable=flippability_preferences_by_variable,
        ):
            if yielded_count >= config.max_paddings_per_packing:
                return
            exterior_config[exterior_variable] = value
            assigned[exterior_variable] = True
            touched_sites = sites_by_exterior_variable[exterior_variable]
            touched_static_plaquettes = static_exterior_plaquettes_by_variable[exterior_variable]
            if all(partial_site_check(site_id) for site_id in touched_sites) and (
                not touched_static_plaquettes
                or _qdm_static_exterior_checks_pass(
                    touched_static_plaquettes,
                    exterior_config=exterior_config,
                    assigned=assigned,
                )
            ):
                yield from dfs(depth + 1)
            assigned[exterior_variable] = False
            exterior_config[exterior_variable] = 0

    yield from dfs(0)


def _find_qdm_exterior_paddings_for_blocks(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig,
) -> list[MultiLocalQDMPadding]:
    return list(_iter_qdm_exterior_paddings_for_blocks(model, blocks, config=config))


def factorized_qdm_padding_from_multi_padding(
    padding: MultiLocalQDMPadding,
) -> FactorizedLocalQDMPadding:
    """Drop the materialized Cartesian-product support from an old padding."""
    return FactorizedLocalQDMPadding(
        block_ids=padding.block_ids,
        exterior_link_ids=padding.exterior_link_ids,
        exterior_config=padding.exterior_config,
    )


def iter_factorized_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
    max_yielded: int | None = None,
) -> Iterator[FactorizedLocalQDMPadding]:
    """Yield exterior assignments without materializing block support products.

    This mirrors :func:`iter_multi_qdm_block_paddings`, but the returned object
    contains only the block ids and shared exterior configuration.  The search
    therefore remains usable when ``prod(block.support_size)`` is too large to
    enumerate.  ``max_product_support_size`` is intentionally not applied on
    this path because the Cartesian-product support is never materialized.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    yielded_limit = multi_config.max_padding_attempts if max_yielded is None else max_yielded
    if yielded_limit is not None and yielded_limit <= 0:
        return

    blocks_tuple = tuple(block_pool)
    block_ids = [int(block.block_id) for block in blocks_tuple]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("block_pool contains duplicate block_id values.")
    max_blocks = (
        len(blocks_tuple)
        if multi_config.max_blocks is None
        else min(int(multi_config.max_blocks), len(blocks_tuple))
    )
    yielded = 0

    for block_count in range(multi_config.min_blocks, max_blocks + 1):
        for blocks in itertools.combinations(blocks_tuple, block_count):
            if yielded_limit is not None and yielded >= yielded_limit:
                return
            if not _qdm_blocks_are_pairwise_link_disjoint(blocks):
                continue
            separated = _qdm_blocks_are_kinetically_separated(model, blocks)
            if multi_config.require_kinetic_separation and not separated:
                continue
            for padding in _iter_qdm_exterior_paddings_for_blocks(
                model,
                blocks,
                config=multi_config,
                factorized=True,
            ):
                if not isinstance(padding, FactorizedLocalQDMPadding):
                    raise TypeError("factorized padding iterator returned an unexpected object.")
                yield padding
                yielded += 1
                if yielded_limit is not None and yielded >= yielded_limit:
                    return


def find_factorized_qdm_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> list[FactorizedLocalQDMPadding]:
    """Materialize a bounded list of factorized QDM exterior assignments."""
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    if multi_config.max_paddings == 0:
        return []
    return list(
        itertools.islice(
            iter_factorized_qdm_block_paddings(
                model,
                block_pool,
                config=multi_config,
                max_yielded=multi_config.max_paddings,
            ),
            multi_config.max_paddings,
        )
    )


def _factorized_padding_reference_config(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding,
    *,
    support_indices: Sequence[int] | None = None,
) -> npt.NDArray[np.int64]:
    config = np.zeros(int(model.lattice.num_links), dtype=np.int64)
    config[padding.exterior_link_ids] = padding.exterior_config
    indices = [0] * len(blocks) if support_indices is None else list(support_indices)
    if len(indices) != len(blocks):
        raise ValueError("support_indices must have one entry per block.")
    for block, support_index in zip(blocks, indices, strict=True):
        config[block.link_ids] = block.support_configs[int(support_index)]
    return config


def _factorized_padding_validation_reason(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: FactorizedLocalQDMPadding,
    config: LocalQDMMultiPaddingConfig,
) -> tuple[str | None, str, int]:
    fixed_blocks = tuple(blocks)
    if not fixed_blocks:
        return "no_blocks", "not_checked", 0
    if tuple(int(block.block_id) for block in fixed_blocks) != padding.block_ids:
        return "block_id_mismatch", "not_checked", 0
    if not _qdm_blocks_are_pairwise_link_disjoint(fixed_blocks):
        return "overlapping_block_links", "not_checked", 0

    owner_by_link: dict[int, int] = {}
    for block_index, block in enumerate(fixed_blocks):
        for link_id in block.link_ids:
            owner_by_link[int(link_id)] = int(block_index)
    exterior_ids = set(int(link_id) for link_id in padding.exterior_link_ids)
    expected_exterior = set(range(int(model.lattice.num_links))) - set(owner_by_link)
    if exterior_ids != expected_exterior:
        return "incomplete_link_partition", "not_checked", 0

    max_touched = 0
    for action in _qdm_global_plaquette_actions(model):
        owners = {
            owner_by_link[int(link_id)] for link_id in action.links if int(link_id) in owner_by_link
        }
        max_touched = max(max_touched, len(owners))
    if max_touched > 1:
        return "plaquette_touches_multiple_blocks", "not_checked", max_touched

    reference = _factorized_padding_reference_config(model, fixed_blocks, padding)
    if not _global_configs_satisfy_qdm_constraints(model, reference):
        return "constraint_violation", "not_checked", max_touched

    sector_validation = "disabled"
    if config.include_sectors:
        sector_validation = "reference_and_single_block_variations"
        if not _global_configs_satisfy_model_sectors(model, reference):
            return "sector_violation", sector_validation, max_touched
        for block_index, block in enumerate(fixed_blocks):
            for support_index in range(block.support_size):
                support_indices = [0] * len(fixed_blocks)
                support_indices[block_index] = int(support_index)
                varied = _factorized_padding_reference_config(
                    model,
                    fixed_blocks,
                    padding,
                    support_indices=support_indices,
                )
                if not _global_configs_satisfy_model_sectors(model, varied):
                    return "sector_variation", sector_validation, max_touched

    if config.require_static_exterior:
        block_link_set = set(owner_by_link)
        for action in _qdm_global_plaquette_actions(model):
            if any(int(link_id) in block_link_set for link_id in action.links):
                continue
            if _qdm_plaquette_is_flippable_from_action(reference, action):
                return "nonstatic_exterior", sector_validation, max_touched

    return None, sector_validation, max_touched


def _make_qdm_multi_padding_from_exterior(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    exterior_link_ids: npt.NDArray[np.int64],
    exterior_config: npt.NDArray[np.int64],
) -> MultiLocalQDMPadding:
    fixed_blocks = tuple(blocks)
    support_ranges = [range(int(block.support_size)) for block in fixed_blocks]
    support_tuples = list(itertools.product(*support_ranges))
    n_support = len(support_tuples)
    n_global_links = int(model.lattice.num_links)

    full_configs = np.zeros((n_support, n_global_links), dtype=np.int64)
    amplitudes = np.ones(n_support, dtype=np.complex128)
    block_support_indices = np.zeros((n_support, len(fixed_blocks)), dtype=np.int64)
    exterior_link_ids = np.asarray(exterior_link_ids, dtype=np.int64)
    exterior_config = np.asarray(exterior_config, dtype=np.int64)

    for row_index, support_tuple in enumerate(support_tuples):
        if exterior_link_ids.size:
            full_configs[row_index, exterior_link_ids] = exterior_config
        for block_position, (block, support_index) in enumerate(
            zip(fixed_blocks, support_tuple, strict=True)
        ):
            support_index = int(support_index)
            full_configs[row_index, np.asarray(block.link_ids, dtype=np.int64)] = (
                block.support_configs[support_index]
            )
            amplitudes[row_index] *= complex(block.amplitudes[support_index])
            block_support_indices[row_index, block_position] = support_index

    return MultiLocalQDMPadding(
        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
        exterior_link_ids=exterior_link_ids.copy(),
        exterior_config=exterior_config.copy(),
        global_support_configs=full_configs,
        global_amplitudes=amplitudes,
        block_support_indices=block_support_indices,
    )


def _multi_padding_passes_global_filters(
    model: object,
    padding: MultiLocalQDMPadding,
    blocks: Sequence[LocalQDMCageBlock],
    config: LocalQDMMultiPaddingConfig,
) -> bool:
    if not _global_configs_satisfy_qdm_constraints(model, padding.global_support_configs):
        return False
    if config.include_sectors and not _global_configs_satisfy_model_sectors(
        model,
        padding.global_support_configs,
    ):
        return False
    if config.require_static_exterior and not _multi_padding_has_static_exterior(
        model,
        padding,
        blocks,
    ):
        return False
    return True


def _qdm_multi_block_certification_actions(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    config: LocalQDMMultiPaddingConfig,
) -> tuple[_QDMGlobalPlaquetteAction, ...]:
    actions = _qdm_global_plaquette_actions(model)
    if not config.require_static_exterior:
        return actions

    block_link_set = {
        int(link_id) for block in blocks for link_id in np.asarray(block.link_ids, dtype=np.int64)
    }
    return tuple(
        action
        for action in actions
        if any(int(link_id) in block_link_set for link_id in action.links)
    )


def _multi_padding_has_static_exterior(
    model: object,
    padding: MultiLocalQDMPadding,
    blocks: Sequence[LocalQDMCageBlock],
) -> bool:
    block_link_set = {
        int(link_id) for block in blocks for link_id in np.asarray(block.link_ids, dtype=np.int64)
    }
    if padding.global_support_configs.shape[0] == 0:
        return True

    # Plaquettes disjoint from every block only see the shared exterior config,
    # so one support row is enough.  Avoid constructing flipped configs here;
    # we only need to know whether an exterior plaquette is flippable.
    reference_config = padding.global_support_configs[0]
    for action in _qdm_global_plaquette_actions(model):
        if any(int(link_id) in block_link_set for link_id in action.links):
            continue
        if _qdm_plaquette_is_flippable_from_action(reference_config, action):
            return False
    return True


def find_shared_qdm_exterior_paddings(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    config: LocalQDMPaddingConfig | None = None,
) -> list[LocalQDMPadding]:
    """Find shared exterior configurations compatible with a local QDM cage.

    A shared exterior is a single assignment on all nonlocal links such that
    every local support configuration becomes a full valid dimer covering.  This
    is the simplest product padding that preserves the local superposition.
    """
    padding_config = LocalQDMPaddingConfig() if config is None else config
    local_link_ids = np.asarray(local_record.local_link_ids, dtype=np.int64)
    local_link_set = set(int(link_id) for link_id in local_link_ids)
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(local_link_ids)}

    n_global_links = int(model.lattice.num_links)
    exterior_link_ids = np.asarray(
        [link_id for link_id in range(n_global_links) if link_id not in local_link_set],
        dtype=np.int64,
    )
    exterior_index_by_link = {int(link_id): i for i, link_id in enumerate(exterior_link_ids)}
    n_exterior = int(exterior_link_ids.size)

    support_configs = np.asarray(local_record.support_configs, dtype=np.int64)
    if support_configs.ndim != 2:
        raise ValueError("local_record.support_configs must have shape (support, n_local_links).")

    required_count = int(getattr(model, "required_count", 1))
    site_targets: dict[int, int] = {}
    site_exterior_links: dict[int, npt.NDArray[np.int64]] = {}

    for site_id in range(int(model.lattice.num_sites)):
        incident = [int(link_id) for link_id in model.lattice.incident_links(int(site_id))]
        local_incident = [
            local_index_by_link[link_id] for link_id in incident if link_id in local_index_by_link
        ]
        exterior_incident = [
            exterior_index_by_link[link_id]
            for link_id in incident
            if link_id in exterior_index_by_link
        ]

        if local_incident:
            local_counts = np.sum(support_configs[:, local_incident], axis=1).astype(np.int64)
        else:
            local_counts = np.zeros(support_configs.shape[0], dtype=np.int64)

        if np.unique(local_counts).size != 1:
            return []

        target = required_count - int(local_counts[0])
        if target < 0 or target > len(exterior_incident):
            return []

        site_targets[int(site_id)] = int(target)
        site_exterior_links[int(site_id)] = np.asarray(exterior_incident, dtype=np.int64)

    if n_exterior == 0:
        exterior_config = np.zeros(0, dtype=np.int64)
        padding = _make_qdm_padding_from_exterior(
            model,
            local_record,
            exterior_link_ids=exterior_link_ids,
            exterior_config=exterior_config,
        )
        if _padding_passes_global_filters(model, padding, local_record, padding_config):
            return [padding]
        return []

    variable_order = _qdm_exterior_variable_order(
        model,
        exterior_link_ids,
        site_exterior_links,
        site_targets,
        fixed_link_sets=[local_link_set],
        require_static_exterior=padding_config.require_static_exterior,
    )

    exterior_config = np.zeros(n_exterior, dtype=np.int64)
    assigned = np.zeros(n_exterior, dtype=bool)
    sites_by_exterior_variable: list[list[int]] = [[] for _ in range(n_exterior)]
    for site_id, exterior_indices in site_exterior_links.items():
        for exterior_index in exterior_indices:
            sites_by_exterior_variable[int(exterior_index)].append(int(site_id))

    static_exterior_plaquettes_by_variable = (
        _qdm_static_exterior_plaquettes_by_variable(
            model,
            exterior_link_ids,
            fixed_link_set=local_link_set,
        )
        if padding_config.require_static_exterior
        else [[] for _ in range(n_exterior)]
    )

    paddings: list[LocalQDMPadding] = []
    nodes_visited = 0

    def partial_site_check(site_id: int) -> bool:
        exterior_indices = site_exterior_links[site_id]
        target = site_targets[site_id]
        if exterior_indices.size == 0:
            return target == 0
        assigned_local = assigned[exterior_indices]
        occupied = int(np.sum(exterior_config[exterior_indices[assigned_local]]))
        unassigned = int(exterior_indices.size - np.count_nonzero(assigned_local))
        if occupied > target:
            return False
        if occupied + unassigned < target:
            return False
        if unassigned == 0 and occupied != target:
            return False
        return True

    def full_check() -> bool:
        for site_id in range(int(model.lattice.num_sites)):
            if not partial_site_check(int(site_id)):
                return False
        return True

    def dfs(depth: int) -> None:
        nonlocal nodes_visited
        if len(paddings) >= padding_config.max_paddings_per_record:
            return
        if (
            padding_config.max_dfs_nodes is not None
            and nodes_visited >= padding_config.max_dfs_nodes
        ):
            return
        nodes_visited += 1

        if depth == n_exterior:
            if full_check():
                padding = _make_qdm_padding_from_exterior(
                    model,
                    local_record,
                    exterior_link_ids=exterior_link_ids,
                    exterior_config=exterior_config.copy(),
                )
                if _padding_passes_global_filters(model, padding, local_record, padding_config):
                    paddings.append(padding)
            return

        exterior_variable = int(variable_order[depth])
        for value in _qdm_exterior_value_order(
            exterior_variable,
            exterior_config=exterior_config,
            assigned=assigned,
            sites_by_exterior_variable=sites_by_exterior_variable,
            site_exterior_links=site_exterior_links,
            site_targets=site_targets,
        ):
            if len(paddings) >= padding_config.max_paddings_per_record:
                return
            exterior_config[exterior_variable] = value
            assigned[exterior_variable] = True
            touched_sites = sites_by_exterior_variable[exterior_variable]
            touched_static_plaquettes = static_exterior_plaquettes_by_variable[exterior_variable]
            if all(partial_site_check(site_id) for site_id in touched_sites) and (
                not touched_static_plaquettes
                or _qdm_static_exterior_checks_pass(
                    touched_static_plaquettes,
                    exterior_config=exterior_config,
                    assigned=assigned,
                )
            ):
                dfs(depth + 1)
            assigned[exterior_variable] = False
            exterior_config[exterior_variable] = 0

    dfs(0)
    return paddings


def _make_qdm_padding_from_exterior(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    exterior_link_ids: npt.NDArray[np.int64],
    exterior_config: npt.NDArray[np.int64],
) -> LocalQDMPadding:
    local_link_ids = np.asarray(local_record.local_link_ids, dtype=np.int64)
    support_configs = np.asarray(local_record.support_configs, dtype=np.int64)
    full_configs = np.zeros(
        (support_configs.shape[0], int(model.lattice.num_links)),
        dtype=np.int64,
    )
    full_configs[:, local_link_ids] = support_configs
    if exterior_link_ids.size:
        full_configs[:, exterior_link_ids] = np.asarray(exterior_config, dtype=np.int64)
    return LocalQDMPadding(
        exterior_link_ids=np.asarray(exterior_link_ids, dtype=np.int64).copy(),
        exterior_config=np.asarray(exterior_config, dtype=np.int64).copy(),
        global_support_configs=full_configs,
    )


def _padding_passes_global_filters(
    model: object,
    padding: LocalQDMPadding,
    local_record: LocalQDMCageRecord,
    config: LocalQDMPaddingConfig,
) -> bool:
    if not _padding_satisfies_qdm_constraints(model, padding):
        return False
    if config.include_sectors and not _padding_satisfies_model_sectors(model, padding):
        return False
    if config.require_static_exterior and not _padding_has_static_exterior(
        model,
        padding,
        local_record,
    ):
        return False
    return True


def _padding_satisfies_qdm_constraints(model: object, padding: LocalQDMPadding) -> bool:
    required_count = int(getattr(model, "required_count", 1))
    for config_row in padding.global_support_configs:
        for site_id in range(int(model.lattice.num_sites)):
            incident = np.asarray(model.lattice.incident_links(int(site_id)), dtype=np.int64)
            if int(np.sum(config_row[incident])) != required_count:
                return False
    return True


def _padding_satisfies_model_sectors(model: object, padding: LocalQDMPadding) -> bool:
    sectors = tuple(model.make_sectors())
    if not sectors:
        return True
    for config_row in padding.global_support_configs:
        for sector in sectors:
            if not sector.is_satisfied(config_row):
                return False
    return True


def _padding_has_static_exterior(
    model: object,
    padding: LocalQDMPadding,
    local_record: LocalQDMCageRecord,
) -> bool:
    local_link_set = set(int(link_id) for link_id in local_record.local_link_ids)
    if padding.global_support_configs.shape[0] == 0:
        return True

    reference_config = padding.global_support_configs[0]
    for action in _qdm_global_plaquette_actions(model):
        if any(int(link_id) in local_link_set for link_id in action.links):
            continue
        if _qdm_plaquette_is_flippable_from_action(reference_config, action):
            return False
    return True


find_qdm_multi_block_paddings = find_multi_qdm_block_paddings
