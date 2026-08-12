"""Pure region-geometry helpers for local cage-search proposals.

These helpers manipulate plaquette/link regions, stripe and snake geometry, and local index
layouts. They do not enumerate cage states or certify a cage and therefore form a lower-level
dependency of :mod:`qlinks.caging.local_search`.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.caging.local_search_types import (
    AdaptiveRegionProposalRecord,
    LocalQDMRegion,
    SnakeStripeKindPattern,
    StripeMotifComponentSubsetMode,
    StripeMotifSubsetMode,
)
from qlinks.variables import LocalSpace, VariableLayout


def _ordered_straight_stripe_plaquettes(
    model: object,
    plaquette_ids: Sequence[int] | npt.ArrayLike,
    *,
    direction: int,
) -> tuple[int, ...]:
    """Order plaquettes by anchor coordinate along a straight stripe."""
    ids = tuple(int(pid) for pid in np.asarray(plaquette_ids, dtype=np.int64))
    return tuple(
        sorted(
            ids,
            key=lambda pid: (
                _stripe_anchor_cell(model, int(pid))[int(direction)],
                _stripe_anchor_cell(model, int(pid)),
                int(pid),
            ),
        )
    )


def _stripe_motif_subsets(
    ordered_plaquette_ids: Sequence[int],
    *,
    motif_sizes: Sequence[int],
    subset_mode: StripeMotifSubsetMode,
) -> Iterator[tuple[int, ...]]:
    """Yield small plaquette subsets from a cyclic stripe skeleton."""
    ordered = tuple(int(pid) for pid in ordered_plaquette_ids)
    n_plaquettes = len(ordered)
    if n_plaquettes == 0:
        return

    for motif_size in sorted({int(size) for size in motif_sizes}):
        if motif_size <= 0 or motif_size > n_plaquettes:
            continue
        if subset_mode == "windows":
            for start in range(n_plaquettes):
                yield tuple(
                    ordered[(start + offset) % n_plaquettes] for offset in range(motif_size)
                )
            continue

        for combination in itertools.combinations(ordered, motif_size):
            yield tuple(int(pid) for pid in combination)


def _stripe_component_subsets(
    ordered_plaquette_ids: Sequence[int],
    *,
    component_sizes: Sequence[int] | None,
    subset_mode: StripeMotifComponentSubsetMode,
) -> Iterator[tuple[int, ...]]:
    """Yield merged stripe component subsets from a cyclic skeleton."""
    ordered = tuple(int(pid) for pid in ordered_plaquette_ids)
    n_plaquettes = len(ordered)
    if n_plaquettes == 0:
        return

    if subset_mode == "full":
        yield ordered
        return

    if component_sizes is None:
        sizes = (n_plaquettes,)
    else:
        sizes = tuple(sorted({int(size) for size in component_sizes}))

    for component_size in sizes:
        if component_size <= 0 or component_size > n_plaquettes:
            continue
        if subset_mode == "windows":
            for start in range(n_plaquettes):
                yield tuple(
                    ordered[(start + offset) % n_plaquettes] for offset in range(component_size)
                )
            continue

        for combination in itertools.combinations(ordered, component_size):
            yield tuple(int(pid) for pid in combination)


def _ordered_cycle_plaquettes_from_shared_graph(
    model: object,
    plaquette_ids: Sequence[int] | npt.ArrayLike,
) -> tuple[int, ...]:
    """Return one cyclic order when selected plaquettes form a degree-2 cycle."""
    selected = tuple(sorted({int(pid) for pid in np.asarray(plaquette_ids, dtype=np.int64)}))
    if len(selected) <= 2:
        return selected

    selected_set = set(selected)
    neighbor_map = _plaquette_shared_link_neighbor_map(model)
    induced: dict[int, list[int]] = {
        pid: sorted(int(nbr) for nbr in neighbor_map.get(pid, ()) if int(nbr) in selected_set)
        for pid in selected
    }
    if any(len(neighbors) != 2 for neighbors in induced.values()):
        return ()

    start = min(selected)
    ordered = [start]
    previous: int | None = None
    current = start
    while True:
        choices = [neighbor for neighbor in induced[current] if neighbor != previous]
        if not choices:
            return ()
        next_pid = choices[0]
        if next_pid == start:
            break
        if next_pid in ordered:
            return ()
        ordered.append(next_pid)
        previous, current = current, next_pid

    if len(ordered) != len(selected):
        return ()
    return tuple(int(pid) for pid in ordered)


def _stripe_plaquette_data(
    model: object,
    plaquette_kinds: tuple[str, ...] | None,
) -> list[tuple[int, tuple[int, ...], str]]:
    """Return ``(plaquette_id, anchor_cell, kind)`` entries for QDM stripe proposals."""
    allowed_kinds = None if plaquette_kinds is None else set(str(kind) for kind in plaquette_kinds)
    entries: list[tuple[int, tuple[int, ...], str]] = []

    for plaquette_id in model.plaquette_ids():
        plaquette_id = int(plaquette_id)
        plaquette = model.lattice.plaquettes[plaquette_id]
        kind = str(plaquette.kind)
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue

        entries.append(
            (
                plaquette_id,
                _stripe_anchor_cell(model, plaquette_id),
                kind,
            )
        )

    return entries


def _stripe_anchor_cell(model: object, plaquette_id: int) -> tuple[int, ...]:
    """Return a stable plaquette cell used by stripe proposals.

    Most lattices store ``anchor_cell`` directly.  Older triangular-rhombus
    plaquettes did not, so we fall back to the first boundary site's cell, which
    matches the construction anchor for those rhombi.
    """
    try:
        return tuple(int(value) for value in model.lattice.plaquette_anchor_cell(int(plaquette_id)))
    except ValueError:
        plaquette = model.lattice.plaquettes[int(plaquette_id)]
        first_site = model.lattice.sites[int(plaquette.sites[0])]
        return tuple(int(value) for value in first_site.cell)


def _default_stripe_directions(
    plaquette_data: Sequence[tuple[int, tuple[int, ...], str]],
) -> tuple[int, ...]:
    ndim = max(len(cell) for _, cell, _ in plaquette_data)
    directions: list[int] = []
    for axis in range(ndim):
        values = {int(cell[axis]) for _, cell, _ in plaquette_data if len(cell) > axis}
        if len(values) > 1:
            directions.append(axis)

    if directions:
        return tuple(directions)
    return tuple(range(ndim))


def _validate_stripe_direction(
    direction: int,
    plaquette_data: Sequence[tuple[int, tuple[int, ...], str]],
) -> None:
    if direction < 0:
        raise ValueError("Stripe direction must be non-negative.")
    if any(len(cell) <= direction for _, cell, _ in plaquette_data):
        raise ValueError(f"Stripe direction {direction} is outside plaquette anchor dimension.")


def _transverse_coordinates(cell: tuple[int, ...], direction: int) -> tuple[int, ...]:
    return tuple(int(value) for axis, value in enumerate(cell) if axis != int(direction))


def _cell_in_stripe_band(
    model: object,
    cell: tuple[int, ...],
    *,
    direction: int,
    transverse_origin: tuple[int, ...],
    width: int,
) -> bool:
    transverse_axes = [axis for axis in range(len(cell)) if axis != int(direction)]
    if len(transverse_axes) != len(transverse_origin):
        raise ValueError("transverse_origin has the wrong dimension for this stripe direction.")

    periodic = _lattice_is_periodic(model)
    for origin, axis in zip(transverse_origin, transverse_axes, strict=True):
        value = int(cell[axis])
        origin = int(origin)
        period = _lattice_axis_period(model, axis) if periodic else None

        if period is None:
            if value < origin or value >= origin + int(width):
                return False
            continue

        if int(width) >= period:
            continue

        distance = (value - origin) % period
        if distance < 0 or distance >= int(width):
            return False

    return True


def _lattice_is_periodic(model: object) -> bool:
    boundary_condition = getattr(model.lattice, "boundary_condition", None)
    value = getattr(boundary_condition, "value", boundary_condition)
    return str(value) == "periodic"


def _lattice_axis_period(model: object, axis: int) -> int | None:
    if axis == 0 and hasattr(model.lattice, "lx"):
        return int(model.lattice.lx)
    if axis == 1 and hasattr(model.lattice, "ly"):
        return int(model.lattice.ly)
    return None


def _adaptive_seed_plaquette_ids(
    model: object,
    seed_plaquette_ids: Sequence[int] | npt.ArrayLike | None,
) -> npt.NDArray[np.int64]:
    if seed_plaquette_ids is None:
        ids = np.asarray([int(pid) for pid in model.plaquette_ids()], dtype=np.int64)
    else:
        ids = _unique_int_array(seed_plaquette_ids, name="seed_plaquette_ids")
    _validate_plaquette_ids(model, ids)
    return ids


def _plaquette_shared_link_neighbor_map(model: object) -> dict[int, frozenset[int]]:
    link_to_plaquettes: dict[int, list[int]] = defaultdict(list)
    plaquette_ids = tuple(int(pid) for pid in model.plaquette_ids())
    for plaquette_id in plaquette_ids:
        for link_id in model.lattice.plaquette_links(int(plaquette_id)):
            link_to_plaquettes[int(link_id)].append(int(plaquette_id))

    neighbors: dict[int, set[int]] = {int(pid): set() for pid in plaquette_ids}
    for incident_plaquettes in link_to_plaquettes.values():
        for left in incident_plaquettes:
            for right in incident_plaquettes:
                if int(left) != int(right):
                    neighbors[int(left)].add(int(right))

    return {
        int(plaquette_id): frozenset(sorted(neighbor_ids))
        for plaquette_id, neighbor_ids in neighbors.items()
    }


def _plaquette_shared_link_neighbor_edges(
    model: object,
    *,
    plaquette_kinds: tuple[str, ...] | None,
    allow_kind_changes: bool,
) -> dict[int, tuple[tuple[int, tuple[int, ...]], ...]]:
    """Return shared-link plaquette-neighbor edges with lifted-cell steps.

    Each edge is ``(neighbor_plaquette_id, anchor_cell_displacement)``.  The
    displacement is chosen as the short periodic step from the source
    plaquette's anchor cell to the neighbor's anchor cell.
    """
    allowed_kinds = None if plaquette_kinds is None else set(str(kind) for kind in plaquette_kinds)
    plaquette_ids = tuple(
        int(plaquette_id)
        for plaquette_id in model.plaquette_ids()
        if allowed_kinds is None
        or str(model.lattice.plaquettes[int(plaquette_id)].kind) in allowed_kinds
    )
    allowed_id_set = set(plaquette_ids)
    if not allowed_id_set:
        return {}

    kind_by_id = {
        int(plaquette_id): str(model.lattice.plaquettes[int(plaquette_id)].kind)
        for plaquette_id in plaquette_ids
    }
    cell_by_id = {
        int(plaquette_id): _stripe_anchor_cell(model, int(plaquette_id))
        for plaquette_id in plaquette_ids
    }

    link_to_plaquettes: dict[int, list[int]] = defaultdict(list)
    for plaquette_id in plaquette_ids:
        for link_id in model.lattice.plaquette_links(int(plaquette_id)):
            link_to_plaquettes[int(link_id)].append(int(plaquette_id))

    edges: dict[int, set[tuple[int, tuple[int, ...]]]] = {
        int(plaquette_id): set() for plaquette_id in plaquette_ids
    }
    for incident_plaquettes in link_to_plaquettes.values():
        for source in incident_plaquettes:
            source = int(source)
            if source not in allowed_id_set:
                continue
            for target in incident_plaquettes:
                target = int(target)
                if target == source or target not in allowed_id_set:
                    continue
                if not allow_kind_changes and kind_by_id[int(source)] != kind_by_id[int(target)]:
                    continue
                step = _periodic_anchor_cell_displacement(
                    model,
                    cell_by_id[int(source)],
                    cell_by_id[int(target)],
                )
                edges[int(source)].add((int(target), step))

    return {
        int(plaquette_id): tuple(
            sorted(
                edge_items,
                key=lambda item: (
                    _cell_displacement_norm(item[1]),
                    tuple(int(value) for value in item[1]),
                    int(item[0]),
                ),
            )
        )
        for plaquette_id, edge_items in edges.items()
    }


def _snake_edge_neighbor_sets(
    edge_map: dict[int, tuple[tuple[int, tuple[int, ...]], ...]],
) -> dict[int, frozenset[int]]:
    """Return undirected neighbor sets used by snake-cycle filters."""
    neighbors: dict[int, set[int]] = {int(pid): set() for pid in edge_map}
    for source, edge_items in edge_map.items():
        source = int(source)
        neighbors.setdefault(source, set())
        for target, _ in edge_items:
            target = int(target)
            neighbors.setdefault(target, set()).add(source)
            neighbors[source].add(target)

    return {int(pid): frozenset(sorted(values)) for pid, values in neighbors.items()}


def _snake_path_extension_is_induced(
    path: tuple[int, ...],
    candidate: int,
    edge_neighbors: dict[int, frozenset[int]],
) -> bool:
    """Return whether appending ``candidate`` keeps a chordless snake path."""
    candidate = int(candidate)
    if candidate in path:
        return False
    if not path:
        return True

    previous = int(path[-1])
    candidate_neighbors = edge_neighbors.get(candidate, frozenset())
    seed = int(path[0])
    for existing in path[:-1]:
        # The last vertex of an induced cycle must also touch the seed.  Allow
        # this prospective closing edge while pruning all other chords.
        if int(existing) == seed:
            continue
        if int(existing) in candidate_neighbors:
            return False
    return previous in candidate_neighbors


def _snake_path_is_induced_cycle(
    path: tuple[int, ...],
    edge_neighbors: dict[int, frozenset[int]],
) -> bool:
    """Return whether ``path`` is a chordless cycle on the plaquette graph."""
    if len(path) < 3:
        return False
    selected = {int(pid) for pid in path}
    if len(selected) != len(path):
        return False

    for plaquette_id in selected:
        degree = len(selected.intersection(edge_neighbors.get(int(plaquette_id), frozenset())))
        if degree != 2:
            return False
    return True


def _snake_path_kinds(model: object, path: Sequence[int]) -> tuple[str, ...]:
    return tuple(str(model.lattice.plaquettes[int(pid)].kind) for pid in path)


def _snake_partial_kind_pattern_possible(
    model: object,
    path: Sequence[int],
    pattern: SnakeStripeKindPattern,
) -> bool:
    """Cheap partial-path pruning for kind-pattern-restricted snakes."""
    if pattern == "any" or len(path) <= 1:
        return True

    kinds = _snake_path_kinds(model, path)
    distinct = set(kinds)
    if pattern == "constant":
        return len(distinct) == 1
    if pattern == "alternating":
        return len(distinct) <= 2 and all(
            left != right for left, right in zip(kinds, kinds[1:], strict=False)
        )

    # ``constant_or_alternating`` keeps both possibilities alive until closure.
    constant_possible = len(distinct) == 1
    alternating_possible = len(distinct) <= 2 and all(
        left != right for left, right in zip(kinds, kinds[1:], strict=False)
    )
    return constant_possible or alternating_possible


def _snake_path_kind_pattern_allowed(
    model: object,
    path: Sequence[int],
    pattern: SnakeStripeKindPattern,
) -> bool:
    """Return whether a closed snake has the requested plaquette-kind pattern."""
    if pattern == "any":
        return True
    if len(path) == 0:
        return False

    kinds = _snake_path_kinds(model, path)
    distinct = set(kinds)
    if pattern == "constant":
        return len(distinct) == 1

    alternating = (
        len(distinct) == 2
        and len(kinds) % 2 == 0
        and all(left != right for left, right in zip(kinds, kinds[1:], strict=False))
        and kinds[-1] != kinds[0]
    )
    if pattern == "alternating":
        return alternating

    return len(distinct) == 1 or alternating


def _zero_cell_displacement(model: object) -> tuple[int, ...]:
    ndim = _lattice_anchor_dimension(model)
    return tuple(0 for _ in range(ndim))


def _lattice_anchor_dimension(model: object) -> int:
    cells = [_stripe_anchor_cell(model, int(pid)) for pid in model.plaquette_ids()]
    if not cells:
        return 0
    return max(len(cell) for cell in cells)


def _pad_cell(cell: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if len(cell) > ndim:
        raise ValueError("cell dimension exceeds requested dimension.")
    return tuple(int(cell[axis]) if axis < len(cell) else 0 for axis in range(ndim))


def _periodic_anchor_cell_displacement(
    model: object,
    source_cell: tuple[int, ...],
    target_cell: tuple[int, ...],
) -> tuple[int, ...]:
    ndim = max(len(source_cell), len(target_cell), _lattice_anchor_dimension(model))
    source = _pad_cell(source_cell, ndim)
    target = _pad_cell(target_cell, ndim)
    periodic = _lattice_is_periodic(model)

    displacement: list[int] = []
    for axis, (source_value, target_value) in enumerate(zip(source, target, strict=True)):
        raw = int(target_value) - int(source_value)
        period = _lattice_axis_period(model, axis) if periodic else None
        if period is None or period <= 0:
            displacement.append(raw)
            continue

        candidates = (raw - period, raw, raw + period)
        best = min(
            candidates,
            key=lambda value: (abs(int(value)), 0 if int(value) >= 0 else 1),
        )
        displacement.append(int(best))

    return tuple(displacement)


def _add_cell_displacements(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    ndim = max(len(left), len(right))
    left_padded = _pad_cell(left, ndim)
    right_padded = _pad_cell(right, ndim)
    return tuple(
        int(left_value) + int(right_value)
        for left_value, right_value in zip(left_padded, right_padded, strict=True)
    )


def _cell_displacement_norm(displacement: tuple[int, ...]) -> int:
    return int(sum(abs(int(value)) for value in displacement))


def _canonical_snake_step(step: tuple[int, ...]) -> tuple[int, ...]:
    norm = _cell_displacement_norm(step)
    if norm == 0:
        return tuple(0 for _ in step)
    # Keep the integer direction.  Shared-link plaquette steps on the current
    # lattices are primitive, so no gcd reduction is needed for the intended use.
    return tuple(int(value) for value in step)


def _snake_step_turn_increment(
    previous_step: tuple[int, ...] | None,
    next_step: tuple[int, ...],
) -> int:
    if previous_step is None:
        return 0
    return int(_canonical_snake_step(previous_step) != _canonical_snake_step(next_step))


def _winding_from_lifted_displacement(
    model: object,
    displacement: tuple[int, ...],
) -> tuple[int, ...] | None:
    periodic = _lattice_is_periodic(model)
    if not periodic:
        return None

    winding: list[int] = []
    for axis, value in enumerate(displacement):
        period = _lattice_axis_period(model, axis)
        if period is None or period <= 0:
            if int(value) != 0:
                return None
            winding.append(0)
            continue
        if int(value) % int(period) != 0:
            return None
        winding.append(int(value) // int(period))

    return tuple(winding)


def _adaptive_region_frontier(
    plaquette_ids: frozenset[int],
    neighbor_map: dict[int, frozenset[int]],
) -> tuple[int, ...]:
    frontier: set[int] = set()
    for plaquette_id in plaquette_ids:
        frontier.update(int(neighbor) for neighbor in neighbor_map.get(int(plaquette_id), ()))
    frontier.difference_update(plaquette_ids)
    return tuple(sorted(frontier))


def _top_adaptive_records(
    records: Sequence[AdaptiveRegionProposalRecord],
    limit: int,
) -> list[AdaptiveRegionProposalRecord]:
    return sorted(
        records,
        key=lambda record: (
            -float(record.score),
            int(record.link_count),
            tuple(int(pid) for pid in record.plaquette_ids),
        ),
    )[: int(limit)]


def _adaptive_region_score(
    region: LocalQDMRegion,
    *,
    plaquette_ids: npt.NDArray[np.int64],
    neighbor_map: dict[int, frozenset[int]],
    feedback_bonus: float,
) -> float:
    selected = {int(pid) for pid in np.asarray(plaquette_ids, dtype=np.int64)}
    internal_edges = 0
    for plaquette_id in selected:
        internal_edges += sum(
            1 for neighbor in neighbor_map.get(plaquette_id, ()) if neighbor in selected
        )
    internal_edges //= 2

    n_plaquettes = int(len(selected))
    n_links = int(region.link_ids.size)
    n_unresolved = int(region.unresolved_boundary_plaquette_ids.size)
    n_closed_sites = int(region.closed_site_ids.size)

    # Cheap closure/compactness heuristic.  The weights are intentionally mild:
    # hard limits still come from max_plaquettes/max_links, while this score
    # merely ranks which growth paths survive the beam.
    return float(
        feedback_bonus
        + 1.0 * n_plaquettes
        + 0.75 * internal_edges
        + 0.10 * n_closed_sites
        - 1.0 * n_unresolved
        - 0.05 * n_links
    )


def _unique_int_array(values: Sequence[int] | npt.ArrayLike, *, name: str) -> npt.NDArray[np.int64]:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return np.unique(arr).astype(np.int64)


def _validate_link_ids(model: object, link_ids: npt.NDArray[np.int64]) -> None:
    if link_ids.size == 0:
        raise ValueError("At least one local link is required.")
    if np.any(link_ids < 0) or np.any(link_ids >= int(model.lattice.num_links)):
        raise ValueError("link_ids contain ids outside the model lattice.")


def _validate_plaquette_ids(model: object, plaquette_ids: npt.NDArray[np.int64]) -> None:
    allowed = set(int(pid) for pid in model.plaquette_ids())
    if plaquette_ids.size == 0:
        raise ValueError("At least one plaquette id is required.")
    bad = [int(pid) for pid in plaquette_ids if int(pid) not in allowed]
    if bad:
        raise ValueError(f"plaquette ids are not valid QDM plaquettes for this model: {bad}")


def _require_plaquettes_inside_links(
    model: object,
    plaquette_ids: npt.NDArray[np.int64],
    local_link_set: set[int],
    *,
    name: str,
) -> None:
    bad = []
    for plaquette_id in plaquette_ids:
        links = set(int(link_id) for link_id in model.lattice.plaquette_links(int(plaquette_id)))
        if not links.issubset(local_link_set):
            bad.append(int(plaquette_id))
    if bad:
        raise ValueError(f"{name} contains plaquettes not covered by link_ids: {bad}")


def _plaquette_union_links(
    model: object,
    plaquette_ids: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    return np.unique(
        np.asarray(
            [link for pid in plaquette_ids for link in model.lattice.plaquette_links(int(pid))],
            dtype=np.int64,
        )
    ).astype(np.int64)


def _expand_plaquettes_by_shared_links(model: object, plaquette_ids: set[int]) -> set[int]:
    links = set(
        int(link_id)
        for plaquette_id in plaquette_ids
        for link_id in model.lattice.plaquette_links(int(plaquette_id))
    )
    expanded = set(plaquette_ids)
    for candidate in model.plaquette_ids():
        candidate_links = set(
            int(link_id) for link_id in model.lattice.plaquette_links(int(candidate))
        )
        if links.intersection(candidate_links):
            expanded.add(int(candidate))
    return expanded


def _site_partition_for_local_links(
    model: object,
    local_link_set: set[int],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    touched_sites: set[int] = set()
    for link_id in local_link_set:
        source, target = model.lattice.link_endpoints[int(link_id)]
        touched_sites.add(int(source))
        touched_sites.add(int(target))

    closed: list[int] = []
    boundary: list[int] = []
    for site_id in sorted(touched_sites):
        incident = set(int(link_id) for link_id in model.lattice.incident_links(int(site_id)))
        if incident.issubset(local_link_set):
            closed.append(int(site_id))
        else:
            boundary.append(int(site_id))

    return np.asarray(closed, dtype=np.int64), np.asarray(boundary, dtype=np.int64)


def _unresolved_boundary_plaquettes(
    model: object,
    *,
    local_link_set: set[int],
    active_plaquette_ids: set[int],
) -> npt.NDArray[np.int64]:
    unresolved: list[int] = []
    for plaquette_id in model.plaquette_ids():
        plaquette_id = int(plaquette_id)
        if plaquette_id in active_plaquette_ids:
            continue
        links = set(int(link_id) for link_id in model.lattice.plaquette_links(plaquette_id))
        if links.intersection(local_link_set):
            unresolved.append(plaquette_id)
    return np.asarray(unresolved, dtype=np.int64)


def _local_binary_layout(n_links: int):
    return VariableLayout.from_links(int(n_links), LocalSpace.binary())


def _plaquette_local_indices(
    model: object,
    plaquette_id: int,
    local_index_by_link: dict[int, int],
) -> npt.NDArray[np.int64]:
    try:
        return np.asarray(
            [
                local_index_by_link[int(link_id)]
                for link_id in model.lattice.plaquette_links(plaquette_id)
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"Plaquette {plaquette_id} is not contained in the local link set."
        ) from exc
