"""QDM adapter and local-region algebra for local-first cage search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis, DFSBasisSolver
from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.local_search.core import (
    register_local_cage_adapter_factory,
)
from qlinks.caging.local_search.geometry import (
    _expand_plaquettes_by_shared_links,
    _local_binary_layout,
    _plaquette_local_indices,
    _plaquette_union_links,
    _require_plaquettes_inside_links,
    _site_partition_for_local_links,
    _unique_int_array,
    _unresolved_boundary_plaquettes,
    _validate_link_ids,
    _validate_plaquette_ids,
)
from qlinks.caging.local_search.types import (
    LocalBoundaryMode,
    LocalCageModelAdapter,
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    LocalQDMRegion,
    _LocalQDMActivePlaquetteObserver,
    _LocalQDMCountConstraint,
)
from qlinks.caging.results import CageState
from qlinks.models.couplings import DirectedPlaquetteCoupling
from qlinks.operators.plaquette import alternating_binary_patterns
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class QDMLocalCageAdapter:
    """QDM implementation of the local variable interface.

    This is intentionally the only place where the generic local searcher needs
    to know how QDM variables/plaquette flips are represented.  Later QLM/PXP
    adapters can implement the same protocol without modifying the solver core.
    """

    model: object
    source_label: str = "qdm"

    def normalize_config(
        self,
        config: LocalQDMCageSearchConfig,
    ) -> LocalQDMCageSearchConfig:
        return _with_inferred_potential_signature_unit(config, self.model)

    def build_region_from_plaquettes(
        self,
        *,
        plaquette_ids: Sequence[int] | npt.ArrayLike,
        config: LocalQDMCageSearchConfig,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    ) -> LocalQDMRegion:
        return build_qdm_local_region_from_plaquettes(
            self.model,
            plaquette_ids=plaquette_ids,
            halo_layers=config.halo_layers,
            boundary_mode=config.boundary_mode,
            scoring_plaquette_ids=scoring_plaquette_ids,
        )

    def build_region_from_links(
        self,
        *,
        link_ids: Sequence[int] | npt.ArrayLike,
        config: LocalQDMCageSearchConfig,
        active_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    ) -> LocalQDMRegion:
        return build_qdm_local_region_from_links(
            self.model,
            link_ids=link_ids,
            boundary_mode=config.boundary_mode,
            active_plaquette_ids=active_plaquette_ids,
            scoring_plaquette_ids=scoring_plaquette_ids,
        )

    def full_model_region(
        self,
        *,
        config: LocalQDMCageSearchConfig,
    ) -> LocalQDMRegion:
        return self.build_region_from_links(
            link_ids=np.arange(self.model.lattice.num_links, dtype=np.int64),
            active_plaquette_ids=self.model.plaquette_ids(),
            scoring_plaquette_ids=self.model.plaquette_ids(),
            config=config,
        )

    def enumerate_local_basis(
        self,
        region: LocalQDMRegion,
        config: LocalQDMCageSearchConfig,
    ) -> Basis:
        return enumerate_qdm_local_basis(
            self.model,
            region,
            include_sectors_when_full=config.include_sectors_when_full,
            prune_inactive_states=(
                config.prune_inactive_local_basis_states and config.min_component_size > 1
            ),
            max_states=config.max_local_states,
            sort=config.sort_basis,
        )

    def build_local_kinetic_matrix(
        self,
        region: LocalQDMRegion,
        local_basis: Basis,
    ) -> scipy_sparse.csr_array:
        return build_qdm_local_kinetic_matrix(self.model, region, local_basis)

    def local_self_loop_values(
        self,
        region: LocalQDMRegion,
        local_basis: Basis,
    ) -> npt.NDArray[np.complex128]:
        return qdm_local_self_loop_values(self.model, region, local_basis)

    def make_local_record(
        self,
        *,
        cage_state: CageState,
        signature: tuple[int, int],
        candidate: CandidateSubgraph,
        local_basis: Basis,
        region: LocalQDMRegion,
    ) -> LocalQDMCageRecord:
        support_configs = np.asarray(local_basis.states[cage_state.support], dtype=np.int64)
        return LocalQDMCageRecord(
            cage_state=cage_state,
            signature=signature,
            candidate=candidate,
            support_configs=support_configs,
            local_link_ids=region.link_ids.copy(),
            active_plaquette_ids=region.active_plaquette_ids.copy(),
            scoring_plaquette_ids=region.scoring_plaquette_ids.copy(),
            unresolved_boundary_plaquette_ids=region.unresolved_boundary_plaquette_ids.copy(),
        )


def _qdm_local_cage_adapter_factory(model: object) -> LocalCageModelAdapter | None:
    lattice = getattr(model, "lattice", None)
    if lattice is None:
        return None
    required_model_attrs = (
        "plaquette_ids",
        "make_sectors",
        "_coup_kin_at",
        "_coup_pot_at",
    )
    required_lattice_attrs = (
        "num_links",
        "num_sites",
        "incident_links",
        "plaquette_links",
        "link_endpoints",
    )
    if not all(hasattr(model, name) for name in required_model_attrs):
        return None
    if not all(hasattr(lattice, name) for name in required_lattice_attrs):
        return None
    if not hasattr(model, "required_count"):
        return None
    return QDMLocalCageAdapter(model)


def build_qdm_local_region_from_plaquettes(
    model: object,
    *,
    plaquette_ids: Sequence[int] | npt.ArrayLike,
    halo_layers: int,
    boundary_mode: LocalBoundaryMode,
    scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
) -> LocalQDMRegion:
    """Build a local QDM region by expanding seed plaquettes by shared-link halo."""
    seed = _unique_int_array(plaquette_ids, name="plaquette_ids")
    _validate_plaquette_ids(model, seed)

    active = set(int(pid) for pid in seed)
    for _ in range(int(halo_layers)):
        active = _expand_plaquettes_by_shared_links(model, active)

    active_arr = np.asarray(sorted(active), dtype=np.int64)
    link_ids = _plaquette_union_links(model, active_arr)

    if scoring_plaquette_ids is None:
        scoring = active_arr
    else:
        scoring = _unique_int_array(scoring_plaquette_ids, name="scoring_plaquette_ids")
        _validate_plaquette_ids(model, scoring)

    return build_qdm_local_region_from_links(
        model,
        link_ids=link_ids,
        boundary_mode=boundary_mode,
        active_plaquette_ids=active_arr,
        scoring_plaquette_ids=scoring,
        seed_plaquette_ids=seed,
    )


def build_qdm_local_region_from_links(
    model: object,
    *,
    link_ids: Sequence[int] | npt.ArrayLike,
    boundary_mode: LocalBoundaryMode,
    active_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    seed_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
) -> LocalQDMRegion:
    """Build a local QDM region from explicit link ids."""
    local_links = _unique_int_array(link_ids, name="link_ids")
    _validate_link_ids(model, local_links)
    local_link_set = set(int(link_id) for link_id in local_links)

    contained_plaquettes = np.asarray(
        [
            int(pid)
            for pid in model.plaquette_ids()
            if set(int(link) for link in model.lattice.plaquette_links(int(pid))).issubset(
                local_link_set
            )
        ],
        dtype=np.int64,
    )

    if active_plaquette_ids is None:
        active = contained_plaquettes
    else:
        active = _unique_int_array(active_plaquette_ids, name="active_plaquette_ids")
        _validate_plaquette_ids(model, active)
        _require_plaquettes_inside_links(model, active, local_link_set, name="active_plaquette_ids")

    if scoring_plaquette_ids is None:
        scoring = active
    else:
        scoring = _unique_int_array(scoring_plaquette_ids, name="scoring_plaquette_ids")
        _validate_plaquette_ids(model, scoring)
        _require_plaquettes_inside_links(
            model, scoring, local_link_set, name="scoring_plaquette_ids"
        )

    if seed_plaquette_ids is None:
        seed = active
    else:
        seed = _unique_int_array(seed_plaquette_ids, name="seed_plaquette_ids")
        _validate_plaquette_ids(model, seed)

    closed_sites, boundary_sites = _site_partition_for_local_links(model, local_link_set)

    if boundary_mode == "closed" and boundary_sites.size:
        raise ValueError(
            "boundary_mode='closed' requires every touched site's incident links "
            "to be included in the local link set."
        )

    unresolved = _unresolved_boundary_plaquettes(
        model,
        local_link_set=local_link_set,
        active_plaquette_ids=set(int(pid) for pid in active),
    )

    return LocalQDMRegion(
        link_ids=local_links,
        seed_plaquette_ids=seed,
        active_plaquette_ids=active,
        scoring_plaquette_ids=scoring,
        closed_site_ids=closed_sites,
        boundary_site_ids=boundary_sites,
        unresolved_boundary_plaquette_ids=unresolved,
    )


def _qdm_active_plaquette_closure_variable_order(
    model: object,
    region: LocalQDMRegion,
    *,
    n_local_variables: int,
) -> npt.NDArray[np.int64]:
    """Order local variables so active plaquettes are decided early.

    The local-basis DFS may otherwise spend a long time enumerating boundary
    dimer completions before assigning enough links to decide whether any active
    plaquette can be flippable.  For local cage searches with the active-state
    observer enabled, grouping links plaquette-by-plaquette exposes inactive
    branches to the observer much earlier.  The basis remains sorted afterward
    when ``sort=True``, so this only changes traversal cost, not the public basis
    order.
    """
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(region.link_ids)}
    ordered: list[int] = []
    seen: set[int] = set()

    def append_variable(variable_index: int) -> None:
        variable_index = int(variable_index)
        if variable_index in seen:
            return
        if variable_index < 0 or variable_index >= int(n_local_variables):
            return
        seen.add(variable_index)
        ordered.append(variable_index)

    for plaquette_id in region.active_plaquette_ids:
        for variable_index in _plaquette_local_indices(
            model,
            int(plaquette_id),
            local_index_by_link,
        ):
            append_variable(int(variable_index))

    for variable_index in range(int(n_local_variables)):
        append_variable(variable_index)

    return np.asarray(ordered, dtype=np.int64)


def _qdm_active_plaquette_observer(
    model: object,
    region: LocalQDMRegion,
) -> _LocalQDMActivePlaquetteObserver | None:
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(region.link_ids)}
    variable_indices_by_plaquette: list[npt.NDArray[np.int64]] = []
    patterns_by_plaquette: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []

    for plaquette_id in region.active_plaquette_ids:
        local_variables = _plaquette_local_indices(model, int(plaquette_id), local_index_by_link)
        if local_variables.size == 0:
            continue
        pattern0, pattern1 = alternating_binary_patterns(int(local_variables.size))
        variable_indices_by_plaquette.append(np.asarray(local_variables, dtype=np.int64))
        patterns_by_plaquette.append(
            (
                np.asarray(pattern0, dtype=np.int64),
                np.asarray(pattern1, dtype=np.int64),
            )
        )

    if not variable_indices_by_plaquette:
        return None

    return _LocalQDMActivePlaquetteObserver(
        plaquette_variable_indices=tuple(variable_indices_by_plaquette),
        plaquette_patterns=tuple(patterns_by_plaquette),
    )


def _qdm_local_basis_constraints(
    model: object,
    region: LocalQDMRegion,
    *,
    layout: VariableLayout,
) -> tuple[_LocalQDMCountConstraint, ...]:
    """Build DFS constraints for local QDM basis enumeration."""
    link_ids = np.asarray(region.link_ids, dtype=np.int64)
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(link_ids)}

    touched_sites = np.unique(
        np.asarray(
            [site for link_id in link_ids for site in model.lattice.link_endpoints[int(link_id)]],
            dtype=np.int64,
        )
    )
    closed_site_set = set(int(site_id) for site_id in region.closed_site_ids)
    required_count = int(getattr(model, "required_count", 1))

    constraints: list[_LocalQDMCountConstraint] = []
    for site_id in touched_sites:
        incident_local = [
            local_index_by_link[int(link_id)]
            for link_id in model.lattice.incident_links(int(site_id))
            if int(link_id) in local_index_by_link
        ]
        local_indices = np.asarray(incident_local, dtype=np.int64)

        is_closed = int(site_id) in closed_site_set
        constraints.append(
            _LocalQDMCountConstraint(
                layout=layout,
                site_id=int(site_id),
                variable_indices=local_indices,
                min_count=required_count if is_closed else None,
                max_count=required_count,
                name=(
                    "local_qdm_closed_site_count" if is_closed else "local_qdm_boundary_site_count"
                ),
            )
        )

    return tuple(constraints)


def enumerate_qdm_local_basis(
    model: object,
    region: LocalQDMRegion,
    *,
    include_sectors_when_full: bool,
    prune_inactive_states: bool = False,
    max_states: int | None = None,
    sort: bool = True,
) -> Basis:
    """Enumerate local dimer configurations on ``region.link_ids``.

    The local-search layer deliberately reuses :class:`DFSBasisSolver` rather
    than maintaining a separate DFS.  QDM-specific local rules are represented
    as lightweight constraints on the local binary-link layout, so future DFS
    optimizations immediately benefit both full-basis enumeration and local cage
    searches.
    """
    if max_states is not None and max_states < 0:
        raise ValueError("max_states must be non-negative or None.")

    link_ids = np.asarray(region.link_ids, dtype=np.int64)
    n_local = int(link_ids.size)
    layout = _local_binary_layout(n_local)

    constraints = _qdm_local_basis_constraints(
        model,
        region,
        layout=layout,
    )

    full_link_region = n_local == int(model.lattice.num_links) and np.array_equal(
        np.sort(link_ids),
        np.arange(model.lattice.num_links, dtype=np.int64),
    )
    sectors = (
        tuple(model.make_sectors()) if (include_sectors_when_full and full_link_region) else ()
    )

    observers = ()
    variable_order = None
    if prune_inactive_states and not full_link_region:
        observer = _qdm_active_plaquette_observer(model, region)
        observers = () if observer is None else (observer,)
        variable_order = _qdm_active_plaquette_closure_variable_order(
            model,
            region,
            n_local_variables=n_local,
        )

    return DFSBasisSolver(sort=sort, variable_order=variable_order).solve(
        layout,
        constraints=constraints,
        sectors=sectors,
        observers=observers,
        max_states=max_states,
    )


def build_qdm_local_kinetic_matrix(
    model: object,
    region: LocalQDMRegion,
    local_basis: Basis,
) -> scipy_sparse.csr_array:
    """Build the local kinetic matrix without using a global basis/Hamiltonian."""
    n = int(local_basis.n_states)
    if n == 0:
        return scipy_sparse.csr_array((0, 0), dtype=np.complex128)

    local_index_by_link = {int(link_id): i for i, link_id in enumerate(region.link_ids)}
    state_index = {tuple(int(x) for x in state): i for i, state in enumerate(local_basis.states)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for col, config in enumerate(local_basis.states):
        for plaquette_id in region.active_plaquette_ids:
            plaquette_id = int(plaquette_id)
            local_variables = _plaquette_local_indices(model, plaquette_id, local_index_by_link)
            local_values = config[local_variables]
            p0, p1 = alternating_binary_patterns(local_variables.size)
            coupling = model._coup_kin_at(plaquette_id)  # qlinks QDM internal API.

            if np.array_equal(local_values, p0):
                final = np.asarray(config, dtype=np.int64).copy()
                final[local_variables] = p1
                row = state_index.get(tuple(int(x) for x in final))
                if row is not None:
                    rows.append(int(row))
                    cols.append(int(col))
                    data.append(_forward_coefficient(coupling))
            elif np.array_equal(local_values, p1):
                final = np.asarray(config, dtype=np.int64).copy()
                final[local_variables] = p0
                row = state_index.get(tuple(int(x) for x in final))
                if row is not None:
                    rows.append(int(row))
                    cols.append(int(col))
                    data.append(_backward_coefficient(coupling))

    return scipy_sparse.coo_array(
        (np.asarray(data, dtype=np.complex128), (rows, cols)),
        shape=(n, n),
        dtype=np.complex128,
    ).tocsr()


def qdm_local_self_loop_values(
    model: object,
    region: LocalQDMRegion,
    local_basis: Basis,
) -> npt.NDArray[np.complex128]:
    """Return local potential/self-loop values on the scoring plaquettes."""
    values = np.zeros(int(local_basis.n_states), dtype=np.complex128)
    if local_basis.n_states == 0:
        return values

    local_index_by_link = {int(link_id): i for i, link_id in enumerate(region.link_ids)}

    for basis_index, config in enumerate(local_basis.states):
        total = 0.0 + 0.0j
        for plaquette_id in region.scoring_plaquette_ids:
            plaquette_id = int(plaquette_id)
            local_variables = _plaquette_local_indices(model, plaquette_id, local_index_by_link)
            local_values = config[local_variables]
            p0, p1 = alternating_binary_patterns(local_variables.size)
            if np.array_equal(local_values, p0) or np.array_equal(local_values, p1):
                total += complex(model._coup_pot_at(plaquette_id))
        values[basis_index] = total

    return values


def _infer_potential_unit_from_model(model: object) -> complex:
    coupling = getattr(model, "coup_pot", None)
    if coupling is None or callable(coupling) or isinstance(coupling, dict):
        return 1.0 + 0.0j
    try:
        value = complex(coupling)
    except (TypeError, ValueError):
        return 1.0 + 0.0j
    if value == 0:
        return 1.0 + 0.0j
    return value


def _forward_coefficient(coupling: DirectedPlaquetteCoupling) -> complex:
    return complex(coupling.resolved_forward())


def _backward_coefficient(coupling: DirectedPlaquetteCoupling) -> complex:
    return complex(coupling.resolved_backward())


def _with_inferred_potential_signature_unit(
    config: LocalQDMCageSearchConfig,
    model: object,
) -> LocalQDMCageSearchConfig:
    if complex(config.potential_signature_unit) != complex(1.0):
        return config

    coupling = getattr(model, "coup_pot", None)
    if coupling is None or callable(coupling) or isinstance(coupling, dict):
        return config

    try:
        potential_unit = complex(coupling)
    except (TypeError, ValueError):
        return config

    if potential_unit == 0:
        return config

    return replace(config, potential_signature_unit=potential_unit)


register_local_cage_adapter_factory(_qdm_local_cage_adapter_factory)
