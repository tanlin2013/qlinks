"""Local-first shortcut cage searchers.

This module intentionally does **not** depend on a globally built Hilbert-space
Hamiltonian.  The first implementation targets QDM-like models, where the
configuration variables are binary link occupations and the kinetic term is a
plaquette flip between alternating dimer patterns.

The central object is a local cage certificate: enumerate configurations only
on a chosen link/plaquette region, build the local kinetic graph inside that
region, and run the usual type-1 caging algebra on that small graph.  A later
padding/global-certification layer can decide whether a certificate embeds into
one or more global winding sectors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis
from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.partition import type1_candidates_from_bipartite_self_loops
from qlinks.caging.results import CageState
from qlinks.caging.search import (
    bipartition_labels,
    signature_from_energy_and_self_loop,
)
from qlinks.caging.solver import CageSolverConfig, solve_candidate_for_kinetic_targets
from qlinks.models.couplings import DirectedPlaquetteCoupling
from qlinks.operators.plaquette import alternating_binary_patterns

LocalBoundaryMode = Literal["relaxed", "closed"]


@dataclass(frozen=True, slots=True)
class LocalQDMCageSearchConfig:
    """Configuration for the QDM local-first type-1 cage search.

    Parameters
    ----------
    tolerance:
        Numerical tolerance used by the local candidate solver.

    allowed_kappas:
        Kinetic eigenvalues to target.  The first shortcut path is intended for
        type-1 cages, so the default is ``(0,)``.

    halo_layers:
        Number of plaquette-neighbor expansions used when the search region is
        supplied by plaquettes.  Neighboring plaquettes are those sharing at
        least one link.  ``halo_layers=1`` means: include the seed plaquettes
        and their one-plaquette kinetic halo.

    boundary_mode:
        ``"relaxed"`` enforces exact dimer constraints only at sites whose all
        incident links lie in the local link set.  Boundary sites only enforce
        an at-most constraint, leaving room for an exterior padding.

        ``"closed"`` requires every site touched by the local link set to have
        all incident links included, and then enforces exact local constraints.
        This is useful for full-lattice regression tests or deliberately closed
        stripe/torus regions.

    include_sectors_when_full:
        If the local link set is the full model link set, apply the model's
        sector conditions during local basis generation.  For genuine local
        regions sector checks are deferred to the padding/global-certification
        layer.
    """

    tolerance: float = 1.0e-10
    allowed_kappas: tuple[int, ...] = (0,)
    min_component_size: int = 2
    halo_layers: int = 1
    boundary_mode: LocalBoundaryMode = "relaxed"
    include_sectors_when_full: bool = True
    max_local_states: int | None = None
    sort_basis: bool = True
    validate_full_residual: bool = True
    deduplicate_by_rank: bool = True
    rank_tolerance_factor: float = 100.0
    signature_tolerance_factor: float = 10.0
    potential_signature_unit: complex = 1.0

    def __post_init__(self) -> None:
        if self.halo_layers < 0:
            raise ValueError("halo_layers must be non-negative.")
        if self.boundary_mode not in {"relaxed", "closed"}:
            raise ValueError("boundary_mode must be 'relaxed' or 'closed'.")
        if self.max_local_states is not None and self.max_local_states < 0:
            raise ValueError("max_local_states must be non-negative or None.")


@dataclass(frozen=True, slots=True)
class LocalQDMRegion:
    """A real-space region used by :class:`LocalQDMCageSearcher`."""

    link_ids: npt.NDArray[np.int64]
    seed_plaquette_ids: npt.NDArray[np.int64]
    active_plaquette_ids: npt.NDArray[np.int64]
    scoring_plaquette_ids: npt.NDArray[np.int64]
    closed_site_ids: npt.NDArray[np.int64]
    boundary_site_ids: npt.NDArray[np.int64]
    unresolved_boundary_plaquette_ids: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        for field_name in (
            "link_ids",
            "seed_plaquette_ids",
            "active_plaquette_ids",
            "scoring_plaquette_ids",
            "closed_site_ids",
            "boundary_site_ids",
            "unresolved_boundary_plaquette_ids",
        ):
            values = np.asarray(getattr(self, field_name), dtype=np.int64)
            if values.ndim != 1:
                raise ValueError(f"{field_name} must be one-dimensional.")
            object.__setattr__(self, field_name, np.unique(values).astype(np.int64))


@dataclass(frozen=True, slots=True)
class LocalQDMCageRecord:
    """One local cage certificate."""

    cage_state: CageState
    signature: tuple[int, int]
    candidate: CandidateSubgraph
    support_configs: npt.NDArray[np.int64]
    local_link_ids: npt.NDArray[np.int64]
    active_plaquette_ids: npt.NDArray[np.int64]
    scoring_plaquette_ids: npt.NDArray[np.int64]
    unresolved_boundary_plaquette_ids: npt.NDArray[np.int64]

    @property
    def kappa(self) -> int:
        return int(self.signature[0])

    @property
    def potential_value(self) -> int:
        return int(self.signature[1])

    @property
    def support(self) -> npt.NDArray[np.int64]:
        return self.cage_state.support

    @property
    def local_state(self) -> npt.NDArray[np.complex128]:
        return self.cage_state.local_state


@dataclass
class LocalQDMCageSearchResult:
    """Result of a local QDM cage search."""

    records: list[LocalQDMCageRecord]
    region: LocalQDMRegion
    local_basis: Basis
    kinetic_matrix: scipy_sparse.csr_array
    self_loop_values: npt.NDArray[np.complex128]
    config: LocalQDMCageSearchConfig
    type1_candidates: list[CandidateSubgraph] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def local_hilbert_size(self) -> int:
        return int(self.local_basis.n_states)

    @property
    def counts_by_signature(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for record in self.records:
            counts[record.signature] = counts.get(record.signature, 0) + 1
        return counts

    @property
    def signatures(self) -> list[tuple[int, int]]:
        return sorted(self.counts_by_signature)

    def records_by_signature(self, signature: tuple[int, int]) -> list[LocalQDMCageRecord]:
        normalized = (int(signature[0]), int(signature[1]))
        return [record for record in self.records if record.signature == normalized]


@dataclass
class LocalQDMCageSearcher:
    """Local-first type-1 cage searcher for QDM models.

    The searcher consumes a QDM model object and a local region.  It does not
    call ``model.build()`` and does not assemble the full Hilbert-space
    Hamiltonian.  It only enumerates local dimer configurations on the chosen
    links and builds a small local kinetic matrix from plaquette flips whose
    support is contained in that local link set.
    """

    model: object
    region: LocalQDMRegion
    config: LocalQDMCageSearchConfig = field(default_factory=LocalQDMCageSearchConfig)

    @classmethod
    def from_plaquettes(
        cls,
        model: object,
        plaquette_ids: Sequence[int] | npt.ArrayLike,
        *,
        config: LocalQDMCageSearchConfig | None = None,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    ) -> LocalQDMCageSearcher:
        """Construct a local searcher from seed plaquettes."""
        search_config = LocalQDMCageSearchConfig() if config is None else config
        search_config = _with_inferred_potential_signature_unit(search_config, model)
        region = build_qdm_local_region_from_plaquettes(
            model,
            plaquette_ids=plaquette_ids,
            halo_layers=search_config.halo_layers,
            boundary_mode=search_config.boundary_mode,
            scoring_plaquette_ids=scoring_plaquette_ids,
        )
        return cls(model=model, region=region, config=search_config)

    @classmethod
    def from_links(
        cls,
        model: object,
        link_ids: Sequence[int] | npt.ArrayLike,
        *,
        config: LocalQDMCageSearchConfig | None = None,
        active_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    ) -> LocalQDMCageSearcher:
        """Construct a local searcher from explicit link ids."""
        search_config = LocalQDMCageSearchConfig() if config is None else config
        search_config = _with_inferred_potential_signature_unit(search_config, model)
        region = build_qdm_local_region_from_links(
            model,
            link_ids=link_ids,
            boundary_mode=search_config.boundary_mode,
            active_plaquette_ids=active_plaquette_ids,
            scoring_plaquette_ids=scoring_plaquette_ids,
        )
        return cls(model=model, region=region, config=search_config)

    @classmethod
    def full_model_region(
        cls,
        model: object,
        *,
        config: LocalQDMCageSearchConfig | None = None,
    ) -> LocalQDMCageSearcher:
        """Construct a local searcher whose region is the full model.

        This is mostly useful as a regression bridge: the implementation path is
        still local-first/no-full-Hamiltonian, but the local region happens to
        contain every link and plaquette.
        """
        search_config = LocalQDMCageSearchConfig() if config is None else config
        search_config = _with_inferred_potential_signature_unit(search_config, model)
        return cls.from_links(
            model,
            link_ids=np.arange(model.lattice.num_links, dtype=np.int64),
            active_plaquette_ids=model.plaquette_ids(),
            scoring_plaquette_ids=model.plaquette_ids(),
            config=search_config,
        )

    def run(self) -> LocalQDMCageSearchResult:
        """Run the local type-1 cage search."""
        local_basis = enumerate_qdm_local_basis(
            self.model,
            self.region,
            include_sectors_when_full=self.config.include_sectors_when_full,
            max_states=self.config.max_local_states,
            sort=self.config.sort_basis,
        )

        kinetic_matrix = build_qdm_local_kinetic_matrix(
            self.model,
            self.region,
            local_basis,
        )
        self_loop_values = qdm_local_self_loop_values(
            self.model,
            self.region,
            local_basis,
        )

        if local_basis.n_states == 0:
            return LocalQDMCageSearchResult(
                records=[],
                region=self.region,
                local_basis=local_basis,
                kinetic_matrix=kinetic_matrix,
                self_loop_values=self_loop_values,
                config=self.config,
                type1_candidates=[],
            )

        bipartition = bipartition_labels(kinetic_matrix)
        candidates = type1_candidates_from_bipartite_self_loops(
            kinetic_matrix,
            self_loop_values,
            bipartition,
            min_component_size=self.config.min_component_size,
        )

        records = self._solve_candidates(
            candidates=candidates,
            local_basis=local_basis,
            kinetic_matrix=kinetic_matrix,
            self_loop_values=self_loop_values,
        )

        return LocalQDMCageSearchResult(
            records=records,
            region=self.region,
            local_basis=local_basis,
            kinetic_matrix=kinetic_matrix,
            self_loop_values=self_loop_values,
            config=self.config,
            type1_candidates=candidates,
        )

    def _solve_candidates(
        self,
        *,
        candidates: list[CandidateSubgraph],
        local_basis: Basis,
        kinetic_matrix: scipy_sparse.csr_array,
        self_loop_values: npt.NDArray[np.complex128],
    ) -> list[LocalQDMCageRecord]:
        hamiltonian_matrix = kinetic_matrix + scipy_sparse.diags(
            self_loop_values,
            offsets=0,
            shape=kinetic_matrix.shape,
            format="csr",
        )

        solver_config = CageSolverConfig(
            tolerance=self.config.tolerance,
            validate_full_residual=self.config.validate_full_residual,
        )

        records: list[LocalQDMCageRecord] = []

        for candidate in candidates:
            cage_states = solve_candidate_for_kinetic_targets(
                hamiltonian_matrix,
                kinetic_matrix,
                self_loop_values,
                candidate,
                target_kappas=tuple(complex(kappa) for kappa in self.config.allowed_kappas),
                config=solver_config,
            )

            for cage_state in cage_states:
                self_loop_value = self_loop_values[int(candidate.vertices[0])]
                signature = signature_from_energy_and_self_loop(
                    cage_state.energy,
                    self_loop_value,
                    tolerance=self.config.signature_tolerance_factor * self.config.tolerance,
                    potential_unit=self.config.potential_signature_unit,
                )

                if signature is None or signature[0] not in self.config.allowed_kappas:
                    continue

                support_configs = np.asarray(local_basis.states[cage_state.support], dtype=np.int64)
                records.append(
                    LocalQDMCageRecord(
                        cage_state=cage_state,
                        signature=signature,
                        candidate=candidate,
                        support_configs=support_configs,
                        local_link_ids=self.region.link_ids.copy(),
                        active_plaquette_ids=self.region.active_plaquette_ids.copy(),
                        scoring_plaquette_ids=self.region.scoring_plaquette_ids.copy(),
                        unresolved_boundary_plaquette_ids=(
                            self.region.unresolved_boundary_plaquette_ids.copy()
                        ),
                    )
                )

        if self.config.deduplicate_by_rank:
            records = _deduplicate_local_records(
                records,
                hilbert_size=local_basis.n_states,
                tolerance=self.config.rank_tolerance_factor * self.config.tolerance,
            )

        return records


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


def enumerate_qdm_local_basis(
    model: object,
    region: LocalQDMRegion,
    *,
    include_sectors_when_full: bool,
    max_states: int | None = None,
    sort: bool = True,
) -> Basis:
    """Enumerate local dimer configurations on ``region.link_ids``."""
    if max_states is not None and max_states < 0:
        raise ValueError("max_states must be non-negative or None.")
    if max_states == 0:
        return Basis.empty(_local_binary_layout(region.link_ids.size))

    link_ids = np.asarray(region.link_ids, dtype=np.int64)
    local_index_by_link = {int(link_id): i for i, link_id in enumerate(link_ids)}
    n_local = int(link_ids.size)
    layout = _local_binary_layout(n_local)

    touched_sites = np.unique(
        np.asarray(
            [site for link_id in link_ids for site in model.lattice.link_endpoints[int(link_id)]],
            dtype=np.int64,
        )
    )
    closed_site_set = set(int(site_id) for site_id in region.closed_site_ids)
    boundary_site_set = set(int(site_id) for site_id in region.boundary_site_ids)

    required_count = int(getattr(model, "required_count", 1))
    site_local_links: dict[int, npt.NDArray[np.int64]] = {}
    for site_id in touched_sites:
        incident_local = [
            local_index_by_link[int(link_id)]
            for link_id in model.lattice.incident_links(int(site_id))
            if int(link_id) in local_index_by_link
        ]
        site_local_links[int(site_id)] = np.asarray(incident_local, dtype=np.int64)

    # Degree order helps close/highly constrain sites earlier while remaining deterministic.
    scores = np.zeros(n_local, dtype=np.int64)
    for site_id in touched_sites:
        weight = 2 if int(site_id) in closed_site_set else 1
        for local_index in site_local_links[int(site_id)]:
            scores[int(local_index)] += weight
    variable_order = np.lexsort((np.arange(n_local), -scores)).astype(np.int64)

    config = np.zeros(n_local, dtype=np.int64)
    assigned = np.zeros(n_local, dtype=bool)
    states: list[npt.NDArray[np.int64]] = []

    full_link_region = n_local == int(model.lattice.num_links) and np.array_equal(
        np.sort(link_ids),
        np.arange(model.lattice.num_links, dtype=np.int64),
    )
    sectors = (
        tuple(model.make_sectors()) if (include_sectors_when_full and full_link_region) else ()
    )

    def partial_site_check(site_id: int) -> bool:
        local = site_local_links[site_id]
        if local.size == 0:
            return True
        assigned_local = assigned[local]
        assigned_values = config[local[assigned_local]]
        occupied = int(np.sum(assigned_values))
        unassigned = int(local.size - np.count_nonzero(assigned_local))

        if occupied > required_count:
            return False

        if site_id in closed_site_set:
            if occupied + unassigned < required_count:
                return False
            if unassigned == 0 and occupied != required_count:
                return False

        # Boundary sites can be completed by exterior links, so only the at-most
        # part is local.  If all incident links are local, they would have been
        # classified as closed above.
        return True

    sites_by_variable: list[list[int]] = [[] for _ in range(n_local)]
    for site_id, local_indices in site_local_links.items():
        for local_index in local_indices:
            sites_by_variable[int(local_index)].append(int(site_id))

    def full_check() -> bool:
        for site_id in closed_site_set:
            local = site_local_links[site_id]
            if int(np.sum(config[local])) != required_count:
                return False
        for site_id in boundary_site_set:
            local = site_local_links[site_id]
            if int(np.sum(config[local])) > required_count:
                return False
        if sectors:
            # In a full-link QDM region, local link order equals global link order.
            for sector in sectors:
                if not sector.is_satisfied(config):
                    return False
        return True

    def dfs(depth: int) -> None:
        if max_states is not None and len(states) >= max_states:
            return
        if depth == n_local:
            if full_check():
                states.append(config.copy())
            return

        local_variable = int(variable_order[depth])
        for value in (0, 1):
            if max_states is not None and len(states) >= max_states:
                return
            config[local_variable] = value
            assigned[local_variable] = True

            if all(partial_site_check(site_id) for site_id in sites_by_variable[local_variable]):
                dfs(depth + 1)

            assigned[local_variable] = False
            config[local_variable] = 0

    dfs(0)

    if not states:
        return Basis.empty(layout)

    arr = np.asarray(states, dtype=np.int64)
    if sort:
        order = np.lexsort(arr.T[::-1])
        arr = arr[order]
    return Basis.from_states(layout, arr)


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


def _deduplicate_local_records(
    records: list[LocalQDMCageRecord],
    *,
    hilbert_size: int,
    tolerance: float,
) -> list[LocalQDMCageRecord]:
    # Small, dependency-light rank deduplication by signature.  This mirrors the
    # global searcher semantics without importing its private selector class.
    kept: list[LocalQDMCageRecord] = []
    matrices_by_signature: dict[tuple[int, int], list[npt.NDArray[np.complex128]]] = defaultdict(
        list
    )

    for record in records:
        vector = np.zeros(hilbert_size, dtype=np.complex128)
        vector[record.support] = record.local_state
        group = matrices_by_signature[record.signature]

        if not group:
            group.append(vector)
            kept.append(record)
            continue

        old_matrix = np.vstack(group)
        new_matrix = np.vstack([old_matrix, vector])
        old_rank = np.linalg.matrix_rank(old_matrix, tol=tolerance)
        new_rank = np.linalg.matrix_rank(new_matrix, tol=tolerance)

        if new_rank > old_rank:
            group.append(vector)
            kept.append(record)

    return kept


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
    from qlinks.variables import LocalSpace, VariableLayout

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
