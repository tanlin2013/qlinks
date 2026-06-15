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
    CageRecord,
    CageSearchConfig,
    CageSearchResult,
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


@dataclass(frozen=True, slots=True)
class LocalQDMPaddingConfig:
    """Configuration for global padding/certification of local QDM cages.

    The first certification backend is intentionally conservative: it searches
    for a single shared exterior product configuration that can be tensored with
    every local support configuration of the cage state.  It then verifies the
    resulting global state by applying all QDM plaquette flips reachable in one
    kinetic step from the support, keyed by configurations rather than by a
    globally enumerated Hilbert space.
    """

    max_paddings_per_record: int = 1
    max_dfs_nodes: int | None = None
    include_sectors: bool = True
    require_static_exterior: bool = False
    tolerance: float = 1.0e-10
    sort_limited_basis: bool = True
    store_full_states: bool = True

    def __post_init__(self) -> None:
        if self.max_paddings_per_record < 0:
            raise ValueError("max_paddings_per_record must be non-negative.")
        if self.max_dfs_nodes is not None and self.max_dfs_nodes < 0:
            raise ValueError("max_dfs_nodes must be non-negative or None.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")


@dataclass(frozen=True, slots=True)
class LocalQDMPadding:
    """One shared-exterior padding of a local QDM cage record."""

    exterior_link_ids: npt.NDArray[np.int64]
    exterior_config: npt.NDArray[np.int64]
    global_support_configs: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LocalQDMCertificationReport:
    """Numerical certificate for one padded local QDM cage."""

    local_record_index: int
    padding_index: int
    signature: tuple[int, int]
    energy: complex
    kinetic_eigenvalue: complex
    self_loop_value: complex
    support_size: int
    one_hop_shell_size: int
    leakage_residual: float
    support_kinetic_residual: float
    support_hamiltonian_residual: float
    full_residual: float
    padding: LocalQDMPadding
    leakage_configs: npt.NDArray[np.int64]


@dataclass
class CertifiedLocalQDMCageSearchResult:
    """A certified local-first result with CageSearchResult-compatible records.

    ``cage_search_result`` is an ordinary :class:`CageSearchResult` whose
    Hilbert space is the limited certified basis, not the full global Hilbert
    space.  The companion ``basis``, ``kinetic_matrix``, and ``self_loop_values``
    are the limited objects needed by visualizers/classifiers/adapters.
    """

    cage_search_result: CageSearchResult
    basis: Basis
    kinetic_matrix: scipy_sparse.csr_array
    self_loop_values: npt.NDArray[np.complex128]
    reports: list[LocalQDMCertificationReport]
    padding_config: LocalQDMPaddingConfig

    def __len__(self) -> int:
        return len(self.cage_search_result)

    def __iter__(self):
        return iter(self.cage_search_result)

    def __getitem__(self, index):
        return self.cage_search_result[index]

    @property
    def records(self) -> list[CageRecord]:
        return self.cage_search_result.records

    @property
    def hilbert_size(self) -> int:
        return self.cage_search_result.hilbert_size

    @property
    def config(self) -> CageSearchConfig:
        return self.cage_search_result.config

    @property
    def counts_by_signature(self) -> dict[tuple[int, int], int]:
        return self.cage_search_result.counts_by_signature

    @property
    def signatures(self) -> list[tuple[int, int]]:
        return self.cage_search_result.signatures

    def records_by_signature(self, signature: tuple[int, int]) -> list[CageRecord]:
        return self.cage_search_result.records_by_signature(signature)

    def by_signature(self, signature: tuple[int, int]):
        return self.cage_search_result.by_signature(signature)

    def first(self, signature: tuple[int, int] | None = None) -> CageRecord:
        return self.cage_search_result.first(signature)

    def full_state_matrix(
        self,
        signature: tuple[int, int] | None = None,
    ) -> npt.NDArray[np.complex128]:
        return self.cage_search_result.full_state_matrix(signature)

    def cage_states(self) -> list[CageState]:
        return self.cage_search_result.cage_states()

    def as_cage_search_result(self) -> CageSearchResult:
        """Return the underlying ordinary CageSearchResult."""
        return self.cage_search_result


@dataclass
class LocalQDMCageSearchResult:
    """Result of a local QDM cage search."""

    records: list[LocalQDMCageRecord]
    region: LocalQDMRegion
    local_basis: Basis
    kinetic_matrix: scipy_sparse.csr_array
    self_loop_values: npt.NDArray[np.complex128]
    config: LocalQDMCageSearchConfig
    model: object | None = None
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

    def certify_paddings(
        self,
        *,
        config: LocalQDMPaddingConfig | None = None,
    ) -> CertifiedLocalQDMCageSearchResult:
        """Globally certify local records using shared-exterior QDM padding.

        The returned object carries ordinary ``CageRecord`` entries on a
        limited global basis containing the certified support states and their
        one-hop interference-zero shell.
        """
        if self.model is None:
            raise ValueError(
                "This LocalQDMCageSearchResult does not carry a model reference. "
                "Use LocalQDMCageSearcher.run() from the current API or call "
                "certify_qdm_local_result(..., model=...)."
            )
        return certify_qdm_local_result(self.model, self, config=config)


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
                model=self.model,
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
            model=self.model,
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


def certify_qdm_local_result(
    model: object,
    local_result: LocalQDMCageSearchResult,
    *,
    config: LocalQDMPaddingConfig | None = None,
) -> CertifiedLocalQDMCageSearchResult:
    """Pad and certify all local QDM records without a full basis/Hamiltonian.

    The certification uses a limited global basis made from the union of each
    certified support and its one-hop kinetic shell.  It returns ordinary
    ``CageRecord`` objects inside a ``CageSearchResult`` so downstream code that
    only depends on the cage-result protocol can consume the output.
    """
    padding_config = LocalQDMPaddingConfig() if config is None else config

    certified_items: list[tuple[LocalQDMCageRecord, LocalQDMCertificationReport]] = []
    limited_config_keys: set[tuple[int, ...]] = set()

    for local_record_index, local_record in enumerate(local_result.records):
        reports = certify_qdm_local_record(
            model,
            local_record,
            local_record_index=local_record_index,
            config=padding_config,
        )
        for report in reports:
            certified_items.append((local_record, report))
            for config_row in report.padding.global_support_configs:
                limited_config_keys.add(_config_key(config_row))
            for config_row in report.leakage_configs:
                limited_config_keys.add(_config_key(config_row))

    layout = model.layout

    if not certified_items:
        limited_basis = Basis.empty(layout)
        empty_matrix = scipy_sparse.csr_array((0, 0), dtype=np.complex128)
        search_config = _cage_search_config_from_local_and_padding(
            local_result.config,
            padding_config,
        )
        return CertifiedLocalQDMCageSearchResult(
            cage_search_result=CageSearchResult(
                records=[],
                hilbert_size=0,
                config=search_config,
                type1_candidates=[],
                type2_candidates=[],
                search_stage_seconds={},
            ),
            basis=limited_basis,
            kinetic_matrix=empty_matrix,
            self_loop_values=np.zeros(0, dtype=np.complex128),
            reports=[],
            padding_config=padding_config,
        )

    limited_configs = np.asarray([list(key) for key in limited_config_keys], dtype=np.int64)
    if padding_config.sort_limited_basis:
        order = np.lexsort(limited_configs.T[::-1])
        limited_configs = limited_configs[order]

    limited_basis = Basis.from_states(layout, limited_configs)
    limited_index = {_config_key(row): i for i, row in enumerate(limited_basis.states)}
    limited_kinetic = build_qdm_global_limited_kinetic_matrix(model, limited_basis)
    limited_self_loops = qdm_global_self_loop_values(model, limited_basis.states)

    search_config = _cage_search_config_from_local_and_padding(
        local_result.config,
        padding_config,
    )

    cage_records: list[CageRecord] = []
    candidate_by_signature: dict[tuple[int, int], list[CandidateSubgraph]] = defaultdict(list)

    for item_index, (local_record, report) in enumerate(certified_items):
        support_indices: list[int] = []
        support_amplitudes: list[complex] = []
        for config_row, amplitude in zip(
            report.padding.global_support_configs,
            local_record.local_state,
            strict=True,
        ):
            support_indices.append(int(limited_index[_config_key(config_row)]))
            support_amplitudes.append(complex(amplitude))

        support_arr = np.asarray(support_indices, dtype=np.int64)
        amplitude_arr = np.asarray(support_amplitudes, dtype=np.complex128)
        support_order = np.argsort(support_arr)
        support_arr = support_arr[support_order]
        amplitude_arr = amplitude_arr[support_order]

        norm = float(np.linalg.norm(amplitude_arr))
        if norm == 0.0:
            continue
        amplitude_arr = amplitude_arr / norm

        candidate = CandidateSubgraph(
            vertices=support_arr,
            label=f"local_qdm_certified_{item_index}",
            metadata={
                "source": "LocalQDMCageSearcher",
                "local_signature": local_record.signature,
                "local_link_ids": local_record.local_link_ids.copy(),
                "active_plaquette_ids": local_record.active_plaquette_ids.copy(),
                "scoring_plaquette_ids": local_record.scoring_plaquette_ids.copy(),
                "unresolved_boundary_plaquette_ids": (
                    local_record.unresolved_boundary_plaquette_ids.copy()
                ),
                "padding_exterior_link_ids": report.padding.exterior_link_ids.copy(),
                "padding_index": report.padding_index,
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )
        candidate_by_signature[report.signature].append(candidate)

        cage_state = CageState(
            energy=complex(report.energy),
            local_state=amplitude_arr,
            support=support_arr,
            boundary_residual=float(report.leakage_residual),
            eigen_residual=float(report.support_hamiltonian_residual),
            full_residual=float(report.full_residual),
            metadata={
                "source": "LocalQDMCageSearcher.certify_paddings",
                "local_record_index": report.local_record_index,
                "padding_index": report.padding_index,
                "kinetic_eigenvalue": report.kinetic_eigenvalue,
                "self_loop_value": report.self_loop_value,
                "support_kinetic_residual": report.support_kinetic_residual,
                "support_hamiltonian_residual": report.support_hamiltonian_residual,
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )

        full_state = None
        if padding_config.store_full_states:
            full_state = np.zeros(int(limited_basis.n_states), dtype=np.complex128)
            full_state[support_arr] = amplitude_arr

        cage_records.append(
            CageRecord(
                cage_state=cage_state,
                signature=report.signature,
                candidate=candidate,
                full_state=full_state,
            )
        )

    return CertifiedLocalQDMCageSearchResult(
        cage_search_result=CageSearchResult(
            records=cage_records,
            hilbert_size=int(limited_basis.n_states),
            config=search_config,
            type1_candidates=[
                candidate
                for signature in sorted(candidate_by_signature)
                for candidate in candidate_by_signature[signature]
            ],
            type2_candidates=[],
            search_stage_seconds={},
        ),
        basis=limited_basis,
        kinetic_matrix=limited_kinetic,
        self_loop_values=limited_self_loops,
        reports=[report for _record, report in certified_items],
        padding_config=padding_config,
    )


def certify_qdm_local_record(
    model: object,
    local_record: LocalQDMCageRecord,
    *,
    local_record_index: int = 0,
    config: LocalQDMPaddingConfig | None = None,
) -> list[LocalQDMCertificationReport]:
    """Return certified shared-exterior paddings for one local QDM record."""
    padding_config = LocalQDMPaddingConfig() if config is None else config
    if padding_config.max_paddings_per_record == 0:
        return []

    paddings = find_shared_qdm_exterior_paddings(
        model,
        local_record,
        config=padding_config,
    )

    reports: list[LocalQDMCertificationReport] = []
    for padding_index, padding in enumerate(paddings):
        report = _certify_qdm_padding(
            model,
            local_record,
            padding,
            local_record_index=local_record_index,
            padding_index=padding_index,
            config=padding_config,
        )
        if report is not None:
            reports.append(report)

    return reports


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

    scores = np.zeros(n_exterior, dtype=np.int64)
    for site_id, exterior_indices in site_exterior_links.items():
        target = site_targets[int(site_id)]
        weight = 2 if target in {0, len(exterior_indices)} else 1
        for exterior_index in exterior_indices:
            scores[int(exterior_index)] += weight
    variable_order = np.lexsort((np.arange(n_exterior), -scores)).astype(np.int64)

    exterior_config = np.zeros(n_exterior, dtype=np.int64)
    assigned = np.zeros(n_exterior, dtype=bool)
    sites_by_exterior_variable: list[list[int]] = [[] for _ in range(n_exterior)]
    for site_id, exterior_indices in site_exterior_links.items():
        for exterior_index in exterior_indices:
            sites_by_exterior_variable[int(exterior_index)].append(int(site_id))

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
        for value in (0, 1):
            if len(paddings) >= padding_config.max_paddings_per_record:
                return
            exterior_config[exterior_variable] = value
            assigned[exterior_variable] = True
            touched_sites = sites_by_exterior_variable[exterior_variable]
            if all(partial_site_check(site_id) for site_id in touched_sites):
                dfs(depth + 1)
            assigned[exterior_variable] = False
            exterior_config[exterior_variable] = 0

    dfs(0)
    return paddings


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

    for col, config_row in enumerate(basis.states):
        for plaquette_id in model.plaquette_ids():
            transition = _qdm_flip_transition(model, config_row, int(plaquette_id))
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
    arr = np.asarray(configs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    values = np.zeros(arr.shape[0], dtype=np.complex128)
    for row_index, config_row in enumerate(arr):
        values[row_index] = _qdm_global_self_loop_value(model, config_row)
    return values


def _certify_qdm_padding(
    model: object,
    local_record: LocalQDMCageRecord,
    padding: LocalQDMPadding,
    *,
    local_record_index: int,
    padding_index: int,
    config: LocalQDMPaddingConfig,
) -> LocalQDMCertificationReport | None:
    amplitudes = np.asarray(local_record.local_state, dtype=np.complex128)
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        return None
    amplitudes = amplitudes / norm

    support_configs = np.asarray(padding.global_support_configs, dtype=np.int64)
    support_keys = [_config_key(config_row) for config_row in support_configs]
    support_amplitude_by_key = {
        key: complex(amplitude) for key, amplitude in zip(support_keys, amplitudes, strict=True)
    }

    action_by_key: dict[tuple[int, ...], complex] = defaultdict(complex)
    touched_keys: set[tuple[int, ...]] = set(support_keys)

    for source_config, source_amplitude in zip(support_configs, amplitudes, strict=True):
        for plaquette_id in model.plaquette_ids():
            transition = _qdm_flip_transition(model, source_config, int(plaquette_id))
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = _config_key(final_config)
            action_by_key[final_key] += complex(coefficient) * complex(source_amplitude)
            touched_keys.add(final_key)

    kappa = complex(local_record.kappa)
    support_kinetic_residuals: list[complex] = []
    leakage_values: list[complex] = []
    leakage_configs: list[npt.NDArray[np.int64]] = []

    for key in sorted(touched_keys):
        action = complex(action_by_key.get(key, 0.0 + 0.0j))
        if key in support_amplitude_by_key:
            expected = kappa * support_amplitude_by_key[key]
            support_kinetic_residuals.append(action - expected)
        else:
            leakage_values.append(action)
            leakage_configs.append(np.asarray(key, dtype=np.int64))

    support_kinetic_residual = float(np.linalg.norm(np.asarray(support_kinetic_residuals)))
    leakage_residual = float(np.linalg.norm(np.asarray(leakage_values, dtype=np.complex128)))

    if leakage_residual > config.tolerance:
        return None
    if support_kinetic_residual > config.tolerance:
        return None

    support_self_loops = qdm_global_self_loop_values(model, support_configs)
    self_loop_value = complex(support_self_loops[0]) if support_self_loops.size else 0.0 + 0.0j
    if np.linalg.norm(support_self_loops - self_loop_value) > config.tolerance:
        return None

    energy = self_loop_value + kappa
    support_h_residuals = []
    for key, amplitude, self_loop in zip(
        support_keys,
        amplitudes,
        support_self_loops,
        strict=True,
    ):
        kinetic_action = complex(action_by_key.get(key, 0.0 + 0.0j))
        support_h_residuals.append(
            kinetic_action + complex(self_loop) * amplitude - energy * amplitude
        )
    support_hamiltonian_residual = float(
        np.linalg.norm(np.asarray(support_h_residuals, dtype=np.complex128))
    )
    full_residual = float(np.hypot(support_hamiltonian_residual, leakage_residual))
    if full_residual > config.tolerance:
        return None

    signature = signature_from_energy_and_self_loop(
        energy,
        self_loop_value,
        tolerance=max(config.tolerance, 1.0e-15) * 10.0,
        potential_unit=_infer_potential_unit_from_model(model),
    )
    if signature is None:
        return None

    leakage_arr = (
        np.asarray(leakage_configs, dtype=np.int64)
        if leakage_configs
        else np.empty((0, int(model.lattice.num_links)), dtype=np.int64)
    )

    return LocalQDMCertificationReport(
        local_record_index=int(local_record_index),
        padding_index=int(padding_index),
        signature=signature,
        energy=energy,
        kinetic_eigenvalue=kappa,
        self_loop_value=self_loop_value,
        support_size=int(support_configs.shape[0]),
        one_hop_shell_size=int(len(touched_keys)),
        leakage_residual=leakage_residual,
        support_kinetic_residual=support_kinetic_residual,
        support_hamiltonian_residual=support_hamiltonian_residual,
        full_residual=full_residual,
        padding=padding,
        leakage_configs=leakage_arr,
    )


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
    for plaquette_id in model.plaquette_ids():
        plaquette_links = set(
            int(link_id) for link_id in model.lattice.plaquette_links(int(plaquette_id))
        )
        if plaquette_links.intersection(local_link_set):
            continue
        for config_row in padding.global_support_configs:
            if _qdm_flip_transition(model, config_row, int(plaquette_id)) is not None:
                return False
    return True


def _qdm_flip_transition(
    model: object,
    config_row: npt.ArrayLike,
    plaquette_id: int,
) -> tuple[npt.NDArray[np.int64], complex] | None:
    config_arr = np.asarray(config_row, dtype=np.int64)
    links = np.asarray(model.lattice.plaquette_links(int(plaquette_id)), dtype=np.int64)
    values = config_arr[links]
    p0, p1 = alternating_binary_patterns(links.size)
    coupling = model._coup_kin_at(int(plaquette_id))

    if np.array_equal(values, p0):
        final = config_arr.copy()
        final[links] = p1
        return final, _forward_coefficient(coupling)
    if np.array_equal(values, p1):
        final = config_arr.copy()
        final[links] = p0
        return final, _backward_coefficient(coupling)
    return None


def _qdm_global_self_loop_value(model: object, config_row: npt.ArrayLike) -> complex:
    config_arr = np.asarray(config_row, dtype=np.int64)
    total = 0.0 + 0.0j
    for plaquette_id in model.plaquette_ids():
        links = np.asarray(model.lattice.plaquette_links(int(plaquette_id)), dtype=np.int64)
        values = config_arr[links]
        p0, p1 = alternating_binary_patterns(links.size)
        if np.array_equal(values, p0) or np.array_equal(values, p1):
            total += complex(model._coup_pot_at(int(plaquette_id)))
    return complex(total)


def _config_key(config_row: npt.ArrayLike) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(config_row, dtype=np.int64))


def _cage_search_config_from_local_and_padding(
    local_config: LocalQDMCageSearchConfig,
    padding_config: LocalQDMPaddingConfig,
) -> CageSearchConfig:
    return CageSearchConfig(
        search_type="type1",
        tolerance=min(local_config.tolerance, padding_config.tolerance),
        min_component_size=local_config.min_component_size,
        validate_full_residual=local_config.validate_full_residual,
        type1_kappas=local_config.allowed_kappas,
        deduplicate_by_rank=False,
        potential_signature_unit=local_config.potential_signature_unit,
        store_full_states=padding_config.store_full_states,
    )


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
