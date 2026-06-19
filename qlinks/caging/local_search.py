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

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis, DFSBasisSolver
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
from qlinks.caging.types import DegenerateBasisStrategy
from qlinks.constraints import ConstraintPropagation, ConstraintResult
from qlinks.models.couplings import DirectedPlaquetteCoupling
from qlinks.operators.plaquette import alternating_binary_patterns
from qlinks.variables import VariableLayout

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

    prune_inactive_local_basis_states:
        For genuine local regions, ask the shared DFS to prune branches that can
        no longer produce a configuration flippable on any active plaquette.
        Such configurations are isolated in the local kinetic graph and cannot
        enter a nontrivial type-1 component when ``min_component_size > 1``.  It
        is opt-in because observer calls add Python overhead on very small
        regions.
    """

    tolerance: float = 1.0e-10
    allowed_kappas: tuple[int, ...] = (0,)
    min_component_size: int = 2
    halo_layers: int = 1
    boundary_mode: LocalBoundaryMode = "relaxed"
    include_sectors_when_full: bool = True
    prune_inactive_local_basis_states: bool = False
    max_local_states: int | None = None
    sort_basis: bool = True
    validate_full_residual: bool = True

    # Degenerate local cage handling.  ``"ipr"`` rotates a degenerate
    # fixed-kappa nullspace toward compact high-IPR representatives before
    # support trimming, preventing one large mixed support from representing
    # several smaller cages.
    degenerate_basis_strategy: DegenerateBasisStrategy = "none"
    ipr_n_restarts: int = 128
    ipr_max_iter: int = 1000
    ipr_step_size: float = 0.1
    ipr_candidate_count: int = 64
    ipr_rank_completion_patience: int | None = None
    ipr_batch_size: int = 16
    ipr_random_seed: int | None = None

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
        if self.degenerate_basis_strategy not in {"none", "ipr"}:
            raise ValueError("degenerate_basis_strategy must be 'none' or 'ipr'.")
        if self.ipr_n_restarts < 0:
            raise ValueError("ipr_n_restarts must be non-negative.")
        if self.ipr_max_iter < 0:
            raise ValueError("ipr_max_iter must be non-negative.")
        if self.ipr_step_size <= 0:
            raise ValueError("ipr_step_size must be positive.")
        if self.ipr_candidate_count < 0:
            raise ValueError("ipr_candidate_count must be non-negative.")
        if self.ipr_rank_completion_patience is not None and self.ipr_rank_completion_patience < 0:
            raise ValueError("ipr_rank_completion_patience must be non-negative or None.")
        if self.ipr_batch_size <= 0:
            raise ValueError("ipr_batch_size must be positive.")


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
class StripeRegionProposalRecord:
    """One plaquette-stripe local-region proposal.

    ``direction`` is the anchor-coordinate axis along which the stripe runs.
    ``transverse_origin`` labels the first transverse coordinate included in the
    band.  For periodic lattices and ``width > 1``, the band is thickened by
    wrapping forward from this origin.
    """

    region: LocalQDMRegion
    plaquette_ids: npt.NDArray[np.int64]
    direction: int
    transverse_origin: tuple[int, ...]
    width: int
    plaquette_kind: str

    def __post_init__(self) -> None:
        plaquette_ids = np.asarray(self.plaquette_ids, dtype=np.int64)
        if plaquette_ids.ndim != 1:
            raise ValueError("plaquette_ids must be one-dimensional.")
        object.__setattr__(
            self,
            "plaquette_ids",
            np.unique(plaquette_ids).astype(np.int64),
        )
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(self, "width", int(self.width))
        if self.width <= 0:
            raise ValueError("width must be positive.")
        object.__setattr__(
            self,
            "transverse_origin",
            tuple(int(value) for value in self.transverse_origin),
        )
        object.__setattr__(self, "plaquette_kind", str(self.plaquette_kind))


@dataclass(frozen=True, slots=True)
class SnakeStripeRegionProposalRecord:
    """One width-one noncontractible snake-stripe region proposal.

    Unlike :class:`StripeRegionProposalRecord`, this record is generated from
    simple noncontractible cycles on the plaquette shared-link graph.  It is
    useful on lattices where natural stripe cages wrap around the torus but do
    not follow a straight anchor-coordinate line.
    """

    region: LocalQDMRegion
    plaquette_ids: npt.NDArray[np.int64]
    seed_plaquette_id: int
    winding: tuple[int, ...]
    length: int
    turn_count: int
    plaquette_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        plaquette_ids = np.asarray(self.plaquette_ids, dtype=np.int64)
        if plaquette_ids.ndim != 1:
            raise ValueError("plaquette_ids must be one-dimensional.")
        if plaquette_ids.size == 0:
            raise ValueError("plaquette_ids must be non-empty.")
        object.__setattr__(
            self,
            "plaquette_ids",
            np.unique(plaquette_ids).astype(np.int64),
        )
        object.__setattr__(self, "seed_plaquette_id", int(self.seed_plaquette_id))
        object.__setattr__(self, "winding", tuple(int(value) for value in self.winding))
        object.__setattr__(self, "length", int(self.length))
        object.__setattr__(self, "turn_count", int(self.turn_count))
        object.__setattr__(
            self,
            "plaquette_kinds",
            tuple(str(kind) for kind in self.plaquette_kinds),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRegionProposalRecord:
    """One dynamically grown plaquette-region proposal.

    The adaptive proposal stores the seed plaquettes, the selected plaquette set,
    and the cheap heuristic score that was used by the beam search.  Optional
    local-search feedback is filled only when ``use_search_feedback=True`` on
    :class:`AdaptiveRegionProposal`.
    """

    region: LocalQDMRegion
    plaquette_ids: npt.NDArray[np.int64]
    seed_plaquette_ids: npt.NDArray[np.int64]
    generation: int
    score: float
    link_count: int
    unresolved_boundary_count: int
    local_hilbert_size: int | None = None
    n_records: int | None = None
    counts_by_signature: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plaquette_ids = np.asarray(self.plaquette_ids, dtype=np.int64)
        if plaquette_ids.ndim != 1:
            raise ValueError("plaquette_ids must be one-dimensional.")
        if plaquette_ids.size == 0:
            raise ValueError("plaquette_ids must be non-empty.")
        object.__setattr__(
            self,
            "plaquette_ids",
            np.unique(plaquette_ids).astype(np.int64),
        )

        seed_ids = np.asarray(self.seed_plaquette_ids, dtype=np.int64)
        if seed_ids.ndim != 1:
            raise ValueError("seed_plaquette_ids must be one-dimensional.")
        if seed_ids.size == 0:
            raise ValueError("seed_plaquette_ids must be non-empty.")
        object.__setattr__(
            self,
            "seed_plaquette_ids",
            np.unique(seed_ids).astype(np.int64),
        )
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "link_count", int(self.link_count))
        object.__setattr__(
            self,
            "unresolved_boundary_count",
            int(self.unresolved_boundary_count),
        )
        if self.local_hilbert_size is not None:
            object.__setattr__(self, "local_hilbert_size", int(self.local_hilbert_size))
        if self.n_records is not None:
            object.__setattr__(self, "n_records", int(self.n_records))
        object.__setattr__(
            self,
            "counts_by_signature",
            {
                (int(signature[0]), int(signature[1])): int(count)
                for signature, count in self.counts_by_signature.items()
            },
        )


@dataclass(frozen=True, slots=True)
class ConnectedRegionProposalRecord:
    """One connected plaquette-set local-region proposal.

    Unlike the adaptive beam proposal, this record comes from exhaustive
    connected-region enumeration under explicit size limits.  It is intended as
    a robust fallback when the cage shape is not known a priori.
    """

    region: LocalQDMRegion
    plaquette_ids: npt.NDArray[np.int64]
    seed_plaquette_id: int
    size: int
    link_count: int
    unresolved_boundary_count: int

    def __post_init__(self) -> None:
        plaquette_ids = np.asarray(self.plaquette_ids, dtype=np.int64)
        if plaquette_ids.ndim != 1:
            raise ValueError("plaquette_ids must be one-dimensional.")
        if plaquette_ids.size == 0:
            raise ValueError("plaquette_ids must be non-empty.")
        object.__setattr__(
            self,
            "plaquette_ids",
            np.unique(plaquette_ids).astype(np.int64),
        )
        object.__setattr__(self, "seed_plaquette_id", int(self.seed_plaquette_id))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "link_count", int(self.link_count))
        object.__setattr__(
            self,
            "unresolved_boundary_count",
            int(self.unresolved_boundary_count),
        )


class LocalRegionProposal(Protocol):
    """Protocol for objects that propose local regions to the local cage searcher."""

    def iter_regions(self) -> Iterator[LocalQDMRegion]:
        """Yield candidate local regions."""
        ...


@dataclass(frozen=True, slots=True)
class StripeRegionProposal:
    """Generate QDM stripe/band local regions from plaquette anchor coordinates.

    A stripe is selected on the plaquette-anchor lattice.  For ``direction=0``
    on a square torus, the proposal keeps all plaquettes along the x direction
    at fixed y; for ``direction=1`` it keeps all plaquettes along y at fixed x.
    ``width`` thickens the stripe in the transverse coordinate.

    The default search config uses ``halo_layers=0`` because the stripe itself
    is meant to be the active region.  Passing a config with ``halo_layers > 0``
    intentionally asks for the old shared-link halo around each stripe.
    """

    model: object
    config: LocalQDMCageSearchConfig = field(
        default_factory=lambda: LocalQDMCageSearchConfig(halo_layers=0)
    )
    directions: tuple[int, ...] | None = None
    width: int = 1
    plaquette_kinds: tuple[str, ...] | None = None
    adapter: LocalCageModelAdapter | None = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive.")

        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        config = adapter.normalize_config(self.config)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "config", config)

        if self.directions is not None:
            directions = tuple(int(direction) for direction in self.directions)
            if not directions:
                raise ValueError("directions must be non-empty when provided.")
            object.__setattr__(self, "directions", directions)

        if self.plaquette_kinds is not None:
            kinds = tuple(str(kind) for kind in self.plaquette_kinds)
            if not kinds:
                raise ValueError("plaquette_kinds must be non-empty when provided.")
            object.__setattr__(self, "plaquette_kinds", kinds)

    def iter_records(self) -> Iterator[StripeRegionProposalRecord]:
        """Yield stripe proposal records, including metadata and regions."""
        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        plaquette_data = _stripe_plaquette_data(self.model, self.plaquette_kinds)
        if not plaquette_data:
            return

        directions = self.directions
        if directions is None:
            directions = _default_stripe_directions(plaquette_data)

        seen: set[tuple[int, str, tuple[int, ...]]] = set()

        for direction in directions:
            direction = int(direction)
            _validate_stripe_direction(direction, plaquette_data)

            for kind in sorted({item[2] for item in plaquette_data}):
                kind_items = [item for item in plaquette_data if item[2] == kind]
                origins = sorted(
                    {_transverse_coordinates(cell, direction) for _, cell, _ in kind_items}
                )

                for origin in origins:
                    plaquette_ids = np.asarray(
                        [
                            plaquette_id
                            for plaquette_id, cell, _ in kind_items
                            if _cell_in_stripe_band(
                                self.model,
                                cell,
                                direction=direction,
                                transverse_origin=origin,
                                width=self.width,
                            )
                        ],
                        dtype=np.int64,
                    )
                    if plaquette_ids.size == 0:
                        continue

                    key = (
                        direction,
                        kind,
                        tuple(int(pid) for pid in np.unique(plaquette_ids)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    region = adapter.build_region_from_plaquettes(
                        plaquette_ids=plaquette_ids,
                        config=self.config,
                        scoring_plaquette_ids=plaquette_ids,
                    )
                    yield StripeRegionProposalRecord(
                        region=region,
                        plaquette_ids=plaquette_ids,
                        direction=direction,
                        transverse_origin=origin,
                        width=self.width,
                        plaquette_kind=kind,
                    )

    def iter_regions(self) -> Iterator[LocalQDMRegion]:
        """Yield only the local regions from :meth:`iter_records`."""
        for record in self.iter_records():
            yield record.region

    def iter_searchers(self) -> Iterator[LocalCageSearcher]:
        """Yield ready-to-run local cage searchers for each stripe region."""
        for record in self.iter_records():
            yield LocalCageSearcher(
                model=self.model,
                region=record.region,
                config=self.config,
                adapter=self.adapter,
            )


@dataclass(frozen=True, slots=True)
class SnakeStripeRegionProposal:
    """Generate width-one noncontractible snake stripes on the plaquette graph.

    A snake stripe is a simple cycle of plaquettes, adjacent by shared links,
    whose lifted anchor-cell displacement winds around a periodic lattice.  This
    proposal does not assume the stripe is straight in anchor coordinates; it is
    therefore a better first pass for honeycomb and triangular QDM where useful
    width-one stripes can turn while wrapping the torus.

    The enumeration is intentionally budgeted by ``max_plaquettes``,
    ``max_links``, ``max_turns``, and ``max_records``.
    """

    model: object
    max_plaquettes: int
    config: LocalQDMCageSearchConfig = field(
        default_factory=lambda: LocalQDMCageSearchConfig(halo_layers=0)
    )
    min_plaquettes: int = 3
    seed_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None
    max_records: int | None = None
    max_links: int | None = None
    max_turns: int | None = None
    plaquette_kinds: tuple[str, ...] | None = None
    allow_kind_changes: bool = False
    winding_vectors: tuple[tuple[int, ...], ...] | None = None
    adapter: LocalCageModelAdapter | None = None

    def __post_init__(self) -> None:
        if self.max_plaquettes <= 0:
            raise ValueError("max_plaquettes must be positive.")
        if self.min_plaquettes <= 0:
            raise ValueError("min_plaquettes must be positive.")
        if self.min_plaquettes > self.max_plaquettes:
            raise ValueError("min_plaquettes cannot exceed max_plaquettes.")
        if self.max_records is not None and self.max_records < 0:
            raise ValueError("max_records must be non-negative or None.")
        if self.max_links is not None and self.max_links <= 0:
            raise ValueError("max_links must be positive or None.")
        if self.max_turns is not None and self.max_turns < 0:
            raise ValueError("max_turns must be non-negative or None.")

        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        config = adapter.normalize_config(self.config)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "config", config)

        if self.seed_plaquette_ids is not None:
            seed_ids = _unique_int_array(self.seed_plaquette_ids, name="seed_plaquette_ids")
            _validate_plaquette_ids(self.model, seed_ids)
            object.__setattr__(self, "seed_plaquette_ids", seed_ids)

        if self.plaquette_kinds is not None:
            kinds = tuple(str(kind) for kind in self.plaquette_kinds)
            if not kinds:
                raise ValueError("plaquette_kinds must be non-empty when provided.")
            object.__setattr__(self, "plaquette_kinds", kinds)

        if self.winding_vectors is not None:
            windings = tuple(
                tuple(int(value) for value in winding) for winding in self.winding_vectors
            )
            if not windings:
                raise ValueError("winding_vectors must be non-empty when provided.")
            object.__setattr__(self, "winding_vectors", windings)

    def iter_records(self) -> Iterator[SnakeStripeRegionProposalRecord]:
        """Yield snake-stripe records in deterministic DFS order."""
        seed_ids = _adaptive_seed_plaquette_ids(self.model, self.seed_plaquette_ids)
        edge_map = _plaquette_shared_link_neighbor_edges(
            self.model,
            plaquette_kinds=self.plaquette_kinds,
            allow_kind_changes=self.allow_kind_changes,
        )
        allowed_windings = (
            None
            if self.winding_vectors is None
            else {tuple(int(value) for value in winding) for winding in self.winding_vectors}
        )

        emitted: set[tuple[int, ...]] = set()
        n_emitted = 0

        for seed_id in seed_ids:
            seed_id = int(seed_id)
            if seed_id not in edge_map:
                continue

            stack: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None, int]]
            stack = [((seed_id,), _zero_cell_displacement(self.model), None, 0)]

            while stack:
                path, lifted_position, previous_step, turn_count = stack.pop()
                current = int(path[-1])

                for neighbor, step in reversed(edge_map.get(current, ())):
                    neighbor = int(neighbor)
                    step = tuple(int(value) for value in step)
                    next_lifted = _add_cell_displacements(lifted_position, step)
                    next_turn_count = turn_count + _snake_step_turn_increment(previous_step, step)
                    if self.max_turns is not None and next_turn_count > int(self.max_turns):
                        continue

                    if neighbor == seed_id:
                        if len(path) < int(self.min_plaquettes):
                            continue
                        winding = _winding_from_lifted_displacement(self.model, next_lifted)
                        if winding is None or not any(int(value) != 0 for value in winding):
                            continue
                        if allowed_windings is not None and winding not in allowed_windings:
                            continue

                        key = tuple(sorted(int(pid) for pid in path))
                        if key in emitted:
                            continue

                        record = self._make_record(
                            plaquette_ids=path,
                            seed_plaquette_id=seed_id,
                            winding=winding,
                            turn_count=next_turn_count,
                        )
                        if record is None:
                            continue

                        emitted.add(key)
                        n_emitted += 1
                        yield record
                        if self.max_records is not None and n_emitted >= int(self.max_records):
                            return
                        continue

                    if neighbor in path or len(path) >= int(self.max_plaquettes):
                        continue

                    stack.append(
                        (
                            (*path, neighbor),
                            next_lifted,
                            step,
                            next_turn_count,
                        )
                    )

    def iter_regions(self) -> Iterator[LocalQDMRegion]:
        """Yield only the local regions from :meth:`iter_records`."""
        for record in self.iter_records():
            yield record.region

    def iter_searchers(self) -> Iterator[LocalCageSearcher]:
        """Yield ready-to-run local cage searchers for proposed regions."""
        for record in self.iter_records():
            yield LocalCageSearcher(
                model=self.model,
                region=record.region,
                config=self.config,
                adapter=self.adapter,
            )

    def _make_record(
        self,
        *,
        plaquette_ids: Sequence[int],
        seed_plaquette_id: int,
        winding: tuple[int, ...],
        turn_count: int,
    ) -> SnakeStripeRegionProposalRecord | None:
        selected = np.asarray(tuple(sorted({int(pid) for pid in plaquette_ids})), dtype=np.int64)
        if selected.size == 0:
            return None

        region = self.adapter.build_region_from_plaquettes(
            plaquette_ids=selected,
            config=self.config,
            scoring_plaquette_ids=selected,
        )
        if self.max_links is not None and region.link_ids.size > int(self.max_links):
            return None

        kinds = tuple(
            sorted({str(self.model.lattice.plaquettes[int(pid)].kind) for pid in selected})
        )
        return SnakeStripeRegionProposalRecord(
            region=region,
            plaquette_ids=selected,
            seed_plaquette_id=int(seed_plaquette_id),
            winding=winding,
            length=int(selected.size),
            turn_count=int(turn_count),
            plaquette_kinds=kinds,
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRegionProposal:
    """Dynamically grow local QDM regions with a beam-search heuristic.

    Unlike :class:`StripeRegionProposal`, this strategy does not assume a fixed
    region shape.  It starts from one seed plaquette at a time, repeatedly adds
    neighboring plaquettes sharing links with the current region, and keeps only
    the best-scoring partial regions under hard size limits.

    ``use_search_feedback=False`` keeps proposal generation cheap and scores
    regions by structural proxies: small kinetic boundary, moderate link count,
    and compact shared-link connectivity.  Setting ``use_search_feedback=True``
    additionally runs the local cage searcher while growing and boosts regions
    that already contain candidate local cages.
    """

    model: object
    max_plaquettes: int
    config: LocalQDMCageSearchConfig = field(
        default_factory=lambda: LocalQDMCageSearchConfig(halo_layers=0)
    )
    seed_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None
    min_plaquettes: int = 1
    beam_width: int = 8
    branch_factor: int = 8
    max_regions: int | None = None
    max_links: int | None = None
    use_search_feedback: bool = False
    adapter: LocalCageModelAdapter | None = None

    def __post_init__(self) -> None:
        if self.max_plaquettes <= 0:
            raise ValueError("max_plaquettes must be positive.")
        if self.min_plaquettes <= 0:
            raise ValueError("min_plaquettes must be positive.")
        if self.min_plaquettes > self.max_plaquettes:
            raise ValueError("min_plaquettes cannot exceed max_plaquettes.")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.branch_factor <= 0:
            raise ValueError("branch_factor must be positive.")
        if self.max_regions is not None and self.max_regions < 0:
            raise ValueError("max_regions must be non-negative or None.")
        if self.max_links is not None and self.max_links <= 0:
            raise ValueError("max_links must be positive or None.")

        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        config = adapter.normalize_config(self.config)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "config", config)

        if self.seed_plaquette_ids is not None:
            seed_ids = _unique_int_array(self.seed_plaquette_ids, name="seed_plaquette_ids")
            _validate_plaquette_ids(self.model, seed_ids)
            object.__setattr__(self, "seed_plaquette_ids", seed_ids)

    def iter_records(self) -> Iterator[AdaptiveRegionProposalRecord]:
        """Yield adaptive proposal records in beam-search order."""
        plaquette_ids = _adaptive_seed_plaquette_ids(self.model, self.seed_plaquette_ids)
        neighbor_map = _plaquette_shared_link_neighbor_map(self.model)

        beam: list[AdaptiveRegionProposalRecord] = []
        for plaquette_id in plaquette_ids:
            record = self._make_record(
                plaquette_ids=(int(plaquette_id),),
                seed_plaquette_ids=(int(plaquette_id),),
                generation=1,
                neighbor_map=neighbor_map,
            )
            if record is not None:
                beam.append(record)

        beam = _top_adaptive_records(beam, self.beam_width)
        emitted: set[tuple[int, ...]] = set()
        considered: set[tuple[int, ...]] = {
            tuple(int(pid) for pid in record.plaquette_ids) for record in beam
        }

        for generation in range(1, int(self.max_plaquettes) + 1):
            for record in beam:
                key = tuple(int(pid) for pid in record.plaquette_ids)
                if key in emitted or len(key) < int(self.min_plaquettes):
                    continue
                emitted.add(key)
                yield record
                if self.max_regions is not None and len(emitted) >= int(self.max_regions):
                    return

            if generation >= int(self.max_plaquettes):
                break

            next_records: list[AdaptiveRegionProposalRecord] = []
            for parent in beam:
                parent_set = frozenset(int(pid) for pid in parent.plaquette_ids)
                expansions: list[AdaptiveRegionProposalRecord] = []
                for plaquette_id in _adaptive_region_frontier(parent_set, neighbor_map):
                    child = tuple(sorted((*parent_set, int(plaquette_id))))
                    if child in considered:
                        continue
                    considered.add(child)
                    record = self._make_record(
                        plaquette_ids=child,
                        seed_plaquette_ids=parent.seed_plaquette_ids,
                        generation=generation + 1,
                        neighbor_map=neighbor_map,
                    )
                    if record is not None:
                        expansions.append(record)

                next_records.extend(_top_adaptive_records(expansions, self.branch_factor))

            beam = _top_adaptive_records(next_records, self.beam_width)
            if not beam:
                break

    def iter_regions(self) -> Iterator[LocalQDMRegion]:
        """Yield only the local regions from :meth:`iter_records`."""
        for record in self.iter_records():
            yield record.region

    def iter_searchers(self) -> Iterator[LocalCageSearcher]:
        """Yield ready-to-run local cage searchers for proposed regions."""
        for record in self.iter_records():
            yield LocalCageSearcher(
                model=self.model,
                region=record.region,
                config=self.config,
                adapter=self.adapter,
            )

    def _make_record(
        self,
        *,
        plaquette_ids: Sequence[int],
        seed_plaquette_ids: Sequence[int] | npt.ArrayLike,
        generation: int,
        neighbor_map: dict[int, frozenset[int]],
    ) -> AdaptiveRegionProposalRecord | None:
        selected = np.asarray(tuple(sorted({int(pid) for pid in plaquette_ids})), dtype=np.int64)
        if selected.size == 0 or selected.size > int(self.max_plaquettes):
            return None

        region = self.adapter.build_region_from_plaquettes(
            plaquette_ids=selected,
            config=self.config,
            scoring_plaquette_ids=selected,
        )
        if self.max_links is not None and region.link_ids.size > int(self.max_links):
            return None

        local_hilbert_size: int | None = None
        n_records: int | None = None
        counts_by_signature: dict[tuple[int, int], int] = {}
        feedback_bonus = 0.0
        if self.use_search_feedback:
            result = LocalCageSearcher(
                model=self.model,
                region=region,
                config=self.config,
                adapter=self.adapter,
            ).run()
            local_hilbert_size = result.local_hilbert_size
            n_records = len(result.records)
            counts_by_signature = result.counts_by_signature
            feedback_bonus = 10.0 * float(n_records)

        score = _adaptive_region_score(
            region,
            plaquette_ids=selected,
            neighbor_map=neighbor_map,
            feedback_bonus=feedback_bonus,
        )
        return AdaptiveRegionProposalRecord(
            region=region,
            plaquette_ids=selected,
            seed_plaquette_ids=np.asarray(seed_plaquette_ids, dtype=np.int64),
            generation=int(generation),
            score=score,
            link_count=int(region.link_ids.size),
            unresolved_boundary_count=int(region.unresolved_boundary_plaquette_ids.size),
            local_hilbert_size=local_hilbert_size,
            n_records=n_records,
            counts_by_signature=counts_by_signature,
        )


@dataclass(frozen=True, slots=True)
class ConnectedRegionProposal:
    """Enumerate connected plaquette regions under explicit size budgets.

    This is the robust, shape-agnostic counterpart of the stripe/adaptive
    proposals.  It exhaustively enumerates connected plaquette sets on the
    shared-link plaquette graph up to ``max_plaquettes`` and optionally
    ``max_links``.  It is deliberately simple: the only physics assumption is
    connectedness on the kinetic plaquette graph, while the local solver and
    global certification decide which regions are useful.
    """

    model: object
    max_plaquettes: int
    config: LocalQDMCageSearchConfig = field(
        default_factory=lambda: LocalQDMCageSearchConfig(halo_layers=0)
    )
    min_plaquettes: int = 1
    seed_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None
    max_regions: int | None = None
    max_links: int | None = None
    adapter: LocalCageModelAdapter | None = None

    def __post_init__(self) -> None:
        if self.max_plaquettes <= 0:
            raise ValueError("max_plaquettes must be positive.")
        if self.min_plaquettes <= 0:
            raise ValueError("min_plaquettes must be positive.")
        if self.min_plaquettes > self.max_plaquettes:
            raise ValueError("min_plaquettes cannot exceed max_plaquettes.")
        if self.max_regions is not None and self.max_regions < 0:
            raise ValueError("max_regions must be non-negative or None.")
        if self.max_links is not None and self.max_links <= 0:
            raise ValueError("max_links must be positive or None.")

        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        config = adapter.normalize_config(self.config)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "config", config)

        if self.seed_plaquette_ids is not None:
            seed_ids = _unique_int_array(self.seed_plaquette_ids, name="seed_plaquette_ids")
            _validate_plaquette_ids(self.model, seed_ids)
            object.__setattr__(self, "seed_plaquette_ids", seed_ids)

    def iter_records(self) -> Iterator[ConnectedRegionProposalRecord]:
        """Yield connected plaquette-set records in increasing size order."""
        seeds = _adaptive_seed_plaquette_ids(self.model, self.seed_plaquette_ids)
        neighbor_map = _plaquette_shared_link_neighbor_map(self.model)
        emitted: set[tuple[int, ...]] = set()
        queued: set[tuple[int, ...]] = set()
        queue: list[tuple[int, tuple[int, ...]]] = []

        for seed in seeds:
            key = (int(seed),)
            if key in queued:
                continue
            queued.add(key)
            queue.append((int(seed), key))

        yielded = 0
        head = 0
        while head < len(queue):
            seed, current = queue[head]
            head += 1

            if len(current) >= int(self.min_plaquettes) and current not in emitted:
                record = self._make_record(seed_plaquette_id=seed, plaquette_ids=current)
                if record is not None:
                    emitted.add(current)
                    yield record
                    yielded += 1
                    if self.max_regions is not None and yielded >= int(self.max_regions):
                        return

            if len(current) >= int(self.max_plaquettes):
                continue

            current_set = frozenset(int(pid) for pid in current)
            for plaquette_id in _adaptive_region_frontier(current_set, neighbor_map):
                child = tuple(sorted((*current_set, int(plaquette_id))))
                if child in queued:
                    continue
                queued.add(child)
                queue.append((seed, child))

    def iter_regions(self) -> Iterator[LocalQDMRegion]:
        """Yield only local regions from :meth:`iter_records`."""
        for record in self.iter_records():
            yield record.region

    def iter_searchers(self) -> Iterator[LocalCageSearcher]:
        """Yield ready-to-run local cage searchers for enumerated regions."""
        for record in self.iter_records():
            yield LocalCageSearcher(
                model=self.model,
                region=record.region,
                config=self.config,
                adapter=self.adapter,
            )

    def _make_record(
        self,
        *,
        seed_plaquette_id: int,
        plaquette_ids: Sequence[int],
    ) -> ConnectedRegionProposalRecord | None:
        selected = np.asarray(tuple(sorted({int(pid) for pid in plaquette_ids})), dtype=np.int64)
        if selected.size == 0 or selected.size > int(self.max_plaquettes):
            return None

        region = self.adapter.build_region_from_plaquettes(
            plaquette_ids=selected,
            config=self.config,
            scoring_plaquette_ids=selected,
        )
        if self.max_links is not None and region.link_ids.size > int(self.max_links):
            return None

        return ConnectedRegionProposalRecord(
            region=region,
            plaquette_ids=selected,
            seed_plaquette_id=int(seed_plaquette_id),
            size=int(selected.size),
            link_count=int(region.link_ids.size),
            unresolved_boundary_count=int(region.unresolved_boundary_plaquette_ids.size),
        )


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
class LocalQDMMultiPaddingConfig:
    """Configuration for Lego-style multi-block QDM padding.

    The multi-block path chooses compatible, disjoint local cage blocks from a
    pool, solves one shared static exterior for the union of their fixed
    boundary charges, and then certifies the resulting product state by applying
    all global QDM flips.  Every selected block must have support-independent
    site counts; otherwise an independent tensor-product block cannot be padded
    by one shared exterior configuration.
    """

    min_blocks: int = 2
    max_blocks: int | None = None
    max_paddings: int = 1
    max_padding_attempts: int | None = None
    max_paddings_per_packing: int = 1
    max_dfs_nodes: int | None = None
    include_sectors: bool = True
    require_static_exterior: bool = False
    tolerance: float = 1.0e-10
    max_product_support_size: int | None = 512
    require_kinetic_separation: bool = True
    sort_limited_basis: bool = True
    store_full_states: bool = True

    def __post_init__(self) -> None:
        if self.min_blocks < 1:
            raise ValueError("min_blocks must be positive.")
        if self.max_blocks is not None and self.max_blocks < self.min_blocks:
            raise ValueError("max_blocks must be None or at least min_blocks.")
        if self.max_paddings < 0:
            raise ValueError("max_paddings must be non-negative.")
        if self.max_padding_attempts is not None and self.max_padding_attempts < 0:
            raise ValueError("max_padding_attempts must be non-negative or None.")
        if self.max_paddings_per_packing < 0:
            raise ValueError("max_paddings_per_packing must be non-negative.")
        if self.max_dfs_nodes is not None and self.max_dfs_nodes < 0:
            raise ValueError("max_dfs_nodes must be non-negative or None.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if self.max_product_support_size is not None and self.max_product_support_size < 1:
            raise ValueError("max_product_support_size must be None or positive.")

    def as_single_padding_config(self) -> LocalQDMPaddingConfig:
        """Return the shared options in the single-block padding config form."""
        return LocalQDMPaddingConfig(
            max_paddings_per_record=self.max_paddings_per_packing,
            max_dfs_nodes=self.max_dfs_nodes,
            include_sectors=self.include_sectors,
            require_static_exterior=self.require_static_exterior,
            tolerance=self.tolerance,
            sort_limited_basis=self.sort_limited_basis,
            store_full_states=self.store_full_states,
        )


@dataclass(frozen=True, slots=True)
class RobustQDMLocalCageSearchConfig:
    """Budget-oriented configuration for robust local QDM cage discovery.

    This config intentionally exposes budgets and strategy choices rather than
    delicate geometry assumptions.  ``robust_qdm_local_cage_search`` uses a
    portfolio of region proposals, collects compatible local-cage blocks, then
    runs a schedule of permissive-to-strict multi-block padding configurations
    and lets global certification decide which candidates survive.
    """

    local_config: LocalQDMCageSearchConfig = field(
        default_factory=lambda: LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
        )
    )
    region_strategies: tuple[str, ...] = ("stripe", "snake_stripe", "connected", "adaptive")
    max_region_plaquettes: int = 6
    min_region_plaquettes: int = 1
    max_region_links: int | None = None
    max_regions_per_strategy: int | None = 128
    stripe_widths: tuple[int, ...] = (1, 2)
    stripe_directions: tuple[int, ...] | None = None
    snake_stripe_max_turns: int | None = None
    snake_stripe_allow_kind_changes: bool = False
    snake_stripe_plaquette_kinds: tuple[str, ...] | None = None
    snake_stripe_winding_vectors: tuple[tuple[int, ...], ...] | None = None
    adaptive_beam_width: int = 8
    adaptive_branch_factor: int = 8
    adaptive_seed_plaquette_ids: tuple[int, ...] | None = None
    adaptive_use_search_feedback: bool = False
    block_signatures: tuple[tuple[int, int], ...] | None = None
    max_records_per_region: int | None = 2
    max_blocks: int | None = 4
    min_blocks: int = 1
    max_product_support_size: int | None = 2048
    max_paddings_per_stage: int = 64
    max_padding_attempts_per_stage: int | None = None
    max_paddings_per_packing: int = 4
    max_dfs_nodes: int | None = None
    include_sectors: bool = True
    padding_stages: tuple[str, ...] = ("loose", "static", "strict")
    tolerance: float = 1.0e-9
    sort_limited_basis: bool = True
    store_full_states: bool = True
    skip_incompatible_blocks: bool = True

    def __post_init__(self) -> None:
        if self.max_region_plaquettes <= 0:
            raise ValueError("max_region_plaquettes must be positive.")
        if self.min_region_plaquettes <= 0:
            raise ValueError("min_region_plaquettes must be positive.")
        if self.min_region_plaquettes > self.max_region_plaquettes:
            raise ValueError("min_region_plaquettes cannot exceed max_region_plaquettes.")
        if self.max_region_links is not None and self.max_region_links <= 0:
            raise ValueError("max_region_links must be positive or None.")
        if self.max_regions_per_strategy is not None and self.max_regions_per_strategy < 0:
            raise ValueError("max_regions_per_strategy must be non-negative or None.")
        if not self.region_strategies:
            raise ValueError("region_strategies must be non-empty.")
        valid_strategies = {"stripe", "snake_stripe", "connected", "adaptive"}
        bad_strategies = [
            strategy for strategy in self.region_strategies if strategy not in valid_strategies
        ]
        if bad_strategies:
            raise ValueError(f"Unsupported region strategies: {bad_strategies}.")
        if not self.stripe_widths:
            raise ValueError("stripe_widths must be non-empty.")
        if any(int(width) <= 0 for width in self.stripe_widths):
            raise ValueError("stripe_widths must contain positive integers.")
        if self.snake_stripe_max_turns is not None and self.snake_stripe_max_turns < 0:
            raise ValueError("snake_stripe_max_turns must be non-negative or None.")
        if self.snake_stripe_plaquette_kinds is not None and not self.snake_stripe_plaquette_kinds:
            raise ValueError("snake_stripe_plaquette_kinds must be non-empty or None.")
        if self.snake_stripe_winding_vectors is not None and not self.snake_stripe_winding_vectors:
            raise ValueError("snake_stripe_winding_vectors must be non-empty or None.")
        if self.adaptive_beam_width <= 0:
            raise ValueError("adaptive_beam_width must be positive.")
        if self.adaptive_branch_factor <= 0:
            raise ValueError("adaptive_branch_factor must be positive.")
        if self.max_records_per_region is not None and self.max_records_per_region < 0:
            raise ValueError("max_records_per_region must be non-negative or None.")
        if self.max_blocks is not None and self.max_blocks < self.min_blocks:
            raise ValueError("max_blocks must be None or at least min_blocks.")
        if self.min_blocks < 1:
            raise ValueError("min_blocks must be positive.")
        if self.max_product_support_size is not None and self.max_product_support_size < 1:
            raise ValueError("max_product_support_size must be None or positive.")
        if self.max_paddings_per_stage < 0:
            raise ValueError("max_paddings_per_stage must be non-negative.")
        if (
            self.max_padding_attempts_per_stage is not None
            and self.max_padding_attempts_per_stage < 0
        ):
            raise ValueError("max_padding_attempts_per_stage must be non-negative or None.")
        if self.max_paddings_per_packing < 0:
            raise ValueError("max_paddings_per_packing must be non-negative.")
        if self.max_dfs_nodes is not None and self.max_dfs_nodes < 0:
            raise ValueError("max_dfs_nodes must be non-negative or None.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        valid_stages = {"base", "loose", "static", "strict"}
        bad_stages = [stage for stage in self.padding_stages if stage not in valid_stages]
        if bad_stages:
            raise ValueError(f"Unsupported padding stages: {bad_stages}.")

    def as_multi_padding_config(self) -> LocalQDMMultiPaddingConfig:
        """Return the base multi-padding budget used by the stage schedule."""
        return LocalQDMMultiPaddingConfig(
            min_blocks=self.min_blocks,
            max_blocks=self.max_blocks,
            max_paddings=self.max_paddings_per_stage,
            max_padding_attempts=self.max_padding_attempts_per_stage,
            max_paddings_per_packing=self.max_paddings_per_packing,
            max_dfs_nodes=self.max_dfs_nodes,
            include_sectors=self.include_sectors,
            require_static_exterior=False,
            tolerance=self.tolerance,
            max_product_support_size=self.max_product_support_size,
            require_kinetic_separation=False,
            sort_limited_basis=self.sort_limited_basis,
            store_full_states=self.store_full_states,
        )


@dataclass(frozen=True, slots=True)
class LocalQDMCageBlock:
    """A placed local QDM cage usable as one independent padding block."""

    block_id: int
    record: LocalQDMCageRecord
    link_ids: npt.NDArray[np.int64]
    active_plaquette_ids: npt.NDArray[np.int64]
    guard_plaquette_ids: npt.NDArray[np.int64]
    support_configs: npt.NDArray[np.int64]
    amplitudes: npt.NDArray[np.complex128]
    site_counts: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        link_ids = np.asarray(self.link_ids, dtype=np.int64)
        if link_ids.ndim != 1:
            raise ValueError("link_ids must be one-dimensional.")
        if np.unique(link_ids).size != link_ids.size:
            raise ValueError("link_ids must not contain duplicates.")
        object.__setattr__(self, "link_ids", link_ids.copy())

        for field_name in ("active_plaquette_ids", "guard_plaquette_ids"):
            arr = np.asarray(getattr(self, field_name), dtype=np.int64)
            if arr.ndim != 1:
                raise ValueError(f"{field_name} must be one-dimensional.")
            object.__setattr__(self, field_name, np.unique(arr).astype(np.int64))

        support_configs = np.asarray(self.support_configs, dtype=np.int64)
        if support_configs.ndim != 2:
            raise ValueError("support_configs must have shape (support, n_block_links).")
        if support_configs.shape[1] != np.asarray(self.link_ids).size:
            raise ValueError("support_configs width must match link_ids size.")
        object.__setattr__(self, "support_configs", support_configs.copy())

        amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        if amplitudes.ndim != 1 or amplitudes.size != support_configs.shape[0]:
            raise ValueError("amplitudes must have one entry per support configuration.")
        norm = float(np.linalg.norm(amplitudes))
        if norm == 0.0:
            raise ValueError("block amplitudes must have nonzero norm.")
        object.__setattr__(self, "amplitudes", (amplitudes / norm).astype(np.complex128))

        site_counts = np.asarray(self.site_counts, dtype=np.int64)
        if site_counts.ndim != 1:
            raise ValueError("site_counts must be one-dimensional.")
        if np.any(site_counts < 0):
            raise ValueError("site_counts must be non-negative.")
        object.__setattr__(self, "site_counts", site_counts.copy())

    @property
    def support_size(self) -> int:
        return int(self.support_configs.shape[0])

    @property
    def kappa(self) -> int:
        return int(self.record.kappa)

    @property
    def potential_value(self) -> int:
        return int(self.record.potential_value)

    @property
    def signature(self) -> tuple[int, int]:
        return self.record.signature


@dataclass(frozen=True, slots=True)
class MultiLocalQDMPadding:
    """One shared-exterior padding for a product of several local QDM blocks."""

    block_ids: tuple[int, ...]
    exterior_link_ids: npt.NDArray[np.int64]
    exterior_config: npt.NDArray[np.int64]
    global_support_configs: npt.NDArray[np.int64]
    global_amplitudes: npt.NDArray[np.complex128]
    block_support_indices: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        for field_name in ("exterior_link_ids", "exterior_config"):
            arr = np.asarray(getattr(self, field_name), dtype=np.int64)
            if arr.ndim != 1:
                raise ValueError(f"{field_name} must be one-dimensional.")
            object.__setattr__(self, field_name, arr.copy())

        configs = np.asarray(self.global_support_configs, dtype=np.int64)
        if configs.ndim != 2:
            raise ValueError("global_support_configs must be two-dimensional.")
        object.__setattr__(self, "global_support_configs", configs.copy())

        amplitudes = np.asarray(self.global_amplitudes, dtype=np.complex128)
        if amplitudes.ndim != 1 or amplitudes.size != configs.shape[0]:
            raise ValueError("global_amplitudes must have one entry per global support config.")
        norm = float(np.linalg.norm(amplitudes))
        if norm == 0.0 and amplitudes.size:
            raise ValueError("global_amplitudes must have nonzero norm.")
        if norm != 0.0:
            amplitudes = amplitudes / norm
        object.__setattr__(self, "global_amplitudes", amplitudes.astype(np.complex128))

        indices = np.asarray(self.block_support_indices, dtype=np.int64)
        if indices.ndim != 2:
            raise ValueError("block_support_indices must be two-dimensional.")
        if indices.shape[0] != configs.shape[0]:
            raise ValueError("block_support_indices must align with global support configs.")
        if indices.shape[1] != len(self.block_ids):
            raise ValueError("block_support_indices width must match block_ids length.")
        object.__setattr__(self, "block_support_indices", indices.copy())


@dataclass(frozen=True, slots=True)
class MultiLocalQDMCertificationReport:
    """Numerical certificate for one multi-block QDM padding."""

    block_ids: tuple[int, ...]
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
    padding: MultiLocalQDMPadding
    leakage_configs: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class QDMMultiPaddingFailureReport:
    """Reason one candidate multi-block padding failed certification."""

    block_ids: tuple[int, ...]
    padding_index: int
    reason: str
    padding: MultiLocalQDMPadding
    leakage_residual: float | None = None
    support_kinetic_residual: float | None = None
    support_hamiltonian_residual: float | None = None
    full_residual: float | None = None
    leakage_counts_by_class: dict[str, int] = field(default_factory=dict)
    leakage_norms_by_class: dict[str, float] = field(default_factory=dict)

    @property
    def dominant_leakage_class(self) -> str | None:
        """Return the plaquette class with the largest leakage norm, if known."""
        if not self.leakage_norms_by_class:
            return None
        return max(
            self.leakage_norms_by_class,
            key=lambda key: self.leakage_norms_by_class[key],
        )


@dataclass(frozen=True, slots=True)
class QDMMultiPaddingDiagnostics:
    """Certification diagnostics for a pool of multi-block padding candidates."""

    paddings: list[MultiLocalQDMPadding]
    reports: list[MultiLocalQDMCertificationReport]
    failures: list[QDMMultiPaddingFailureReport]
    config: LocalQDMMultiPaddingConfig
    padding_attempts: int | None = None
    first_certified_padding_index: int | None = None

    @property
    def n_paddings(self) -> int:
        return len(self.paddings)

    @property
    def n_padding_attempts(self) -> int:
        if self.padding_attempts is None:
            return len(self.paddings)
        return int(self.padding_attempts)

    @property
    def first_certified_attempt_index(self) -> int | None:
        return self.first_certified_padding_index

    @property
    def n_certified(self) -> int:
        return len(self.reports)

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def counts_by_failure_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in self.failures:
            counts[failure.reason] = counts.get(failure.reason, 0) + 1
        return counts

    @property
    def leakage_failure_counts_by_class(self) -> dict[str, int]:
        """Count leakage failures by their dominant plaquette class."""
        counts: dict[str, int] = {}
        for failure in self.failures:
            if failure.reason != "leakage_residual":
                continue
            leakage_class = failure.dominant_leakage_class or "unknown"
            counts[leakage_class] = counts.get(leakage_class, 0) + 1
        return counts

    @property
    def leakage_failure_norms_by_class(self) -> dict[str, float]:
        """Sum leakage norms by plaquette class over all leakage failures."""
        norms: dict[str, float] = {}
        for failure in self.failures:
            if failure.reason != "leakage_residual":
                continue
            for leakage_class, norm in failure.leakage_norms_by_class.items():
                norms[leakage_class] = norms.get(leakage_class, 0.0) + float(norm)
        return norms


@dataclass(frozen=True, slots=True)
class RobustQDMLocalCageSearchContext:
    """Debug context for :func:`robust_qdm_local_cage_search`.

    The ordinary robust search returns a ``CertifiedLocalQDMCageSearchResult`` so
    downstream tools can consume it directly.  When ``return_context=True``,
    this companion object exposes the intermediate proposal scan, block pool,
    and per-padding-stage diagnostics that explain where candidates were found
    or rejected.
    """

    config: RobustQDMLocalCageSearchConfig
    scan: LocalRegionProposalSearchResult
    blocks: list[LocalQDMCageBlock]
    padding_config: LocalQDMMultiPaddingConfig
    diagnostics_by_stage: dict[str, QDMMultiPaddingDiagnostics]

    @property
    def n_regions(self) -> int:
        return len(self.scan)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(self.diagnostics_by_stage)

    @property
    def n_paddings_by_stage(self) -> dict[str, int]:
        return {
            stage: diagnostics.n_paddings
            for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def n_padding_attempts_by_stage(self) -> dict[str, int]:
        return {
            stage: diagnostics.n_padding_attempts
            for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def n_certified_by_stage(self) -> dict[str, int]:
        return {
            stage: diagnostics.n_certified
            for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def first_certified_attempt_by_stage(self) -> dict[str, int | None]:
        return {
            stage: diagnostics.first_certified_attempt_index
            for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def failure_counts_by_stage(self) -> dict[str, dict[str, int]]:
        return {
            stage: diagnostics.counts_by_failure_reason
            for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def leakage_failure_counts_by_stage(self) -> dict[str, dict[str, int]]:
        return {
            stage: diagnostics.leakage_failure_counts_by_class
            for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def leakage_failure_norms_by_stage(self) -> dict[str, dict[str, float]]:
        return {
            stage: diagnostics.leakage_failure_norms_by_class
            for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def reports_by_stage(self) -> dict[str, list[MultiLocalQDMCertificationReport]]:
        return {
            stage: diagnostics.reports for stage, diagnostics in self.diagnostics_by_stage.items()
        }

    @property
    def reports(self) -> list[MultiLocalQDMCertificationReport]:
        return [
            report
            for diagnostics in self.diagnostics_by_stage.values()
            for report in diagnostics.reports
        ]


@dataclass(frozen=True, slots=True)
class _QDMGlobalPlaquetteAction:
    """Cached data needed to test/apply one global QDM plaquette flip."""

    plaquette_id: int
    links: npt.NDArray[np.int64]
    pattern0: npt.NDArray[np.int64]
    pattern1: npt.NDArray[np.int64]
    forward: complex
    backward: complex
    potential: complex


@dataclass(frozen=True, slots=True)
class _QDMExteriorStaticPlaquette:
    """Exterior-only plaquette represented in exterior-link coordinates."""

    plaquette_id: int
    exterior_indices: npt.NDArray[np.int64]
    pattern0: npt.NDArray[np.int64]
    pattern1: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _QDMExteriorFlippabilityPreference:
    """Possible flippable plaquette pattern in exterior-link coordinates."""

    plaquette_id: int
    plaquette_class: str
    exterior_indices: npt.NDArray[np.int64]
    dangerous_patterns: tuple[npt.NDArray[np.int64], ...]
    weight: int


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
    reports: list[LocalQDMCertificationReport | MultiLocalQDMCertificationReport]
    padding_config: LocalQDMPaddingConfig | LocalQDMMultiPaddingConfig

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
    adapter: LocalCageModelAdapter | None = None
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
                "This local cage result does not carry a model reference. "
                "Use LocalCageSearcher.run() from the current API or call the "
                "model-specific certification helper directly."
            )
        adapter = self.adapter or local_cage_adapter_for_model(self.model)
        return adapter.certify_result(self, config=config)


class LocalCageModelAdapter(Protocol):
    """Model-specific local variable interface used by :class:`LocalCageSearcher`.

    The generic local searcher owns the caging algebra.  The adapter owns the
    model/lattice details: how to build a local region, enumerate compatible
    local configurations, construct local kinetic transitions, and compute the
    local diagonal/self-loop values.  New models should add an adapter rather
    than adding branches to ``LocalCageSearcher``.
    """

    model: object
    source_label: str

    def normalize_config(
        self,
        config: LocalQDMCageSearchConfig,
    ) -> LocalQDMCageSearchConfig:
        """Return a model-normalized search config."""
        ...

    def build_region_from_plaquettes(
        self,
        *,
        plaquette_ids: Sequence[int] | npt.ArrayLike,
        config: LocalQDMCageSearchConfig,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    ) -> LocalQDMRegion:
        """Build a local region from seed plaquettes/local kinetic terms."""
        ...

    def build_region_from_links(
        self,
        *,
        link_ids: Sequence[int] | npt.ArrayLike,
        config: LocalQDMCageSearchConfig,
        active_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
    ) -> LocalQDMRegion:
        """Build a local region from explicit local variables."""
        ...

    def full_model_region(
        self,
        *,
        config: LocalQDMCageSearchConfig,
    ) -> LocalQDMRegion:
        """Build the full-model region for exact-regression mode."""
        ...

    def enumerate_local_basis(
        self,
        region: LocalQDMRegion,
        config: LocalQDMCageSearchConfig,
    ) -> Basis:
        """Enumerate locally valid configurations for ``region``."""
        ...

    def build_local_kinetic_matrix(
        self,
        region: LocalQDMRegion,
        local_basis: Basis,
    ) -> scipy_sparse.csr_array:
        """Build the local kinetic matrix on ``local_basis``."""
        ...

    def local_self_loop_values(
        self,
        region: LocalQDMRegion,
        local_basis: Basis,
    ) -> npt.NDArray[np.complex128]:
        """Compute local diagonal/self-loop values."""
        ...

    def make_local_record(
        self,
        *,
        cage_state: CageState,
        signature: tuple[int, int],
        candidate: CandidateSubgraph,
        local_basis: Basis,
        region: LocalQDMRegion,
    ) -> LocalQDMCageRecord:
        """Wrap one solved local cage state in a model-specific record."""
        ...

    def certify_result(
        self,
        local_result: LocalQDMCageSearchResult,
        *,
        config: LocalQDMPaddingConfig | None = None,
    ) -> CertifiedLocalQDMCageSearchResult:
        """Pad/certify local records for this model, when available."""
        ...


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

    def certify_result(
        self,
        local_result: LocalQDMCageSearchResult,
        *,
        config: LocalQDMPaddingConfig | None = None,
    ) -> CertifiedLocalQDMCageSearchResult:
        return certify_qdm_local_result(self.model, local_result, config=config)


LocalCageAdapterFactory = Callable[[object], LocalCageModelAdapter | None]
_LOCAL_CAGE_ADAPTER_FACTORIES: list[LocalCageAdapterFactory] = []


def register_local_cage_adapter_factory(
    factory: LocalCageAdapterFactory,
    *,
    prepend: bool = False,
) -> None:
    """Register a factory that can adapt models for ``LocalCageSearcher``.

    Factories receive a model and return either a ``LocalCageModelAdapter`` or
    ``None`` when they do not support that model.  The built-in QDM factory is
    registered by default; future model families can register their adapters
    without branching inside the solver core.
    """
    if prepend:
        _LOCAL_CAGE_ADAPTER_FACTORIES.insert(0, factory)
    else:
        _LOCAL_CAGE_ADAPTER_FACTORIES.append(factory)


def local_cage_adapter_for_model(
    model: object,
    adapter: LocalCageModelAdapter | None = None,
) -> LocalCageModelAdapter:
    """Return a local-search adapter for ``model``.

    Passing ``adapter`` is the explicit, model-generic path.  Without an
    explicit adapter, the registered factories are tried in order.
    """
    if adapter is not None:
        return adapter
    for factory in _LOCAL_CAGE_ADAPTER_FACTORIES:
        candidate = factory(model)
        if candidate is not None:
            return candidate
    raise ValueError(
        "No LocalCageModelAdapter is registered for this model. "
        "Pass adapter=... explicitly or register a factory with "
        "register_local_cage_adapter_factory(...)."
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


register_local_cage_adapter_factory(_qdm_local_cage_adapter_factory)


@dataclass
class LocalCageSearcher:
    """Local-first type-1 cage searcher over a model adapter.

    The searcher owns only the generic caging algebra: build a local kinetic
    graph, find bipartite/uniform-self-loop type-1 candidates, and solve the
    fixed-kappa cage problem.  The adapter owns all model/lattice details such
    as local variable ids, constraints, local kinetic moves, and padding.
    """

    model: object
    region: LocalQDMRegion
    config: LocalQDMCageSearchConfig = field(default_factory=LocalQDMCageSearchConfig)
    adapter: LocalCageModelAdapter | None = None

    def __post_init__(self) -> None:
        self.adapter = local_cage_adapter_for_model(self.model, self.adapter)
        self.config = self.adapter.normalize_config(self.config)

    @classmethod
    def from_plaquettes(
        cls,
        model: object,
        plaquette_ids: Sequence[int] | npt.ArrayLike,
        *,
        config: LocalQDMCageSearchConfig | None = None,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
        adapter: LocalCageModelAdapter | None = None,
    ) -> LocalCageSearcher:
        """Construct a local searcher from seed plaquettes/local kinetic terms."""
        adapter = local_cage_adapter_for_model(model, adapter)
        search_config = LocalQDMCageSearchConfig() if config is None else config
        search_config = adapter.normalize_config(search_config)
        region = adapter.build_region_from_plaquettes(
            plaquette_ids=plaquette_ids,
            config=search_config,
            scoring_plaquette_ids=scoring_plaquette_ids,
        )
        return cls(model=model, region=region, config=search_config, adapter=adapter)

    @classmethod
    def from_links(
        cls,
        model: object,
        link_ids: Sequence[int] | npt.ArrayLike,
        *,
        config: LocalQDMCageSearchConfig | None = None,
        active_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
        scoring_plaquette_ids: Sequence[int] | npt.ArrayLike | None = None,
        adapter: LocalCageModelAdapter | None = None,
    ) -> LocalCageSearcher:
        """Construct a local searcher from explicit local variable ids."""
        adapter = local_cage_adapter_for_model(model, adapter)
        search_config = LocalQDMCageSearchConfig() if config is None else config
        search_config = adapter.normalize_config(search_config)
        region = adapter.build_region_from_links(
            link_ids=link_ids,
            config=search_config,
            active_plaquette_ids=active_plaquette_ids,
            scoring_plaquette_ids=scoring_plaquette_ids,
        )
        return cls(model=model, region=region, config=search_config, adapter=adapter)

    @classmethod
    def full_model_region(
        cls,
        model: object,
        *,
        config: LocalQDMCageSearchConfig | None = None,
        adapter: LocalCageModelAdapter | None = None,
    ) -> LocalCageSearcher:
        """Construct a local searcher whose region is the full model.

        This is mostly useful as a regression bridge: the implementation path is
        still local-first/no-full-Hamiltonian, but the local region happens to
        contain every variable and local kinetic term.
        """
        adapter = local_cage_adapter_for_model(model, adapter)
        search_config = LocalQDMCageSearchConfig() if config is None else config
        search_config = adapter.normalize_config(search_config)
        region = adapter.full_model_region(config=search_config)
        return cls(model=model, region=region, config=search_config, adapter=adapter)

    def run(self) -> LocalQDMCageSearchResult:
        """Run the local type-1 cage search."""
        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        local_basis = adapter.enumerate_local_basis(self.region, self.config)
        kinetic_matrix = adapter.build_local_kinetic_matrix(self.region, local_basis)
        self_loop_values = adapter.local_self_loop_values(self.region, local_basis)

        if local_basis.n_states == 0:
            return LocalQDMCageSearchResult(
                records=[],
                region=self.region,
                local_basis=local_basis,
                kinetic_matrix=kinetic_matrix,
                self_loop_values=self_loop_values,
                config=self.config,
                model=self.model,
                adapter=adapter,
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
            adapter=adapter,
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
            degenerate_basis_strategy=self.config.degenerate_basis_strategy,
            ipr_n_restarts=self.config.ipr_n_restarts,
            ipr_max_iter=self.config.ipr_max_iter,
            ipr_step_size=self.config.ipr_step_size,
            ipr_candidate_count=self.config.ipr_candidate_count,
            ipr_rank_completion_patience=self.config.ipr_rank_completion_patience,
            ipr_batch_size=self.config.ipr_batch_size,
            ipr_random_seed=self.config.ipr_random_seed,
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

                adapter = local_cage_adapter_for_model(self.model, self.adapter)
                records.append(
                    adapter.make_local_record(
                        cage_state=cage_state,
                        signature=signature,
                        candidate=candidate,
                        local_basis=local_basis,
                        region=self.region,
                    )
                )

        if self.config.deduplicate_by_rank:
            records = _deduplicate_local_records(
                records,
                hilbert_size=local_basis.n_states,
                tolerance=self.config.rank_tolerance_factor * self.config.tolerance,
            )

        return records


class LocalQDMCageSearcher(LocalCageSearcher):
    """Backward-compatible QDM name for :class:`LocalCageSearcher`.

    New code should prefer ``LocalCageSearcher``.  The old name remains as a
    thin subclass so existing notebooks/tests keep working while the core
    solver is routed through the model-adapter interface.
    """


# Generic public names.  The current implementation still returns the QDM
# concrete record/region/result classes, but the solver consumes them through
# ``LocalCageModelAdapter`` rather than by branching on a model type.
LocalCageSearchConfig = LocalQDMCageSearchConfig
LocalCageRegion = LocalQDMRegion
LocalCageRecord = LocalQDMCageRecord
LocalCageSearchResult = LocalQDMCageSearchResult
CertifiedLocalCageSearchResult = CertifiedLocalQDMCageSearchResult


@dataclass(frozen=True, slots=True)
class LocalRegionProposalSearchRecord:
    """Result for one local region emitted by a proposal."""

    proposal_index: int
    region_index: int
    region: LocalQDMRegion
    result: LocalQDMCageSearchResult
    proposal_record: object | None = None

    @property
    def records(self) -> list[LocalQDMCageRecord]:
        return self.result.records

    @property
    def local_hilbert_size(self) -> int:
        return self.result.local_hilbert_size

    @property
    def counts_by_signature(self) -> dict[tuple[int, int], int]:
        return self.result.counts_by_signature


@dataclass(frozen=True, slots=True)
class LocalRegionProposalSearchResult:
    """Container returned by proposal-driven local cage scans."""

    records: list[LocalRegionProposalSearchRecord]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @property
    def local_results(self) -> list[LocalQDMCageSearchResult]:
        return [record.result for record in self.records]

    @property
    def cage_records(self) -> list[LocalQDMCageRecord]:
        return [cage_record for record in self.records for cage_record in record.result.records]

    @property
    def counts_by_signature(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for cage_record in self.cage_records:
            counts[cage_record.signature] = counts.get(cage_record.signature, 0) + 1
        return counts

    @property
    def nonempty_records(self) -> list[LocalRegionProposalSearchRecord]:
        return [record for record in self.records if len(record.result) > 0]

    def qdm_cage_blocks(
        self,
        model: object | None = None,
        *,
        block_id_start: int = 0,
        signatures: Sequence[tuple[int, int]] | None = None,
        max_records_per_region: int | None = None,
        max_blocks: int | None = None,
        skip_incompatible_blocks: bool = True,
    ) -> list[LocalQDMCageBlock]:
        """Convert compatible local QDM records from this scan into blocks.

        Records whose boundary site contribution changes across support
        configurations are not independent product blocks.  They are skipped by
        default because such records may still be valid local cages, just not
        valid Lego blocks for the current independent multi-padding ansatz.
        """
        if block_id_start < 0:
            raise ValueError("block_id_start must be non-negative.")
        if max_records_per_region is not None and max_records_per_region < 0:
            raise ValueError("max_records_per_region must be non-negative or None.")
        if max_blocks is not None and max_blocks < 0:
            raise ValueError("max_blocks must be non-negative or None.")

        signature_filter = None
        if signatures is not None:
            signature_filter = {(int(kappa), int(potential)) for kappa, potential in signatures}

        blocks: list[LocalQDMCageBlock] = []
        next_block_id = int(block_id_start)
        for proposal_record in self.records:
            block_model = model if model is not None else proposal_record.result.model
            if block_model is None:
                raise ValueError("A model is required to convert proposal records into QDM blocks.")

            region_records = proposal_record.result.records
            if signature_filter is not None:
                region_records = [
                    record for record in region_records if record.signature in signature_filter
                ]
            if max_records_per_region is not None:
                region_records = region_records[:max_records_per_region]

            for local_record in region_records:
                if max_blocks is not None and len(blocks) >= max_blocks:
                    return blocks
                try:
                    block = make_qdm_cage_block(
                        block_model,
                        local_record,
                        block_id=next_block_id,
                    )
                except ValueError:
                    if skip_incompatible_blocks:
                        continue
                    raise
                blocks.append(block)
                next_block_id += 1

        return blocks


def run_local_region_proposal(
    proposal: LocalRegionProposal,
    *,
    model: object | None = None,
    config: LocalQDMCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    max_regions: int | None = None,
) -> LocalRegionProposalSearchResult:
    """Run the local cage searcher over every region emitted by one proposal."""
    return run_local_region_proposals(
        [proposal],
        model=model,
        config=config,
        adapter=adapter,
        max_regions=max_regions,
    )


def run_local_region_proposals(
    proposals: Sequence[LocalRegionProposal],
    *,
    model: object | None = None,
    config: LocalQDMCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    max_regions: int | None = None,
) -> LocalRegionProposalSearchResult:
    """Run local cage searches over a stream of proposal-generated regions.

    The helper is intentionally lightweight: proposal objects only need to
    provide ``iter_regions()``.  If they provide richer ``iter_records()``
    records with a ``region`` attribute, that metadata is retained in the scan
    result.  ``StripeRegionProposal`` follows this richer path.
    """
    if max_regions is not None and max_regions < 0:
        raise ValueError("max_regions must be non-negative or None.")

    search_records: list[LocalRegionProposalSearchRecord] = []
    emitted = 0
    for proposal_index, proposal in enumerate(proposals):
        proposal_model = _model_for_region_proposal(proposal, model)
        proposal_adapter = _adapter_for_region_proposal(proposal, adapter)
        proposal_config = _config_for_region_proposal(proposal, config)

        for region_index, proposal_record, region in _iter_region_proposal_records(proposal):
            if max_regions is not None and emitted >= max_regions:
                return LocalRegionProposalSearchResult(records=search_records)
            result = LocalCageSearcher(
                model=proposal_model,
                region=region,
                config=proposal_config,
                adapter=proposal_adapter,
            ).run()
            search_records.append(
                LocalRegionProposalSearchRecord(
                    proposal_index=proposal_index,
                    region_index=region_index,
                    region=region,
                    result=result,
                    proposal_record=proposal_record,
                )
            )
            emitted += 1

    return LocalRegionProposalSearchResult(records=search_records)


def collect_qdm_cage_blocks_from_region_proposals(
    proposals: Sequence[LocalRegionProposal],
    *,
    model: object | None = None,
    config: LocalQDMCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    signatures: Sequence[tuple[int, int]] | None = None,
    max_regions: int | None = None,
    max_records_per_region: int | None = None,
    max_blocks: int | None = None,
    block_id_start: int = 0,
    skip_incompatible_blocks: bool = True,
) -> list[LocalQDMCageBlock]:
    """Run proposal searches and return a QDM block pool for multi-padding."""
    scan = run_local_region_proposals(
        proposals,
        model=model,
        config=config,
        adapter=adapter,
        max_regions=max_regions,
    )
    return scan.qdm_cage_blocks(
        model=model,
        block_id_start=block_id_start,
        signatures=signatures,
        max_records_per_region=max_records_per_region,
        max_blocks=max_blocks,
        skip_incompatible_blocks=skip_incompatible_blocks,
    )


collect_qdm_cage_blocks_from_proposals = collect_qdm_cage_blocks_from_region_proposals


def _model_for_region_proposal(
    proposal: LocalRegionProposal,
    model: object | None,
) -> object:
    if model is not None:
        return model
    proposal_model = getattr(proposal, "model", None)
    if proposal_model is None:
        raise ValueError("model must be provided when proposal has no model attribute.")
    return proposal_model


def _adapter_for_region_proposal(
    proposal: LocalRegionProposal,
    adapter: LocalCageModelAdapter | None,
) -> LocalCageModelAdapter | None:
    if adapter is not None:
        return adapter
    return getattr(proposal, "adapter", None)


def _config_for_region_proposal(
    proposal: LocalRegionProposal,
    config: LocalQDMCageSearchConfig | None,
) -> LocalQDMCageSearchConfig:
    if config is not None:
        return config
    proposal_config = getattr(proposal, "config", None)
    if proposal_config is None:
        return LocalQDMCageSearchConfig()
    return proposal_config


def _iter_region_proposal_records(
    proposal: LocalRegionProposal,
) -> Iterator[tuple[int, object | None, LocalQDMRegion]]:
    if hasattr(proposal, "iter_records"):
        for region_index, proposal_record in enumerate(proposal.iter_records()):
            region = getattr(proposal_record, "region", None)
            if region is None:
                raise ValueError("proposal iter_records() entries must carry a region attribute.")
            yield region_index, proposal_record, region
        return

    for region_index, region in enumerate(proposal.iter_regions()):
        yield region_index, None, region


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


@dataclass(frozen=True, slots=True)
class _LocalQDMCountConstraint:
    """Local dimer-count rule for a site in a local QDM region.

    ``min_count=None`` means only the upper bound is enforced.  This is used at
    open local-region boundary sites, where exterior links may later complete
    the dimer covering.  Closed sites use ``min_count=max_count=required_count``.
    """

    layout: VariableLayout
    site_id: int
    variable_indices: npt.NDArray[np.int64]
    min_count: int | None
    max_count: int
    name: str = "local_qdm_site_count"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")
        if variable_indices.size and (
            np.any(variable_indices < 0) or np.any(variable_indices >= self.layout.n_variables)
        ):
            raise ValueError("variable_indices contains indices outside the local layout.")
        if self.min_count is not None and self.min_count < 0:
            raise ValueError("min_count must be non-negative or None.")
        if self.max_count < 0:
            raise ValueError("max_count must be non-negative.")
        if self.min_count is not None and self.min_count > self.max_count:
            raise ValueError("min_count cannot exceed max_count.")
        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "site_id", int(self.site_id))
        object.__setattr__(self, "max_count", int(self.max_count))
        if self.min_count is not None:
            object.__setattr__(self, "min_count", int(self.min_count))

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = np.asarray(config, dtype=np.int64)
        if arr.shape != self.layout.shape:
            raise ValueError(f"Expected config shape {self.layout.shape}, got {arr.shape}.")
        return int(np.sum(arr[self.variable_indices]))

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        occupied = self.value(config)
        satisfied = occupied <= self.max_count and (
            self.min_count is None or occupied >= self.min_count
        )
        if self.min_count is None:
            rule = f"count<={self.max_count}"
        elif self.min_count == self.max_count:
            rule = f"count={self.min_count}"
        else:
            rule = f"{self.min_count}<=count<={self.max_count}"
        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=occupied,
            message=f"{self.name}(site={self.site_id}): count={occupied}, rule={rule}",
        )

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return self.check(config).satisfied

    def partial_check(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> bool:
        return self.propagate(config, assigned_mask).consistent

    def propagate(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> ConstraintPropagation:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)
        variable_indices = self.variable_indices

        assigned_local = assigned[variable_indices]
        unassigned_variables = variable_indices[~assigned_local]
        occupied = int(np.sum(arr[variable_indices[assigned_local]]))
        unassigned = int(unassigned_variables.size)

        if occupied > self.max_count:
            return ConstraintPropagation.contradiction()

        if self.min_count is not None and occupied + unassigned < self.min_count:
            return ConstraintPropagation.contradiction()

        if unassigned == 0:
            if self.min_count is not None and occupied < self.min_count:
                return ConstraintPropagation.contradiction()
            return ConstraintPropagation.no_change()

        forced: list[tuple[int, int]] = []

        if occupied == self.max_count:
            forced.extend((int(variable_index), 0) for variable_index in unassigned_variables)

        if self.min_count is not None and occupied + unassigned == self.min_count:
            forced.extend((int(variable_index), 1) for variable_index in unassigned_variables)

        if not forced:
            return ConstraintPropagation.no_change()

        forced_by_variable: dict[int, int] = {}
        for variable_index, value in forced:
            previous = forced_by_variable.get(variable_index)
            if previous is not None and previous != value:
                return ConstraintPropagation.contradiction()
            forced_by_variable[variable_index] = value

        return ConstraintPropagation(forced_assignments=tuple(sorted(forced_by_variable.items())))


@dataclass(slots=True)
class _LocalQDMActivePlaquetteObserver:
    """Incremental DFS observer for locally kinetic-relevant QDM states.

    A local QDM configuration with no flippable active plaquette is isolated in
    the local kinetic graph.  For local cage searches with nontrivial component
    size requirements, those states can be filtered before the local kinetic
    matrix is built.

    The observer maintains per-plaquette incompatibility counters for the two
    alternating patterns.  Therefore ``can_continue`` is O(1) rather than
    rescanning every active plaquette after each DFS assignment.
    """

    plaquette_variable_indices: tuple[npt.NDArray[np.int64], ...]
    plaquette_patterns: tuple[
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
        ...,
    ]
    name: str = "local_qdm_active_plaquette_viability"
    variable_to_plaquette_entries: tuple[tuple[tuple[int, int], ...], ...] = field(
        init=False,
        repr=False,
    )
    conflict_counts: npt.NDArray[np.int64] = field(init=False, repr=False)
    viable_plaquette_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if len(self.plaquette_variable_indices) != len(self.plaquette_patterns):
            raise ValueError("plaquette_variable_indices and plaquette_patterns must align.")

        max_variable = -1
        entries_by_variable: dict[int, list[tuple[int, int]]] = defaultdict(list)

        for plaquette_index, (variable_indices, (pattern0, pattern1)) in enumerate(
            zip(
                self.plaquette_variable_indices,
                self.plaquette_patterns,
                strict=True,
            )
        ):
            variable_indices = np.asarray(variable_indices, dtype=np.int64)
            pattern0 = np.asarray(pattern0, dtype=np.int64)
            pattern1 = np.asarray(pattern1, dtype=np.int64)

            if variable_indices.ndim != 1:
                raise ValueError("Each plaquette variable-index array must be one-dimensional.")
            if pattern0.shape != variable_indices.shape or pattern1.shape != variable_indices.shape:
                raise ValueError("Each active-plaquette pattern must match its variable support.")

            for local_position, variable_index in enumerate(variable_indices):
                variable_index = int(variable_index)
                if variable_index < 0:
                    raise ValueError("Local variable indices must be non-negative.")
                max_variable = max(max_variable, variable_index)
                entries_by_variable[variable_index].append(
                    (int(plaquette_index), int(local_position))
                )

        variable_to_plaquette_entries: list[tuple[tuple[int, int], ...]] = []
        for variable_index in range(max_variable + 1):
            variable_to_plaquette_entries.append(tuple(entries_by_variable.get(variable_index, ())))

        self.variable_to_plaquette_entries = tuple(variable_to_plaquette_entries)
        self.conflict_counts = np.zeros((len(self.plaquette_variable_indices), 2), dtype=np.int64)
        self.viable_plaquette_count = int(len(self.plaquette_variable_indices))

    def reset(
        self,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
    ) -> None:
        self.conflict_counts.fill(0)
        self.viable_plaquette_count = int(len(self.plaquette_variable_indices))

        assigned_variables = np.flatnonzero(np.asarray(assigned_mask, dtype=bool))
        for variable_index in assigned_variables:
            self._update_variable_assignment(
                int(variable_index),
                int(config[int(variable_index)]),
                delta=1,
            )

    def on_assignments(
        self,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        changed_variables: Sequence[int],
    ) -> None:
        del assigned_mask
        for variable_index in changed_variables:
            self._update_variable_assignment(
                int(variable_index),
                int(config[int(variable_index)]),
                delta=1,
            )

    def on_unassignments(
        self,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        changed_variables: Sequence[int],
    ) -> None:
        del assigned_mask
        for variable_index in changed_variables:
            self._update_variable_assignment(
                int(variable_index),
                int(config[int(variable_index)]),
                delta=-1,
            )

    def can_continue(
        self,
        config: npt.NDArray[np.int64],
        assigned_mask: npt.NDArray[np.bool_],
        changed_variables: Sequence[int],
    ) -> bool:
        del config, assigned_mask, changed_variables
        return self.viable_plaquette_count > 0

    def accept_solution(
        self,
        config: npt.NDArray[np.int64],
    ) -> bool:
        del config
        return self.viable_plaquette_count > 0

    def _update_variable_assignment(self, variable_index: int, value: int, *, delta: int) -> None:
        if not self.plaquette_variable_indices:
            return
        if variable_index >= len(self.variable_to_plaquette_entries):
            return

        for plaquette_index, local_position in self.variable_to_plaquette_entries[variable_index]:
            pattern0, pattern1 = self.plaquette_patterns[plaquette_index]
            was_viable = self._plaquette_is_viable(plaquette_index)

            if int(value) != int(pattern0[local_position]):
                self.conflict_counts[plaquette_index, 0] += int(delta)
            if int(value) != int(pattern1[local_position]):
                self.conflict_counts[plaquette_index, 1] += int(delta)

            if np.any(self.conflict_counts[plaquette_index] < 0):
                raise RuntimeError("Active-plaquette observer received an unbalanced undo.")

            is_viable = self._plaquette_is_viable(plaquette_index)
            if was_viable and not is_viable:
                self.viable_plaquette_count -= 1
            elif not was_viable and is_viable:
                self.viable_plaquette_count += 1

    def _plaquette_is_viable(self, plaquette_index: int) -> bool:
        return bool(
            self.conflict_counts[int(plaquette_index), 0] == 0
            or self.conflict_counts[int(plaquette_index), 1] == 0
        )


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
    if prune_inactive_states and not full_link_region:
        observer = _qdm_active_plaquette_observer(model, region)
        observers = () if observer is None else (observer,)

    return DFSBasisSolver(sort=sort).solve(
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


# Backward-compatible/readable alias matching the wording used in design notes.
find_qdm_multi_block_paddings = find_multi_qdm_block_paddings


def certify_qdm_multi_block_padding(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: MultiLocalQDMPadding,
    *,
    padding_index: int = 0,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> MultiLocalQDMCertificationReport | None:
    """Certify one multi-block QDM padding by explicit global one-hop action."""
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    return _certify_qdm_multi_padding(
        model,
        tuple(blocks),
        padding,
        padding_index=padding_index,
        config=multi_config,
    )


def _qdm_multi_padding_attempt_limit(config: LocalQDMMultiPaddingConfig) -> int | None:
    """Return the raw-padding attempt cap used by certification loops.

    ``max_paddings`` caps certified successes.  ``max_padding_attempts`` is the
    only raw-attempt cap; ``None`` means the finite padding iterator is allowed
    to run until enough certified reports are found or the search space is
    exhausted.
    """
    if config.max_padding_attempts is None:
        return None
    return int(config.max_padding_attempts)


def certify_qdm_multi_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> list[MultiLocalQDMCertificationReport]:
    """Find and certify Lego-style multi-block QDM paddings from a block pool.

    Candidate padding generation is interleaved with certification.  The search
    stops after ``config.max_paddings`` certified reports or after
    ``config.max_padding_attempts`` raw padding attempts.  If
    ``max_padding_attempts`` is ``None``, there is no separate raw-attempt cap.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    if multi_config.max_paddings == 0:
        return []

    block_by_id = {int(block.block_id): block for block in block_pool}
    reports: list[MultiLocalQDMCertificationReport] = []
    for padding_index, padding in enumerate(
        iter_multi_qdm_block_paddings(
            model,
            block_pool,
            config=multi_config,
            max_yielded=_qdm_multi_padding_attempt_limit(multi_config),
        )
    ):
        blocks = tuple(block_by_id[int(block_id)] for block_id in padding.block_ids)
        report = _certify_qdm_multi_padding(
            model,
            blocks,
            padding,
            padding_index=padding_index,
            config=multi_config,
        )
        if report is not None:
            reports.append(report)
            if len(reports) >= multi_config.max_paddings:
                break
    return reports


def diagnose_qdm_multi_block_paddings(
    model: object,
    block_pool: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> QDMMultiPaddingDiagnostics:
    """Find multi-block paddings and report both successes and failures.

    This diagnostic path uses the same interleaved padding/certification loop as
    :func:`certify_qdm_multi_block_paddings`.  ``paddings`` stores the raw
    candidates actually attempted, while ``n_padding_attempts`` records that
    count explicitly for notebook/debug summaries.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    block_by_id = {int(block.block_id): block for block in block_pool}

    paddings: list[MultiLocalQDMPadding] = []
    reports: list[MultiLocalQDMCertificationReport] = []
    failures: list[QDMMultiPaddingFailureReport] = []
    first_certified_padding_index: int | None = None

    if multi_config.max_paddings == 0:
        return QDMMultiPaddingDiagnostics(
            paddings=[],
            reports=[],
            failures=[],
            config=multi_config,
            padding_attempts=0,
            first_certified_padding_index=None,
        )

    for padding_index, padding in enumerate(
        iter_multi_qdm_block_paddings(
            model,
            block_pool,
            config=multi_config,
            max_yielded=_qdm_multi_padding_attempt_limit(multi_config),
        )
    ):
        paddings.append(padding)
        blocks = tuple(block_by_id[int(block_id)] for block_id in padding.block_ids)
        report = _certify_qdm_multi_padding(
            model,
            blocks,
            padding,
            padding_index=padding_index,
            config=multi_config,
        )
        if report is not None:
            reports.append(report)
            if first_certified_padding_index is None:
                first_certified_padding_index = int(padding_index)
            if len(reports) >= multi_config.max_paddings:
                break
            continue
        failures.append(
            _qdm_multi_padding_failure_report(
                model,
                blocks,
                padding,
                padding_index=padding_index,
                config=multi_config,
            )
        )

    return QDMMultiPaddingDiagnostics(
        paddings=paddings,
        reports=reports,
        failures=failures,
        config=multi_config,
        padding_attempts=len(paddings),
        first_certified_padding_index=first_certified_padding_index,
    )


def qdm_multi_padding_config_schedule(
    config: LocalQDMMultiPaddingConfig | None = None,
    *,
    stages: Sequence[str] = ("loose", "static", "strict"),
) -> list[tuple[str, LocalQDMMultiPaddingConfig]]:
    """Return a permissive-to-strict schedule of multi-padding configs."""
    base = LocalQDMMultiPaddingConfig() if config is None else config
    scheduled: list[tuple[str, LocalQDMMultiPaddingConfig]] = []
    for stage in stages:
        stage = str(stage)
        if stage == "base":
            stage_config = base
        elif stage == "loose":
            stage_config = replace(
                base,
                require_static_exterior=False,
                require_kinetic_separation=False,
            )
        elif stage == "static":
            stage_config = replace(
                base,
                require_static_exterior=True,
                require_kinetic_separation=False,
            )
        elif stage == "strict":
            stage_config = replace(
                base,
                require_static_exterior=True,
                require_kinetic_separation=True,
            )
        else:
            raise ValueError(f"Unsupported multi-padding stage: {stage!r}.")
        scheduled.append((stage, stage_config))
    return scheduled


def _qdm_multi_block_report_key(
    report: MultiLocalQDMCertificationReport,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, int]]:
    support_key = tuple(
        sorted(_config_key(config_row) for config_row in report.padding.global_support_configs)
    )
    return (
        tuple(int(block_id) for block_id in report.block_ids),
        support_key,
        report.signature,
    )


def _deduplicate_qdm_multi_block_reports(
    reports: Sequence[MultiLocalQDMCertificationReport],
) -> list[MultiLocalQDMCertificationReport]:
    deduplicated: list[MultiLocalQDMCertificationReport] = []
    seen: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, int]]] = set()
    for report in reports:
        key = _qdm_multi_block_report_key(report)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(report)
    return deduplicated


def robust_certify_qdm_multi_block_result(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
    stages: Sequence[str] = ("loose", "static", "strict"),
) -> CertifiedLocalQDMCageSearchResult:
    """Certify a block pool with a multi-stage padding schedule.

    The early stages are deliberately permissive; exact global certification is
    still the only acceptance criterion.  Duplicate certified supports found at
    multiple stages are deduplicated before wrapping into a limited-basis result.
    """
    base_config = LocalQDMMultiPaddingConfig() if config is None else config
    all_reports: list[MultiLocalQDMCertificationReport] = []

    for _stage_name, stage_config in qdm_multi_padding_config_schedule(
        base_config,
        stages=stages,
    ):
        all_reports.extend(certify_qdm_multi_block_paddings(model, blocks, config=stage_config))

    return certified_qdm_result_from_multi_block_reports(
        model,
        _deduplicate_qdm_multi_block_reports(all_reports),
        config=base_config,
    )


def robust_qdm_local_cage_search(
    model: object,
    *,
    config: RobustQDMLocalCageSearchConfig | None = None,
    adapter: LocalCageModelAdapter | None = None,
    return_context: bool = False,
) -> (
    CertifiedLocalQDMCageSearchResult
    | tuple[
        CertifiedLocalQDMCageSearchResult,
        RobustQDMLocalCageSearchContext,
    ]
):
    """Run a budgeted robust local QDM cage search.

    The search builds a portfolio of region proposals, converts successful local
    records into independent Lego blocks, then certifies the block pool with a
    permissive-to-strict padding schedule.  By default, the return value is the
    existing ``CertifiedLocalQDMCageSearchResult`` container used by downstream
    tools.  Pass ``return_context=True`` to also receive the intermediate scan,
    block pool, and per-stage diagnostics for debugging.
    """
    robust_config = RobustQDMLocalCageSearchConfig() if config is None else config
    proposals = _robust_qdm_region_proposals(model, robust_config, adapter=adapter)
    padding_config = robust_config.as_multi_padding_config()

    if not return_context:
        blocks = collect_qdm_cage_blocks_from_region_proposals(
            proposals,
            model=model,
            adapter=adapter,
            signatures=robust_config.block_signatures,
            max_regions=None,
            max_records_per_region=robust_config.max_records_per_region,
            max_blocks=robust_config.max_blocks,
            skip_incompatible_blocks=robust_config.skip_incompatible_blocks,
        )
        if not blocks:
            return certified_qdm_result_from_multi_block_reports(
                model,
                [],
                config=padding_config,
            )
        return robust_certify_qdm_multi_block_result(
            model,
            blocks,
            config=padding_config,
            stages=robust_config.padding_stages,
        )

    scan = run_local_region_proposals(
        proposals,
        model=model,
        config=robust_config.local_config,
        adapter=adapter,
    )
    blocks = scan.qdm_cage_blocks(
        model=model,
        signatures=robust_config.block_signatures,
        max_records_per_region=robust_config.max_records_per_region,
        max_blocks=robust_config.max_blocks,
        skip_incompatible_blocks=robust_config.skip_incompatible_blocks,
    )

    diagnostics_by_stage: dict[str, QDMMultiPaddingDiagnostics] = {}
    all_reports: list[MultiLocalQDMCertificationReport] = []
    for stage_name, stage_config in qdm_multi_padding_config_schedule(
        padding_config,
        stages=robust_config.padding_stages,
    ):
        if blocks:
            diagnostics = diagnose_qdm_multi_block_paddings(model, blocks, config=stage_config)
        else:
            diagnostics = QDMMultiPaddingDiagnostics(
                paddings=[],
                reports=[],
                failures=[],
                config=stage_config,
            )
        diagnostics_by_stage[stage_name] = diagnostics
        all_reports.extend(diagnostics.reports)

    certified = certified_qdm_result_from_multi_block_reports(
        model,
        _deduplicate_qdm_multi_block_reports(all_reports),
        config=padding_config,
    )
    context = RobustQDMLocalCageSearchContext(
        config=robust_config,
        scan=scan,
        blocks=blocks,
        padding_config=padding_config,
        diagnostics_by_stage=diagnostics_by_stage,
    )
    return certified, context


# Readable alias mirroring the generic-local naming style.
robust_local_qdm_cage_search = robust_qdm_local_cage_search


def certified_qdm_result_from_multi_block_reports(
    model: object,
    reports: Sequence[MultiLocalQDMCertificationReport],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> CertifiedLocalQDMCageSearchResult:
    """Wrap multi-block QDM certificates as a limited-basis cage result.

    The returned object uses the same ``CertifiedLocalQDMCageSearchResult``
    container as the single-block local-padding path. Its basis is the limited
    union of certified support configurations and their one-hop kinetic shell,
    so downstream classification and visualization tools can consume it without
    enumerating the full global Hilbert space.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    report_list = list(reports)
    limited_config_keys: set[tuple[int, ...]] = set()

    for report in report_list:
        for config_row in report.padding.global_support_configs:
            limited_config_keys.add(_config_key(config_row))
        for config_row in report.leakage_configs:
            limited_config_keys.add(_config_key(config_row))

    search_config = _cage_search_config_from_multi_padding(
        model,
        multi_config,
        report_list,
    )

    if not limited_config_keys:
        limited_basis = Basis.empty(model.layout)
        empty_matrix = scipy_sparse.csr_array((0, 0), dtype=np.complex128)
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
            padding_config=multi_config,
        )

    limited_configs = np.asarray([list(key) for key in limited_config_keys], dtype=np.int64)
    if multi_config.sort_limited_basis:
        order = np.lexsort(limited_configs.T[::-1])
        limited_configs = limited_configs[order]

    limited_basis = Basis.from_states(model.layout, limited_configs)
    limited_index = {_config_key(row): i for i, row in enumerate(limited_basis.states)}
    limited_kinetic = build_qdm_global_limited_kinetic_matrix(model, limited_basis)
    limited_self_loops = qdm_global_self_loop_values(model, limited_basis.states)

    cage_records: list[CageRecord] = []
    type1_candidates: list[CandidateSubgraph] = []

    for report_index, report in enumerate(report_list):
        support_indices: list[int] = []
        support_amplitudes: list[complex] = []
        for config_row, amplitude in zip(
            report.padding.global_support_configs,
            report.padding.global_amplitudes,
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
            label=f"multi_qdm_certified_{report_index}",
            metadata={
                "source": "certified_qdm_result_from_multi_block_reports",
                "block_ids": tuple(int(block_id) for block_id in report.block_ids),
                "padding_index": report.padding_index,
                "kinetic_eigenvalue": report.kinetic_eigenvalue,
                "self_loop_value": report.self_loop_value,
                "padding_exterior_link_ids": report.padding.exterior_link_ids.copy(),
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )
        type1_candidates.append(candidate)

        cage_state = CageState(
            energy=complex(report.energy),
            local_state=amplitude_arr,
            support=support_arr,
            boundary_residual=float(report.leakage_residual),
            eigen_residual=float(report.support_hamiltonian_residual),
            full_residual=float(report.full_residual),
            metadata={
                "source": "certify_qdm_multi_block_result",
                "block_ids": tuple(int(block_id) for block_id in report.block_ids),
                "padding_index": report.padding_index,
                "kinetic_eigenvalue": report.kinetic_eigenvalue,
                "self_loop_value": report.self_loop_value,
                "support_kinetic_residual": report.support_kinetic_residual,
                "support_hamiltonian_residual": report.support_hamiltonian_residual,
                "one_hop_shell_size": report.one_hop_shell_size,
            },
        )

        full_state = None
        if multi_config.store_full_states:
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
            type1_candidates=type1_candidates,
            type2_candidates=[],
            search_stage_seconds={},
        ),
        basis=limited_basis,
        kinetic_matrix=limited_kinetic,
        self_loop_values=limited_self_loops,
        reports=report_list,
        padding_config=multi_config,
    )


def certify_qdm_multi_block_result(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    *,
    config: LocalQDMMultiPaddingConfig | None = None,
) -> CertifiedLocalQDMCageSearchResult:
    """Find/certify multi-block QDM paddings and return a certified result.

    This is the multi-block analogue of ``certify_qdm_local_result``: it keeps
    the basis limited to the certified product support plus one-hop shell, but
    exposes ordinary ``CageRecord`` entries for existing tools.
    """
    multi_config = LocalQDMMultiPaddingConfig() if config is None else config
    reports = certify_qdm_multi_block_paddings(model, blocks, config=multi_config)
    return certified_qdm_result_from_multi_block_reports(
        model,
        reports,
        config=multi_config,
    )


def _robust_qdm_region_proposals(
    model: object,
    config: RobustQDMLocalCageSearchConfig,
    *,
    adapter: LocalCageModelAdapter | None = None,
) -> list[LocalRegionProposal]:
    proposals: list[LocalRegionProposal] = []
    if "stripe" in config.region_strategies:
        for width in config.stripe_widths:
            proposals.append(
                StripeRegionProposal(
                    model,
                    directions=config.stripe_directions,
                    width=int(width),
                    config=config.local_config,
                    adapter=adapter,
                )
            )
    if "snake_stripe" in config.region_strategies:
        proposals.append(
            SnakeStripeRegionProposal(
                model,
                max_plaquettes=config.max_region_plaquettes,
                min_plaquettes=config.min_region_plaquettes,
                max_records=config.max_regions_per_strategy,
                max_links=config.max_region_links,
                max_turns=config.snake_stripe_max_turns,
                plaquette_kinds=config.snake_stripe_plaquette_kinds,
                allow_kind_changes=config.snake_stripe_allow_kind_changes,
                winding_vectors=config.snake_stripe_winding_vectors,
                config=config.local_config,
                adapter=adapter,
            )
        )
    if "connected" in config.region_strategies:
        proposals.append(
            ConnectedRegionProposal(
                model,
                max_plaquettes=config.max_region_plaquettes,
                min_plaquettes=config.min_region_plaquettes,
                max_regions=config.max_regions_per_strategy,
                max_links=config.max_region_links,
                config=config.local_config,
                adapter=adapter,
            )
        )
    if "adaptive" in config.region_strategies:
        proposals.append(
            AdaptiveRegionProposal(
                model,
                max_plaquettes=config.max_region_plaquettes,
                seed_plaquette_ids=config.adaptive_seed_plaquette_ids,
                min_plaquettes=config.min_region_plaquettes,
                beam_width=config.adaptive_beam_width,
                branch_factor=config.adaptive_branch_factor,
                max_regions=config.max_regions_per_strategy,
                max_links=config.max_region_links,
                use_search_feedback=config.adaptive_use_search_feedback,
                config=config.local_config,
                adapter=adapter,
            )
        )
    return proposals


def _qdm_multi_padding_failure_report(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: MultiLocalQDMPadding,
    *,
    padding_index: int,
    config: LocalQDMMultiPaddingConfig,
) -> QDMMultiPaddingFailureReport:
    fixed_blocks = tuple(blocks)
    if config.require_static_exterior and not _multi_padding_has_static_exterior(
        model,
        padding,
        fixed_blocks,
    ):
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="static_exterior",
            padding=padding,
        )

    amplitudes = np.asarray(padding.global_amplitudes, dtype=np.complex128)
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="zero_norm",
            padding=padding,
        )
    amplitudes = amplitudes / norm

    support_configs = np.asarray(padding.global_support_configs, dtype=np.int64)
    plaquette_actions = _qdm_multi_block_certification_actions(model, fixed_blocks, config)
    support_keys = [_config_key(config_row) for config_row in support_configs]
    support_amplitude_by_key = {
        key: complex(amplitude) for key, amplitude in zip(support_keys, amplitudes, strict=True)
    }

    action_by_key: dict[tuple[int, ...], complex] = defaultdict(complex)
    action_by_key_and_class: dict[str, dict[tuple[int, ...], complex]] = defaultdict(
        lambda: defaultdict(complex)
    )
    touched_keys: set[tuple[int, ...]] = set(support_keys)
    for source_config, source_amplitude in zip(support_configs, amplitudes, strict=True):
        for action in plaquette_actions:
            transition = _qdm_flip_transition_from_action(source_config, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = _config_key(final_config)
            contribution = complex(coefficient) * complex(source_amplitude)
            action_by_key[final_key] += contribution
            action_class = _qdm_action_plaquette_class(action, fixed_blocks)
            action_by_key_and_class[action_class][final_key] += contribution
            touched_keys.add(final_key)

    kappa = complex(sum(int(block.kappa) for block in fixed_blocks))
    support_kinetic_residuals: list[complex] = []
    leakage_values: list[complex] = []
    leakage_values_by_class: dict[str, list[complex]] = defaultdict(list)
    leakage_counts_by_class: dict[str, int] = defaultdict(int)
    for key in sorted(touched_keys):
        action_value = complex(action_by_key.get(key, 0.0 + 0.0j))
        if key in support_amplitude_by_key:
            support_kinetic_residuals.append(action_value - kappa * support_amplitude_by_key[key])
        else:
            leakage_values.append(action_value)
            for action_class, class_action_by_key in action_by_key_and_class.items():
                class_value = complex(class_action_by_key.get(key, 0.0 + 0.0j))
                if abs(class_value) <= config.tolerance:
                    continue
                leakage_values_by_class[action_class].append(class_value)
                leakage_counts_by_class[action_class] += 1

    support_kinetic_residual = float(
        np.linalg.norm(np.asarray(support_kinetic_residuals, dtype=np.complex128))
    )
    leakage_residual = float(np.linalg.norm(np.asarray(leakage_values, dtype=np.complex128)))
    leakage_norms_by_class = {
        action_class: float(np.linalg.norm(np.asarray(values, dtype=np.complex128)))
        for action_class, values in leakage_values_by_class.items()
    }
    if leakage_residual > config.tolerance:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="leakage_residual",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
            leakage_counts_by_class=dict(leakage_counts_by_class),
            leakage_norms_by_class=leakage_norms_by_class,
        )
    if support_kinetic_residual > config.tolerance:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="support_kinetic_residual",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
        )

    support_self_loops = _qdm_global_self_loop_values_from_actions(
        support_configs,
        plaquette_actions,
    )
    self_loop_value = complex(support_self_loops[0]) if support_self_loops.size else 0.0 + 0.0j
    if np.linalg.norm(support_self_loops - self_loop_value) > config.tolerance:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="nonuniform_self_loop",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
        )

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
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="full_residual",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
            support_hamiltonian_residual=support_hamiltonian_residual,
            full_residual=full_residual,
        )

    signature = signature_from_energy_and_self_loop(
        energy,
        self_loop_value,
        tolerance=max(config.tolerance, 1.0e-15) * 10.0,
        potential_unit=_infer_potential_unit_from_model(model),
    )
    if signature is None:
        return QDMMultiPaddingFailureReport(
            block_ids=tuple(int(block_id) for block_id in padding.block_ids),
            padding_index=int(padding_index),
            reason="signature_inference_failed",
            padding=padding,
            leakage_residual=leakage_residual,
            support_kinetic_residual=support_kinetic_residual,
            support_hamiltonian_residual=support_hamiltonian_residual,
            full_residual=full_residual,
        )

    return QDMMultiPaddingFailureReport(
        block_ids=tuple(int(block_id) for block_id in padding.block_ids),
        padding_index=int(padding_index),
        reason="unknown",
        padding=padding,
        leakage_residual=leakage_residual,
        support_kinetic_residual=support_kinetic_residual,
        support_hamiltonian_residual=support_hamiltonian_residual,
        full_residual=full_residual,
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
) -> Iterator[MultiLocalQDMPadding]:
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
                padding = _make_qdm_multi_padding_from_exterior(
                    model,
                    fixed_blocks,
                    exterior_link_ids=exterior_link_ids,
                    exterior_config=exterior_config.copy(),
                )
                if _multi_padding_passes_global_filters(model, padding, fixed_blocks, config):
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


def _certify_qdm_multi_padding(
    model: object,
    blocks: Sequence[LocalQDMCageBlock],
    padding: MultiLocalQDMPadding,
    *,
    padding_index: int,
    config: LocalQDMMultiPaddingConfig,
) -> MultiLocalQDMCertificationReport | None:
    fixed_blocks = tuple(blocks)
    if tuple(int(block.block_id) for block in fixed_blocks) != tuple(
        int(x) for x in padding.block_ids
    ):
        raise ValueError("blocks must match padding.block_ids and order.")

    amplitudes = np.asarray(padding.global_amplitudes, dtype=np.complex128)
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        return None
    amplitudes = amplitudes / norm

    support_configs = np.asarray(padding.global_support_configs, dtype=np.int64)
    if config.require_static_exterior and not _multi_padding_has_static_exterior(
        model,
        padding,
        fixed_blocks,
    ):
        return None

    plaquette_actions = _qdm_multi_block_certification_actions(model, fixed_blocks, config)
    support_keys = [_config_key(config_row) for config_row in support_configs]
    support_amplitude_by_key = {
        key: complex(amplitude) for key, amplitude in zip(support_keys, amplitudes, strict=True)
    }

    action_by_key: dict[tuple[int, ...], complex] = defaultdict(complex)
    touched_keys: set[tuple[int, ...]] = set(support_keys)

    for source_config, source_amplitude in zip(support_configs, amplitudes, strict=True):
        for action in plaquette_actions:
            transition = _qdm_flip_transition_from_action(source_config, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = _config_key(final_config)
            action_by_key[final_key] += complex(coefficient) * complex(source_amplitude)
            touched_keys.add(final_key)

    kappa = complex(sum(int(block.kappa) for block in fixed_blocks))
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

    support_self_loops = _qdm_global_self_loop_values_from_actions(
        support_configs,
        plaquette_actions,
    )
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

    return MultiLocalQDMCertificationReport(
        block_ids=tuple(int(block.block_id) for block in fixed_blocks),
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
    if padding.global_support_configs.shape[0] == 0:
        return True

    reference_config = padding.global_support_configs[0]
    for action in _qdm_global_plaquette_actions(model):
        if any(int(link_id) in local_link_set for link_id in action.links):
            continue
        if _qdm_plaquette_is_flippable_from_action(reference_config, action):
            return False
    return True


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


def _cage_search_config_from_multi_padding(
    model: object,
    padding_config: LocalQDMMultiPaddingConfig,
    reports: Sequence[MultiLocalQDMCertificationReport],
) -> CageSearchConfig:
    kappas = tuple(sorted({int(report.signature[0]) for report in reports})) or (0,)
    return CageSearchConfig(
        search_type="type1",
        tolerance=padding_config.tolerance,
        min_component_size=1,
        validate_full_residual=True,
        type1_kappas=kappas,
        deduplicate_by_rank=False,
        potential_signature_unit=_infer_potential_unit_from_model(model),
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
