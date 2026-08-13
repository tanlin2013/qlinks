"""Local-region proposal generation and proposal-driven cage scans."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from qlinks.caging.local_search_certification import make_qdm_cage_block
from qlinks.caging.local_search_core import (
    LocalCageSearcher,
    LocalQDMCageSearchResult,
    local_cage_adapter_for_model,
)
from qlinks.caging.local_search_geometry import (
    _adaptive_region_frontier,
    _adaptive_region_score,
    _adaptive_seed_plaquette_ids,
    _add_cell_displacements,
    _cell_in_stripe_band,
    _default_stripe_directions,
    _ordered_cycle_plaquettes_from_shared_graph,
    _ordered_straight_stripe_plaquettes,
    _plaquette_shared_link_neighbor_edges,
    _plaquette_shared_link_neighbor_map,
    _snake_edge_neighbor_sets,
    _snake_partial_kind_pattern_possible,
    _snake_path_extension_is_induced,
    _snake_path_is_induced_cycle,
    _snake_path_kind_pattern_allowed,
    _snake_step_turn_increment,
    _stripe_component_subsets,
    _stripe_motif_subsets,
    _stripe_plaquette_data,
    _top_adaptive_records,
    _transverse_coordinates,
    _unique_int_array,
    _validate_plaquette_ids,
    _validate_stripe_direction,
    _winding_from_lifted_displacement,
    _zero_cell_displacement,
)
from qlinks.caging.local_search_types import (
    AdaptiveRegionProposalRecord,
    ConnectedRegionProposalRecord,
    LocalCageModelAdapter,
    LocalQDMCageBlock,
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    LocalQDMRegion,
    LocalRegionProposal,
    LocalRegionProposalSearchRecord,
    RobustQDMLocalCageSearchConfig,
    SnakeStripeKindPattern,
    SnakeStripeRegionProposalRecord,
    StripeMotifComponentRegionProposalRecord,
    StripeMotifComponentSubsetMode,
    StripeMotifRegionProposalRecord,
    StripeMotifSource,
    StripeMotifSubsetMode,
    StripeRegionProposalRecord,
)


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

    Optional ``require_induced_cycle`` and ``kind_pattern`` filters turn the
    broad cycle enumerator into a more motif-like proposal: the examples seen
    in exact QDM cages are usually chordless width-one cycles whose plaquette
    kinds are either constant or strictly alternating between two kinds.

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
    kind_pattern: SnakeStripeKindPattern = "constant_or_alternating"
    require_induced_cycle: bool = False
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
        if self.kind_pattern not in {"any", "constant", "alternating", "constant_or_alternating"}:
            raise ValueError(
                "kind_pattern must be 'any', 'constant', 'alternating', "
                "or 'constant_or_alternating'."
            )

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
        edge_neighbors = _snake_edge_neighbor_sets(edge_map)
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
                        if self.require_induced_cycle and not _snake_path_is_induced_cycle(
                            path, edge_neighbors
                        ):
                            continue
                        if not _snake_path_kind_pattern_allowed(
                            self.model, path, self.kind_pattern
                        ):
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
                    next_path = (*path, neighbor)
                    if self.require_induced_cycle and not _snake_path_extension_is_induced(
                        path, neighbor, edge_neighbors
                    ):
                        continue
                    if not _snake_partial_kind_pattern_possible(
                        self.model, next_path, self.kind_pattern
                    ):
                        continue

                    stack.append(
                        (
                            next_path,
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
class StripeMotifRegionProposal:
    """Generate small QDM motif regions cut from stripe-like plaquette paths.

    This is a fast path for QDM cages whose real-space organization is a
    width-one stripe but whose coherent local objects are only small two- or
    three-plaquette singlet/triplet motifs.  It first constructs cheap straight
    and/or snake stripe skeletons, then emits small motif subsets from each
    skeleton.  The ordinary local cage algebra is still used afterward, but on
    much smaller regions than a full stripe.
    """

    model: object
    config: LocalQDMCageSearchConfig = field(
        default_factory=lambda: LocalQDMCageSearchConfig(halo_layers=0)
    )
    motif_sizes: tuple[int, ...] = (2, 3)
    sources: tuple[StripeMotifSource, ...] = ("stripe", "snake_stripe")
    subset_mode: StripeMotifSubsetMode = "all"
    max_motifs_per_stripe: int | None = None
    max_records: int | None = None
    max_links: int | None = None
    stripe_widths: tuple[int, ...] = (1,)
    stripe_directions: tuple[int, ...] | None = None
    plaquette_kinds: tuple[str, ...] | None = None
    snake_max_plaquettes: int | None = None
    snake_min_plaquettes: int = 3
    snake_max_turns: int | None = None
    snake_allow_kind_changes: bool = False
    snake_kind_pattern: SnakeStripeKindPattern = "constant_or_alternating"
    snake_require_induced_cycle: bool = False
    snake_winding_vectors: tuple[tuple[int, ...], ...] | None = None
    adapter: LocalCageModelAdapter | None = None

    def __post_init__(self) -> None:
        if not self.motif_sizes:
            raise ValueError("motif_sizes must be non-empty.")
        motif_sizes = tuple(int(size) for size in self.motif_sizes)
        if any(size <= 0 for size in motif_sizes):
            raise ValueError("motif_sizes must contain positive integers.")
        object.__setattr__(self, "motif_sizes", motif_sizes)

        if not self.sources:
            raise ValueError("sources must be non-empty.")
        bad_sources = [
            source for source in self.sources if source not in {"stripe", "snake_stripe"}
        ]
        if bad_sources:
            raise ValueError(f"Unsupported stripe motif sources: {bad_sources}.")
        object.__setattr__(self, "sources", tuple(str(source) for source in self.sources))

        if self.subset_mode not in {"windows", "all"}:
            raise ValueError("subset_mode must be 'windows' or 'all'.")
        if self.max_motifs_per_stripe is not None and self.max_motifs_per_stripe < 0:
            raise ValueError("max_motifs_per_stripe must be non-negative or None.")
        if self.max_records is not None and self.max_records < 0:
            raise ValueError("max_records must be non-negative or None.")
        if self.max_links is not None and self.max_links <= 0:
            raise ValueError("max_links must be positive or None.")
        if not self.stripe_widths:
            raise ValueError("stripe_widths must be non-empty.")
        stripe_widths = tuple(int(width) for width in self.stripe_widths)
        if any(width <= 0 for width in stripe_widths):
            raise ValueError("stripe_widths must contain positive integers.")
        object.__setattr__(self, "stripe_widths", stripe_widths)
        if self.snake_max_plaquettes is not None and self.snake_max_plaquettes <= 0:
            raise ValueError("snake_max_plaquettes must be positive or None.")
        if self.snake_min_plaquettes <= 0:
            raise ValueError("snake_min_plaquettes must be positive.")
        if (
            self.snake_max_plaquettes is not None
            and self.snake_min_plaquettes > self.snake_max_plaquettes
        ):
            raise ValueError("snake_min_plaquettes cannot exceed snake_max_plaquettes.")
        if self.snake_max_turns is not None and self.snake_max_turns < 0:
            raise ValueError("snake_max_turns must be non-negative or None.")
        if self.snake_kind_pattern not in {
            "any",
            "constant",
            "alternating",
            "constant_or_alternating",
        }:
            raise ValueError(
                "snake_kind_pattern must be 'any', 'constant', 'alternating', "
                "or 'constant_or_alternating'."
            )

        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        config = adapter.normalize_config(self.config)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "config", config)

        if self.stripe_directions is not None:
            directions = tuple(int(direction) for direction in self.stripe_directions)
            if not directions:
                raise ValueError("stripe_directions must be non-empty when provided.")
            object.__setattr__(self, "stripe_directions", directions)

        if self.plaquette_kinds is not None:
            kinds = tuple(str(kind) for kind in self.plaquette_kinds)
            if not kinds:
                raise ValueError("plaquette_kinds must be non-empty when provided.")
            object.__setattr__(self, "plaquette_kinds", kinds)

        if self.snake_winding_vectors is not None:
            windings = tuple(
                tuple(int(value) for value in winding) for winding in self.snake_winding_vectors
            )
            if not windings:
                raise ValueError("snake_winding_vectors must be non-empty or None.")
            object.__setattr__(self, "snake_winding_vectors", windings)

    def iter_records(self) -> Iterator[StripeMotifRegionProposalRecord]:
        """Yield small motif records in deterministic stripe-skeleton order."""
        n_emitted = 0
        seen: set[tuple[int, ...]] = set()
        for source_index, source, ordered_ids in self._iter_source_stripes():
            n_from_source = 0
            for motif_index, motif_ids in enumerate(
                _stripe_motif_subsets(
                    ordered_ids,
                    motif_sizes=self.motif_sizes,
                    subset_mode=self.subset_mode,
                )
            ):
                if self.max_motifs_per_stripe is not None and n_from_source >= int(
                    self.max_motifs_per_stripe
                ):
                    break
                key = tuple(sorted(int(pid) for pid in motif_ids))
                if key in seen:
                    continue
                record = self._make_record(
                    plaquette_ids=motif_ids,
                    source=source,
                    source_index=source_index,
                    source_plaquette_ids=ordered_ids,
                    motif_index=motif_index,
                )
                if record is None:
                    continue
                seen.add(key)
                n_from_source += 1
                n_emitted += 1
                yield record
                if self.max_records is not None and n_emitted >= int(self.max_records):
                    return

    def iter_regions(self) -> Iterator[LocalQDMRegion]:
        """Yield only the local regions from :meth:`iter_records`."""
        for record in self.iter_records():
            yield record.region

    def iter_searchers(self) -> Iterator[LocalCageSearcher]:
        """Yield ready-to-run local cage searchers for proposed motifs."""
        for record in self.iter_records():
            yield LocalCageSearcher(
                model=self.model,
                region=record.region,
                config=self.config,
                adapter=self.adapter,
            )

    def _iter_source_stripes(self) -> Iterator[tuple[int, str, tuple[int, ...]]]:
        source_index = 0
        if "stripe" in self.sources:
            for width in self.stripe_widths:
                proposal = StripeRegionProposal(
                    self.model,
                    directions=self.stripe_directions,
                    width=int(width),
                    plaquette_kinds=self.plaquette_kinds,
                    config=self.config,
                    adapter=self.adapter,
                )
                for record in proposal.iter_records():
                    ordered = _ordered_straight_stripe_plaquettes(
                        self.model,
                        record.plaquette_ids,
                        direction=record.direction,
                    )
                    yield source_index, "stripe", ordered
                    source_index += 1

        if "snake_stripe" in self.sources:
            max_plaquettes = self.snake_max_plaquettes
            if max_plaquettes is None:
                max_plaquettes = max(self.snake_min_plaquettes, max(self.motif_sizes) + 1)
            proposal = SnakeStripeRegionProposal(
                self.model,
                max_plaquettes=int(max_plaquettes),
                min_plaquettes=int(self.snake_min_plaquettes),
                max_records=self.max_records,
                max_links=None,
                max_turns=self.snake_max_turns,
                plaquette_kinds=self.plaquette_kinds,
                allow_kind_changes=self.snake_allow_kind_changes,
                kind_pattern=self.snake_kind_pattern,
                require_induced_cycle=self.snake_require_induced_cycle,
                winding_vectors=self.snake_winding_vectors,
                config=self.config,
                adapter=self.adapter,
            )
            for record in proposal.iter_records():
                ordered = _ordered_cycle_plaquettes_from_shared_graph(
                    self.model,
                    record.plaquette_ids,
                )
                if not ordered:
                    ordered = tuple(int(pid) for pid in record.plaquette_ids)
                yield source_index, "snake_stripe", ordered
                source_index += 1

    def _make_record(
        self,
        *,
        plaquette_ids: Sequence[int],
        source: str,
        source_index: int,
        source_plaquette_ids: Sequence[int],
        motif_index: int,
    ) -> StripeMotifRegionProposalRecord | None:
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
        return StripeMotifRegionProposalRecord(
            region=region,
            plaquette_ids=selected,
            source=str(source),
            source_index=int(source_index),
            source_plaquette_ids=np.asarray(tuple(int(pid) for pid in source_plaquette_ids)),
            motif_size=int(selected.size),
            motif_index=int(motif_index),
        )


@dataclass(frozen=True, slots=True)
class StripeMotifComponentRegionProposal:
    """Generate merged stripe components seeded by small coherent motifs.

    The proposal first cuts small motifs from each straight/snake stripe skeleton
    and runs the existing local cage searcher on those tiny motifs.  If enough
    motifs have local cage records, it emits a larger component region, by
    default the whole stripe skeleton.  This is intended for triangular and
    honeycomb QDM cages where the exact state is a stripe-local object rather
    than a product of independent two-plaquette blocks.
    """

    model: object
    config: LocalQDMCageSearchConfig = field(
        default_factory=lambda: LocalQDMCageSearchConfig(halo_layers=0)
    )
    motif_sizes: tuple[int, ...] = (2, 3)
    motif_subset_mode: StripeMotifSubsetMode = "windows"
    motif_signatures: tuple[tuple[int, int], ...] | None = None
    min_seed_motifs: int = 1
    max_seed_motifs_per_stripe: int | None = None
    component_sizes: tuple[int, ...] | None = None
    component_subset_mode: StripeMotifComponentSubsetMode = "full"
    sources: tuple[StripeMotifSource, ...] = ("snake_stripe",)
    max_components_per_stripe: int | None = 1
    max_records: int | None = None
    max_links: int | None = None
    stripe_widths: tuple[int, ...] = (1,)
    stripe_directions: tuple[int, ...] | None = None
    plaquette_kinds: tuple[str, ...] | None = None
    snake_max_plaquettes: int | None = None
    snake_min_plaquettes: int = 3
    snake_max_turns: int | None = None
    snake_allow_kind_changes: bool = False
    snake_kind_pattern: SnakeStripeKindPattern = "constant_or_alternating"
    snake_require_induced_cycle: bool = False
    snake_winding_vectors: tuple[tuple[int, ...], ...] | None = None
    adapter: LocalCageModelAdapter | None = None

    def __post_init__(self) -> None:
        if not self.motif_sizes:
            raise ValueError("motif_sizes must be non-empty.")
        motif_sizes = tuple(int(size) for size in self.motif_sizes)
        if any(size <= 0 for size in motif_sizes):
            raise ValueError("motif_sizes must contain positive integers.")
        object.__setattr__(self, "motif_sizes", motif_sizes)

        if self.motif_subset_mode not in {"windows", "all"}:
            raise ValueError("motif_subset_mode must be 'windows' or 'all'.")
        if self.min_seed_motifs <= 0:
            raise ValueError("min_seed_motifs must be positive.")
        if self.max_seed_motifs_per_stripe is not None and self.max_seed_motifs_per_stripe < 0:
            raise ValueError("max_seed_motifs_per_stripe must be non-negative or None.")
        if self.component_subset_mode not in {"full", "windows", "all"}:
            raise ValueError("component_subset_mode must be 'full', 'windows', or 'all'.")
        if self.component_sizes is not None:
            component_sizes = tuple(int(size) for size in self.component_sizes)
            if not component_sizes:
                raise ValueError("component_sizes must be non-empty when provided.")
            if any(size <= 0 for size in component_sizes):
                raise ValueError("component_sizes must contain positive integers.")
            object.__setattr__(self, "component_sizes", component_sizes)
        if not self.sources:
            raise ValueError("sources must be non-empty.")
        bad_sources = [
            source for source in self.sources if source not in {"stripe", "snake_stripe"}
        ]
        if bad_sources:
            raise ValueError(f"Unsupported stripe motif component sources: {bad_sources}.")
        object.__setattr__(self, "sources", tuple(str(source) for source in self.sources))
        if self.max_components_per_stripe is not None and self.max_components_per_stripe < 0:
            raise ValueError("max_components_per_stripe must be non-negative or None.")
        if self.max_records is not None and self.max_records < 0:
            raise ValueError("max_records must be non-negative or None.")
        if self.max_links is not None and self.max_links <= 0:
            raise ValueError("max_links must be positive or None.")
        if not self.stripe_widths:
            raise ValueError("stripe_widths must be non-empty.")
        stripe_widths = tuple(int(width) for width in self.stripe_widths)
        if any(width <= 0 for width in stripe_widths):
            raise ValueError("stripe_widths must contain positive integers.")
        object.__setattr__(self, "stripe_widths", stripe_widths)
        if self.snake_max_plaquettes is not None and self.snake_max_plaquettes <= 0:
            raise ValueError("snake_max_plaquettes must be positive or None.")
        if self.snake_min_plaquettes <= 0:
            raise ValueError("snake_min_plaquettes must be positive.")
        if (
            self.snake_max_plaquettes is not None
            and self.snake_min_plaquettes > self.snake_max_plaquettes
        ):
            raise ValueError("snake_min_plaquettes cannot exceed snake_max_plaquettes.")
        if self.snake_max_turns is not None and self.snake_max_turns < 0:
            raise ValueError("snake_max_turns must be non-negative or None.")
        if self.snake_kind_pattern not in {
            "any",
            "constant",
            "alternating",
            "constant_or_alternating",
        }:
            raise ValueError(
                "snake_kind_pattern must be 'any', 'constant', 'alternating', "
                "or 'constant_or_alternating'."
            )

        adapter = local_cage_adapter_for_model(self.model, self.adapter)
        config = adapter.normalize_config(self.config)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "config", config)

        if self.stripe_directions is not None:
            directions = tuple(int(direction) for direction in self.stripe_directions)
            if not directions:
                raise ValueError("stripe_directions must be non-empty when provided.")
            object.__setattr__(self, "stripe_directions", directions)

        if self.plaquette_kinds is not None:
            kinds = tuple(str(kind) for kind in self.plaquette_kinds)
            if not kinds:
                raise ValueError("plaquette_kinds must be non-empty when provided.")
            object.__setattr__(self, "plaquette_kinds", kinds)

        if self.snake_winding_vectors is not None:
            windings = tuple(
                tuple(int(value) for value in winding) for winding in self.snake_winding_vectors
            )
            if not windings:
                raise ValueError("snake_winding_vectors must be non-empty or None.")
            object.__setattr__(self, "snake_winding_vectors", windings)

        if self.motif_signatures is not None:
            signatures = tuple(
                (int(kappa), int(potential)) for kappa, potential in self.motif_signatures
            )
            object.__setattr__(self, "motif_signatures", signatures)

    def iter_records(self) -> Iterator[StripeMotifComponentRegionProposalRecord]:
        """Yield merged component records in deterministic stripe-skeleton order."""
        n_emitted = 0
        seen: set[tuple[int, ...]] = set()
        for source_index, source, ordered_ids in self._iter_source_stripes():
            seed_motifs, seed_signatures = self._seed_motifs_for_stripe(ordered_ids)
            if len(seed_motifs) < int(self.min_seed_motifs):
                continue

            n_from_source = 0
            for component_index, component_ids in enumerate(
                _stripe_component_subsets(
                    ordered_ids,
                    component_sizes=self.component_sizes,
                    subset_mode=self.component_subset_mode,
                )
            ):
                if self.max_components_per_stripe is not None and n_from_source >= int(
                    self.max_components_per_stripe
                ):
                    break
                key = tuple(sorted(int(pid) for pid in component_ids))
                if key in seen:
                    continue
                record = self._make_record(
                    plaquette_ids=component_ids,
                    source=source,
                    source_index=source_index,
                    source_plaquette_ids=ordered_ids,
                    component_index=component_index,
                    seed_motifs=seed_motifs,
                    seed_signatures=seed_signatures,
                )
                if record is None:
                    continue
                seen.add(key)
                n_from_source += 1
                n_emitted += 1
                yield record
                if self.max_records is not None and n_emitted >= int(self.max_records):
                    return

    def iter_regions(self) -> Iterator[LocalQDMRegion]:
        for record in self.iter_records():
            yield record.region

    def iter_searchers(self) -> Iterator[LocalCageSearcher]:
        for record in self.iter_records():
            yield LocalCageSearcher(
                model=self.model,
                region=record.region,
                config=self.config,
                adapter=self.adapter,
            )

    def _iter_source_stripes(self) -> Iterator[tuple[int, str, tuple[int, ...]]]:
        source_index = 0
        if "stripe" in self.sources:
            for width in self.stripe_widths:
                proposal = StripeRegionProposal(
                    self.model,
                    directions=self.stripe_directions,
                    width=int(width),
                    plaquette_kinds=self.plaquette_kinds,
                    config=self.config,
                    adapter=self.adapter,
                )
                for record in proposal.iter_records():
                    ordered = _ordered_straight_stripe_plaquettes(
                        self.model,
                        record.plaquette_ids,
                        direction=record.direction,
                    )
                    yield source_index, "stripe", ordered
                    source_index += 1

        if "snake_stripe" in self.sources:
            max_plaquettes = self.snake_max_plaquettes
            if max_plaquettes is None:
                max_plaquettes = max(self.snake_min_plaquettes, max(self.motif_sizes) + 1)
            proposal = SnakeStripeRegionProposal(
                self.model,
                max_plaquettes=int(max_plaquettes),
                min_plaquettes=int(self.snake_min_plaquettes),
                max_records=self.max_records,
                max_links=None,
                max_turns=self.snake_max_turns,
                plaquette_kinds=self.plaquette_kinds,
                allow_kind_changes=self.snake_allow_kind_changes,
                kind_pattern=self.snake_kind_pattern,
                require_induced_cycle=self.snake_require_induced_cycle,
                winding_vectors=self.snake_winding_vectors,
                config=self.config,
                adapter=self.adapter,
            )
            for record in proposal.iter_records():
                ordered = _ordered_cycle_plaquettes_from_shared_graph(
                    self.model,
                    record.plaquette_ids,
                )
                if not ordered:
                    ordered = tuple(int(pid) for pid in record.plaquette_ids)
                yield source_index, "snake_stripe", ordered
                source_index += 1

    def _seed_motifs_for_stripe(
        self,
        ordered_ids: Sequence[int],
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
        signature_filter = None
        if self.motif_signatures is not None:
            signature_filter = set(self.motif_signatures)

        motifs: list[tuple[int, ...]] = []
        signatures: list[tuple[int, int]] = []
        for motif_ids in _stripe_motif_subsets(
            ordered_ids,
            motif_sizes=self.motif_sizes,
            subset_mode=self.motif_subset_mode,
        ):
            if self.max_seed_motifs_per_stripe is not None and len(motifs) >= int(
                self.max_seed_motifs_per_stripe
            ):
                break
            result = self._run_seed_motif_search(motif_ids)
            if not result.records:
                continue
            motif_signatures = tuple(record.signature for record in result.records)
            if signature_filter is not None:
                motif_signatures = tuple(
                    signature for signature in motif_signatures if signature in signature_filter
                )
            if not motif_signatures:
                continue
            motifs.append(tuple(int(pid) for pid in motif_ids))
            signatures.extend(motif_signatures)
        return tuple(motifs), tuple(signatures)

    def _run_seed_motif_search(self, motif_ids: Sequence[int]) -> LocalQDMCageSearchResult:
        region = self.adapter.build_region_from_plaquettes(
            plaquette_ids=np.asarray(tuple(int(pid) for pid in motif_ids), dtype=np.int64),
            config=self.config,
            scoring_plaquette_ids=np.asarray(tuple(int(pid) for pid in motif_ids), dtype=np.int64),
        )
        seed_config = replace(self.config, degenerate_basis_strategy="none")
        return LocalCageSearcher(
            model=self.model,
            region=region,
            config=seed_config,
            adapter=self.adapter,
        ).run()

    def _make_record(
        self,
        *,
        plaquette_ids: Sequence[int],
        source: str,
        source_index: int,
        source_plaquette_ids: Sequence[int],
        component_index: int,
        seed_motifs: Sequence[Sequence[int]],
        seed_signatures: Sequence[tuple[int, int]],
    ) -> StripeMotifComponentRegionProposalRecord | None:
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
        return StripeMotifComponentRegionProposalRecord(
            region=region,
            plaquette_ids=selected,
            source=str(source),
            source_index=int(source_index),
            source_plaquette_ids=np.asarray(tuple(int(pid) for pid in source_plaquette_ids)),
            component_size=int(selected.size),
            component_index=int(component_index),
            n_seed_motifs=len(seed_motifs),
            seed_motif_plaquette_ids=tuple(
                tuple(int(pid) for pid in motif) for motif in seed_motifs
            ),
            seed_motif_signatures=tuple(
                (int(kappa), int(potential)) for kappa, potential in seed_signatures
            ),
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


def collect_qdm_cage_blocks_with_scan_from_region_proposals(
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
) -> tuple[LocalRegionProposalSearchResult, list[LocalQDMCageBlock]]:
    """Run proposal searches and stream compatible QDM blocks.

    This is the block-oriented counterpart of :func:`run_local_region_proposals`.
    It converts records into ``LocalQDMCageBlock`` objects immediately after each
    region is searched and stops as soon as ``max_blocks`` is reached.  This is
    important for expensive proposal portfolios: the older two-stage workflow
    searched every proposed region first and only then applied the block cap, so
    robust scans could spend most of their time in local DFS branches that would
    never contribute to the requested block pool.
    """
    if block_id_start < 0:
        raise ValueError("block_id_start must be non-negative.")
    if max_regions is not None and max_regions < 0:
        raise ValueError("max_regions must be non-negative or None.")
    if max_records_per_region is not None and max_records_per_region < 0:
        raise ValueError("max_records_per_region must be non-negative or None.")
    if max_blocks is not None and max_blocks < 0:
        raise ValueError("max_blocks must be non-negative or None.")

    signature_filter = None
    if signatures is not None:
        signature_filter = {(int(kappa), int(potential)) for kappa, potential in signatures}

    search_records: list[LocalRegionProposalSearchRecord] = []
    blocks: list[LocalQDMCageBlock] = []
    emitted_regions = 0
    next_block_id = int(block_id_start)

    if max_blocks == 0:
        return LocalRegionProposalSearchResult(records=[]), []

    for proposal_index, proposal in enumerate(proposals):
        proposal_model = _model_for_region_proposal(proposal, model)
        proposal_adapter = _adapter_for_region_proposal(proposal, adapter)
        proposal_config = _config_for_region_proposal(proposal, config)

        for region_index, proposal_record, region in _iter_region_proposal_records(proposal):
            if max_regions is not None and emitted_regions >= max_regions:
                return LocalRegionProposalSearchResult(records=search_records), blocks
            if max_blocks is not None and len(blocks) >= max_blocks:
                return LocalRegionProposalSearchResult(records=search_records), blocks

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
            emitted_regions += 1

            region_records = result.records
            if signature_filter is not None:
                region_records = [
                    record for record in region_records if record.signature in signature_filter
                ]
            if max_records_per_region is not None:
                region_records = region_records[:max_records_per_region]

            for local_record in region_records:
                if max_blocks is not None and len(blocks) >= max_blocks:
                    return LocalRegionProposalSearchResult(records=search_records), blocks
                try:
                    block = make_qdm_cage_block(
                        proposal_model,
                        local_record,
                        block_id=next_block_id,
                    )
                except ValueError:
                    if skip_incompatible_blocks:
                        continue
                    raise
                blocks.append(block)
                next_block_id += 1

    return LocalRegionProposalSearchResult(records=search_records), blocks


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
    _, blocks = collect_qdm_cage_blocks_with_scan_from_region_proposals(
        proposals,
        model=model,
        config=config,
        adapter=adapter,
        signatures=signatures,
        max_regions=max_regions,
        max_records_per_region=max_records_per_region,
        max_blocks=max_blocks,
        block_id_start=block_id_start,
        skip_incompatible_blocks=skip_incompatible_blocks,
    )
    return blocks


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


def _robust_qdm_region_proposals(
    model: object,
    config: RobustQDMLocalCageSearchConfig,
    *,
    adapter: LocalCageModelAdapter | None = None,
) -> list[LocalRegionProposal]:
    proposals: list[LocalRegionProposal] = []
    if "stripe_motif" in config.region_strategies:
        proposals.append(
            StripeMotifRegionProposal(
                model,
                motif_sizes=config.stripe_motif_sizes,
                sources=config.stripe_motif_sources,
                subset_mode=config.stripe_motif_subset_mode,
                max_motifs_per_stripe=config.stripe_motif_max_motifs_per_stripe,
                max_records=config.max_regions_per_strategy,
                max_links=config.max_region_links,
                stripe_widths=config.stripe_widths,
                stripe_directions=config.stripe_directions,
                plaquette_kinds=config.snake_stripe_plaquette_kinds,
                snake_max_plaquettes=config.max_region_plaquettes,
                snake_min_plaquettes=config.min_region_plaquettes,
                snake_max_turns=config.snake_stripe_max_turns,
                snake_allow_kind_changes=config.snake_stripe_allow_kind_changes,
                snake_kind_pattern=config.snake_stripe_kind_pattern,
                snake_require_induced_cycle=config.snake_stripe_require_induced_cycle,
                snake_winding_vectors=config.snake_stripe_winding_vectors,
                config=config.local_config,
                adapter=adapter,
            )
        )
    if "stripe_motif_component" in config.region_strategies:
        proposals.append(
            StripeMotifComponentRegionProposal(
                model,
                motif_sizes=config.stripe_motif_sizes,
                motif_subset_mode=config.stripe_motif_subset_mode,
                motif_signatures=config.stripe_motif_component_motif_signatures,
                min_seed_motifs=config.stripe_motif_component_min_seed_motifs,
                max_seed_motifs_per_stripe=(
                    config.stripe_motif_component_max_seed_motifs_per_stripe
                ),
                component_sizes=config.stripe_motif_component_sizes,
                component_subset_mode=config.stripe_motif_component_subset_mode,
                sources=config.stripe_motif_sources,
                max_components_per_stripe=config.stripe_motif_component_max_components_per_stripe,
                max_records=config.max_regions_per_strategy,
                max_links=config.max_region_links,
                stripe_widths=config.stripe_widths,
                stripe_directions=config.stripe_directions,
                plaquette_kinds=config.snake_stripe_plaquette_kinds,
                snake_max_plaquettes=config.max_region_plaquettes,
                snake_min_plaquettes=config.min_region_plaquettes,
                snake_max_turns=config.snake_stripe_max_turns,
                snake_allow_kind_changes=config.snake_stripe_allow_kind_changes,
                snake_kind_pattern=config.snake_stripe_kind_pattern,
                snake_require_induced_cycle=config.snake_stripe_require_induced_cycle,
                snake_winding_vectors=config.snake_stripe_winding_vectors,
                config=config.local_config,
                adapter=adapter,
            )
        )
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
                kind_pattern=config.snake_stripe_kind_pattern,
                require_induced_cycle=config.snake_stripe_require_induced_cycle,
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


collect_qdm_cage_blocks_from_proposals = collect_qdm_cage_blocks_from_region_proposals
collect_qdm_cage_blocks_with_scan_from_proposals = (
    collect_qdm_cage_blocks_with_scan_from_region_proposals
)
