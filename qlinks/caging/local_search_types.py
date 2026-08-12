"""Data contracts for local-first cage search.

This module owns configuration, record, report, and protocol-like data structures that do
not execute the local-search algorithms themselves. Keeping these contracts separate from
search orchestration makes the numerical implementation reviewable and gives the refactor a
clear API boundary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis
from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.results import CageState
from qlinks.caging.search import CageRecord, CageSearchConfig, CageSearchResult
from qlinks.caging.types import DegenerateBasisStrategy
from qlinks.constraints import ConstraintPropagation, ConstraintResult
from qlinks.variables import VariableLayout

if TYPE_CHECKING:
    from qlinks.caging.local_search import LocalQDMCageSearchResult, LocalRegionProposalSearchResult

LocalBoundaryMode = Literal["relaxed", "closed"]
SnakeStripeKindPattern = Literal["any", "constant", "alternating", "constant_or_alternating"]
StripeMotifSource = Literal["stripe", "snake_stripe"]
StripeMotifSubsetMode = Literal["windows", "all"]
StripeMotifComponentSubsetMode = Literal["full", "windows", "all"]


@dataclass(frozen=True, slots=True)
class LocalQDMCageSearchConfig:
    """Configuration for the QDM local-first type-1 cage search.

    Attributes:
        tolerance: Numerical tolerance used by the local candidate solver.
        allowed_kappas: Kinetic eigenvalues to target.  The local-first path is
            intended for type-1 cages, so the default is ``(0,)``.
        min_component_size: Minimum local kinetic-graph component size.
        halo_layers: Number of plaquette-neighbor expansions applied when the
            search region is supplied by plaquettes.  Neighbors share at least
            one link.
        boundary_mode: ``"relaxed"`` enforces exact dimer constraints only at
            internally complete sites; boundary sites use an at-most constraint.
            ``"closed"`` requires all touched sites to be complete and then
            enforces exact constraints.
        include_sectors_when_full: If the local link set is the full model link
            set, also apply model sector conditions during local basis
            generation.
        prune_inactive_local_basis_states: For genuine local regions, ask DFS
            to prune branches that can no longer produce a configuration
            flippable on any active plaquette.
        max_local_states: Optional early-stop limit for local basis generation.
        sort_basis: Whether to sort the local basis.
        validate_full_residual: Whether local cage states should be validated
            against the full local kinetic graph columns.
        degenerate_basis_strategy: How to choose representatives from degenerate
            local cage subspaces.
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
class StripeMotifRegionProposalRecord:
    """One small stripe-motif local-region proposal.

    The proposal is meant to capture the QDM pattern seen in exact cages: a
    width-one stripe supplies the global organizing structure, but the coherent
    local object is often only a two- or three-plaquette motif on that stripe.
    ``source`` records whether the motif was cut from a straight stripe or from
    a snake-stripe cycle.
    """

    region: LocalQDMRegion
    plaquette_ids: npt.NDArray[np.int64]
    source: str
    source_index: int
    source_plaquette_ids: npt.NDArray[np.int64]
    motif_size: int
    motif_index: int

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

        source_ids = np.asarray(self.source_plaquette_ids, dtype=np.int64)
        if source_ids.ndim != 1:
            raise ValueError("source_plaquette_ids must be one-dimensional.")
        if source_ids.size == 0:
            raise ValueError("source_plaquette_ids must be non-empty.")
        object.__setattr__(self, "source_plaquette_ids", source_ids.copy())
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "motif_size", int(self.motif_size))
        object.__setattr__(self, "motif_index", int(self.motif_index))


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
class StripeMotifComponentRegionProposalRecord:
    """One merged stripe component selected by small local motif probes.

    The record represents the second-stage fast path: small two-/three-plaquette
    motifs are used only as cheap evidence that a stripe skeleton is promising;
    the emitted region is a larger component, usually the whole stripe/snake, so
    coherent units inside the stripe can cancel jointly instead of being forced
    into independent product blocks.
    """

    region: LocalQDMRegion
    plaquette_ids: npt.NDArray[np.int64]
    source: str
    source_index: int
    source_plaquette_ids: npt.NDArray[np.int64]
    component_size: int
    component_index: int
    n_seed_motifs: int
    seed_motif_plaquette_ids: tuple[tuple[int, ...], ...]
    seed_motif_signatures: tuple[tuple[int, int], ...]

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

        source_ids = np.asarray(self.source_plaquette_ids, dtype=np.int64)
        if source_ids.ndim != 1:
            raise ValueError("source_plaquette_ids must be one-dimensional.")
        if source_ids.size == 0:
            raise ValueError("source_plaquette_ids must be non-empty.")
        object.__setattr__(self, "source_plaquette_ids", source_ids.copy())
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "component_size", int(self.component_size))
        object.__setattr__(self, "component_index", int(self.component_index))
        object.__setattr__(self, "n_seed_motifs", int(self.n_seed_motifs))
        object.__setattr__(
            self,
            "seed_motif_plaquette_ids",
            tuple(tuple(int(pid) for pid in motif) for motif in self.seed_motif_plaquette_ids),
        )
        object.__setattr__(
            self,
            "seed_motif_signatures",
            tuple((int(kappa), int(potential)) for kappa, potential in self.seed_motif_signatures),
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
    stripe_motif_sizes: tuple[int, ...] = (2, 3)
    stripe_motif_sources: tuple[StripeMotifSource, ...] = ("stripe", "snake_stripe")
    stripe_motif_subset_mode: StripeMotifSubsetMode = "all"
    stripe_motif_max_motifs_per_stripe: int | None = None
    stripe_motif_component_sizes: tuple[int, ...] | None = None
    stripe_motif_component_subset_mode: StripeMotifComponentSubsetMode = "full"
    stripe_motif_component_min_seed_motifs: int = 1
    stripe_motif_component_max_seed_motifs_per_stripe: int | None = None
    stripe_motif_component_max_components_per_stripe: int | None = 1
    stripe_motif_component_motif_signatures: tuple[tuple[int, int], ...] | None = None
    stripe_widths: tuple[int, ...] = (1, 2)
    stripe_directions: tuple[int, ...] | None = None
    snake_stripe_max_turns: int | None = None
    snake_stripe_allow_kind_changes: bool = False
    snake_stripe_kind_pattern: SnakeStripeKindPattern = "constant_or_alternating"
    snake_stripe_require_induced_cycle: bool = False
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
        if not self.stripe_motif_sizes:
            raise ValueError("stripe_motif_sizes must be non-empty.")
        if any(int(size) <= 0 for size in self.stripe_motif_sizes):
            raise ValueError("stripe_motif_sizes must contain positive integers.")
        if not self.stripe_motif_sources:
            raise ValueError("stripe_motif_sources must be non-empty.")
        bad_motif_sources = [
            source
            for source in self.stripe_motif_sources
            if source not in {"stripe", "snake_stripe"}
        ]
        if bad_motif_sources:
            raise ValueError(f"Unsupported stripe motif sources: {bad_motif_sources}.")
        if self.stripe_motif_subset_mode not in {"windows", "all"}:
            raise ValueError("stripe_motif_subset_mode must be 'windows' or 'all'.")
        if (
            self.stripe_motif_max_motifs_per_stripe is not None
            and self.stripe_motif_max_motifs_per_stripe < 0
        ):
            raise ValueError("stripe_motif_max_motifs_per_stripe must be non-negative or None.")
        if self.stripe_motif_component_sizes is not None:
            component_sizes = tuple(int(size) for size in self.stripe_motif_component_sizes)
            if not component_sizes:
                raise ValueError("stripe_motif_component_sizes must be non-empty or None.")
            if any(size <= 0 for size in component_sizes):
                raise ValueError("stripe_motif_component_sizes must contain positive integers.")
            object.__setattr__(self, "stripe_motif_component_sizes", component_sizes)
        if self.stripe_motif_component_subset_mode not in {"full", "windows", "all"}:
            raise ValueError(
                "stripe_motif_component_subset_mode must be 'full', 'windows', or 'all'."
            )
        if self.stripe_motif_component_min_seed_motifs <= 0:
            raise ValueError("stripe_motif_component_min_seed_motifs must be positive.")
        if (
            self.stripe_motif_component_max_seed_motifs_per_stripe is not None
            and self.stripe_motif_component_max_seed_motifs_per_stripe < 0
        ):
            raise ValueError(
                "stripe_motif_component_max_seed_motifs_per_stripe must be non-negative or None."
            )
        if (
            self.stripe_motif_component_max_components_per_stripe is not None
            and self.stripe_motif_component_max_components_per_stripe < 0
        ):
            raise ValueError(
                "stripe_motif_component_max_components_per_stripe must be non-negative or None."
            )
        if self.stripe_motif_component_motif_signatures is not None:
            motif_signatures = tuple(
                (int(kappa), int(potential))
                for kappa, potential in self.stripe_motif_component_motif_signatures
            )
            object.__setattr__(
                self,
                "stripe_motif_component_motif_signatures",
                motif_signatures,
            )
        valid_strategies = {
            "stripe_motif",
            "stripe_motif_component",
            "stripe",
            "snake_stripe",
            "connected",
            "adaptive",
        }
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
        if self.snake_stripe_kind_pattern not in {
            "any",
            "constant",
            "alternating",
            "constant_or_alternating",
        }:
            raise ValueError(
                "snake_stripe_kind_pattern must be 'any', 'constant', 'alternating', "
                "or 'constant_or_alternating'."
            )
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
class FactorizedLocalQDMPadding:
    """Shared exterior for a product of local blocks without support expansion.

    Unlike :class:`MultiLocalQDMPadding`, this object never forms the Cartesian
    product of block support configurations.  Its memory cost is therefore
    independent of ``prod(block.support_size)``.
    """

    block_ids: tuple[int, ...]
    exterior_link_ids: npt.NDArray[np.int64]
    exterior_config: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        link_ids = np.asarray(self.exterior_link_ids, dtype=np.int64)
        config = np.asarray(self.exterior_config, dtype=np.int64)
        if link_ids.ndim != 1 or config.ndim != 1:
            raise ValueError("exterior_link_ids and exterior_config must be one-dimensional.")
        if link_ids.size != config.size:
            raise ValueError("exterior_config must have one value per exterior link.")
        if np.unique(link_ids).size != link_ids.size:
            raise ValueError("exterior_link_ids must not contain duplicates.")
        if np.any((config != 0) & (config != 1)):
            raise ValueError("exterior_config must be binary.")
        object.__setattr__(self, "block_ids", tuple(int(value) for value in self.block_ids))
        object.__setattr__(self, "exterior_link_ids", link_ids.copy())
        object.__setattr__(self, "exterior_config", config.copy())


@dataclass(frozen=True, slots=True)
class QDMFactorizedProductCertificationReport:
    """Polynomial-cost certificate for a separated product of QDM cage blocks."""

    block_ids: tuple[int, ...]
    padding: FactorizedLocalQDMPadding
    support_size: int
    kinetic_eigenvalue: complex
    self_loop_value: complex
    energy: complex
    kinetic_residual: float
    potential_residual: float
    hamiltonian_residual: float
    signature: tuple[int, int] | None
    n_kinetic_product_terms: int
    n_potential_product_terms: int
    max_blocks_touched_by_plaquette: int
    sector_validation: str
    failure_reason: str | None = None

    @property
    def is_certified(self) -> bool:
        return self.failure_reason is None and self.signature is not None

    @property
    def avoids_support_materialization(self) -> bool:
        return True

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "block_ids": self.block_ids,
            "support_size": self.support_size,
            "kinetic_eigenvalue": self.kinetic_eigenvalue,
            "self_loop_value": self.self_loop_value,
            "energy": self.energy,
            "kinetic_residual": self.kinetic_residual,
            "potential_residual": self.potential_residual,
            "hamiltonian_residual": self.hamiltonian_residual,
            "signature": self.signature,
            "n_kinetic_product_terms": self.n_kinetic_product_terms,
            "n_potential_product_terms": self.n_potential_product_terms,
            "max_blocks_touched_by_plaquette": self.max_blocks_touched_by_plaquette,
            "sector_validation": self.sector_validation,
            "failure_reason": self.failure_reason,
            "is_certified": self.is_certified,
            "avoids_support_materialization": self.avoids_support_materialization,
        }


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
class _FactorizedProductTerm:
    """One coefficient times a tensor product of sparse factor vectors."""

    coefficient: complex
    factors: tuple[dict[tuple[int, ...], complex], ...]


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
