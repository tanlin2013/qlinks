"""Generic local-first cage-search algebra and adapter registry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.basis import Basis
from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.local_search_types import (
    CertifiedLocalQDMCageSearchResult,
    LocalCageModelAdapter,
    LocalQDMCageRecord,
    LocalQDMCageSearchConfig,
    LocalQDMRegion,
)
from qlinks.caging.partition import type1_candidates_from_bipartite_self_loops
from qlinks.caging.search import bipartition_labels, signature_from_energy_and_self_loop
from qlinks.caging.solver import CageSolverConfig, solve_candidate_for_kinetic_targets


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


LocalCageAdapterFactory = Callable[[object], LocalCageModelAdapter | None]
_LOCAL_CAGE_ADAPTER_FACTORIES: list[LocalCageAdapterFactory] = []

LocalCageSearchConfig = LocalQDMCageSearchConfig
LocalCageRegion = LocalQDMRegion
LocalCageRecord = LocalQDMCageRecord
LocalCageSearchResult = LocalQDMCageSearchResult
CertifiedLocalCageSearchResult = CertifiedLocalQDMCageSearchResult
