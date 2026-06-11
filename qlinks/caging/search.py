"""High-level assembly API for interference-cage searches."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.linear_independence import IndependentColumnSelector
from qlinks.caging.partition import (
    type1_candidates_from_bipartite_self_loops,
    type2_candidates_from_self_loops,
)
from qlinks.caging.solver import (
    CageSolverConfig,
    CageState,
    solve_candidate_for_kinetic_targets,
)

CageSearchType = Literal["type1", "type2", "qdm", "qlm", "custom"]


@dataclass(frozen=True, slots=True)
class CageSearchConfig:
    """Configuration for the high-level cage-search workflow."""

    search_type: CageSearchType = "qlm"
    tolerance: float = 1e-10
    min_component_size: int = 2
    validate_full_residual: bool = True

    degenerate_basis_strategy: Literal["none", "ipr"] = "none"
    ipr_n_restarts: int = 128
    ipr_max_iter: int = 1000
    ipr_step_size: float = 0.1
    ipr_candidate_count: int = 64
    ipr_random_seed: int | None = None

    type1_kappas: tuple[int, ...] = (0,)
    type2_kappas: tuple[int, ...] = (-2, 2)

    deduplicate_by_rank: bool = True
    rank_tolerance_factor: float = 100.0
    signature_tolerance_factor: float = 10.0

    include_type1: bool | None = None
    include_type2: bool | None = None


@dataclass(frozen=True, slots=True)
class CageRecord:
    """One discovered cage state together with its metadata."""

    cage_state: CageState
    signature: tuple[int, int]
    candidate: CandidateSubgraph
    full_state: npt.NDArray[np.complex128] | None = None

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
class CageRecordView:
    """Indexable view into a subset of cage records."""

    records: Sequence[CageRecord]
    signature: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[CageRecord]:
        return iter(self.records)

    @overload
    def __getitem__(self, index: int) -> CageRecord: ...

    @overload
    def __getitem__(self, index: slice) -> list[CageRecord]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> CageRecord | list[CageRecord]:
        return self.records[index]

    def first(self) -> CageRecord:
        if len(self.records) == 0:
            if self.signature is None:
                raise ValueError("No cage records are available.")
            raise ValueError(f"No cage record found for signature {self.signature}.")
        return self.records[0]

    def to_list(self) -> list[CageRecord]:
        return list(self.records)


@dataclass
class CageSearchResult:
    """Result of a high-level cage search."""

    records: list[CageRecord]
    hilbert_size: int
    config: CageSearchConfig
    type1_candidates: list[CandidateSubgraph] = field(default_factory=list)
    type2_candidates: list[CandidateSubgraph] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[CageRecord]:
        return iter(self.records)

    @overload
    def __getitem__(self, index: int) -> CageRecord: ...

    @overload
    def __getitem__(self, index: slice) -> list[CageRecord]: ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> CageRecordView: ...

    @overload
    def __getitem__(
        self,
        index: tuple[tuple[int, int], int],
    ) -> CageRecord: ...

    @overload
    def __getitem__(
        self,
        index: tuple[tuple[int, int], slice],
    ) -> list[CageRecord]: ...

    def __getitem__(
        self,
        index: (
            int
            | slice
            | tuple[int, int]
            | tuple[tuple[int, int], int]
            | tuple[tuple[int, int], slice]
        ),
    ) -> CageRecord | list[CageRecord] | CageRecordView:
        """
        Index cage records.

        Supported forms
        ---------------
        result[i]
            Return the i-th record among all records.

        result[i:j]
            Return a list of records among all records.

        result[(kappa, z)]
            Return an indexable view of records with this signature.

        result[(kappa, z), i]
            Return the i-th record with this signature.

        result[(kappa, z), i:j]
            Return records with this signature in the given slice.
        """
        if isinstance(index, Integral):
            return self.records[int(index)]

        if isinstance(index, slice):
            return self.records[index]

        if _is_signature_index(index):
            signature = _normalize_signature(index)
            return CageRecordView(
                records=self.records_by_signature(signature),
                signature=signature,
            )

        if _is_signature_record_index(index):
            signature = _normalize_signature(index[0])
            record_index = index[1]
            records = self.records_by_signature(signature)
            return records[record_index]

        raise TypeError(
            "CageSearchResult indices must be an integer, a slice, "
            "a signature tuple like (kappa, Z), or "
            "a pair like ((kappa, Z), index)."
        )

    @property
    def counts_by_signature(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for record in self.records:
            counts[record.signature] = counts.get(record.signature, 0) + 1
        return counts

    @property
    def signatures(self) -> list[tuple[int, int]]:
        return sorted(self.counts_by_signature)

    def records_by_signature(
        self,
        signature: tuple[int, int],
    ) -> list[CageRecord]:
        signature = _normalize_signature(signature)
        return [record for record in self.records if record.signature == signature]

    def by_signature(
        self,
        signature: tuple[int, int],
    ) -> CageRecordView:
        """Return an indexable view of records with a fixed signature."""
        signature = _normalize_signature(signature)
        return CageRecordView(
            records=self.records_by_signature(signature),
            signature=signature,
        )

    def first(
        self,
        signature: tuple[int, int] | None = None,
    ) -> CageRecord:
        if signature is None:
            if len(self.records) == 0:
                raise ValueError("No cage records are available.")
            return self.records[0]

        return self.by_signature(signature).first()

    def full_state_matrix(
        self,
        signature: tuple[int, int] | None = None,
    ) -> npt.NDArray[np.complex128]:
        records = self.records if signature is None else self.records_by_signature(signature)

        full_matrix = np.zeros(
            (len(records), self.hilbert_size),
            dtype=np.complex128,
        )

        for row_index, record in enumerate(records):
            if record.full_state is not None:
                full_matrix[row_index, :] = record.full_state
            else:
                full_matrix[row_index, record.support] = record.local_state

        return full_matrix

    def cage_states(self) -> list[CageState]:
        return [record.cage_state for record in self.records]


@dataclass
class CageSearcher:
    """High-level cage-search assembly object."""

    hamiltonian_matrix: scipy_sparse.spmatrix | scipy_sparse.sparray
    kinetic_matrix: scipy_sparse.spmatrix | scipy_sparse.sparray
    self_loop_values: npt.NDArray[np.complex128]
    config: CageSearchConfig = field(default_factory=CageSearchConfig)

    @classmethod
    def from_model_build_result(
        cls,
        build_result,
        *,
        config: CageSearchConfig | None = None,
    ) -> CageSearcher:
        """Construct from a model build result containing H, K, and V."""
        if build_result.kinetic is None:
            raise ValueError("build_result.kinetic is required for cage search.")

        if build_result.potential is None:
            raise ValueError("build_result.potential is required for cage search.")

        return cls(
            hamiltonian_matrix=build_result.hamiltonian,
            kinetic_matrix=build_result.kinetic,
            self_loop_values=diagonal_values(build_result.potential),
            config=CageSearchConfig() if config is None else config,
        )

    def run(
        self,
        *,
        type1_candidates: list[CandidateSubgraph] | None = None,
        type2_candidates: list[CandidateSubgraph] | None = None,
    ) -> CageSearchResult:
        """Run the full cage-search workflow."""
        type1_enabled, type2_enabled = self._enabled_candidate_types()

        if type1_candidates is None:
            type1_candidates = self._build_type1_candidates() if type1_enabled else []

        if type2_candidates is None:
            type2_candidates = self._build_type2_candidates() if type2_enabled else []

        records: list[CageRecord] = []

        if type1_enabled:
            records.extend(
                self._solve_candidates(
                    candidates=type1_candidates,
                    allowed_kappas=self.config.type1_kappas,
                )
            )

        if type2_enabled:
            records.extend(
                self._solve_candidates(
                    candidates=type2_candidates,
                    allowed_kappas=self.config.type2_kappas,
                )
            )

        records = self._deduplicate_records_by_signature(records)

        return CageSearchResult(
            records=records,
            hilbert_size=int(self.hamiltonian_matrix.shape[0]),
            config=self.config,
            type1_candidates=type1_candidates,
            type2_candidates=type2_candidates,
        )

    def _enabled_candidate_types(self) -> tuple[bool, bool]:
        if self.config.include_type1 is not None:
            include_type1 = self.config.include_type1
        else:
            include_type1 = self.config.search_type in {
                "type1",
                "qdm",
                "qlm",
                "custom",
            }

        if self.config.include_type2 is not None:
            include_type2 = self.config.include_type2
        else:
            include_type2 = self.config.search_type in {
                "type2",
                "qlm",
                "custom",
            }

        return include_type1, include_type2

    def _build_type1_candidates(self) -> list[CandidateSubgraph]:
        bipartition = bipartition_labels(self.kinetic_matrix)

        return type1_candidates_from_bipartite_self_loops(
            self.kinetic_matrix,
            self.self_loop_values,
            bipartition,
            min_component_size=self.config.min_component_size,
        )

    def _build_type2_candidates(self) -> list[CandidateSubgraph]:
        return type2_candidates_from_self_loops(
            self.kinetic_matrix,
            self.self_loop_values,
            min_component_size=self.config.min_component_size,
        )

    def _solve_candidates(
        self,
        *,
        candidates: list[CandidateSubgraph],
        allowed_kappas: tuple[int, ...],
    ) -> list[CageRecord]:
        solver_config = CageSolverConfig(
            tolerance=self.config.tolerance,
            validate_full_residual=self.config.validate_full_residual,
            degenerate_basis_strategy=self.config.degenerate_basis_strategy,
            ipr_n_restarts=self.config.ipr_n_restarts,
            ipr_max_iter=self.config.ipr_max_iter,
            ipr_step_size=self.config.ipr_step_size,
            ipr_candidate_count=self.config.ipr_candidate_count,
            ipr_random_seed=self.config.ipr_random_seed,
        )

        records: list[CageRecord] = []

        for candidate in candidates:
            cage_states = solve_candidate_for_kinetic_targets(
                self.hamiltonian_matrix,
                self.kinetic_matrix,
                self.self_loop_values,
                candidate,
                target_kappas=tuple(complex(kappa) for kappa in allowed_kappas),
                config=solver_config,
            )

            for cage_state in cage_states:
                self_loop_value = self.self_loop_values[candidate.vertices[0]]
                signature = signature_from_energy_and_self_loop(
                    cage_state.energy,
                    self_loop_value,
                    tolerance=(self.config.signature_tolerance_factor * self.config.tolerance),
                )

                if signature is None:
                    continue

                kinetic_value, _potential_value = signature

                if kinetic_value not in allowed_kappas:
                    continue

                full_state = embed_cage_state(
                    cage_state,
                    hilbert_size=int(self.hamiltonian_matrix.shape[0]),
                )

                records.append(
                    CageRecord(
                        cage_state=cage_state,
                        signature=signature,
                        candidate=candidate,
                        full_state=full_state,
                    )
                )

        return records

    def _deduplicate_records_by_signature(
        self,
        records: list[CageRecord],
    ) -> list[CageRecord]:
        if not self.config.deduplicate_by_rank:
            return records

        grouped_records: dict[tuple[int, int], list[CageRecord]] = defaultdict(list)
        grouped_selectors: dict[tuple[int, int], IndependentColumnSelector] = {}

        rank_tolerance = self.config.rank_tolerance_factor * self.config.tolerance

        for record in records:
            selector = grouped_selectors.setdefault(
                record.signature,
                IndependentColumnSelector(tolerance=rank_tolerance),
            )
            full_state = (
                record.full_state
                if record.full_state is not None
                else embed_cage_state(
                    record.cage_state,
                    hilbert_size=int(self.hamiltonian_matrix.shape[0]),
                )
            )

            if selector.add(full_state):
                grouped_records[record.signature].append(record)

        deduplicated: list[CageRecord] = []

        for signature in sorted(grouped_records):
            deduplicated.extend(grouped_records[signature])

        return deduplicated


def diagonal_values(matrix) -> npt.NDArray[np.complex128]:
    """Return diagonal values from a dense or sparse matrix."""
    if scipy_sparse.issparse(matrix):
        values = matrix.diagonal()
    else:
        values = np.diag(matrix)

    return np.asarray(values, dtype=np.complex128)


def embed_cage_state(
    cage_state: CageState,
    *,
    hilbert_size: int,
) -> npt.NDArray[np.complex128]:
    """Embed a compact cage state into the full Hilbert space."""
    full_state = np.zeros(hilbert_size, dtype=np.complex128)
    full_state[cage_state.support] = cage_state.local_state

    norm = np.linalg.norm(full_state)

    if norm == 0:
        raise ValueError("Cannot embed a zero cage state.")

    return full_state / norm


def signature_from_energy_and_self_loop(
    energy_value: complex,
    self_loop_value: complex,
    *,
    tolerance: float,
) -> tuple[int, int] | None:
    """Infer integer ``(kappa, Z)`` signature from energy and self-loop value."""
    potential_value = int(round(float(np.real(self_loop_value))))
    kinetic_value = float(np.real(energy_value - self_loop_value))
    kinetic_integer = int(round(kinetic_value))

    if not np.isclose(
        float(np.real(self_loop_value)),
        potential_value,
        atol=tolerance,
        rtol=0.0,
    ):
        return None

    if not np.isclose(
        kinetic_value,
        kinetic_integer,
        atol=tolerance,
        rtol=0.0,
    ):
        return None

    return kinetic_integer, potential_value


def bipartition_labels(matrix) -> npt.NDArray[np.int64]:
    """Return bipartition labels for an undirected sparse graph."""
    adjacency = scipy_sparse.csr_array(matrix).copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    adjacency = adjacency.maximum(adjacency.T).tocsr()

    n_vertices = adjacency.shape[0]
    labels = -np.ones(n_vertices, dtype=np.int64)

    for start_index in range(n_vertices):
        if labels[start_index] != -1:
            continue

        labels[start_index] = 0
        queue: deque[int] = deque([start_index])

        while queue:
            vertex_index = queue.popleft()
            neighbors = adjacency.indices[
                adjacency.indptr[vertex_index] : adjacency.indptr[vertex_index + 1]
            ]

            for neighbor_index in neighbors:
                neighbor_index = int(neighbor_index)

                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = 1 - labels[vertex_index]
                    queue.append(neighbor_index)
                elif labels[neighbor_index] == labels[vertex_index]:
                    raise ValueError("Graph is not bipartite.")

    return labels


def _is_signature_index(index: object) -> bool:
    """Return whether index looks like a cage signature ``(kappa, Z)``."""
    return (
        isinstance(index, tuple)
        and len(index) == 2
        and isinstance(index[0], Integral)
        and isinstance(index[1], Integral)
    )


def _is_signature_record_index(index: object) -> bool:
    """Return whether index looks like ``((kappa, Z), record_index)``."""
    return (
        isinstance(index, tuple)
        and len(index) == 2
        and _is_signature_index(index[0])
        and isinstance(index[1], (Integral, slice))
    )


def _normalize_signature(signature: tuple[int, int]) -> tuple[int, int]:
    """Convert numpy integer signatures to plain Python integer tuples."""
    if not _is_signature_index(signature):
        raise TypeError("Signature must be a tuple of two integers: (kappa, Z).")
    return int(signature[0]), int(signature[1])
