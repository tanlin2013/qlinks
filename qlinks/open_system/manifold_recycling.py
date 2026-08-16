"""Local recycler construction and recycled-manifold kernel diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.open_system.manifold_dark import (
    _append_ranked_recycled_candidate,
    _as_csr,
    _combined_operator,
    _default_detector_name,
    _diagonal_vector_if_diagonal,
    _embedded_local_operator_metrics_with_diagonal_right_factor,
    _embedded_local_operator_times_diagonal_as_csr,
    _embedded_matrix_unit_metrics_with_diagonal_right_factor,
    _embedded_matrix_unit_times_diagonal_as_csr,
    _multi_jump_projected_inflow_norm,
    _normalize_detector_coefficients,
    _normalize_state_columns,
    _orthogonal_complement_basis,
    _projected_inflow_norm,
    _recycled_candidate_sort_key,
    _right_kernel_basis,
)
from qlinks.open_system.manifold_detector_types import (
    LocalOperatorMatrixReadout,
    ManifoldDarkOperatorBasisReport,
    RecycledManifoldCandidateFamilyKernelReport,
    RecycledManifoldCollectiveRecyclerGroup,
    RecycledManifoldDarkDetectorCandidate,
    RecycledManifoldDarkDetectorReport,
    RecycledManifoldJumpSelectionStep,
)


@dataclass(frozen=True, slots=True)
class RecycledManifoldJumpSelectionReport:
    """Greedy small-subset selection report for recycled dark-detector jumps.

    The report owns the selected jump operators.  The summary intentionally omits
    raw sparse matrices, but ``report.jumps`` can be passed directly to
    :func:`qlinks.open_system.diagnose_dark_manifold` or a Lindblad solver.
    The stopping criterion uses the common jump kernel in the complement of the
    target manifold.  Reaching zero is a strong sufficient condition that no
    complement vector is dark under all selected jumps.
    """

    manifold_dimension: int
    hilbert_dimension: int
    candidate_pool_size: int
    max_selected_jumps: int
    target_bad_kernel_dimension: int
    dark_tolerance: float
    inflow_tolerance: float
    jumps: tuple[sp.csr_array, ...]
    steps: tuple[RecycledManifoldJumpSelectionStep, ...]
    final_diagnostics: Any | None
    candidate_report: RecycledManifoldDarkDetectorReport
    candidate_report_was_expanded: bool = False
    candidate_pool_was_limited: bool = False
    compression_strategy: str = "none"
    n_compression_passes: int = 0
    n_compressed_jumps_removed: int = 0
    collective_recycler_strategy: str = "none"
    unbundled_n_jumps: int | None = None
    collective_groups: tuple[RecycledManifoldCollectiveRecyclerGroup, ...] = ()
    selected_inflow_norm: float | None = None
    unbundled_inflow_norm: float | None = None

    @property
    def n_selected_jumps(self) -> int:
        return len(self.jumps)

    @property
    def n_unbundled_jumps(self) -> int:
        if self.unbundled_n_jumps is not None:
            return int(self.unbundled_n_jumps)
        return len(self.steps)

    @property
    def n_collective_groups(self) -> int:
        return len(self.collective_groups)

    @property
    def n_bundled_recyclers(self) -> int:
        return int(sum(group.n_bundled_recyclers for group in self.collective_groups))

    @property
    def collective_jump_reduction(self) -> int:
        return max(self.n_unbundled_jumps - self.n_selected_jumps, 0)

    @property
    def uses_collective_recyclers(self) -> bool:
        return bool(self.collective_groups)

    @property
    def selected_candidates(self) -> tuple[RecycledManifoldDarkDetectorCandidate, ...]:
        return tuple(step.candidate for step in self.steps)

    @property
    def n_reported_candidates(self) -> int:
        return len(self.candidate_report.candidates)

    @property
    def n_tested_candidates(self) -> int:
        return int(self.candidate_report.n_tested_candidates)

    @property
    def n_nonzero_candidates(self) -> int:
        return int(self.candidate_report.n_nonzero_candidates)

    @property
    def candidate_report_is_truncated(self) -> bool:
        return self.candidate_report.candidate_report_is_truncated

    @property
    def candidate_pool_is_truncated(self) -> bool:
        return bool(self.candidate_pool_was_limited)

    @property
    def stopped_with_available_candidates(self) -> bool:
        return (
            self.final_bad_common_jump_kernel_dimension is not None
            and self.final_bad_common_jump_kernel_dimension > self.target_bad_kernel_dimension
            and self.candidate_pool_is_truncated
        )

    @property
    def final_bad_kernel_iprs(self) -> tuple[float, ...]:
        if self.final_diagnostics is None:
            return ()
        return tuple(float(value) for value in self.final_diagnostics.bad_common_jump_kernel_iprs)

    @property
    def final_bad_kernel_ipr_min(self) -> float | None:
        values = self.final_bad_kernel_iprs
        if len(values) == 0:
            return None
        return float(min(values))

    @property
    def final_bad_kernel_ipr_max(self) -> float | None:
        values = self.final_bad_kernel_iprs
        if len(values) == 0:
            return None
        return float(max(values))

    @property
    def final_bad_common_jump_kernel_dimension(self) -> int | None:
        if self.final_diagnostics is None:
            return None
        return int(self.final_diagnostics.bad_common_jump_kernel_dimension)

    @property
    def final_inflow_norm(self) -> float | None:
        if self.final_diagnostics is not None:
            return float(self.final_diagnostics.inflow_norm)
        if self.selected_inflow_norm is None:
            return None
        return float(self.selected_inflow_norm)

    @property
    def collective_inflow_ratio(self) -> float | None:
        if (
            self.selected_inflow_norm is None
            or self.unbundled_inflow_norm is None
            or self.unbundled_inflow_norm <= 0.0
        ):
            return None
        return float(self.selected_inflow_norm / self.unbundled_inflow_norm)

    @property
    def complement_common_kernel_removed(self) -> bool | None:
        final_bad = self.final_bad_common_jump_kernel_dimension
        if final_bad is None:
            return None
        return final_bad <= self.target_bad_kernel_dimension

    @property
    def total_jump_nnz(self) -> int:
        return int(sum(jump.nnz for jump in self.jumps))

    @property
    def max_jump_nnz(self) -> int:
        return int(max((jump.nnz for jump in self.jumps), default=0))

    @property
    def selected_region_indices(self) -> tuple[int, ...]:
        if self.collective_groups:
            return tuple(group.region_index for group in self.collective_groups)
        return tuple(step.candidate.region_index for step in self.steps)

    @property
    def selected_detector_indices(self) -> tuple[int, ...]:
        if self.collective_groups:
            return tuple(group.detector_index for group in self.collective_groups)
        return tuple(step.candidate.detector_index for step in self.steps)

    def selected_recycler_readouts(
        self,
        *,
        basis_configs: npt.NDArray[np.integer],
        states: npt.ArrayLike | None = None,
        max_readouts: int | None = None,
        tolerance: float = 1.0e-10,
        rdm_tolerance: float = 1.0e-10,
    ) -> tuple[LocalOperatorMatrixReadout, ...]:
        """Return local-matrix readouts for selected recycled jump recyclers.

        The returned objects can be passed directly to
        ``LocalBasisGridVisualizer.plot_readout``.  For
        ``rdm_support_matrix_units`` recyclers, pass the target state/manifold
        through ``states`` so the local RDM support basis can be reconstructed.
        """
        if self.collective_groups:
            groups = self.collective_groups
            if max_readouts is not None:
                groups = groups[: max(int(max_readouts), 0)]
            return tuple(
                _local_operator_from_collective_recycler_group(
                    group=group,
                    basis_configs=basis_configs,
                )
                for group in groups
            )

        candidates = self.selected_candidates
        if max_readouts is not None:
            candidates = candidates[: max(int(max_readouts), 0)]
        return tuple(
            _local_operator_from_recycler_candidate(
                candidate=candidate,
                basis_configs=basis_configs,
                states=states,
                recycler_source=self.candidate_report.recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
            )
            for candidate in candidates
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "hilbert_dimension": self.hilbert_dimension,
            "candidate_pool_size": self.candidate_pool_size,
            "n_reported_candidates": self.n_reported_candidates,
            "n_nonzero_candidates": self.n_nonzero_candidates,
            "n_tested_candidates": self.n_tested_candidates,
            "candidate_report_was_expanded": self.candidate_report_was_expanded,
            "candidate_pool_was_limited": self.candidate_pool_was_limited,
            "compression_strategy": self.compression_strategy,
            "n_compression_passes": self.n_compression_passes,
            "n_compressed_jumps_removed": self.n_compressed_jumps_removed,
            "candidate_report_is_truncated": self.candidate_report_is_truncated,
            "candidate_pool_is_truncated": self.candidate_pool_is_truncated,
            "stopped_with_available_candidates": self.stopped_with_available_candidates,
            "max_selected_jumps": self.max_selected_jumps,
            "target_bad_kernel_dimension": self.target_bad_kernel_dimension,
            "n_selected_jumps": self.n_selected_jumps,
            "n_unbundled_jumps": self.n_unbundled_jumps,
            "collective_recycler_strategy": self.collective_recycler_strategy,
            "uses_collective_recyclers": self.uses_collective_recyclers,
            "n_collective_groups": self.n_collective_groups,
            "n_bundled_recyclers": self.n_bundled_recyclers,
            "collective_jump_reduction": self.collective_jump_reduction,
            "selected_inflow_norm": self.selected_inflow_norm,
            "unbundled_inflow_norm": self.unbundled_inflow_norm,
            "collective_inflow_ratio": self.collective_inflow_ratio,
            "collective_groups": tuple(group.to_summary_dict() for group in self.collective_groups),
            "selected_region_indices": self.selected_region_indices,
            "selected_detector_indices": self.selected_detector_indices,
            "total_jump_nnz": self.total_jump_nnz,
            "max_jump_nnz": self.max_jump_nnz,
            "final_bad_common_jump_kernel_dimension": (self.final_bad_common_jump_kernel_dimension),
            "final_inflow_norm": self.final_inflow_norm,
            "complement_common_kernel_removed": self.complement_common_kernel_removed,
            "final_bad_kernel_ipr_min": self.final_bad_kernel_ipr_min,
            "final_bad_kernel_ipr_max": self.final_bad_kernel_ipr_max,
            "dark_tolerance": self.dark_tolerance,
            "inflow_tolerance": self.inflow_tolerance,
            "steps": tuple(step.to_summary_dict() for step in self.steps),
            "final_diagnostics": (
                None if self.final_diagnostics is None else self.final_diagnostics.to_summary_dict()
            ),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self, *, max_steps: int = 24):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "RecycledManifoldJumpSelectionReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("candidate pool", str(self.candidate_pool_size))
        overview.add_row(
            "reported/nonzero/tested candidates",
            f"{self.n_reported_candidates}/{self.n_nonzero_candidates}/{self.n_tested_candidates}",
        )
        overview.add_row("expanded report", str(self.candidate_report_was_expanded))
        overview.add_row("pool truncated", str(self.candidate_pool_is_truncated))
        overview.add_row("selected jumps", str(self.n_selected_jumps))
        overview.add_row(
            "compression", f"{self.compression_strategy}, removed={self.n_compressed_jumps_removed}"
        )
        overview.add_row("target bad-kernel dim", str(self.target_bad_kernel_dimension))
        overview.add_row(
            "final bad-kernel dim",
            (
                "not checked"
                if self.final_bad_common_jump_kernel_dimension is None
                else str(self.final_bad_common_jump_kernel_dimension)
            ),
        )
        overview.add_row("complement kernel removed", str(self.complement_common_kernel_removed))
        overview.add_row(
            "final inflow",
            "not checked" if self.final_inflow_norm is None else f"{self.final_inflow_norm:.3e}",
        )
        overview.add_row("total jump nnz", str(self.total_jump_nnz))
        overview.add_row(
            "stopped with available candidates",
            str(self.stopped_with_available_candidates),
        )

        table = Table(title="Greedy selected recycled dark-detector jumps")
        table.add_column("step", justify="right")
        table.add_column("detector")
        table.add_column("region")
        table.add_column("recycler")
        table.add_column("candidate inflow", justify="right")
        table.add_column("bad kernel", justify="right")
        table.add_column("selected", justify="right")
        table.add_column("jump nnz", justify="right")

        for step in self.steps[: max(int(max_steps), 0)]:
            candidate = step.candidate
            table.add_row(
                str(step.step_index),
                candidate.detector_name,
                str(candidate.variable_indices),
                candidate.recycler_name,
                f"{candidate.inflow_norm:.3e}",
                str(step.bad_common_jump_kernel_dimension),
                str(step.n_selected_jumps),
                str(candidate.jump_nnz),
                style=(
                    "green"
                    if step.bad_common_jump_kernel_dimension <= self.target_bad_kernel_dimension
                    else ""
                ),
            )

        if len(self.steps) > max_steps:
            table.add_row(
                "…",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{len(self.steps) - max_steps} more steps",
            )

        return Panel(
            Group(overview, table),
            title=Text("Recycled manifold jump-selection report", style="bold green"),
            border_style="green",
        )


@dataclass(frozen=True, slots=True)
class RecycledFamilyKernelDiagnostics:
    """Lightweight dark-manifold kernel diagnostics for a streamed jump family.

    This mirrors the common-kernel fields of :class:`DarkManifoldDiagnostics`
    without requiring every jump operator to be materialized at once.  It is
    intended for large recycled-detector families where the decisive question is
    whether the whole family removes the complement common jump kernel.
    """

    dim: int
    n_jumps: int
    manifold_dimension: int
    hamiltonian_closure_residual: float
    max_target_jump_residual: float
    target_density_liouvillian_residual: float
    inflow_norm: float
    common_jump_kernel_dimension: int
    target_projection_onto_common_kernel: float
    target_distance_from_common_kernel: float
    target_in_common_jump_kernel: bool
    bad_common_jump_kernel_dimension: int
    bad_common_jump_kernel_iprs: tuple[float, ...]
    internal_hamiltonian_eigenvalues: tuple[complex, ...]
    expected_internal_zero_mode_count: int
    expected_internal_peripheral_mode_count: int
    liouvillian_zero_tolerance: float

    @property
    def target_jump_residuals(self) -> tuple[float, ...]:
        # The streamed diagnostic keeps only the maximum residual to avoid
        # storing one value per candidate jump.
        return ()

    @property
    def expected_internal_liouvillian_eigenvalues(self) -> tuple[complex, ...]:
        return _internal_liouvillian_eigenvalues_from_energies(
            self.internal_hamiltonian_eigenvalues
        )

    @property
    def liouvillian_zero_mode_count(self) -> None:
        return None

    @property
    def liouvillian_zero_mode_count_is_lower_bound(self) -> bool:
        return False

    @property
    def liouvillian_spectral_gap(self) -> None:
        return None

    @property
    def liouvillian_decay_gap(self) -> None:
        return None

    @property
    def liouvillian_peripheral_mode_count(self) -> None:
        return None

    @property
    def liouvillian_spectrum_method(self) -> str:
        return "streamed_kernel"

    @property
    def liouvillian_eigenvalues(self) -> tuple[complex, ...]:
        return ()

    @property
    def matched_internal_nondecaying_mode_count(self) -> None:
        return None

    @property
    def missing_internal_nondecaying_mode_count(self) -> None:
        return None

    @property
    def extra_nondecaying_mode_count(self) -> None:
        return None

    @property
    def extra_zero_mode_count(self) -> None:
        return None

    @property
    def external_decay_gap(self) -> None:
        return None

    @property
    def likely_attractive_dark_manifold(self) -> None:
        return None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "manifold_dimension": self.manifold_dimension,
            "h_closure_residual": self.hamiltonian_closure_residual,
            "max_target_jump_residual": self.max_target_jump_residual,
            "target_density_liouvillian_residual": self.target_density_liouvillian_residual,
            "inflow_norm": self.inflow_norm,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "target_projection_onto_common_kernel": self.target_projection_onto_common_kernel,
            "target_distance_from_common_kernel": self.target_distance_from_common_kernel,
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "bad_common_jump_kernel_dimension": self.bad_common_jump_kernel_dimension,
            "bad_common_jump_kernel_iprs": self.bad_common_jump_kernel_iprs,
            "internal_hamiltonian_eigenvalues": [
                complex(value) for value in self.internal_hamiltonian_eigenvalues
            ],
            "expected_internal_zero_mode_count": self.expected_internal_zero_mode_count,
            "expected_internal_peripheral_mode_count": (
                self.expected_internal_peripheral_mode_count
            ),
            "liouvillian_zero_mode_count": None,
            "liouvillian_zero_mode_count_is_lower_bound": False,
            "liouvillian_spectral_gap": None,
            "liouvillian_decay_gap": None,
            "liouvillian_peripheral_mode_count": None,
            "liouvillian_spectrum_method": self.liouvillian_spectrum_method,
            "matched_internal_nondecaying_mode_count": None,
            "missing_internal_nondecaying_mode_count": None,
            "extra_nondecaying_mode_count": None,
            "extra_zero_mode_count": None,
            "external_decay_gap": None,
            "likely_attractive_dark_manifold": None,
        }


def _normalize_local_regions(
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
) -> tuple[tuple[int, ...], ...]:
    regions: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for region_like in local_regions:
        region = tuple(sorted(int(index) for index in region_like))
        if len(region) == 0:
            raise ValueError("local_regions must not contain empty regions.")
        if region in seen:
            continue
        seen.add(region)
        regions.append(region)
    if len(regions) == 0:
        raise ValueError("local_regions must contain at least one region.")
    return tuple(regions)


def expand_local_regions_to_pair_unions(
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    *,
    pair_mode: Literal["overlap", "all"] = "overlap",
    min_overlap: int = 1,
    max_region_size: int | None = None,
    include_single_regions: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Return unions of pairs of local regions, useful for two-block recyclers.

    The single-plaquette recycled-detector family can leave residual bad kernel
    sectors when the target manifold is built from two-plaquette singlet-like
    structures.  This helper creates bounded two-region supports that can be
    supplied to ``diagnose_recycled_manifold_dark_detectors`` or
    ``diagnose_recycled_manifold_candidate_family_kernel`` as ``local_regions``.

    Args:
        local_regions: Base local regions, usually plaquette supports.
        pair_mode: ``"overlap"`` keeps only pairs sharing at least
            ``min_overlap`` variables; ``"all"`` keeps all unordered pairs.
        min_overlap: Minimum number of shared variables when
            ``pair_mode="overlap"``.
        max_region_size: Optional upper bound on the size of the union.
        include_single_regions: Whether to prepend the original regions.

    Returns:
        Deduplicated sorted variable-index unions.
    """
    if pair_mode not in {"overlap", "all"}:
        raise ValueError('pair_mode must be "overlap" or "all".')
    if min_overlap < 0:
        raise ValueError("min_overlap must be non-negative.")
    if max_region_size is not None and max_region_size <= 0:
        raise ValueError("max_region_size must be positive when provided.")

    base_regions = _normalize_local_regions(local_regions)
    expanded: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def maybe_add(region: tuple[int, ...]) -> None:
        if max_region_size is not None and len(region) > max_region_size:
            return
        if region in seen:
            return
        seen.add(region)
        expanded.append(region)

    if include_single_regions:
        for region in base_regions:
            maybe_add(region)

    region_sets = [set(region) for region in base_regions]
    for left_index, left in enumerate(base_regions):
        left_set = region_sets[left_index]
        for right_index in range(left_index + 1, len(base_regions)):
            right = base_regions[right_index]
            overlap = len(left_set.intersection(region_sets[right_index]))
            if pair_mode == "overlap" and overlap < min_overlap:
                continue
            maybe_add(tuple(sorted(set(left).union(right))))

    if len(expanded) == 0:
        raise ValueError(
            "No pair-union regions were generated. Relax pair_mode/min_overlap "
            "or max_region_size, or pass include_single_regions=True."
        )

    return tuple(expanded)


@lru_cache(maxsize=256)
def _expand_normalized_local_regions_to_cluster_unions_cached(
    base_regions: tuple[tuple[int, ...], ...],
    *,
    cluster_size: int,
    cluster_mode: Literal["overlap_connected", "all"],
    min_overlap: int,
    max_region_size: int | None,
    include_single_regions: bool,
    include_smaller_clusters: bool,
) -> tuple[tuple[int, ...], ...]:
    expanded: list[tuple[int, ...]] = []
    seen_regions: set[tuple[int, ...]] = set()

    def maybe_add(region: tuple[int, ...]) -> None:
        if max_region_size is not None and len(region) > max_region_size:
            return
        if region in seen_regions:
            return
        seen_regions.add(region)
        expanded.append(region)

    if include_single_regions:
        for region in base_regions:
            maybe_add(region)

    region_sets = tuple(frozenset(region) for region in base_regions)
    cluster_sizes = tuple(
        range(2, cluster_size + 1) if include_smaller_clusters else (cluster_size,)
    )

    if cluster_mode == "all":
        for size in cluster_sizes:
            for indices in combinations(range(len(base_regions)), size):
                union: set[int] = set()
                for index in indices:
                    union.update(base_regions[index])
                maybe_add(tuple(sorted(union)))
        return tuple(expanded)

    # Optimized connected-cluster enumeration.  The previous implementation
    # checked every k-combination and then tested overlap connectivity.  That is
    # acceptable for small square/honeycomb runs, but model-regional-unit modes
    # can have many plaquette/bond units.  Enumerating connected clusters by
    # growing along the overlap graph avoids most disconnected combinations.
    neighbors: tuple[tuple[int, ...], ...] = tuple(
        tuple(
            right_index
            for right_index, right_set in enumerate(region_sets)
            if right_index != left_index and len(left_set.intersection(right_set)) >= min_overlap
        )
        for left_index, left_set in enumerate(region_sets)
    )
    target_sizes = set(int(size) for size in cluster_sizes)
    max_cluster_size = max(target_sizes, default=1)
    seen_index_clusters: set[tuple[int, ...]] = set()

    def grow(
        *,
        seed: int,
        cluster: tuple[int, ...],
        union: frozenset[int],
    ) -> None:
        if len(cluster) in target_sizes:
            key = tuple(sorted(cluster))
            if key not in seen_index_clusters:
                seen_index_clusters.add(key)
                maybe_add(tuple(sorted(union)))
        if len(cluster) >= max_cluster_size:
            return

        cluster_set = set(cluster)
        frontier = sorted(
            {
                neighbor
                for index in cluster
                for neighbor in neighbors[index]
                if neighbor > seed and neighbor not in cluster_set
            }
        )
        for neighbor in frontier:
            new_union = frozenset(set(union).union(base_regions[neighbor]))
            if max_region_size is not None and len(new_union) > max_region_size:
                continue
            grow(
                seed=seed,
                cluster=tuple(sorted(cluster + (neighbor,))),
                union=new_union,
            )

    for seed in range(len(base_regions)):
        grow(seed=seed, cluster=(seed,), union=frozenset(base_regions[seed]))

    return tuple(expanded)


def expand_local_regions_to_cluster_unions(
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    *,
    cluster_size: int = 3,
    cluster_mode: Literal["overlap_connected", "all"] = "overlap_connected",
    min_overlap: int = 1,
    max_region_size: int | None = None,
    include_single_regions: bool = False,
    include_smaller_clusters: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Return bounded unions of multiple local regions.

    This generalizes :func:`expand_local_regions_to_pair_unions` to one or
    more regional units.  ``cluster_size=1`` returns the normalized base
    regions themselves, which is useful when each base region is already a
    model-natural non-onsite unit such as a plaquette, rhombus, hexagon, or
    bond.  ``cluster_mode="overlap_connected"`` keeps larger clusters that are
    connected in the overlap graph of the base regions; this is the natural
    setting for connected multi-plaquette QDM patches.

    The normalized expansion is cached and the connected mode grows clusters on
    the overlap graph directly.  This makes repeated
    ``local_region_mode="regional_unit_clusters"`` calls much cheaper in
    notebook sweeps, especially when trying several detector/recycler settings
    with the same model-natural plaquette or bond units.
    """
    if cluster_mode not in {"overlap_connected", "all"}:
        raise ValueError('cluster_mode must be "overlap_connected" or "all".')
    if cluster_size < 1:
        raise ValueError("cluster_size must be at least one.")
    if min_overlap < 0:
        raise ValueError("min_overlap must be non-negative.")
    if max_region_size is not None and max_region_size <= 0:
        raise ValueError("max_region_size must be positive when provided.")

    base_regions = _normalize_local_regions(local_regions)
    expanded = _expand_normalized_local_regions_to_cluster_unions_cached(
        base_regions,
        cluster_size=int(cluster_size),
        cluster_mode=cluster_mode,
        min_overlap=int(min_overlap),
        max_region_size=None if max_region_size is None else int(max_region_size),
        include_single_regions=bool(include_single_regions),
        include_smaller_clusters=bool(include_smaller_clusters),
    )

    if len(expanded) == 0:
        raise ValueError(
            "No cluster-union regions were generated. Relax cluster_mode/min_overlap "
            "or max_region_size, reduce cluster_size, or pass include_single_regions=True."
        )

    return expanded


def _pattern_name(pattern: tuple[int, ...]) -> str:
    return "(" + ",".join(str(int(value)) for value in pattern) + ")"


def _local_patterns_from_basis_configs(
    *,
    basis_configs: npt.NDArray[np.integer],
    variable_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")
    if len(variable_indices) == 0:
        raise ValueError("variable_indices must be nonempty.")
    if any(index < 0 or index >= basis_array.shape[1] for index in variable_indices):
        raise ValueError("variable_indices contains out-of-range entries.")
    variable_index_array = np.asarray(variable_indices, dtype=np.int64)
    return tuple(
        sorted(
            {tuple(int(value) for value in config[variable_index_array]) for config in basis_array}
        )
    )


def _matrix_unit_local_operator(
    *,
    local_dim: int,
    recycler_index: int,
) -> npt.NDArray[np.complex128]:
    if local_dim <= 0:
        raise ValueError("local_dim must be positive.")
    n_matrix_units = int(local_dim) * int(local_dim)
    if recycler_index < 0 or recycler_index >= n_matrix_units:
        raise ValueError("recycler_index is out of range for matrix-unit recyclers.")
    target_index, source_index = divmod(int(recycler_index), int(local_dim))
    local_operator = np.zeros((int(local_dim), int(local_dim)), dtype=np.complex128)
    local_operator[target_index, source_index] = 1.0
    return local_operator


def _local_operator_from_matrix_unit_terms(
    *,
    local_patterns: tuple[tuple[int, ...], ...],
    terms: tuple[Any, ...],
) -> npt.NDArray[np.complex128]:
    pattern_to_index = {pattern: index for index, pattern in enumerate(local_patterns)}
    local_dim = len(local_patterns)
    local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
    for term in terms:
        name = str(term.operator_name)
        coefficient = complex(term.coefficient)
        if "<-" not in name:
            raise ValueError(
                f"Cannot build a matrix readout from non-matrix-unit term name {name!r}."
            )
        target_text, source_text = name.split("<-", 1)
        target_pattern = tuple(
            int(value) for value in target_text.strip()[1:-1].split(",") if value
        )
        source_pattern = tuple(
            int(value) for value in source_text.strip()[1:-1].split(",") if value
        )
        try:
            target_index = pattern_to_index[target_pattern]
            source_index = pattern_to_index[source_pattern]
        except KeyError as exc:
            raise ValueError(
                "matrix-unit term contains a pattern that is absent from local_patterns."
            ) from exc
        local_operator[target_index, source_index] += coefficient
    return local_operator


def _local_operator_from_recycler_candidate(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    basis_configs: npt.NDArray[np.integer],
    states: npt.ArrayLike | None,
    recycler_source: Literal["matrix_units", "rdm_support_matrix_units"],
    tolerance: float,
    rdm_tolerance: float,
) -> LocalOperatorMatrixReadout:
    variable_indices = tuple(int(value) for value in candidate.variable_indices)
    local_patterns = _local_patterns_from_basis_configs(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    if len(local_patterns) != int(candidate.local_dim):
        raise ValueError(
            "basis_configs/local_patterns are incompatible with the selected candidate."
        )

    if recycler_source == "matrix_units":
        local_operator = _matrix_unit_local_operator(
            local_dim=candidate.local_dim,
            recycler_index=candidate.recycler_index,
        )
    elif recycler_source == "rdm_support_matrix_units":
        if states is None:
            raise ValueError("states are required to read rdm_support_matrix_units recyclers.")
        from qlinks.local_structure.reduced_density import (
            _local_pattern_basis_context_from_basis,
            _local_reduced_density_matrix_from_basis_context_and_states,
        )

        state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
        context = _local_pattern_basis_context_from_basis(
            basis_configs=np.asarray(basis_configs),
            variable_indices=variable_indices,
            local_patterns=local_patterns,
        )
        rdm = _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=state_basis,
            tolerance=rdm_tolerance,
        )
        specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")
        _name, local_operator = specs[int(candidate.recycler_index)]
    else:
        raise ValueError("recycler_source must be 'matrix_units' or 'rdm_support_matrix_units'.")

    return LocalOperatorMatrixReadout(
        label=f"recycled_{candidate.candidate_index}_{candidate.recycler_name}",
        source="recycled_recycler",
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        local_operator=np.asarray(local_operator, dtype=np.complex128),
        metadata=(
            ("candidate_index", int(candidate.candidate_index)),
            ("detector_index", int(candidate.detector_index)),
            ("detector_name", candidate.detector_name),
            ("region_index", int(candidate.region_index)),
            ("recycler_index", int(candidate.recycler_index)),
            ("recycler_name", candidate.recycler_name),
            ("inflow_norm", float(candidate.inflow_norm)),
            ("jump_nnz", int(candidate.jump_nnz)),
        ),
    )


def _local_operator_from_collective_recycler_group(
    *,
    group: RecycledManifoldCollectiveRecyclerGroup,
    basis_configs: npt.NDArray[np.integer],
) -> LocalOperatorMatrixReadout:
    variable_indices = tuple(int(value) for value in group.variable_indices)
    local_patterns = _local_patterns_from_basis_configs(
        basis_configs=basis_configs,
        variable_indices=variable_indices,
    )
    if len(local_patterns) != int(group.local_dim):
        raise ValueError("basis_configs/local_patterns are incompatible with the collective group.")
    return LocalOperatorMatrixReadout(
        label=f"collective_recycled_{group.group_index}_{group.detector_name}",
        source="collective_recycled_recycler",
        variable_indices=variable_indices,
        local_patterns=local_patterns,
        local_operator=np.asarray(group.local_operator, dtype=np.complex128),
        metadata=(
            ("group_index", int(group.group_index)),
            ("detector_index", int(group.detector_index)),
            ("detector_name", group.detector_name),
            ("region_index", int(group.region_index)),
            ("n_bundled_recyclers", int(group.n_bundled_recyclers)),
            ("candidate_indices", group.candidate_indices),
            ("recycler_indices", group.recycler_indices),
            ("recycler_names", group.recycler_names),
            ("jump_nnz", int(group.jump_nnz)),
        ),
    )


def _local_recycler_specs(
    *,
    local_patterns: tuple[tuple[int, ...], ...],
    support_basis: npt.NDArray[np.complex128],
    recycler_source: Literal["matrix_units", "rdm_support_matrix_units"],
) -> tuple[tuple[str, npt.NDArray[np.complex128]], ...]:
    local_dim = len(local_patterns)
    specs: list[tuple[str, npt.NDArray[np.complex128]]] = []

    if recycler_source == "matrix_units":
        for target_index, target_pattern in enumerate(local_patterns):
            for source_index, source_pattern in enumerate(local_patterns):
                local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
                local_operator[target_index, source_index] = 1.0
                specs.append(
                    (
                        f"{_pattern_name(target_pattern)}<-{_pattern_name(source_pattern)}",
                        local_operator,
                    )
                )
        return tuple(specs)

    if recycler_source == "rdm_support_matrix_units":
        if support_basis.shape[1] == 0:
            return ()
        for target_index in range(support_basis.shape[1]):
            target_vector = np.asarray(support_basis[:, target_index], dtype=np.complex128)
            for source_index, source_pattern in enumerate(local_patterns):
                source_vector = np.zeros(local_dim, dtype=np.complex128)
                source_vector[source_index] = 1.0
                local_operator = np.outer(target_vector, source_vector.conj())
                specs.append(
                    (
                        f"support_{target_index}<-{_pattern_name(source_pattern)}",
                        local_operator,
                    )
                )
        return tuple(specs)

    raise ValueError("recycler_source must be 'matrix_units' or 'rdm_support_matrix_units'.")


def diagnose_recycled_manifold_dark_detectors(
    *,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    max_detectors: int | None = None,
    max_report_candidates: int | None = 256,
    sort_by_inflow: bool = True,
) -> RecycledManifoldDarkDetectorReport:
    """Test local RDM/matrix-unit recyclers after dark detectors.

    This scans jumps of the form ``J = R D``.  The right detector ``D`` is a
    collective operator satisfying ``D P_M ~= 0``.  The left operator ``R`` is a
    local recycler on one bounded region.  Since target darkness comes from
    ``D``, the recycler can be a general local matrix unit and need not be dark
    on the target manifold by itself.

    Args:
        states: Target manifold basis; columns are orthonormalized.
        basis_configs: Constrained/product basis configurations aligned with
            the Hilbert-space basis used by ``detector_operators``.
        detector_operators: Operator basis used to assemble the detectors.
        local_regions: Variable-index regions where local recyclers are
            embedded.
        detector_coefficients: Optional coefficient matrix defining detectors.
            If omitted, coefficients are read from ``dark_operator_report``.
        dark_operator_report: Report from
            :func:`diagnose_manifold_dark_operator_basis`.
        detector_operator_names: Names for the detector operator basis.
        detector_names: Optional explicit detector names.
        recycler_source: ``"matrix_units"`` scans canonical local matrix units;
            ``"rdm_support_matrix_units"`` maps canonical source patterns into
            the local support eigenvectors of the target manifold RDM.
        tolerance: Orthonormalization and shape-check tolerance.
        rdm_tolerance: Local RDM support threshold.
        dark_tolerance: Relative dark residual threshold for ``J P_M``.
        inflow_tolerance: Threshold for ``||P_M J (I-P_M)||_F``.
        max_detectors: Optional maximum detector columns to scan.
        max_report_candidates: Optional maximum number of best candidates to
            retain in the report.  All candidates are still counted in
            ``n_tested_candidates``.
        sort_by_inflow: If true, report candidates with largest inflow first.

    Returns:
        A report of local-recycler dressed candidates.  Nonzero inflow is a
        necessary, not sufficient, condition for attractive dark-manifold
        dynamics.
    """
    from qlinks.local_structure.embedding import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
    )
    from qlinks.local_structure.reduced_density import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")

    state_basis, gram_residual = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")

    for operator in detector_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(f"operator has incompatible shape: {operator.shape} != {(dim, dim)}.")

    if detector_coefficients is None:
        if dark_operator_report is None:
            raise ValueError(
                "Pass detector_coefficients or dark_operator_report to define detectors."
            )
        detector_coefficients = np.column_stack(
            [candidate.coefficients for candidate in dark_operator_report.candidates]
        )

    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=len(detector_matrices),
    )
    if max_detectors is not None:
        coefficients = coefficients[:, : max(int(max_detectors), 0)]

    if detector_operator_names is None:
        operator_names = tuple(f"O_{index}" for index in range(len(detector_matrices)))
    else:
        operator_names = tuple(str(name) for name in detector_operator_names)
        if len(operator_names) != len(detector_matrices):
            raise ValueError("detector_operator_names length must match detector_operators.")

    if detector_names is None:
        names = tuple(
            _default_detector_name(
                coefficients=coefficients[:, detector_index],
                operator_names=operator_names,
            )
            for detector_index in range(coefficients.shape[1])
        )
    else:
        names = tuple(str(name) for name in detector_names)
        if len(names) != coefficients.shape[1]:
            raise ValueError("detector_names length must match detector count.")

    regions = _normalize_local_regions(local_regions)
    contexts = tuple(
        _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        for region in regions
    )
    embedding_contexts = tuple(
        _embedding_context_from_basis_context(context) for context in contexts
    )
    rdms = tuple(
        _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=state_basis,
            tolerance=rdm_tolerance,
        )
        for context in contexts
    )
    local_dims = tuple(int(rdm.local_dim) for rdm in rdms)

    detectors: list[tuple[sp.csr_array, float, float, npt.NDArray[np.complex128] | None]] = []
    for detector_index in range(coefficients.shape[1]):
        detector = _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        detector_action_residual = float(np.linalg.norm(detector @ state_basis))
        detector_norm = float(sp.linalg.norm(detector))
        detector_relative_residual = detector_action_residual / max(detector_norm, 1.0)
        detector_diagonal = _diagonal_vector_if_diagonal(
            detector,
            tolerance=tolerance,
        )
        detectors.append(
            (
                detector,
                detector_action_residual,
                detector_relative_residual,
                detector_diagonal,
            )
        )

    candidate_buffer: list[RecycledManifoldDarkDetectorCandidate] = []
    n_tested_candidates = 0
    n_nonzero_candidates = 0

    for detector_index, (
        detector,
        detector_action_residual,
        detector_relative_residual,
        detector_diagonal,
    ) in enumerate(detectors):
        for region_index, (embedding_context, rdm) in enumerate(
            zip(embedding_contexts, rdms, strict=True)
        ):
            if recycler_source == "matrix_units" and detector_diagonal is not None:
                for target_index, target_pattern in enumerate(rdm.local_patterns):
                    for source_index, source_pattern in enumerate(rdm.local_patterns):
                        recycler_index = target_index * rdm.local_dim + source_index
                        recycler_name = (
                            f"{_pattern_name(target_pattern)}<-{_pattern_name(source_pattern)}"
                        )
                        n_tested_candidates += 1
                        fast_metrics = _embedded_matrix_unit_metrics_with_diagonal_right_factor(
                            embedding_context=embedding_context,
                            target_local_index=target_index,
                            source_local_index=source_index,
                            right_diagonal=detector_diagonal,
                            state_basis=state_basis,
                            zero_tolerance=0.0,
                        )
                        if fast_metrics is None:
                            continue

                        (
                            inflow_norm,
                            target_block_norm,
                            jump_norm,
                            jump_nnz,
                            recycler_frobenius_norm,
                            recycler_nnz,
                        ) = fast_metrics
                        n_nonzero_candidates += 1
                        dark_residual = float(detector_action_residual * recycler_frobenius_norm)
                        relative_dark_residual = dark_residual / max(jump_norm, 1.0)
                        candidate = RecycledManifoldDarkDetectorCandidate(
                            candidate_index=n_nonzero_candidates - 1,
                            detector_index=int(detector_index),
                            detector_name=names[detector_index],
                            region_index=int(region_index),
                            variable_indices=rdm.variable_indices,
                            local_dim=rdm.local_dim,
                            recycler_index=int(recycler_index),
                            recycler_name=recycler_name,
                            dark_residual=dark_residual,
                            relative_dark_residual=float(relative_dark_residual),
                            inflow_norm=inflow_norm,
                            jump_frobenius_norm=jump_norm,
                            target_block_norm=target_block_norm,
                            detector_action_residual=detector_action_residual,
                            detector_relative_action_residual=float(detector_relative_residual),
                            recycler_frobenius_norm=recycler_frobenius_norm,
                            recycler_nnz=recycler_nnz,
                            jump_nnz=jump_nnz,
                        )
                        if sort_by_inflow:
                            _append_ranked_recycled_candidate(
                                candidate_buffer,
                                candidate,
                                max_report_candidates=max_report_candidates,
                                dark_tolerance=dark_tolerance,
                            )
                        elif max_report_candidates is None or len(candidate_buffer) < max(
                            int(max_report_candidates),
                            0,
                        ):
                            candidate_buffer.append(candidate)
                continue

            recycler_specs = _local_recycler_specs(
                local_patterns=rdm.local_patterns,
                support_basis=rdm.support_basis,
                recycler_source=recycler_source,
            )
            for recycler_index, (recycler_name, local_operator) in enumerate(recycler_specs):
                n_tested_candidates += 1

                fast_metrics = None
                if detector_diagonal is not None:
                    fast_metrics = _embedded_local_operator_metrics_with_diagonal_right_factor(
                        embedding_context=embedding_context,
                        local_operator=local_operator,
                        right_diagonal=detector_diagonal,
                        state_basis=state_basis,
                        zero_tolerance=0.0,
                    )

                if fast_metrics is None:
                    recycler = _embed_local_pattern_operator_from_context(
                        context=embedding_context,
                        local_operator=local_operator,
                    )
                    if recycler.nnz == 0:
                        continue

                    jump = (recycler @ detector).tocsr()
                    if jump.nnz == 0:
                        continue
                    n_nonzero_candidates += 1

                    dark_residual = float(np.linalg.norm(jump @ state_basis))
                    jump_norm = float(sp.linalg.norm(jump))
                    relative_dark_residual = dark_residual / max(jump_norm, 1.0)
                    inflow_norm, target_block_norm = _projected_inflow_norm(
                        jump=jump,
                        state_basis=state_basis,
                    )
                    recycler_frobenius_norm = float(sp.linalg.norm(recycler))
                    recycler_nnz = int(recycler.nnz)
                    jump_nnz = int(jump.nnz)
                else:
                    (
                        inflow_norm,
                        target_block_norm,
                        jump_norm,
                        jump_nnz,
                        recycler_frobenius_norm,
                        recycler_nnz,
                    ) = fast_metrics
                    n_nonzero_candidates += 1
                    # Darkness comes from the right detector: J Q = R (D Q).
                    # This inexpensive bound is exact when the detector is
                    # exactly dark, which is the intended use of this scan.
                    dark_residual = float(detector_action_residual * recycler_frobenius_norm)
                    relative_dark_residual = dark_residual / max(jump_norm, 1.0)

                candidate = RecycledManifoldDarkDetectorCandidate(
                    candidate_index=n_nonzero_candidates - 1,
                    detector_index=int(detector_index),
                    detector_name=names[detector_index],
                    region_index=int(region_index),
                    variable_indices=rdm.variable_indices,
                    local_dim=rdm.local_dim,
                    recycler_index=int(recycler_index),
                    recycler_name=recycler_name,
                    dark_residual=dark_residual,
                    relative_dark_residual=float(relative_dark_residual),
                    inflow_norm=inflow_norm,
                    jump_frobenius_norm=jump_norm,
                    target_block_norm=target_block_norm,
                    detector_action_residual=detector_action_residual,
                    detector_relative_action_residual=float(detector_relative_residual),
                    recycler_frobenius_norm=recycler_frobenius_norm,
                    recycler_nnz=recycler_nnz,
                    jump_nnz=jump_nnz,
                )
                if sort_by_inflow:
                    _append_ranked_recycled_candidate(
                        candidate_buffer,
                        candidate,
                        max_report_candidates=max_report_candidates,
                        dark_tolerance=dark_tolerance,
                    )
                elif max_report_candidates is None or len(candidate_buffer) < max(
                    int(max_report_candidates),
                    0,
                ):
                    candidate_buffer.append(candidate)

    if sort_by_inflow:
        candidate_buffer = sorted(
            candidate_buffer,
            key=lambda candidate: _recycled_candidate_sort_key(
                candidate,
                dark_tolerance=dark_tolerance,
            ),
        )

    if max_report_candidates is not None:
        candidate_buffer = candidate_buffer[: max(int(max_report_candidates), 0)]

    candidate_buffer = [
        RecycledManifoldDarkDetectorCandidate(
            candidate_index=index,
            detector_index=candidate.detector_index,
            detector_name=candidate.detector_name,
            region_index=candidate.region_index,
            variable_indices=candidate.variable_indices,
            local_dim=candidate.local_dim,
            recycler_index=candidate.recycler_index,
            recycler_name=candidate.recycler_name,
            dark_residual=candidate.dark_residual,
            relative_dark_residual=candidate.relative_dark_residual,
            inflow_norm=candidate.inflow_norm,
            jump_frobenius_norm=candidate.jump_frobenius_norm,
            target_block_norm=candidate.target_block_norm,
            detector_action_residual=candidate.detector_action_residual,
            detector_relative_action_residual=candidate.detector_relative_action_residual,
            recycler_frobenius_norm=candidate.recycler_frobenius_norm,
            recycler_nnz=candidate.recycler_nnz,
            jump_nnz=candidate.jump_nnz,
        )
        for index, candidate in enumerate(candidate_buffer)
    ]

    return RecycledManifoldDarkDetectorReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        gram_residual=gram_residual,
        detector_names=names,
        region_variable_indices=regions,
        local_dims=local_dims,
        recycler_source=recycler_source,
        n_tested_candidates=int(n_tested_candidates),
        n_nonzero_candidates=int(n_nonzero_candidates),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        candidates=tuple(candidate_buffer),
    )


def _recycled_jump_for_candidate_from_cache(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    dim: int,
    detector_matrices: tuple[sp.csr_array, ...],
    detector_coefficients: npt.NDArray[np.complex128],
    detector_diagonals: dict[int, npt.NDArray[np.complex128] | None],
    embedding_contexts: dict[int, Any],
    rdms: dict[int, Any],
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ],
    zero_tolerance: float,
) -> sp.csr_array:
    """Rebuild a candidate jump using cached local contexts and detectors."""
    from qlinks.local_structure.embedding import _embed_local_pattern_operator_from_context

    detector_diagonal = detector_diagonals.get(candidate.detector_index)
    embedding_context = embedding_contexts[candidate.region_index]

    if detector_diagonal is not None:
        if recycler_source == "matrix_units":
            target_index, source_index = divmod(
                int(candidate.recycler_index),
                int(candidate.local_dim),
            )
            return _embedded_matrix_unit_times_diagonal_as_csr(
                embedding_context=embedding_context,
                target_local_index=target_index,
                source_local_index=source_index,
                right_diagonal=detector_diagonal,
                dim=dim,
                zero_tolerance=zero_tolerance,
            )

        rdm = rdms[candidate.region_index]
        recycler_specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        _, local_operator = recycler_specs[candidate.recycler_index]
        return _embedded_local_operator_times_diagonal_as_csr(
            embedding_context=embedding_context,
            local_operator=local_operator,
            right_diagonal=detector_diagonal,
            dim=dim,
            zero_tolerance=zero_tolerance,
        )

    detector = _combined_operator(
        operators=detector_matrices,
        coefficients=detector_coefficients[:, candidate.detector_index],
    )

    if recycler_source == "matrix_units":
        local_dim = int(candidate.local_dim)
        if local_dim != int(embedding_context.local_dim):
            raise ValueError(
                "candidate.local_dim is incompatible with the cached embedding context."
            )
        n_matrix_units = local_dim * local_dim
        if candidate.recycler_index < 0 or candidate.recycler_index >= n_matrix_units:
            raise ValueError("candidate.recycler_index is out of range for matrix-unit recyclers.")
        target_index, source_index = divmod(int(candidate.recycler_index), local_dim)
        local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
        local_operator[target_index, source_index] = 1.0
    else:
        rdm = rdms[candidate.region_index]
        recycler_specs = _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")
        _, local_operator = recycler_specs[candidate.recycler_index]

    recycler = _embed_local_pattern_operator_from_context(
        context=embedding_context,
        local_operator=local_operator,
    )
    return (recycler @ detector).tocsr()


def _local_recycler_operator_from_candidate_cache(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    embedding_contexts: dict[int, Any],
    rdms: dict[int, Any],
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ],
) -> tuple[str, npt.NDArray[np.complex128]]:
    """Return a candidate's local recycler matrix from cached local data."""
    embedding_context = embedding_contexts[candidate.region_index]
    if recycler_source == "matrix_units":
        local_dim = int(embedding_context.local_dim)
        if int(candidate.local_dim) != local_dim:
            raise ValueError(
                "candidate.local_dim is incompatible with the cached embedding context."
            )
        n_matrix_units = local_dim * local_dim
        if candidate.recycler_index < 0 or candidate.recycler_index >= n_matrix_units:
            raise ValueError("candidate.recycler_index is out of range for matrix-unit recyclers.")
        target_index, source_index = divmod(int(candidate.recycler_index), local_dim)
        local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
        local_operator[target_index, source_index] = 1.0
        return candidate.recycler_name, local_operator

    rdm = rdms[candidate.region_index]
    recycler_specs = _local_recycler_specs(
        local_patterns=rdm.local_patterns,
        support_basis=rdm.support_basis,
        recycler_source=recycler_source,
    )
    if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
        raise ValueError("candidate.recycler_index is out of range for recycler specs.")
    recycler_name, local_operator = recycler_specs[candidate.recycler_index]
    return recycler_name, np.asarray(local_operator, dtype=np.complex128)


def _collective_recycler_weight(
    candidate: RecycledManifoldDarkDetectorCandidate,
    *,
    weighting: Literal["unit", "inflow", "normalized_inflow"],
) -> complex:
    if weighting == "unit":
        return 1.0 + 0.0j
    if weighting == "inflow":
        return complex(float(candidate.inflow_norm))
    if weighting == "normalized_inflow":
        return complex(
            float(candidate.inflow_norm) / max(float(candidate.jump_frobenius_norm), 1.0e-300)
        )
    raise ValueError(
        'collective_recycler_weighting must be "unit", "inflow", or "normalized_inflow".'
    )


def _bundle_recycled_jumps_by_region_detector(
    *,
    selected_candidates: tuple[RecycledManifoldDarkDetectorCandidate, ...],
    dim: int,
    detector_matrices: tuple[sp.csr_array, ...],
    detector_coefficients: npt.NDArray[np.complex128],
    embedding_contexts: dict[int, Any],
    rdms: dict[int, Any],
    state_basis: npt.NDArray[np.complex128] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ],
    weighting: Literal["unit", "inflow", "normalized_inflow"],
    normalize_recyclers: bool,
    tolerance: float,
) -> tuple[tuple[sp.csr_array, ...], tuple[RecycledManifoldCollectiveRecyclerGroup, ...]]:
    """Bundle selected microscopic recyclers into collective local recyclers.

    Candidates are grouped only by ``(detector_index, region_index)``. This is
    locality-safe: every bundled recycler acts on the same local region as the
    selected microscopic recyclers, and the same dark detector remains on the
    right.
    """
    from qlinks.local_structure.embedding import _embed_local_pattern_operator_from_context

    grouped: dict[tuple[int, int], list[RecycledManifoldDarkDetectorCandidate]] = {}
    group_order: list[tuple[int, int]] = []
    for candidate in selected_candidates:
        key = (int(candidate.detector_index), int(candidate.region_index))
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(candidate)

    detector_cache: dict[int, sp.csr_array] = {}
    jumps: list[sp.csr_array] = []
    groups: list[RecycledManifoldCollectiveRecyclerGroup] = []

    for group_index, key in enumerate(group_order):
        detector_index, region_index = key
        candidates = tuple(grouped[key])
        embedding_context = embedding_contexts[region_index]
        local_dim = int(embedding_context.local_dim)
        local_operator = np.zeros((local_dim, local_dim), dtype=np.complex128)
        raw_weights: list[complex] = []
        recycler_names: list[str] = []

        for candidate in candidates:
            recycler_name, candidate_operator = _local_recycler_operator_from_candidate_cache(
                candidate=candidate,
                embedding_contexts=embedding_contexts,
                rdms=rdms,
                recycler_source=recycler_source,
            )
            weight = _collective_recycler_weight(candidate, weighting=weighting)
            raw_weights.append(weight)
            recycler_names.append(recycler_name)
            local_operator += weight * candidate_operator

        frobenius_norm = float(np.linalg.norm(local_operator))
        if normalize_recyclers and frobenius_norm > tolerance:
            local_operator = local_operator / frobenius_norm
            weights = tuple(complex(weight / frobenius_norm) for weight in raw_weights)
        else:
            weights = tuple(complex(weight) for weight in raw_weights)

        recycler = _embed_local_pattern_operator_from_context(
            context=embedding_context,
            local_operator=local_operator,
        ).tocsr()
        detector = detector_cache.get(detector_index)
        if detector is None:
            detector = _combined_operator(
                operators=detector_matrices,
                coefficients=detector_coefficients[:, detector_index],
            )
            detector_cache[detector_index] = detector
        jump = (recycler @ detector).tocsr()
        if jump.shape != (dim, dim):
            raise ValueError("bundled jump has incompatible shape.")
        bundled_inflow_norm = None
        unbundled_inflow_norm = None
        if state_basis is not None:
            bundled_inflow_norm, _ = _projected_inflow_norm(
                jump=jump,
                state_basis=state_basis,
            )
            unbundled_inflow_norm = float(
                np.sqrt(sum(float(candidate.inflow_norm) ** 2 for candidate in candidates))
            )
        jumps.append(jump)
        groups.append(
            RecycledManifoldCollectiveRecyclerGroup(
                group_index=int(group_index),
                detector_index=int(detector_index),
                detector_name=candidates[0].detector_name,
                region_index=int(region_index),
                variable_indices=tuple(int(value) for value in candidates[0].variable_indices),
                local_dim=local_dim,
                candidate_indices=tuple(int(candidate.candidate_index) for candidate in candidates),
                recycler_indices=tuple(int(candidate.recycler_index) for candidate in candidates),
                recycler_names=tuple(recycler_names),
                weights=weights,
                local_operator=np.asarray(local_operator, dtype=np.complex128),
                jump_frobenius_norm=float(sp.linalg.norm(jump)),
                recycler_frobenius_norm=float(sp.linalg.norm(recycler)),
                recycler_nnz=int(recycler.nnz),
                jump_nnz=int(jump.nnz),
                unbundled_inflow_norm=unbundled_inflow_norm,
                bundled_inflow_norm=bundled_inflow_norm,
            )
        )

    return tuple(jumps), tuple(groups)


def _recycled_jump_for_candidate(
    *,
    candidate: RecycledManifoldDarkDetectorCandidate,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
) -> sp.csr_array:
    """Rebuild the sparse jump operator for a reported recycled candidate."""
    from qlinks.local_structure.embedding import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
    )
    from qlinks.local_structure.reduced_density import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")

    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")

    for operator in detector_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(f"operator has incompatible shape: {operator.shape} != {(dim, dim)}.")

    if detector_coefficients is None:
        if dark_operator_report is None:
            raise ValueError(
                "Pass detector_coefficients or dark_operator_report to define detectors."
            )
        detector_coefficients = np.column_stack(
            [report_candidate.coefficients for report_candidate in dark_operator_report.candidates]
        )

    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=len(detector_matrices),
    )
    if candidate.detector_index < 0 or candidate.detector_index >= coefficients.shape[1]:
        raise ValueError("candidate.detector_index is out of range for detector coefficients.")

    detector = _combined_operator(
        operators=detector_matrices,
        coefficients=coefficients[:, candidate.detector_index],
    )

    regions = _normalize_local_regions(local_regions)
    if candidate.region_index < 0 or candidate.region_index >= len(regions):
        raise ValueError("candidate.region_index is out of range for local_regions.")

    context = _local_pattern_basis_context_from_basis(
        basis_configs=basis_array,
        variable_indices=regions[candidate.region_index],
    )
    embedding_context = _embedding_context_from_basis_context(context)
    rdm = _local_reduced_density_matrix_from_basis_context_and_states(
        context=context,
        states=state_basis,
        tolerance=rdm_tolerance,
    )
    recycler_specs = _local_recycler_specs(
        local_patterns=rdm.local_patterns,
        support_basis=rdm.support_basis,
        recycler_source=recycler_source,
    )
    if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
        raise ValueError("candidate.recycler_index is out of range for recycler specs.")

    _, local_operator = recycler_specs[candidate.recycler_index]
    recycler = _embed_local_pattern_operator_from_context(
        context=embedding_context,
        local_operator=local_operator,
    )
    return (recycler @ detector).tocsr()


def _internal_liouvillian_eigenvalues_from_energies(
    energies: tuple[complex, ...],
) -> tuple[complex, ...]:
    return tuple(-1j * (left - right) for left in energies for right in energies)


def _state_ipr(state: npt.NDArray[np.complex128]) -> float:
    norm_sq = float(np.vdot(state, state).real)
    if norm_sq <= 0.0:
        return 0.0
    probabilities = np.abs(state) ** 2 / norm_sq
    return float(np.sum(probabilities * probabilities))


def _hamiltonian_projector_commutator_residual(
    *,
    hamiltonian: sp.csr_array,
    state_basis: npt.NDArray[np.complex128],
) -> float:
    manifold_dimension = int(state_basis.shape[1])
    density = state_basis @ state_basis.conj().T / float(manifold_dimension)
    action_left = np.asarray(hamiltonian @ density, dtype=np.complex128)
    action_right = np.asarray((density @ hamiltonian).astype(np.complex128))
    return float(np.linalg.norm(-1j * (action_left - action_right)))


def _detector_coefficients_from_report(
    *,
    detector_coefficients: npt.ArrayLike | None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None,
    n_operators: int,
    max_detectors: int | None,
) -> npt.NDArray[np.complex128]:
    if detector_coefficients is None:
        if dark_operator_report is None:
            raise ValueError(
                "Pass detector_coefficients or dark_operator_report to define detectors."
            )
        detector_coefficients = np.column_stack(
            [candidate.coefficients for candidate in dark_operator_report.candidates]
        )
    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=n_operators,
    )
    if max_detectors is not None:
        coefficients = coefficients[:, : max(int(max_detectors), 0)]
    return coefficients


def _stream_recycled_family_kernel_diagnostics(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    candidates: tuple[RecycledManifoldDarkDetectorCandidate, ...],
    detector_coefficients: npt.ArrayLike | None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None,
    recycler_source: Literal["matrix_units", "rdm_support_matrix_units"],
    tolerance: float,
    rdm_tolerance: float,
    kernel_tolerance: float,
    liouvillian_zero_tolerance: float,
    max_detectors: int | None,
) -> tuple[RecycledFamilyKernelDiagnostics, int, int, int]:
    from qlinks.local_structure.embedding import (
        _embed_local_pattern_operator_from_context,
        _embedding_context_from_basis_context,
    )
    from qlinks.local_structure.reduced_density import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    hamiltonian_matrix = _as_csr(hamiltonian)
    if hamiltonian_matrix.shape != (dim, dim):
        raise ValueError("hamiltonian must have shape (hilbert_dimension, hilbert_dimension).")

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")
    for operator in detector_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(f"operator has incompatible shape: {operator.shape} != {(dim, dim)}.")

    coefficients = _detector_coefficients_from_report(
        detector_coefficients=detector_coefficients,
        dark_operator_report=dark_operator_report,
        n_operators=len(detector_matrices),
        max_detectors=max_detectors,
    )
    detectors = tuple(
        _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        for detector_index in range(coefficients.shape[1])
    )

    basis_array = np.asarray(basis_configs)
    if basis_array.ndim != 2 or basis_array.shape[0] != dim:
        raise ValueError("basis_configs must have shape (hilbert_dimension, n_variables).")

    regions = _normalize_local_regions(local_regions)
    contexts = tuple(
        _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=region,
        )
        for region in regions
    )
    embedding_contexts = tuple(
        _embedding_context_from_basis_context(context) for context in contexts
    )
    rdms = tuple(
        _local_reduced_density_matrix_from_basis_context_and_states(
            context=context,
            states=state_basis,
            tolerance=rdm_tolerance,
        )
        for context in contexts
    )
    recycler_specs_by_region = tuple(
        _local_recycler_specs(
            local_patterns=rdm.local_patterns,
            support_basis=rdm.support_basis,
            recycler_source=recycler_source,
        )
        for rdm in rdms
    )
    recycler_cache: dict[tuple[int, int], sp.csr_array] = {}

    complement_basis = _orthogonal_complement_basis(state_basis, tolerance=kernel_tolerance)
    complement_dimension = int(complement_basis.shape[1])
    family_gram = np.zeros(
        (complement_dimension, complement_dimension),
        dtype=np.complex128,
    )
    total_jump_nnz = 0
    max_jump_nnz = 0
    max_target_jump_residual = 0.0
    inflow_squared = 0.0
    n_jumps = 0

    for candidate in candidates:
        if candidate.detector_index < 0 or candidate.detector_index >= len(detectors):
            raise ValueError("candidate.detector_index is out of range for detector coefficients.")
        if candidate.region_index < 0 or candidate.region_index >= len(regions):
            raise ValueError("candidate.region_index is out of range for local_regions.")
        recycler_specs = recycler_specs_by_region[candidate.region_index]
        if candidate.recycler_index < 0 or candidate.recycler_index >= len(recycler_specs):
            raise ValueError("candidate.recycler_index is out of range for recycler specs.")

        cache_key = (int(candidate.region_index), int(candidate.recycler_index))
        recycler = recycler_cache.get(cache_key)
        if recycler is None:
            _, local_operator = recycler_specs[candidate.recycler_index]
            recycler = _embed_local_pattern_operator_from_context(
                context=embedding_contexts[candidate.region_index],
                local_operator=local_operator,
            ).tocsr()
            recycler_cache[cache_key] = recycler

        jump = (recycler @ detectors[candidate.detector_index]).tocsr()
        if jump.nnz == 0:
            continue
        n_jumps += 1
        total_jump_nnz += int(jump.nnz)
        max_jump_nnz = max(max_jump_nnz, int(jump.nnz))
        max_target_jump_residual = max(
            max_target_jump_residual,
            float(np.linalg.norm(jump @ state_basis)),
        )
        inflow_squared += float(candidate.inflow_norm) ** 2

        if complement_dimension > 0:
            image = np.asarray(jump @ complement_basis, dtype=np.complex128)
            family_gram += image.conj().T @ image

    if complement_dimension == 0:
        bad_basis = np.zeros((dim, 0), dtype=np.complex128)
    else:
        family_gram = 0.5 * (family_gram + family_gram.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(family_gram)
        largest = float(np.max(np.maximum(eigenvalues.real, 0.0))) if eigenvalues.size else 0.0
        cutoff = max(float(kernel_tolerance), float(kernel_tolerance) * max(largest, 1.0))
        kernel_mask = np.asarray(eigenvalues.real <= cutoff, dtype=bool)
        bad_basis = complement_basis @ eigenvectors[:, kernel_mask]

    bad_dimension = int(bad_basis.shape[1])
    common_dimension = int(manifold_dimension + bad_dimension)
    bad_iprs = tuple(_state_ipr(bad_basis[:, index]) for index in range(bad_dimension))

    hamiltonian_action = np.asarray(hamiltonian_matrix @ state_basis, dtype=np.complex128)
    internal_hamiltonian = state_basis.conj().T @ hamiltonian_action
    projected_hamiltonian_action = state_basis @ internal_hamiltonian
    hamiltonian_closure_residual = float(
        np.linalg.norm(hamiltonian_action - projected_hamiltonian_action)
    )
    internal_hamiltonian = 0.5 * (internal_hamiltonian + internal_hamiltonian.conj().T)
    internal_hamiltonian_eigenvalues = tuple(
        complex(value) for value in np.linalg.eigvalsh(internal_hamiltonian)
    )
    expected_internal_liouvillian_eigenvalues = _internal_liouvillian_eigenvalues_from_energies(
        internal_hamiltonian_eigenvalues
    )
    expected_internal_zero_mode_count = int(
        sum(
            abs(value) <= liouvillian_zero_tolerance
            for value in expected_internal_liouvillian_eigenvalues
        )
    )
    expected_internal_peripheral_mode_count = int(
        len(expected_internal_liouvillian_eigenvalues) - expected_internal_zero_mode_count
    )

    target_density_liouvillian_residual = _hamiltonian_projector_commutator_residual(
        hamiltonian=hamiltonian_matrix,
        state_basis=state_basis,
    )
    target_in_common_jump_kernel = max_target_jump_residual <= kernel_tolerance
    target_distance = 0.0 if target_in_common_jump_kernel else float("nan")
    target_projection = float(np.sqrt(manifold_dimension)) if target_in_common_jump_kernel else 0.0

    diagnostics = RecycledFamilyKernelDiagnostics(
        dim=dim,
        n_jumps=n_jumps,
        manifold_dimension=manifold_dimension,
        hamiltonian_closure_residual=hamiltonian_closure_residual,
        max_target_jump_residual=max_target_jump_residual,
        target_density_liouvillian_residual=target_density_liouvillian_residual,
        inflow_norm=float(np.sqrt(max(inflow_squared, 0.0))),
        common_jump_kernel_dimension=common_dimension,
        target_projection_onto_common_kernel=target_projection,
        target_distance_from_common_kernel=target_distance,
        target_in_common_jump_kernel=bool(target_in_common_jump_kernel),
        bad_common_jump_kernel_dimension=bad_dimension,
        bad_common_jump_kernel_iprs=bad_iprs,
        internal_hamiltonian_eigenvalues=internal_hamiltonian_eigenvalues,
        expected_internal_zero_mode_count=expected_internal_zero_mode_count,
        expected_internal_peripheral_mode_count=expected_internal_peripheral_mode_count,
        liouvillian_zero_tolerance=float(liouvillian_zero_tolerance),
    )
    return diagnostics, n_jumps, total_jump_nnz, max_jump_nnz


def diagnose_recycled_manifold_candidate_family_kernel(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    kernel_tolerance: float = 1.0e-10,
    liouvillian_zero_tolerance: float = 1.0e-9,
    max_detectors: int | None = None,
    expand_candidate_report: bool = True,
    kernel_method: Literal["streamed", "diagnostics"] = "streamed",
    store_candidate_jumps: bool = False,
) -> RecycledManifoldCandidateFamilyKernelReport:
    """Diagnose the common jump kernel of the full recycled-detector family.

    This is the decisive follow-up when greedy selection saturates at a
    nonzero complement kernel.  If the family of all eligible local candidates
    still has a bad common jump kernel, no subset selected from that family can
    remove it.  If the family removes the kernel but the greedy subset does not,
    the problem is the subset-selection heuristic rather than the operator
    family.
    """
    from qlinks.open_system.diagnostics.dark import diagnose_dark_manifold

    regions = _normalize_local_regions(local_regions)
    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])
    candidate_report_was_expanded = False
    if candidate_report is None:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=None,
            sort_by_inflow=True,
        )
    elif expand_candidate_report and candidate_report.candidate_report_is_truncated:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=None,
            sort_by_inflow=True,
        )
        candidate_report_was_expanded = True

    eligible_candidates = tuple(
        candidate
        for candidate in candidate_report.candidates
        if candidate.relative_dark_residual <= dark_tolerance
        and candidate.inflow_norm > inflow_tolerance
    )
    if kernel_method not in {"streamed", "diagnostics"}:
        raise ValueError('kernel_method must be "streamed" or "diagnostics".')

    candidate_jumps: tuple[sp.csr_array, ...] = ()
    candidate_jump_count: int | None = None
    candidate_total_jump_nnz: int | None = None
    candidate_max_jump_nnz: int | None = None

    if kernel_method == "streamed":
        diagnostics, jump_count, total_jump_nnz, max_jump_nnz = (
            _stream_recycled_family_kernel_diagnostics(
                hamiltonian=hamiltonian,
                states=state_basis,
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=regions,
                candidates=eligible_candidates,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                recycler_source=recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
                kernel_tolerance=kernel_tolerance,
                liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                max_detectors=max_detectors,
            )
        )
        candidate_jump_count = jump_count
        candidate_total_jump_nnz = total_jump_nnz
        candidate_max_jump_nnz = max_jump_nnz
        if store_candidate_jumps:
            candidate_jumps = tuple(
                _recycled_jump_for_candidate(
                    candidate=candidate,
                    states=state_basis,
                    basis_configs=basis_configs,
                    detector_operators=detector_operators,
                    local_regions=regions,
                    detector_coefficients=detector_coefficients,
                    dark_operator_report=dark_operator_report,
                    recycler_source=recycler_source,
                    tolerance=tolerance,
                    rdm_tolerance=rdm_tolerance,
                )
                for candidate in eligible_candidates
            )
    else:
        candidate_jumps = tuple(
            _recycled_jump_for_candidate(
                candidate=candidate,
                states=state_basis,
                basis_configs=basis_configs,
                detector_operators=detector_operators,
                local_regions=regions,
                detector_coefficients=detector_coefficients,
                dark_operator_report=dark_operator_report,
                recycler_source=recycler_source,
                tolerance=tolerance,
                rdm_tolerance=rdm_tolerance,
            )
            for candidate in eligible_candidates
        )

        diagnostics = diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=candidate_jumps,
            target_states=state_basis,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=False,
            liouvillian_spectrum_method="none",
        )

    return RecycledManifoldCandidateFamilyKernelReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        candidate_report=candidate_report,
        candidate_report_was_expanded=candidate_report_was_expanded,
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        candidate_jumps=candidate_jumps,
        diagnostics=diagnostics,
        candidate_jump_count=candidate_jump_count,
        candidate_total_jump_nnz=candidate_total_jump_nnz,
        candidate_max_jump_nnz=candidate_max_jump_nnz,
    )


def select_recycled_manifold_dark_detector_jumps(
    *,
    hamiltonian: Any,
    states: npt.ArrayLike,
    basis_configs: npt.NDArray[np.integer],
    detector_operators: tuple[Any, ...] | list[Any],
    local_regions: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | list[list[int]],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    candidate_report: RecycledManifoldDarkDetectorReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    recycler_source: Literal[
        "matrix_units",
        "rdm_support_matrix_units",
    ] = "rdm_support_matrix_units",
    tolerance: float = 1.0e-10,
    rdm_tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    kernel_tolerance: float = 1.0e-10,
    liouvillian_zero_tolerance: float = 1.0e-9,
    max_detectors: int | None = None,
    max_candidate_pool: int | None = 128,
    max_selected_jumps: int = 16,
    target_bad_kernel_dimension: int = 0,
    allow_non_improving: bool = False,
    expand_candidate_report: bool = False,
    selection_strategy: Literal[
        "diagnostics", "kernel_projection", "ranked_inflow"
    ] = "diagnostics",
    compression_strategy: Literal["none", "h_invariant"] = "none",
    max_compression_passes: int = 1,
    collective_recycler_strategy: Literal["none", "bundle_by_region_detector"] = "none",
    collective_recycler_weighting: Literal["unit", "inflow", "normalized_inflow"] = "unit",
    normalize_collective_recyclers: bool = True,
    check_final_diagnostics: bool | None = None,
) -> RecycledManifoldJumpSelectionReport:
    """Greedily select a small recycled-detector jump subset.

    The candidate family is ``J=R D``: ``D`` is a collective detector dark on
    the target manifold, and ``R`` is a local matrix-unit/RDM-support recycler.
    The selector first keeps the best direct-inflow candidates, then adds jumps
    one at a time to minimize the complement common jump-kernel dimension

        dim( intersection_mu ker J_mu  ∩  P_M^perp ).

    Reaching ``target_bad_kernel_dimension=0`` is a strong, jump-only sufficient
    condition that no complement vector remains dark under all selected jumps.
    It is stronger than the true invariant-subspace condition including ``H``,
    but cheaper and useful before expensive Liouvillian spectrum checks.

    If ``candidate_report`` is a truncated diagnostic report, set
    ``expand_candidate_report=True`` and ``max_candidate_pool=None`` to rescan
    and use the full local recycler candidate family.  This is often the right
    follow-up when a small inflow-ranked pool leaves a low-dimensional bad
    complement kernel.

    ``selection_strategy="kernel_projection"`` is faster for large two-region
    recycler pools: it updates the current complement common kernel directly by
    applying each candidate to the current bad subspace, and runs the full
    diagnostics only once at the end.  ``"ranked_inflow"`` is the production
    preselection mode for large scans: it trusts the inflow-ranked candidate
    report and selects the top candidates directly.  By default this production
    mode skips the expensive final common-kernel diagnostic; pass
    ``check_final_diagnostics=True`` when you want the full certificate.
    """
    from qlinks.open_system.diagnostics.dark import diagnose_dark_manifold

    regions = _normalize_local_regions(local_regions)
    state_basis, _ = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])
    if selection_strategy not in {"diagnostics", "kernel_projection", "ranked_inflow"}:
        raise ValueError(
            'selection_strategy must be "diagnostics", "kernel_projection", or "ranked_inflow".'
        )
    if compression_strategy not in {"none", "h_invariant"}:
        raise ValueError('compression_strategy must be "none" or "h_invariant".')
    if collective_recycler_strategy not in {"none", "bundle_by_region_detector"}:
        raise ValueError(
            'collective_recycler_strategy must be "none" or "bundle_by_region_detector".'
        )
    if collective_recycler_weighting not in {"unit", "inflow", "normalized_inflow"}:
        raise ValueError(
            'collective_recycler_weighting must be "unit", "inflow", or "normalized_inflow".'
        )
    if check_final_diagnostics is None:
        check_final_diagnostics = selection_strategy != "ranked_inflow"

    candidate_report_was_expanded = False
    if candidate_report is None:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=max_candidate_pool,
            sort_by_inflow=True,
        )
    elif expand_candidate_report and candidate_report.candidate_report_is_truncated:
        candidate_report = diagnose_recycled_manifold_dark_detectors(
            states=state_basis,
            basis_configs=basis_configs,
            detector_operators=detector_operators,
            local_regions=regions,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            detector_operator_names=detector_operator_names,
            detector_names=detector_names,
            recycler_source=recycler_source,
            tolerance=tolerance,
            rdm_tolerance=rdm_tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            max_report_candidates=max_candidate_pool,
            sort_by_inflow=True,
        )
        candidate_report_was_expanded = True

    eligible_pool = [
        candidate
        for candidate in candidate_report.candidates
        if candidate.relative_dark_residual <= dark_tolerance
        and candidate.inflow_norm > inflow_tolerance
    ]
    candidate_pool_was_limited = False
    if max_candidate_pool is None:
        pool = eligible_pool
    else:
        pool_limit = max(int(max_candidate_pool), 0)
        candidate_pool_was_limited = len(eligible_pool) > pool_limit
        pool = eligible_pool[:pool_limit]

    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    if detector_coefficients is None:
        if dark_operator_report is None:
            raise ValueError(
                "Pass detector_coefficients or dark_operator_report to define detectors."
            )
        detector_coefficients = np.column_stack(
            [candidate.coefficients for candidate in dark_operator_report.candidates]
        )
    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=len(detector_matrices),
    )

    from qlinks.local_structure.embedding import _embedding_context_from_basis_context
    from qlinks.local_structure.reduced_density import (
        _local_pattern_basis_context_from_basis,
        _local_reduced_density_matrix_from_basis_context_and_states,
    )

    basis_array = np.asarray(basis_configs)
    used_region_indices = sorted({int(candidate.region_index) for candidate in pool})
    contexts = {
        region_index: _local_pattern_basis_context_from_basis(
            basis_configs=basis_array,
            variable_indices=regions[region_index],
        )
        for region_index in used_region_indices
    }
    embedding_contexts = {
        region_index: _embedding_context_from_basis_context(context)
        for region_index, context in contexts.items()
    }
    rdms = (
        {}
        if recycler_source == "matrix_units"
        else {
            region_index: _local_reduced_density_matrix_from_basis_context_and_states(
                context=context,
                states=state_basis,
                tolerance=rdm_tolerance,
            )
            for region_index, context in contexts.items()
        }
    )

    used_detector_indices = {candidate.detector_index for candidate in pool}
    detector_diagonals: dict[int, npt.NDArray[np.complex128] | None] = {}
    for detector_index in used_detector_indices:
        detector = _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        detector_diagonals[detector_index] = _diagonal_vector_if_diagonal(
            detector,
            tolerance=tolerance,
        )

    candidate_jumps = {
        id(candidate): _recycled_jump_for_candidate_from_cache(
            candidate=candidate,
            dim=dim,
            detector_matrices=detector_matrices,
            detector_coefficients=coefficients,
            detector_diagonals=detector_diagonals,
            embedding_contexts=embedding_contexts,
            rdms=rdms,
            recycler_source=recycler_source,
            zero_tolerance=0.0,
        )
        for candidate in pool
    }

    selected_candidates: list[RecycledManifoldDarkDetectorCandidate] = []
    selected_jumps: list[sp.csr_array] = []
    selected_ids: set[int] = set()
    steps: list[RecycledManifoldJumpSelectionStep] = []
    current_bad_dimension = dim - manifold_dimension
    final_diagnostics = None

    if selection_strategy == "ranked_inflow":
        selected_candidates = list(pool[: max(int(max_selected_jumps), 0)])
        selected_jumps = [candidate_jumps[id(candidate)] for candidate in selected_candidates]
        selected_ids = {id(candidate) for candidate in selected_candidates}

        cumulative_inflow_squared = 0.0
        max_target_jump_residual = 0.0
        for selected_candidate in selected_candidates:
            cumulative_inflow_squared += float(selected_candidate.inflow_norm) ** 2
            max_target_jump_residual = max(
                max_target_jump_residual,
                float(selected_candidate.dark_residual),
            )
            steps.append(
                RecycledManifoldJumpSelectionStep(
                    step_index=len(steps),
                    candidate=selected_candidate,
                    bad_common_jump_kernel_dimension=current_bad_dimension,
                    inflow_norm=float(np.sqrt(cumulative_inflow_squared)),
                    max_target_jump_residual=max_target_jump_residual,
                    n_selected_jumps=len(steps) + 1,
                )
            )

    elif selection_strategy == "kernel_projection":
        current_bad_basis = _orthogonal_complement_basis(
            state_basis,
            tolerance=kernel_tolerance,
        )
        current_bad_dimension = int(current_bad_basis.shape[1])
        cumulative_inflow_squared = 0.0
        max_target_jump_residual = 0.0

        for _step_index in range(max(int(max_selected_jumps), 0)):
            best_entry = None
            for candidate in pool:
                candidate_id = id(candidate)
                if candidate_id in selected_ids:
                    continue
                jump = candidate_jumps[candidate_id]
                image = np.asarray(jump @ current_bad_basis, dtype=np.complex128)
                kernel_basis = _right_kernel_basis(image, tolerance=kernel_tolerance)
                next_bad_dimension = int(kernel_basis.shape[1])
                score = (
                    next_bad_dimension,
                    -candidate.inflow_norm,
                    candidate.jump_nnz,
                    candidate.detector_index,
                    candidate.region_index,
                    candidate.recycler_index,
                )
                if best_entry is None or score < best_entry[0]:
                    best_entry = (score, candidate, jump, kernel_basis)

            if best_entry is None:
                break

            _, best_candidate, best_jump, best_kernel_basis = best_entry
            next_bad_dimension = int(best_kernel_basis.shape[1])
            if not allow_non_improving and next_bad_dimension >= current_bad_dimension:
                break

            selected_candidates.append(best_candidate)
            selected_jumps.append(best_jump)
            selected_ids.add(id(best_candidate))
            current_bad_basis = current_bad_basis @ best_kernel_basis
            current_bad_dimension = next_bad_dimension
            cumulative_inflow_squared += float(best_candidate.inflow_norm) ** 2
            max_target_jump_residual = max(
                max_target_jump_residual,
                float(best_candidate.dark_residual),
            )
            steps.append(
                RecycledManifoldJumpSelectionStep(
                    step_index=len(steps),
                    candidate=best_candidate,
                    bad_common_jump_kernel_dimension=current_bad_dimension,
                    inflow_norm=float(np.sqrt(cumulative_inflow_squared)),
                    max_target_jump_residual=max_target_jump_residual,
                    n_selected_jumps=len(selected_jumps),
                )
            )

            if current_bad_dimension <= target_bad_kernel_dimension:
                break
    else:
        for _step_index in range(max(int(max_selected_jumps), 0)):
            best_entry = None
            for candidate in pool:
                candidate_id = id(candidate)
                if candidate_id in selected_ids:
                    continue
                trial_jumps = tuple(selected_jumps + [candidate_jumps[candidate_id]])
                diagnostics = diagnose_dark_manifold(
                    hamiltonian=hamiltonian,
                    jumps=trial_jumps,
                    target_states=state_basis,
                    kernel_tolerance=kernel_tolerance,
                    liouvillian_zero_tolerance=liouvillian_zero_tolerance,
                    check_liouvillian_spectrum=False,
                    liouvillian_spectrum_method="none",
                )
                score = (
                    diagnostics.bad_common_jump_kernel_dimension,
                    diagnostics.max_target_jump_residual,
                    -diagnostics.inflow_norm,
                    -candidate.inflow_norm,
                    candidate.jump_nnz,
                    candidate.detector_index,
                    candidate.region_index,
                    candidate.recycler_index,
                )
                if best_entry is None or score < best_entry[0]:
                    best_entry = (score, candidate, candidate_jumps[candidate_id], diagnostics)

            if best_entry is None:
                break

            _, best_candidate, best_jump, best_diagnostics = best_entry
            if (
                not allow_non_improving
                and best_diagnostics.bad_common_jump_kernel_dimension >= current_bad_dimension
            ):
                break

            selected_candidates.append(best_candidate)
            selected_jumps.append(best_jump)
            selected_ids.add(id(best_candidate))
            current_bad_dimension = int(best_diagnostics.bad_common_jump_kernel_dimension)
            final_diagnostics = best_diagnostics
            steps.append(
                RecycledManifoldJumpSelectionStep(
                    step_index=len(steps),
                    candidate=best_candidate,
                    bad_common_jump_kernel_dimension=current_bad_dimension,
                    inflow_norm=float(best_diagnostics.inflow_norm),
                    max_target_jump_residual=float(best_diagnostics.max_target_jump_residual),
                    n_selected_jumps=len(selected_jumps),
                )
            )

            if current_bad_dimension <= target_bad_kernel_dimension:
                break

    n_compression_passes = 0
    n_compressed_jumps_removed = 0
    if compression_strategy == "h_invariant" and len(selected_jumps) > 0:
        from qlinks.open_system.diagnostics.dark import diagnose_common_kernel_h_invariant_sector

        current_h_report = diagnose_common_kernel_h_invariant_sector(
            hamiltonian=hamiltonian,
            jumps=tuple(selected_jumps),
            target_states=state_basis,
            kernel_tolerance=kernel_tolerance,
        )
        if current_h_report.likely_attractive_by_h_invariant_kernel:
            max_passes = max(int(max_compression_passes), 0)
            for _pass_index in range(max_passes):
                removed_this_pass = False
                n_compression_passes += 1
                # Try weak/inexpensive jumps first.  A successful removal keeps
                # the physical H-invariant certificate true while reducing the
                # number of implemented recyclers.
                order = sorted(
                    range(len(selected_jumps)),
                    key=lambda index: (
                        float(selected_candidates[index].inflow_norm),
                        int(selected_candidates[index].jump_nnz),
                        int(selected_candidates[index].region_index),
                        int(selected_candidates[index].recycler_index),
                    ),
                )
                for remove_index in order:
                    if len(selected_jumps) <= 1:
                        break
                    trial_jumps = tuple(
                        jump for index, jump in enumerate(selected_jumps) if index != remove_index
                    )
                    trial_report = diagnose_common_kernel_h_invariant_sector(
                        hamiltonian=hamiltonian,
                        jumps=trial_jumps,
                        target_states=state_basis,
                        kernel_tolerance=kernel_tolerance,
                    )
                    if not trial_report.likely_attractive_by_h_invariant_kernel:
                        continue
                    selected_jumps.pop(remove_index)
                    selected_candidates.pop(remove_index)
                    selected_ids = {id(candidate) for candidate in selected_candidates}
                    current_h_report = trial_report
                    n_compressed_jumps_removed += 1
                    removed_this_pass = True
                    break
                if not removed_this_pass:
                    break

            # Keep the selected-candidate readouts aligned with the compressed
            # jump list.  The per-step kernel metadata comes from the original
            # selection trajectory, but the final diagnostics below is recomputed
            # on the compressed list when requested.
            old_step_by_candidate_id = {id(step.candidate): step for step in steps}
            compressed_steps: list[RecycledManifoldJumpSelectionStep] = []
            cumulative_inflow_squared = 0.0
            max_target_jump_residual = 0.0
            for candidate in selected_candidates:
                old_step = old_step_by_candidate_id.get(id(candidate))
                cumulative_inflow_squared += float(candidate.inflow_norm) ** 2
                max_target_jump_residual = max(
                    max_target_jump_residual, float(candidate.dark_residual)
                )
                compressed_steps.append(
                    RecycledManifoldJumpSelectionStep(
                        step_index=len(compressed_steps),
                        candidate=candidate,
                        bad_common_jump_kernel_dimension=(
                            current_bad_dimension
                            if old_step is None
                            else old_step.bad_common_jump_kernel_dimension
                        ),
                        inflow_norm=float(np.sqrt(cumulative_inflow_squared)),
                        max_target_jump_residual=max_target_jump_residual,
                        n_selected_jumps=len(compressed_steps) + 1,
                    )
                )
            steps = compressed_steps

    unbundled_n_jumps = len(selected_jumps)
    unbundled_inflow_norm = (
        _multi_jump_projected_inflow_norm(
            jumps=tuple(selected_jumps),
            state_basis=state_basis,
        )
        if selected_jumps
        else 0.0
    )
    collective_groups: tuple[RecycledManifoldCollectiveRecyclerGroup, ...] = ()
    if selected_jumps and collective_recycler_strategy == "bundle_by_region_detector":
        selected_jumps, collective_groups = _bundle_recycled_jumps_by_region_detector(
            selected_candidates=tuple(selected_candidates),
            dim=dim,
            detector_matrices=detector_matrices,
            detector_coefficients=coefficients,
            embedding_contexts=embedding_contexts,
            rdms=rdms,
            state_basis=state_basis,
            recycler_source=recycler_source,
            weighting=collective_recycler_weighting,
            normalize_recyclers=normalize_collective_recyclers,
            tolerance=tolerance,
        )
        final_diagnostics = None

    selected_inflow_norm = (
        _multi_jump_projected_inflow_norm(
            jumps=tuple(selected_jumps),
            state_basis=state_basis,
        )
        if selected_jumps
        else 0.0
    )

    if selected_jumps and check_final_diagnostics:
        final_diagnostics = diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=tuple(selected_jumps),
            target_states=state_basis,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=False,
            liouvillian_spectrum_method="none",
        )

    return RecycledManifoldJumpSelectionReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        candidate_pool_size=len(pool),
        max_selected_jumps=int(max_selected_jumps),
        target_bad_kernel_dimension=int(target_bad_kernel_dimension),
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        jumps=tuple(selected_jumps),
        steps=tuple(steps),
        final_diagnostics=final_diagnostics,
        candidate_report=candidate_report,
        candidate_report_was_expanded=candidate_report_was_expanded,
        candidate_pool_was_limited=candidate_pool_was_limited,
        compression_strategy=compression_strategy,
        n_compression_passes=n_compression_passes,
        n_compressed_jumps_removed=n_compressed_jumps_removed,
        collective_recycler_strategy=collective_recycler_strategy,
        unbundled_n_jumps=unbundled_n_jumps,
        collective_groups=collective_groups,
        selected_inflow_norm=selected_inflow_norm,
        unbundled_inflow_norm=unbundled_inflow_norm,
    )
