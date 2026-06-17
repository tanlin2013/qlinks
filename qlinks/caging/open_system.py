from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.basis import basis_configs_from_build_result
from qlinks.caging.classification import (
    CageClassificationReport,
    InterferenceZeroReport,
    LocalTransitionPattern,
    ReducedIZMonitorDecomposition,
    group_reduced_iz_monitor_reports,
    group_reduced_iz_reports_by_connected_support,
    group_reduced_iz_reports_by_exact_support,
    select_reduced_iz_monitor_reports,
    support_key_for_zero_report,
    support_key_from_mask,
)
from qlinks.caging.support import CageRegionSupport, extract_cage_region_support
from qlinks.encoded import BinaryEncodedBasis
from qlinks.models.base import (
    HamiltonianBuilderName,
    ModelBuildResult,
    SparseBackendName,
)
from qlinks.models.local_terms import LocalTermDescriptor
from qlinks.open_system import (
    LindbladProblem,
    LocalMatrixUnitTerm,
    LocalRecyclingBuildResult,
    OpenSystemBackendName,
    RecyclingJumpSource,
    build_local_recycling_jumps_from_regions,
    density_matrix_from_state,
    diagnose_absorbing_projector_symmetry,
    diagnose_dark_subspace,
    diagnose_monitor_kernel_closure,
    embed_local_pattern_operator,
    lindblad_rhs_density_matrix,
    local_operator_matrix_unit_expansion,
    local_rank_one_matrix_unit_expansion,
    local_reduced_density_matrix_from_state,
    verify_lindblad_final_state,
)

MonitorSource = Literal[
    "local_hamiltonian_terms",
    "reduced_iz_operators",
]
MonitorPlaquettePolicy = Literal[
    "strict_inside",
    "touching",
]
ReducedIZMonitorContent = Literal[
    "offdiagonal_only",
    "offdiagonal_plus_potential",
]
JumpOperatorDesign = Literal[
    "kinetic_times_monitor",
    "kinetic_outside_monitor_inside",
    "hamiltonian_outside_monitor_inside",
    "monitor_recycler",
    "local_rdm_parent_projector",
    "local_rdm_parent_projector_recycling",
    "local_rdm_parent_projector_block_reset",
]
JumpPlaquettePolicy = Literal[
    "disjoint_outside",
    "crossing",
    "outside_or_crossing",
    "not_strictly_inside",
]
MonitorRecyclerHamiltonianShift = Literal["none", "target_energy", "local_expectation"]
MonitorRecyclerHamiltonianClosureSource = Literal[
    "global_hamiltonian",
    "local_hamiltonian_terms",
    "boundary_hamiltonian_terms",
    "touching_hamiltonian_terms",
]


def _record_construction_stage(
    timing_collector: dict[str, float] | None,
    stage_name: str,
    start_time: float,
) -> None:
    if timing_collector is None:
        return

    timing_collector[stage_name] = (
        timing_collector.get(stage_name, 0.0) + time.perf_counter() - start_time
    )


@dataclass(frozen=True, slots=True)
class _LocalTermMatrixCache:
    """Cache local Hamiltonian term matrices for one construction.

    Cage Lindblad constructions repeatedly reuse the same plaquette terms
    while assembling monitors, jumps, and diagnostic residuals. Keeping this
    tiny cache avoids rebuilding identical sparse matrices and also centralizes
    the builder choice needed for encoded bitmask build results.
    """

    model: Any
    build_result: ModelBuildResult
    builder: HamiltonianBuilderName
    backend: SparseBackendName
    _matrices: dict[tuple[str, str, int, tuple[int, ...]], sp.csr_array] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_matrices", {})

    @property
    def effective_builder(self) -> HamiltonianBuilderName:
        return _resolve_local_term_builder(
            self.builder,
            self.build_result,
        )

    def get(self, term: LocalTermDescriptor) -> sp.csr_array:
        key = _local_term_cache_key(term)

        if key not in self._matrices:
            matrix = self.model.build_local_term(
                term,
                self.build_result,
                builder=self.effective_builder,
                backend=self.backend,
            )

            if matrix is None:
                shape = self.build_result.hamiltonian.shape
                matrix = sp.csr_array(shape, dtype=np.complex128)

            self._matrices[key] = matrix.tocsr()

        return self._matrices[key]


def _resolve_local_term_builder(
    builder: HamiltonianBuilderName,
    build_result: ModelBuildResult,
) -> HamiltonianBuilderName:
    """Return a local-term builder compatible with ``build_result.basis``.

    ``build_type1_cage_lindblad_construction`` historically defaulted to
    ``builder="sparse"``. That default is convenient for array-basis builds,
    but it is incompatible with a ``ModelBuildResult`` produced by the bitmask
    builder because local terms must be assembled in the encoded basis ordering.
    In that case, use the bitmask local-term path automatically.
    """

    if isinstance(build_result.basis, BinaryEncodedBasis):
        return "bitmask"

    return builder


def _local_term_cache_key(
    term: LocalTermDescriptor,
) -> tuple[str, str, int, tuple[int, ...]]:
    return (
        str(term.operator_kind),
        str(term.term_kind),
        int(term.term_id),
        tuple(int(index) for index in term.support_links),
    )


def _local_term_matrix(
    *,
    model: Any,
    build_result: ModelBuildResult | None,
    term: LocalTermDescriptor,
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    local_term_cache: _LocalTermMatrixCache | None = None,
) -> sp.csr_array:
    if local_term_cache is not None:
        return local_term_cache.get(term)

    if build_result is None:
        matrix = model.build_local_term(
            term,
            build_result,
            builder=builder,
            backend=backend,
        )
    else:
        matrix = model.build_local_term(
            term,
            build_result,
            builder=_resolve_local_term_builder(builder, build_result),
            backend=backend,
        )

    if matrix is None:
        if build_result is None:
            raise ValueError("Cannot infer shape for an empty local term without build_result.")
        matrix = sp.csr_array(build_result.hamiltonian.shape, dtype=np.complex128)

    return matrix.tocsr()


@dataclass(frozen=True, slots=True)
class _SparseColumnAction:
    """Column-action representation for monomial sparse left factors.

    Local kinetic plaquette terms in QDM/QLM are partial permutations in the
    constrained basis: each input basis state maps to at most one output state.
    For products ``K @ M``, exploiting this column action avoids generic sparse
    matrix multiplication and turns the product into a vectorized remapping of
    the nonzero rows of ``M``.
    """

    row_by_column: NDArray[np.int64]
    data_by_column: NDArray[np.complex128]
    output_shape: tuple[int, int]

    @classmethod
    def from_left(cls, left: Any) -> "_SparseColumnAction | None":
        left_csc = left.tocsc()
        left_csc.sum_duplicates()
        counts = np.diff(left_csc.indptr)

        if np.any(counts > 1):
            return None

        n_columns = int(left_csc.shape[1])
        row_by_column = np.full(n_columns, -1, dtype=np.int64)
        data_by_column = np.zeros(n_columns, dtype=np.complex128)

        active_columns = np.flatnonzero(counts == 1)
        if len(active_columns) > 0:
            starts = left_csc.indptr[active_columns]
            row_by_column[active_columns] = left_csc.indices[starts].astype(
                np.int64,
                copy=False,
            )
            data_by_column[active_columns] = left_csc.data[starts].astype(
                np.complex128,
                copy=False,
            )

        return cls(
            row_by_column=row_by_column,
            data_by_column=data_by_column,
            output_shape=(int(left_csc.shape[0]), int(left_csc.shape[1])),
        )

    def apply(self, right: Any) -> sp.csr_array:
        right_coo = right.tocoo()
        if int(right_coo.shape[0]) != self.output_shape[1]:
            raise ValueError(
                "Sparse column action shape mismatch: "
                f"left has {self.output_shape[1]} columns but right has "
                f"{right_coo.shape[0]} rows."
            )

        if right_coo.nnz == 0:
            return sp.csr_array(
                (self.output_shape[0], int(right_coo.shape[1])),
                dtype=np.complex128,
            )

        mapped_rows = self.row_by_column[right_coo.row]
        nonzero_mask = mapped_rows >= 0

        if not np.any(nonzero_mask):
            return sp.csr_array(
                (self.output_shape[0], int(right_coo.shape[1])),
                dtype=np.complex128,
            )

        source_rows = right_coo.row[nonzero_mask]
        rows = mapped_rows[nonzero_mask]
        cols = right_coo.col[nonzero_mask]
        data = self.data_by_column[source_rows] * right_coo.data[nonzero_mask]

        return sp.coo_array(
            (data.astype(np.complex128, copy=False), (rows, cols)),
            shape=(self.output_shape[0], int(right_coo.shape[1])),
            dtype=np.complex128,
        ).tocsr()


@dataclass(slots=True)
class _LazySparseProductOperator:
    """Lazy sparse product ``left @ right`` for component jump operators.

    Reduced-IZ component decompositions can create hundreds of products
    ``K_p M_i``.  During construction we usually only need their residuals on
    the cage state, which can be checked from ``K_p (M_i |psi>)`` without
    materializing the full sparse product.  The full matrix is therefore built
    only when a downstream solver or caller asks for a sparse/dense matrix.
    """

    left: Any
    right: Any
    _matrix: sp.csr_array | None = field(default=None, init=False, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.left.shape[0]), int(self.right.shape[1]))

    @property
    def dtype(self):
        return np.result_type(
            getattr(self.left, "dtype", np.complex128),
            getattr(self.right, "dtype", np.complex128),
        )

    @property
    def T(self):
        return self.tocsr().T

    def tocsr(self) -> sp.csr_array:
        if self._matrix is None:
            self._matrix = _left_multiply_sparse_csr(self.left, self.right)
        return self._matrix

    def asformat(self, format: str):
        return self.tocsr().asformat(format)

    def astype(self, dtype):
        return self.tocsr().astype(dtype)

    def toarray(self):
        return self.tocsr().toarray()

    def conj(self):
        return self.tocsr().conj()

    def __matmul__(self, other):
        return self.tocsr() @ other


def _as_csr_matrix(operator: Any) -> sp.csr_array:
    if hasattr(operator, "tocsr"):
        return operator.tocsr()
    return sp.csr_array(operator)


@dataclass(slots=True)
class _LazySparseSumOperator:
    """Lazy sparse sum used for decomposed reduced-IZ monitors."""

    terms: tuple[Any, ...]
    shape: tuple[int, int]
    _state_action: NDArray[np.complex128] | None = None
    _matrix: sp.csr_array | None = field(default=None, init=False, repr=False)

    @property
    def dtype(self):
        return np.complex128

    @property
    def T(self):
        return self.tocsr().T

    def tocsr(self) -> sp.csr_array:
        if self._matrix is None:
            if len(self.terms) == 0:
                self._matrix = sp.csr_array(self.shape, dtype=np.complex128)
            else:
                total = sp.csr_array(self.shape, dtype=np.complex128)
                for term in self.terms:
                    total = total + _as_csr_matrix(term)
                self._matrix = total.tocsr()
        return self._matrix

    def asformat(self, format: str):
        return self.tocsr().asformat(format)

    def astype(self, dtype):
        return self.tocsr().astype(dtype)

    def toarray(self):
        return self.tocsr().toarray()

    def conj(self):
        return self.tocsr().conj()

    def __matmul__(self, other):
        return self.tocsr() @ other


@dataclass(slots=True)
class _LazyReducedIZMonitorOperator:
    """Lazy reduced-IZ sparse monitor assembled from classification reports."""

    reports: tuple[InterferenceZeroReport, ...]
    basis_configs: NDArray[np.integer] | None
    config_to_index: dict[tuple[int, ...], int] | None
    shape: tuple[int, int]
    use_collective_coefficients: bool
    build_result: ModelBuildResult | None = None
    assembly_cache: Any | None = None
    _state_action: NDArray[np.complex128] | None = None
    _matrix: sp.csr_array | None = field(default=None, init=False, repr=False)

    @property
    def dtype(self):
        return np.complex128

    @property
    def T(self):
        return self.tocsr().T

    def tocsr(self) -> sp.csr_array:
        if self._matrix is None:
            basis_configs = self.basis_configs
            if basis_configs is None:
                if self.build_result is None:
                    raise ValueError(
                        "Lazy reduced-IZ monitor materialization requires "
                        "basis_configs or build_result."
                    )
                basis_configs = basis_configs_from_build_result(self.build_result)
                self.basis_configs = basis_configs

            assembly_cache = self.assembly_cache
            if assembly_cache is None:
                assembly_cache = _ReducedIZAssemblyCache(
                    basis_configs=basis_configs,
                    config_to_index=self.config_to_index,
                )
                self.assembly_cache = assembly_cache

            self._matrix = _build_reduced_iz_monitor_from_reports(
                reports=self.reports,
                basis_configs=basis_configs,
                config_to_index=self.config_to_index,
                shape=self.shape,
                use_collective_coefficients=self.use_collective_coefficients,
                assembly_cache=assembly_cache,
            ).tocsr()
        return self._matrix

    def asformat(self, format: str):
        return self.tocsr().asformat(format)

    def astype(self, dtype):
        return self.tocsr().astype(dtype)

    def toarray(self):
        return self.tocsr().toarray()

    def conj(self):
        return self.tocsr().conj()

    def __matmul__(self, other):
        return self.tocsr() @ other


@dataclass(slots=True)
class _LazyLocalTermMonitorProductOperator:
    """Lazy product ``K_p @ M_i`` without eager local-term matrix assembly."""

    model: Any
    build_result: ModelBuildResult
    term: LocalTermDescriptor
    builder: HamiltonianBuilderName
    backend: SparseBackendName
    local_term_cache: _LocalTermMatrixCache | None
    monitor: Any
    _matrix: sp.csr_array | None = field(default=None, init=False, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.build_result.hamiltonian.shape[0]), int(self.monitor.shape[1]))

    @property
    def dtype(self):
        return np.complex128

    @property
    def T(self):
        return self.tocsr().T

    def tocsr(self) -> sp.csr_array:
        if self._matrix is None:
            kinetic_matrix = _local_term_matrix(
                model=self.model,
                build_result=self.build_result,
                term=self.term,
                builder=self.builder,
                backend=self.backend,
                local_term_cache=self.local_term_cache,
            )
            self._matrix = _left_multiply_sparse_csr(kinetic_matrix, self.monitor)
        return self._matrix

    def asformat(self, format: str):
        return self.tocsr().asformat(format)

    def astype(self, dtype):
        return self.tocsr().astype(dtype)

    def toarray(self):
        return self.tocsr().toarray()

    def conj(self):
        return self.tocsr().conj()

    def __matmul__(self, other):
        return self.tocsr() @ other


def _lazy_local_term_left_multiply_monitor(
    *,
    model: Any,
    build_result: ModelBuildResult,
    term: LocalTermDescriptor,
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    local_term_cache: _LocalTermMatrixCache | None,
    monitor: Any,
) -> _LazyLocalTermMonitorProductOperator:
    return _LazyLocalTermMonitorProductOperator(
        model=model,
        build_result=build_result,
        term=term,
        builder=builder,
        backend=backend,
        local_term_cache=local_term_cache,
        monitor=monitor,
    )


def _state_action_from_lazy_operator(operator: Any) -> NDArray[np.complex128] | None:
    action = getattr(operator, "_state_action", None)
    if action is None:
        return None
    return np.asarray(action, dtype=np.complex128)


def _state_action_from_component_group(
    component_group: Any,
    *,
    state: NDArray[np.complex128],
) -> NDArray[np.complex128] | None:
    action = getattr(component_group, "state_action_vector", None)
    if action is None:
        return None

    action_array = np.asarray(action, dtype=np.complex128)
    if action_array.shape != state.shape:
        return None

    return action_array


def _report_groups_from_component_groups(
    *,
    selected_reports: tuple[InterferenceZeroReport, ...],
    component_groups: tuple[Any, ...],
) -> tuple[tuple[InterferenceZeroReport, ...], ...]:
    reports_by_zero_index = {int(report.zero_index): report for report in selected_reports}
    report_groups: list[tuple[InterferenceZeroReport, ...]] = []

    for component_group in component_groups:
        group_reports: list[InterferenceZeroReport] = []
        for zero_index in component_group.zero_indices:
            try:
                group_reports.append(reports_by_zero_index[int(zero_index)])
            except KeyError as exc:
                raise ValueError(
                    "Reduced-IZ component group references a zero index that "
                    "is absent from the selected monitor reports: "
                    f"{int(zero_index)}."
                ) from exc
        report_groups.append(tuple(group_reports))

    return tuple(report_groups)


def _lazy_left_multiply_sparse_csr(left: Any, right: Any) -> _LazySparseProductOperator:
    return _LazySparseProductOperator(left=left, right=right)


def _left_multiply_sparse_csr(left: Any, right: Any) -> sp.csr_array:
    """Return ``left @ right`` as a CSR sparse array.

    Local plaquette kinetic terms are usually partial permutations in the
    constrained basis.  In that common case, use a vectorized column-action
    remapping of ``right`` rather than generic sparse matrix multiplication.
    Fall back to SciPy's sparse product for general sparse left factors.
    """

    right_csr = _as_csr_matrix(right)
    action = _SparseColumnAction.from_left(left)
    if action is not None:
        return action.apply(right_csr)

    result = left.tocsr() @ right_csr
    return result.tocsr()


@dataclass(frozen=True, slots=True)
class ReducedIZMonitorComponent:
    """One frustration-free reduced-IZ monitor component."""

    component_id: int
    monitor: Any
    zero_indices: tuple[int, ...]
    support_variables: tuple[int, ...]
    support_plaquette_ids: tuple[int, ...]
    monitor_plaquette_ids: tuple[int, ...]
    jump_plaquette_ids: tuple[int, ...]
    z_value: complex | None
    n_potential_terms: int
    monitor_residual: float
    jump_residuals: tuple[float, ...]

    @property
    def max_jump_residual(self) -> float:
        return max(self.jump_residuals) if self.jump_residuals else 0.0

    @property
    def n_terms(self) -> int:
        return len(self.zero_indices)

    @property
    def n_jumps(self) -> int:
        return len(self.jump_plaquette_ids)


@dataclass(frozen=True, slots=True)
class LocalRecyclerReadout:
    """Notebook-friendly description of one selected local recycler.

    The local recycler is a rank-one operator ``|alpha><beta|`` embedded in the
    constrained global basis.  ``alpha`` lies in the local support of the target
    reduced density matrix, while ``beta`` lies in the local null space.  For a
    monitor-recycler jump, the actual Lindblad jump is ``|alpha><beta| P_i``.
    """

    recycler_index: int
    component_index: int | None
    variable_indices: tuple[int, ...]
    alpha_index: int
    beta_index: int
    local_patterns: tuple[tuple[int, ...], ...]
    local_alpha_vector: NDArray[np.complex128]
    local_beta_vector: NDArray[np.complex128]
    alpha_support_indices: tuple[int, ...]
    beta_support_indices: tuple[int, ...]
    alpha_support_patterns: tuple[tuple[int, ...], ...]
    beta_support_patterns: tuple[tuple[int, ...], ...]
    inflow_norm: float
    outflow_norm: float
    target_residual: float
    projector_commutator_norm: float
    jump_nnz: int
    two_pattern_structure: Any | None
    n_matrix_unit_terms: int
    matrix_unit_terms: tuple[LocalMatrixUnitTerm, ...]
    matrix_unit_terms_truncated: bool
    hamiltonian_closure_orders: tuple[int, ...] = ()
    hamiltonian_closure_source: MonitorRecyclerHamiltonianClosureSource = "global_hamiltonian"
    n_hamiltonian_closure_jumps: int = 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "recycler_index": self.recycler_index,
            "component_index": self.component_index,
            "variable_indices": self.variable_indices,
            "alpha_index": self.alpha_index,
            "beta_index": self.beta_index,
            "n_local_patterns": len(self.local_patterns),
            "alpha_support_indices": self.alpha_support_indices,
            "beta_support_indices": self.beta_support_indices,
            "alpha_support_patterns": self.alpha_support_patterns,
            "beta_support_patterns": self.beta_support_patterns,
            "inflow_norm": self.inflow_norm,
            "outflow_norm": self.outflow_norm,
            "target_residual": self.target_residual,
            "projector_commutator_norm": self.projector_commutator_norm,
            "jump_nnz": self.jump_nnz,
            "has_two_pattern_structure": self.two_pattern_structure is not None,
            "n_matrix_unit_terms": self.n_matrix_unit_terms,
            "matrix_unit_terms_truncated": self.matrix_unit_terms_truncated,
            "hamiltonian_closure_orders": self.hamiltonian_closure_orders,
            "hamiltonian_closure_source": self.hamiltonian_closure_source,
            "n_hamiltonian_closure_jumps": self.n_hamiltonian_closure_jumps,
            "matrix_unit_terms": tuple(
                {
                    "coefficient": term.coefficient,
                    "target_pattern": term.target_pattern,
                    "source_pattern": term.source_pattern,
                }
                for term in self.matrix_unit_terms
            ),
        }


@dataclass(frozen=True, slots=True)
class CageLindbladConstruction:
    """Open-system construction associated with one cage state."""

    cage_state: NDArray[np.complex128]
    region: CageRegionSupport
    z_value: complex | None

    inside_plaquette_ids: tuple[int, ...]
    outside_plaquette_ids: tuple[int, ...]
    crossing_plaquette_ids: tuple[int, ...]

    monitor: Any
    jumps: tuple[Any, ...]
    n_jumps: int
    n_component_jumps: int
    n_global_jump_terms: int

    open_system_backend: OpenSystemBackendName

    monitor_source: MonitorSource
    reduced_iz_monitor_decomposition: ReducedIZMonitorDecomposition
    reduced_iz_monitor_content: ReducedIZMonitorContent
    n_reduced_iz_monitor_terms: int
    reduced_iz_monitor_zero_indices: tuple[int, ...]
    monitor_components: tuple[ReducedIZMonitorComponent, ...]
    component_z_values: tuple[complex | None, ...]

    jump_operator_design: JumpOperatorDesign
    monitor_plaquette_policy: MonitorPlaquettePolicy
    jump_plaquette_policy: JumpPlaquettePolicy

    monitor_plaquette_ids: tuple[int, ...]
    jump_plaquette_ids: tuple[int, ...]

    kinetic_terms_monitor: tuple[LocalTermDescriptor, ...]
    potential_terms_monitor: tuple[LocalTermDescriptor, ...]
    kinetic_terms_jump: tuple[LocalTermDescriptor, ...]

    recycling_jump_source: RecyclingJumpSource
    n_recycling_jumps: int
    recycling_jump_variable_indices: tuple[tuple[int, ...], ...]
    recycling_jump_alpha_beta_indices: tuple[tuple[int, int], ...]
    recycling_two_pattern_count: int
    recycling_build_result: LocalRecyclingBuildResult | None

    monitor_residual: float
    max_jump_residual: float
    jump_residuals: tuple[float, ...]
    local_rdm_parent_projector_rate: float = 1.0
    local_rdm_reset_rate: float = 1.0
    n_local_rdm_parent_projector_jumps: int = 0
    n_local_rdm_block_reset_jumps: int = 0
    liouvillian_residual: float | None = None
    monitor_recycler_hamiltonian_closure_order: int = 0
    monitor_recycler_hamiltonian_shift: MonitorRecyclerHamiltonianShift = "target_energy"
    monitor_recycler_hamiltonian_closure_source: MonitorRecyclerHamiltonianClosureSource = (
        "global_hamiltonian"
    )
    monitor_recycler_jump_closure_orders: tuple[int, ...] = ()

    def __repr__(self) -> str:
        return (
            "CageLindbladConstruction("
            f"monitor_source={self.monitor_source!r}, "
            f"n_monitor_components={len(self.monitor_components)}, "
            f"jump_operator_design={self.jump_operator_design!r}, "
            f"region_size={self.region.region_size}, "
            f"n_jumps={self.n_jumps}, "
            f"n_component_jumps={self.n_component_jumps}, "
            f"n_global_jump_terms={self.n_global_jump_terms}, "
            f"n_recycling_jumps={self.n_recycling_jumps}, "
            f"monitor_recycler_hamiltonian_closure_order={
                self.monitor_recycler_hamiltonian_closure_order
            }, "
            f"monitor_residual={self.monitor_residual:.3e}, "
            f"max_jump_residual={self.max_jump_residual:.3e}, "
            f"liouvillian_residual={self.liouvillian_residual!r}"
            ")"
        )

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact summary useful for notebooks and logging."""
        return {
            "region_size": self.region.region_size,
            "monitor_source": self.monitor_source,
            "jump_operator_design": self.jump_operator_design,
            "monitor_recycler_hamiltonian_closure_order": (
                self.monitor_recycler_hamiltonian_closure_order
            ),
            "monitor_recycler_hamiltonian_shift": self.monitor_recycler_hamiltonian_shift,
            "monitor_recycler_hamiltonian_closure_source": (
                self.monitor_recycler_hamiltonian_closure_source
            ),
            "monitor_recycler_jump_closure_orders": self.monitor_recycler_jump_closure_orders,
            "monitor_plaquette_policy": self.monitor_plaquette_policy,
            "jump_plaquette_policy": self.jump_plaquette_policy,
            "n_inside_plaquettes": len(self.inside_plaquette_ids),
            "n_outside_plaquettes": len(self.outside_plaquette_ids),
            "n_crossing_plaquettes": len(self.crossing_plaquette_ids),
            "reduced_iz_monitor_decomposition": self.reduced_iz_monitor_decomposition,
            "reduced_iz_monitor_content": self.reduced_iz_monitor_content,
            "n_monitor_components": len(self.monitor_components),
            "component_support_plaquette_ids": tuple(
                component.support_plaquette_ids for component in self.monitor_components
            ),
            "component_support_sizes": tuple(
                len(component.support_variables) for component in self.monitor_components
            ),
            "component_support_plaquette_counts": tuple(
                len(component.support_plaquette_ids) for component in self.monitor_components
            ),
            "component_z_values": self.component_z_values,
            "component_monitor_residuals": tuple(
                component.monitor_residual for component in self.monitor_components
            ),
            "component_max_jump_residuals": tuple(
                component.max_jump_residual for component in self.monitor_components
            ),
            "n_monitor_plaquettes": len(self.monitor_plaquette_ids),
            "n_jump_plaquettes": len(self.jump_plaquette_ids),
            "n_jumps": self.n_jumps,
            "n_component_jumps": self.n_component_jumps,
            "n_global_jump_terms": self.n_global_jump_terms,
            "n_reduced_iz_monitor_terms": self.n_reduced_iz_monitor_terms,
            "z_value": self.z_value,
            "recycling_jump_source": self.recycling_jump_source,
            "n_recycling_jumps": self.n_recycling_jumps,
            "recycling_jump_variable_indices": self.recycling_jump_variable_indices,
            "recycling_jump_alpha_beta_indices": self.recycling_jump_alpha_beta_indices,
            "recycling_two_pattern_count": self.recycling_two_pattern_count,
            "local_rdm_parent_projector_rate": self.local_rdm_parent_projector_rate,
            "local_rdm_reset_rate": self.local_rdm_reset_rate,
            "n_local_rdm_parent_projector_jumps": self.n_local_rdm_parent_projector_jumps,
            "n_local_rdm_block_reset_jumps": self.n_local_rdm_block_reset_jumps,
            "monitor_residual": self.monitor_residual,
            "max_jump_residual": self.max_jump_residual,
            "liouvillian_residual": self.liouvillian_residual,
        }

    def recycler_readouts(
        self,
        *,
        tolerance: float = 1e-10,
        max_matrix_unit_terms: int = 64,
    ) -> tuple[LocalRecyclerReadout, ...]:
        """Return notebook-friendly descriptions of selected local recyclers.

        For ``jump_operator_design='monitor_recycler'``, these readouts describe
        the ``V_i`` factors in ``L_i = V_i P_i``.  For older jump designs, they
        describe any appended local recycling jumps.
        """
        if self.recycling_build_result is None:
            return ()

        component_index_by_region = {
            tuple(int(index) for index in component.support_variables): component_index
            for component_index, component in enumerate(self.monitor_components)
        }

        readouts: list[LocalRecyclerReadout] = []
        for recycler_index, selection in enumerate(self.recycling_build_result.selections):
            candidate = selection.candidate
            variable_indices = tuple(int(index) for index in candidate.variable_indices)
            scan_result = _find_recycling_scan_result(
                self.recycling_build_result,
                variable_indices=variable_indices,
            )
            local_patterns = scan_result.reduced_density_matrix.local_patterns
            local_operator = getattr(candidate, "local_operator", None)
            if local_operator is None:
                alpha_support_indices = _local_vector_support_indices(
                    candidate.local_alpha_vector,
                    tolerance=tolerance,
                )
                beta_support_indices = _local_vector_support_indices(
                    candidate.local_beta_vector,
                    tolerance=tolerance,
                )
                matrix_unit_terms = local_rank_one_matrix_unit_expansion(
                    local_patterns=local_patterns,
                    alpha=candidate.local_alpha_vector,
                    beta=candidate.local_beta_vector,
                    tolerance=tolerance,
                )
            else:
                operator = np.asarray(local_operator, dtype=np.complex128)
                alpha_support_indices = tuple(
                    int(index)
                    for index in np.flatnonzero(np.linalg.norm(operator, axis=1) > tolerance)
                )
                beta_support_indices = tuple(
                    int(index)
                    for index in np.flatnonzero(np.linalg.norm(operator, axis=0) > tolerance)
                )
                matrix_unit_terms = local_operator_matrix_unit_expansion(
                    local_patterns=local_patterns,
                    local_operator=operator,
                    tolerance=tolerance,
                )
            truncated = len(matrix_unit_terms) > int(max_matrix_unit_terms)
            if truncated:
                shown_terms = matrix_unit_terms[: int(max_matrix_unit_terms)]
            else:
                shown_terms = matrix_unit_terms

            readouts.append(
                LocalRecyclerReadout(
                    recycler_index=int(recycler_index),
                    component_index=component_index_by_region.get(variable_indices),
                    variable_indices=variable_indices,
                    alpha_index=int(candidate.alpha_index),
                    beta_index=int(candidate.beta_index),
                    local_patterns=local_patterns,
                    local_alpha_vector=candidate.local_alpha_vector,
                    local_beta_vector=candidate.local_beta_vector,
                    alpha_support_indices=alpha_support_indices,
                    beta_support_indices=beta_support_indices,
                    alpha_support_patterns=tuple(
                        local_patterns[index] for index in alpha_support_indices
                    ),
                    beta_support_patterns=tuple(
                        local_patterns[index] for index in beta_support_indices
                    ),
                    inflow_norm=float(candidate.inflow_norm),
                    outflow_norm=float(candidate.outflow_norm),
                    target_residual=float(candidate.target_residual),
                    projector_commutator_norm=float(candidate.projector_commutator_norm),
                    jump_nnz=int(candidate.jump.nnz),
                    two_pattern_structure=selection.two_pattern_structure,
                    n_matrix_unit_terms=len(matrix_unit_terms),
                    matrix_unit_terms=tuple(shown_terms),
                    matrix_unit_terms_truncated=bool(truncated),
                    hamiltonian_closure_orders=(
                        tuple(range(int(self.monitor_recycler_hamiltonian_closure_order) + 1))
                        if self.jump_operator_design == "monitor_recycler"
                        else ()
                    ),
                    hamiltonian_closure_source=(self.monitor_recycler_hamiltonian_closure_source),
                    n_hamiltonian_closure_jumps=(
                        int(self.monitor_recycler_hamiltonian_closure_order) + 1
                        if self.jump_operator_design == "monitor_recycler"
                        else 0
                    ),
                )
            )

        return tuple(readouts)

    def recycler_summary(self, *, tolerance: float = 1e-10) -> tuple[dict[str, object], ...]:
        """Return compact dictionaries describing the selected local recyclers."""
        return tuple(
            readout.to_summary_dict() for readout in self.recycler_readouts(tolerance=tolerance)
        )

    def diagnose_monitor_kernel_closure(
        self,
        *,
        hamiltonian: Any,
        closure_order: int = 1,
        tolerance: float = 1e-10,
    ):
        """Diagnose monitor-kernel closure for this construction."""
        monitors = tuple(component.monitor for component in self.monitor_components)
        if len(monitors) == 0:
            monitors = (self.monitor,)
        return diagnose_monitor_kernel_closure(
            hamiltonian=hamiltonian,
            monitors=monitors,
            target_state=self.cage_state,
            closure_order=closure_order,
            tolerance=tolerance,
        )

    def to_rich(self):
        """Return a rich renderable summary of the construction."""
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "CageLindbladConstruction.to_rich() requires the optional "
                "`rich` package. Install it with `pip install rich`."
            ) from exc

        title = Text("Cage Lindblad Construction", style="bold cyan")

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()

        overview.add_row("monitor source", str(self.monitor_source))
        overview.add_row(
            "reduced-IZ decomposition",
            str(self.reduced_iz_monitor_decomposition),
        )
        overview.add_row(
            "reduced-IZ content",
            str(self.reduced_iz_monitor_content),
        )
        overview.add_row("jump design", str(self.jump_operator_design))
        overview.add_row("monitor policy", str(self.monitor_plaquette_policy))
        overview.add_row("jump policy", str(self.jump_plaquette_policy))
        overview.add_row("z value", _format_complex_or_none(self.z_value))
        overview.add_row("open-system backend", str(self.open_system_backend))

        geometry = Table(title="Region / plaquette partition")
        geometry.add_column("quantity", style="bold")
        geometry.add_column("value", justify="right")

        geometry.add_row("region variables", str(self.region.region_size))
        geometry.add_row("inside plaquettes", str(len(self.inside_plaquette_ids)))
        geometry.add_row("outside plaquettes", str(len(self.outside_plaquette_ids)))
        geometry.add_row("crossing plaquettes", str(len(self.crossing_plaquette_ids)))
        geometry.add_row("monitor plaquettes", str(len(self.monitor_plaquette_ids)))
        geometry.add_row("jump plaquettes", str(len(self.jump_plaquette_ids)))
        geometry.add_row("jumps", str(self.n_jumps))
        geometry.add_row("component jumps", str(self.n_component_jumps))
        geometry.add_row("global outside/crossing jumps", str(self.n_global_jump_terms))

        diagnostics = Table(title="Dark-state diagnostics")
        diagnostics.add_column("quantity", style="bold")
        diagnostics.add_column("value", justify="right")
        diagnostics.add_column("status", justify="center")

        diagnostics.add_row(
            "||M_R psi||",
            _format_float(self.monitor_residual),
            _status_for_residual(self.monitor_residual),
        )
        diagnostics.add_row(
            "max_p ||J_p psi||",
            _format_float(self.max_jump_residual),
            _status_for_residual(self.max_jump_residual),
        )
        diagnostics.add_row(
            "||L(rho_psi)||",
            _format_float_or_none(self.liouvillian_residual),
            _status_for_residual(self.liouvillian_residual),
        )

        reduced_iz = Table(title="Reduced-IZ monitor")
        reduced_iz.add_column("quantity", style="bold")
        reduced_iz.add_column("value", justify="right")

        reduced_iz.add_row(
            "n reduced IZ terms",
            str(self.n_reduced_iz_monitor_terms),
        )
        reduced_iz.add_row(
            "n monitor components",
            str(len(self.monitor_components)),
        )
        reduced_iz.add_row(
            "zero indices",
            _format_index_tuple(self.reduced_iz_monitor_zero_indices),
        )

        recycling = Table(title="Local RDM recycling")
        recycling.add_column("quantity", style="bold")
        recycling.add_column("value", justify="right")
        recycling.add_row("source", str(self.recycling_jump_source))
        recycling.add_row("n jumps", str(self.n_recycling_jumps))
        recycling.add_row("two-pattern jumps", str(self.recycling_two_pattern_count))
        recycling.add_row(
            "alpha/beta",
            _format_tuple_of_pairs(self.recycling_jump_alpha_beta_indices),
        )

        plaquette_ids = Table(title="Plaquette ids")
        plaquette_ids.add_column("group", style="bold")
        plaquette_ids.add_column("ids")

        plaquette_ids.add_row(
            "inside",
            _format_index_tuple(self.inside_plaquette_ids),
        )
        plaquette_ids.add_row(
            "outside",
            _format_index_tuple(self.outside_plaquette_ids),
        )
        plaquette_ids.add_row(
            "crossing",
            _format_index_tuple(self.crossing_plaquette_ids),
        )
        plaquette_ids.add_row(
            "monitor",
            _format_index_tuple(self.monitor_plaquette_ids),
        )
        plaquette_ids.add_row(
            "jump",
            _format_index_tuple(self.jump_plaquette_ids),
        )

        components = Table(title="Monitor components")
        components.add_column("id", justify="right")
        components.add_column("n terms", justify="right")
        components.add_column("|links|", justify="right")
        components.add_column("R_i plaquettes")
        components.add_column("n jumps", justify="right")
        components.add_column("z_i", justify="right")
        components.add_column("n V terms", justify="right")
        components.add_column("||M_i psi||", justify="right")
        components.add_column("max ||J psi||", justify="right")

        for component in self.monitor_components:
            components.add_row(
                str(component.component_id),
                str(component.n_terms),
                str(len(component.support_variables)),
                _format_index_tuple(component.support_plaquette_ids),
                str(component.n_jumps),
                _format_complex_or_none(component.z_value),
                str(component.n_potential_terms),
                _format_float(component.monitor_residual),
                _format_float(component.max_jump_residual),
            )

        return Panel(
            Group(
                overview,
                geometry,
                diagnostics,
                reduced_iz,
                recycling,
                plaquette_ids,
                components,
            ),
            title=title,
            border_style="cyan",
        )

    def diagnose_dark_subspace(
        self,
        *,
        hamiltonian: Any,
        kernel_tolerance: float = 1e-10,
        liouvillian_zero_tolerance: float = 1e-9,
        check_liouvillian_spectrum: bool = True,
        max_liouvillian_dense_dimension: int = 4096,
        liouvillian_spectrum_method: str = "auto",
        sparse_liouvillian_eigenvalue_count: int = 16,
    ):
        return diagnose_dark_subspace(
            hamiltonian=hamiltonian,
            jumps=list(self.jumps),
            target_state=self.cage_state,
            backend=self.open_system_backend,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=check_liouvillian_spectrum,
            max_liouvillian_dense_dimension=max_liouvillian_dense_dimension,
            liouvillian_spectrum_method=liouvillian_spectrum_method,
            sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
        )

    def diagnose_absorbing_projector_symmetry(
        self,
        *,
        hamiltonian: Any,
        tolerance: float = 1e-10,
    ):
        """Diagnose whether P_psi is an absorbing-projector symmetry."""
        return diagnose_absorbing_projector_symmetry(
            hamiltonian=hamiltonian,
            jumps=list(self.jumps),
            target_state=self.cage_state,
            backend=self.open_system_backend,
            tolerance=tolerance,
        )

    def to_lindblad_problem(
        self,
        *,
        hamiltonian: Any,
        backend: str | None = None,
    ) -> LindbladProblem:
        return LindbladProblem(
            hamiltonian=hamiltonian,
            jumps=self.jumps,
            backend=self.open_system_backend if backend is None else backend,
        )

    def build_liouvillian(
        self,
        hamiltonian: Any,
        *,
        sparse_format: str = "csc",
        backend: str | None = None,
    ) -> Any:
        problem = self.to_lindblad_problem(
            hamiltonian=hamiltonian,
            backend=backend,
        )
        return problem.build_liouvillian(sparse_format=sparse_format)

    def verify_final_state(
        self,
        rho,
        *,
        hamiltonian: Any,
        atol: float = 1e-10,
        backend: str | None = None,
    ):
        return verify_lindblad_final_state(
            rho,
            hamiltonian=hamiltonian,
            jumps=list(self.jumps),
            target_state=self.cage_state,
            atol=atol,
            backend=self.open_system_backend if backend is None else backend,
        )

    def evolve(
        self,
        *,
        hamiltonian: Any,
        density_matrix_initial: Any,
        times: NDArray[np.float64],
        options=None,
    ):
        problem = self.to_lindblad_problem(
            hamiltonian=hamiltonian,
        )
        return problem.evolve(
            density_matrix_initial,
            times,
            options=options,
        )


def build_type1_cage_lindblad_construction(
    *,
    model: Any,
    build_result: ModelBuildResult,
    cage_state: NDArray[np.complex128],
    classification_report: CageClassificationReport,
    z_value: complex | None = None,
    builder: HamiltonianBuilderName = "sparse",
    backend: SparseBackendName = "scipy",
    open_system_backend: OpenSystemBackendName = "scipy",
    monitor_source: MonitorSource = "local_hamiltonian_terms",
    reduced_iz_monitor_decomposition: ReducedIZMonitorDecomposition = "single_sum",
    reduced_iz_monitor_content: ReducedIZMonitorContent = "offdiagonal_only",
    monitor_plaquette_policy: MonitorPlaquettePolicy = "strict_inside",
    jump_plaquette_policy: JumpPlaquettePolicy = "outside_or_crossing",
    jump_operator_design: JumpOperatorDesign = "kinetic_outside_monitor_inside",
    recycling_jump_source: RecyclingJumpSource = "none",
    max_recycling_jumps_per_region: int = 1,
    deduplicate_recycling_regions: bool = False,
    recycling_rdm_tolerance: float = 1e-10,
    recycling_dark_tolerance: float = 1e-10,
    recycling_inflow_tolerance: float = 1e-12,
    recycling_prefer_sparse: bool = True,
    recycling_two_pattern_tolerance: float = 1e-8,
    monitor_recycler_hamiltonian_closure_order: int = 0,
    monitor_recycler_hamiltonian_shift: MonitorRecyclerHamiltonianShift = "target_energy",
    monitor_recycler_hamiltonian_closure_source: (
        MonitorRecyclerHamiltonianClosureSource
    ) = "global_hamiltonian",
    local_rdm_parent_projector_rate: float = 1.0,
    local_rdm_reset_rate: float = 1.0,
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
    use_collective_coefficients: bool = True,
    check_liouvillian: bool = True,
    compute_jump_residuals: bool = True,
    timing_collector: dict[str, float] | None = None,
    residual_tolerance: float = 1e-10,
) -> CageLindbladConstruction:
    """Build M_R and J_p = K_p M_R for a type-1 cage.

    This assumes the cage has type-1 signature (0, z), so that K|psi> = 0
    and the relevant potential value is z on the contributing basis states.
    """
    psi = np.asarray(cage_state, dtype=np.complex128)
    norm = np.linalg.norm(psi)

    if norm == 0:
        raise ValueError("cage_state must be nonzero.")

    psi = psi / norm

    if local_rdm_parent_projector_rate < 0.0:
        raise ValueError("local_rdm_parent_projector_rate must be nonnegative.")
    if local_rdm_reset_rate < 0.0:
        raise ValueError("local_rdm_reset_rate must be nonnegative.")

    if monitor_recycler_hamiltonian_closure_order < 0:
        raise ValueError("monitor_recycler_hamiltonian_closure_order must be nonnegative.")
    local_hamiltonian_closure_sources = {
        "local_hamiltonian_terms",
        "boundary_hamiltonian_terms",
        "touching_hamiltonian_terms",
    }
    if monitor_recycler_hamiltonian_closure_source not in {
        "global_hamiltonian",
        *local_hamiltonian_closure_sources,
    }:
        raise ValueError(
            'monitor_recycler_hamiltonian_closure_source must be "global_hamiltonian", '
            '"local_hamiltonian_terms", "boundary_hamiltonian_terms", or '
            '"touching_hamiltonian_terms".'
        )
    if (
        monitor_recycler_hamiltonian_closure_source == "global_hamiltonian"
        and monitor_recycler_hamiltonian_shift == "local_expectation"
    ):
        raise ValueError(
            'monitor_recycler_hamiltonian_shift="local_expectation" is only '
            "valid with local monitor-recycler Hamiltonian closure sources."
        )
    if (
        monitor_recycler_hamiltonian_closure_source in local_hamiltonian_closure_sources
        and monitor_recycler_hamiltonian_shift == "target_energy"
    ):
        raise ValueError(
            'monitor_recycler_hamiltonian_shift="target_energy" is only valid '
            'for global-H closure. Use "local_expectation" or "none" '
            "with local-H closure."
        )

    stage_start = time.perf_counter()
    region = extract_cage_region_support(
        classification_report,
        include_collective_cancellation=include_collective_cancellation,
    )
    _record_construction_stage(timing_collector, "extract_region", stage_start)

    stage_start = time.perf_counter()
    kinetic_terms, potential_terms, potential_by_pid = _plaquette_local_terms_by_operator_kind(
        model
    )

    inside_kinetic, outside_kinetic, crossing_kinetic = _partition_plaquette_terms_by_region(
        kinetic_terms,
        region.variable_index_set,
    )
    monitor_kinetic_terms = _select_monitor_terms(
        inside_terms=inside_kinetic,
        outside_terms=outside_kinetic,
        crossing_terms=crossing_kinetic,
        policy=monitor_plaquette_policy,
    )
    monitor_potential_terms = tuple(
        potential_by_pid[int(term.term_id)]
        for term in monitor_kinetic_terms
        if int(term.term_id) in potential_by_pid
    )
    jump_kinetic_terms = _select_jump_terms(
        inside_terms=inside_kinetic,
        outside_terms=outside_kinetic,
        crossing_terms=crossing_kinetic,
        policy=jump_plaquette_policy,
    )
    _record_construction_stage(timing_collector, "select_terms", stage_start)

    dim = build_result.hamiltonian.shape[0]
    shape = build_result.hamiltonian.shape
    if (
        monitor_source == "reduced_iz_operators"
        and reduced_iz_monitor_content == "offdiagonal_only"
    ):
        identity = None
    else:
        identity = sp.identity(dim, format="csr", dtype=np.complex128)
    local_term_cache = _LocalTermMatrixCache(
        model=model,
        build_result=build_result,
        builder=builder,
        backend=backend,
    )

    monitor_components: tuple[ReducedIZMonitorComponent, ...] = ()
    component_jumps: tuple[Any, ...] | None = None

    stage_start = time.perf_counter()

    if monitor_source == "local_hamiltonian_terms":
        kinetic_inside_matrix = _sum_local_terms(
            model=model,
            build_result=build_result,
            terms=monitor_kinetic_terms,
            builder=builder,
            backend=backend,
            shape=shape,
            local_term_cache=local_term_cache,
        )
        potential_inside_matrix = _sum_local_terms(
            model=model,
            build_result=build_result,
            terms=monitor_potential_terms,
            builder=builder,
            backend=backend,
            shape=shape,
            local_term_cache=local_term_cache,
        )

        if z_value is None:
            v_psi = potential_inside_matrix @ psi
            z_value = complex(np.vdot(psi, v_psi))
            potential_residual = float(np.linalg.norm(v_psi - z_value * psi))
            if potential_residual > residual_tolerance:
                raise ValueError(
                    "Could not infer a sharp regional z value: "
                    f"||V_R psi - z psi||={potential_residual:.3e}."
                )

        monitor = (
            kinetic_inside_matrix + potential_inside_matrix - complex(z_value) * identity
        ).tocsr()

        reduced_iz_monitor_reports = ()
        n_reduced_iz_monitor_terms = 0
        reduced_iz_monitor_zero_indices = ()

    elif monitor_source == "reduced_iz_operators":
        if (
            reduced_iz_monitor_decomposition != "single_sum"
            and reduced_iz_monitor_content == "offdiagonal_only"
        ):
            basis_configs = None
        else:
            basis_configs = basis_configs_from_build_result(build_result)

        if reduced_iz_monitor_decomposition == "single_sum":
            monitor, reduced_iz_monitor_reports, reduced_iz_z_value = _build_reduced_iz_monitor(
                classification_report=classification_report,
                basis_configs=basis_configs,
                shape=shape,
                model=model,
                build_result=build_result,
                potential_terms=monitor_potential_terms,
                state=psi,
                identity=identity,
                builder=builder,
                backend=backend,
                local_term_cache=local_term_cache,
                reduced_iz_monitor_content=reduced_iz_monitor_content,
                residual_tolerance=residual_tolerance,
                include_q_empty=include_q_empty,
                include_closed_by_known_zeros=include_closed_by_known_zeros,
                include_projector_like=include_projector_like,
                include_collective_cancellation=include_collective_cancellation,
                use_collective_coefficients=use_collective_coefficients,
            )

            if reduced_iz_z_value is not None:
                if (
                    z_value is not None
                    and abs(complex(z_value) - reduced_iz_z_value) > residual_tolerance
                ):
                    raise ValueError(
                        "Provided z_value disagrees with inferred reduced-IZ potential "
                        f"value: provided={z_value}, inferred={reduced_iz_z_value}."
                    )
                z_value = reduced_iz_z_value

            single_component_groups = classification_report.reduced_iz_component_groups(
                decomposition="single_sum",
                include_q_empty=include_q_empty,
                include_closed_by_known_zeros=include_closed_by_known_zeros,
                include_projector_like=include_projector_like,
                include_collective_cancellation=include_collective_cancellation,
                use_collective_coefficients=use_collective_coefficients,
            )
            single_component_support = (
                single_component_groups[0].support_variables if single_component_groups else ()
            )

            single_component_support_set = frozenset(single_component_support)

            single_component_support_plaquette_ids = _plaquette_ids_inside_variable_support(
                kinetic_terms,
                single_component_support_set,
            )

            monitor_components = (
                ReducedIZMonitorComponent(
                    component_id=0,
                    monitor=monitor,
                    zero_indices=tuple(
                        int(report.zero_index) for report in reduced_iz_monitor_reports
                    ),
                    support_variables=single_component_support,
                    support_plaquette_ids=single_component_support_plaquette_ids,
                    monitor_plaquette_ids=tuple(
                        int(term.term_id) for term in monitor_kinetic_terms
                    ),
                    jump_plaquette_ids=tuple(int(term.term_id) for term in jump_kinetic_terms),
                    z_value=reduced_iz_z_value,
                    n_potential_terms=(
                        len(monitor_potential_terms)
                        if reduced_iz_monitor_content == "offdiagonal_plus_potential"
                        else 0
                    ),
                    monitor_residual=float(np.linalg.norm(monitor @ psi)),
                    jump_residuals=(),
                ),
            )

        else:
            (
                monitor,
                reduced_iz_monitor_reports,
                monitor_components,
                component_jumps,
            ) = _build_reduced_iz_monitor_components(
                classification_report=classification_report,
                basis_configs=basis_configs,
                shape=shape,
                decomposition=reduced_iz_monitor_decomposition,
                reduced_iz_monitor_content=reduced_iz_monitor_content,
                kinetic_terms=kinetic_terms,
                potential_terms=potential_terms,
                model=model,
                build_result=build_result,
                state=psi,
                identity=identity,
                builder=builder,
                backend=backend,
                local_term_cache=local_term_cache,
                residual_tolerance=residual_tolerance,
                include_q_empty=include_q_empty,
                include_closed_by_known_zeros=include_closed_by_known_zeros,
                include_projector_like=include_projector_like,
                include_collective_cancellation=include_collective_cancellation,
                use_collective_coefficients=use_collective_coefficients,
                compute_jump_residuals=compute_jump_residuals,
                timing_collector=timing_collector,
            )

        component_z_values = tuple(component.z_value for component in monitor_components)

        if (
            monitor_source == "reduced_iz_operators"
            and reduced_iz_monitor_content == "offdiagonal_plus_potential"
            and len(component_z_values) > 0
            and all(value is not None for value in component_z_values)
        ):
            summed_component_z = sum(
                complex(value) for value in component_z_values if value is not None
            )

            if (
                z_value is not None
                and abs(summed_component_z - complex(z_value)) > residual_tolerance
            ):
                raise ValueError(
                    "Component potential shifts do not sum to the global z value: "
                    f"sum_i z_i={summed_component_z}, z={z_value}."
                )

            if z_value is None:
                z_value = summed_component_z

        n_reduced_iz_monitor_terms = len(reduced_iz_monitor_reports)
        reduced_iz_monitor_zero_indices = tuple(
            int(report.zero_index) for report in reduced_iz_monitor_reports
        )

    else:
        raise ValueError(f"Unknown monitor_source: {monitor_source!r}")

    _record_construction_stage(timing_collector, "monitor_assembly", stage_start)

    stage_start = time.perf_counter()
    recycling_build_result = None
    recycling_jumps: tuple[Any, ...] = ()
    monitor_recycler_component_jumps = False
    monitor_recycler_jump_closure_orders: tuple[int, ...] = ()

    n_local_rdm_parent_projector_jumps = 0
    n_local_rdm_block_reset_jumps = 0

    if jump_operator_design in {
        "local_rdm_parent_projector",
        "local_rdm_parent_projector_recycling",
        "local_rdm_parent_projector_block_reset",
    }:
        basis_configs = basis_configs_from_build_result(build_result)
        region_specs = _monitor_recycler_region_specs(
            region=region,
            monitor=monitor,
            monitor_components=monitor_components,
        )
        parent_projector_jumps = _build_local_rdm_parent_projector_jump_operators(
            basis_configs=basis_configs,
            state=psi,
            region_specs=region_specs,
            rdm_tolerance=recycling_rdm_tolerance,
            rate=local_rdm_parent_projector_rate,
        )
        n_local_rdm_parent_projector_jumps = len(parent_projector_jumps)
        jumps = parent_projector_jumps

        if jump_operator_design == "local_rdm_parent_projector_recycling":
            if recycling_jump_source == "none":
                raise ValueError(
                    "jump_operator_design='local_rdm_parent_projector_recycling' "
                    "requires recycling_jump_source to be local_rdm_rank_one, "
                    "local_rdm_two_pattern, local_rdm_null_basis, or "
                    "local_rdm_block_reset."
                )

            recycling_build_result = build_local_recycling_jumps_from_regions(
                basis_configs=basis_configs,
                target_state=psi,
                regions=tuple(region_key for region_key, _ in region_specs),
                source=recycling_jump_source,
                max_jumps_per_region=max_recycling_jumps_per_region,
                deduplicate_regions=deduplicate_recycling_regions,
                rdm_tolerance=recycling_rdm_tolerance,
                dark_tolerance=recycling_dark_tolerance,
                inflow_tolerance=recycling_inflow_tolerance,
                prefer_sparse=recycling_prefer_sparse,
                two_pattern_tolerance=recycling_two_pattern_tolerance,
            )
            recycling_jumps = recycling_build_result.jumps
            if local_rdm_reset_rate != 1.0:
                reset_scale = float(np.sqrt(local_rdm_reset_rate))
                recycling_jumps = tuple((reset_scale * jump).tocsr() for jump in recycling_jumps)
            jumps = tuple(parent_projector_jumps) + tuple(recycling_jumps)

        elif jump_operator_design == "local_rdm_parent_projector_block_reset":
            block_reset_jumps = _build_local_rdm_block_reset_jump_operators(
                basis_configs=basis_configs,
                state=psi,
                region_specs=region_specs,
                rdm_tolerance=recycling_rdm_tolerance,
                rate=local_rdm_reset_rate,
            )
            n_local_rdm_block_reset_jumps = len(block_reset_jumps)
            recycling_jumps = block_reset_jumps
            jumps = tuple(parent_projector_jumps) + tuple(block_reset_jumps)

        monitor_recycler_component_jumps = True

    elif jump_operator_design == "monitor_recycler":
        basis_configs = basis_configs_from_build_result(build_result)
        region_specs = _monitor_recycler_region_specs(
            region=region,
            monitor=monitor,
            monitor_components=monitor_components,
        )
        local_closure_operators_by_region = None
        if (
            monitor_recycler_hamiltonian_closure_order > 0
            and monitor_recycler_hamiltonian_closure_source
            in {
                "local_hamiltonian_terms",
                "boundary_hamiltonian_terms",
                "touching_hamiltonian_terms",
            }
        ):
            local_closure_operators_by_region = (
                _build_monitor_recycler_local_hamiltonian_closure_operators(
                    model=model,
                    build_result=build_result,
                    regions=tuple(region_key for region_key, _ in region_specs),
                    kinetic_terms=kinetic_terms,
                    potential_terms=potential_terms,
                    builder=builder,
                    backend=backend,
                    shape=shape,
                    state=psi,
                    hamiltonian_shift=monitor_recycler_hamiltonian_shift,
                    hamiltonian_closure_source=monitor_recycler_hamiltonian_closure_source,
                    local_term_cache=local_term_cache,
                )
            )

        (
            jumps,
            recycling_build_result,
            monitor_recycler_jump_closure_orders,
        ) = _build_monitor_recycler_jump_operators(
            basis_configs=basis_configs,
            state=psi,
            hamiltonian=build_result.hamiltonian,
            local_closure_operators_by_region=local_closure_operators_by_region,
            region_specs=region_specs,
            source=recycling_jump_source,
            max_jumps_per_region=max_recycling_jumps_per_region,
            deduplicate_regions=deduplicate_recycling_regions,
            rdm_tolerance=recycling_rdm_tolerance,
            dark_tolerance=recycling_dark_tolerance,
            inflow_tolerance=recycling_inflow_tolerance,
            prefer_sparse=recycling_prefer_sparse,
            two_pattern_tolerance=recycling_two_pattern_tolerance,
            hamiltonian_closure_order=monitor_recycler_hamiltonian_closure_order,
            hamiltonian_shift=monitor_recycler_hamiltonian_shift,
            hamiltonian_closure_source=monitor_recycler_hamiltonian_closure_source,
        )
        monitor_recycler_component_jumps = True
    elif component_jumps is not None:
        jumps = _build_component_decomposition_jump_operators(
            model=model,
            build_result=build_result,
            component_jumps=component_jumps,
            jump_kinetic_terms=jump_kinetic_terms,
            potential_terms_by_plaquette_id=potential_by_pid,
            builder=builder,
            backend=backend,
            local_term_cache=local_term_cache,
            jump_operator_design=jump_operator_design,
        )
    else:
        jumps = _build_jump_operators(
            model=model,
            build_result=build_result,
            monitor=monitor,
            monitor_kinetic_terms=monitor_kinetic_terms,
            jump_kinetic_terms=jump_kinetic_terms,
            potential_terms_by_plaquette_id=potential_by_pid,
            builder=builder,
            backend=backend,
            local_term_cache=local_term_cache,
            jump_operator_design=jump_operator_design,
        )

    _record_construction_stage(timing_collector, "jump_assembly", stage_start)

    recycling_regions = _recycling_regions_from_construction_context(
        region=region,
        monitor_components=monitor_components,
    )

    stage_start = time.perf_counter()
    if (
        jump_operator_design
        not in {
            "monitor_recycler",
            "local_rdm_parent_projector",
            "local_rdm_parent_projector_recycling",
            "local_rdm_parent_projector_block_reset",
        }
        and recycling_jump_source != "none"
    ):
        basis_configs = basis_configs_from_build_result(build_result)

        recycling_build_result = build_local_recycling_jumps_from_regions(
            basis_configs=basis_configs,
            target_state=psi,
            regions=recycling_regions,
            source=recycling_jump_source,
            max_jumps_per_region=max_recycling_jumps_per_region,
            deduplicate_regions=deduplicate_recycling_regions,
            rdm_tolerance=recycling_rdm_tolerance,
            dark_tolerance=recycling_dark_tolerance,
            inflow_tolerance=recycling_inflow_tolerance,
            prefer_sparse=recycling_prefer_sparse,
            two_pattern_tolerance=recycling_two_pattern_tolerance,
        )

        recycling_jumps = recycling_build_result.jumps
        jumps = tuple(jumps) + tuple(recycling_jumps)

    _record_construction_stage(timing_collector, "recycling", stage_start)

    stage_start = time.perf_counter()
    monitor_state = _state_action_from_lazy_operator(monitor)
    if monitor_state is None:
        monitor_state = monitor @ psi
    monitor_residual = float(np.linalg.norm(monitor_state))
    if compute_jump_residuals:
        if component_jumps is None or monitor_recycler_component_jumps:
            jump_residuals = _jump_residuals(
                state=psi,
                jumps=jumps,
            )
        else:
            component_jump_residuals = _component_jump_residuals_from_components(monitor_components)
            remaining_jumps = tuple(jumps[len(component_jumps) :])
            jump_residuals = component_jump_residuals + _jump_residuals(
                state=psi,
                jumps=remaining_jumps,
            )
    else:
        jump_residuals = ()
    max_jump_residual = max(jump_residuals) if jump_residuals else 0.0
    bad_component_residuals = [
        component
        for component in monitor_components
        if component.monitor_residual > residual_tolerance
    ]

    if bad_component_residuals:
        summary = ", ".join(
            (f"id={component.component_id}: " f"||M_i psi||={component.monitor_residual:.3e}")
            for component in bad_component_residuals[:8]
        )
        raise ValueError(
            "The reduced-IZ monitor decomposition is not frustration-free: " f"{summary}."
        )

    if recycling_build_result is None:
        n_recycling_jumps = len(recycling_jumps)
        recycling_jump_variable_indices = ()
        recycling_jump_alpha_beta_indices = ()
        recycling_two_pattern_count = 0
    else:
        n_recycling_jumps = recycling_build_result.n_jumps
        recycling_jump_variable_indices = recycling_build_result.variable_indices
        recycling_jump_alpha_beta_indices = recycling_build_result.alpha_beta_indices
        recycling_two_pattern_count = sum(
            selection.two_pattern_structure is not None
            for selection in recycling_build_result.selections
        )

    _record_construction_stage(timing_collector, "diagnostics", stage_start)

    liouvillian_residual = None
    stage_start = time.perf_counter()
    if check_liouvillian:
        target_density_matrix = density_matrix_from_state(
            psi,
            normalize=False,
        )
        target_rhs = lindblad_rhs_density_matrix(
            target_density_matrix,
            hamiltonian=build_result.hamiltonian,
            jumps=list(jumps),
            backend=open_system_backend,
        )
        liouvillian_residual = float(np.linalg.norm(target_rhs))

    _record_construction_stage(timing_collector, "liouvillian_check", stage_start)

    if monitor_residual > residual_tolerance:
        raise ValueError(
            "The inferred regional monitor does not annihilate the cage state: "
            f"||M_R psi||={monitor_residual:.3e}. "
            "This may mean z_value is wrong, R is incomplete, or the cage is "
            "not compatible with the current type-1 construction."
        )

    if compute_jump_residuals and max_jump_residual > residual_tolerance:
        raise ValueError(
            "The inferred jump operators do not annihilate the cage state: "
            f"max_p ||J_p psi||={max_jump_residual:.3e}."
        )

    if jump_operator_design in {
        "monitor_recycler",
        "local_rdm_parent_projector",
        "local_rdm_parent_projector_recycling",
        "local_rdm_parent_projector_block_reset",
    }:
        n_component_jumps = len(jumps)
        n_global_jump_terms = 0
    else:
        n_component_jumps = len(component_jumps) if component_jumps is not None else 0
        n_global_jump_terms = (
            len(jump_kinetic_terms)
            if component_jumps is not None
            and jump_operator_design
            in {
                "kinetic_outside_monitor_inside",
                "hamiltonian_outside_monitor_inside",
            }
            else 0
        )

    return CageLindbladConstruction(
        cage_state=psi,
        region=region,
        z_value=complex(z_value) if z_value is not None else None,
        inside_plaquette_ids=tuple(int(term.term_id) for term in inside_kinetic),
        outside_plaquette_ids=tuple(int(term.term_id) for term in outside_kinetic),
        crossing_plaquette_ids=tuple(int(term.term_id) for term in crossing_kinetic),
        monitor_plaquette_ids=tuple(int(term.term_id) for term in monitor_kinetic_terms),
        jump_plaquette_ids=tuple(int(term.term_id) for term in jump_kinetic_terms),
        kinetic_terms_monitor=monitor_kinetic_terms,
        potential_terms_monitor=monitor_potential_terms,
        kinetic_terms_jump=jump_kinetic_terms,
        monitor=monitor,
        jumps=jumps,
        n_jumps=len(jumps),
        n_component_jumps=n_component_jumps,
        n_global_jump_terms=n_global_jump_terms,
        open_system_backend=open_system_backend,
        monitor_source=monitor_source,
        reduced_iz_monitor_decomposition=reduced_iz_monitor_decomposition,
        reduced_iz_monitor_content=reduced_iz_monitor_content,
        n_reduced_iz_monitor_terms=n_reduced_iz_monitor_terms,
        reduced_iz_monitor_zero_indices=reduced_iz_monitor_zero_indices,
        monitor_components=monitor_components,
        component_z_values=component_z_values,
        jump_operator_design=jump_operator_design,
        monitor_plaquette_policy=monitor_plaquette_policy,
        jump_plaquette_policy=jump_plaquette_policy,
        recycling_jump_source=recycling_jump_source,
        n_recycling_jumps=n_recycling_jumps,
        recycling_jump_variable_indices=recycling_jump_variable_indices,
        recycling_jump_alpha_beta_indices=recycling_jump_alpha_beta_indices,
        recycling_two_pattern_count=int(recycling_two_pattern_count),
        recycling_build_result=recycling_build_result,
        local_rdm_parent_projector_rate=float(local_rdm_parent_projector_rate),
        local_rdm_reset_rate=float(local_rdm_reset_rate),
        n_local_rdm_parent_projector_jumps=int(n_local_rdm_parent_projector_jumps),
        n_local_rdm_block_reset_jumps=int(n_local_rdm_block_reset_jumps),
        monitor_residual=monitor_residual,
        max_jump_residual=max_jump_residual,
        jump_residuals=jump_residuals,
        liouvillian_residual=liouvillian_residual,
        monitor_recycler_hamiltonian_closure_order=int(monitor_recycler_hamiltonian_closure_order),
        monitor_recycler_hamiltonian_shift=monitor_recycler_hamiltonian_shift,
        monitor_recycler_hamiltonian_closure_source=(monitor_recycler_hamiltonian_closure_source),
        monitor_recycler_jump_closure_orders=monitor_recycler_jump_closure_orders,
    )


def _build_jump_operators(
    *,
    model: Any,
    build_result: ModelBuildResult,
    monitor: Any,
    monitor_kinetic_terms: tuple[LocalTermDescriptor, ...],
    jump_kinetic_terms: tuple[LocalTermDescriptor, ...],
    potential_terms_by_plaquette_id: dict[int, LocalTermDescriptor],
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    jump_operator_design: JumpOperatorDesign,
    local_term_cache: _LocalTermMatrixCache | None = None,
) -> tuple[Any, ...]:
    if jump_operator_design == "kinetic_times_monitor":
        return tuple(
            _left_multiply_sparse_csr(
                _local_term_matrix(
                    model=model,
                    build_result=build_result,
                    term=term,
                    builder=builder,
                    backend=backend,
                    local_term_cache=local_term_cache,
                ),
                monitor,
            )
            for term in jump_kinetic_terms
        )

    if jump_operator_design == "kinetic_outside_monitor_inside":
        monitor_terms_by_id = {int(term.term_id): term for term in monitor_kinetic_terms}
        jump_terms_by_id = {int(term.term_id): term for term in jump_kinetic_terms}
        all_jump_ids = sorted(set(monitor_terms_by_id) | set(jump_terms_by_id))

        jumps: list[Any] = []

        for plaquette_id in all_jump_ids:
            if plaquette_id in monitor_terms_by_id:
                kinetic = _local_term_matrix(
                    model=model,
                    build_result=build_result,
                    term=monitor_terms_by_id[plaquette_id],
                    builder=builder,
                    backend=backend,
                    local_term_cache=local_term_cache,
                )
                jumps.append(_left_multiply_sparse_csr(kinetic, monitor))
            else:
                jumps.append(
                    _local_term_matrix(
                        model=model,
                        build_result=build_result,
                        term=jump_terms_by_id[plaquette_id],
                        builder=builder,
                        backend=backend,
                        local_term_cache=local_term_cache,
                    )
                )

        return tuple(jumps)

    if jump_operator_design in {
        "monitor_recycler",
        "local_rdm_parent_projector",
        "local_rdm_parent_projector_recycling",
        "local_rdm_parent_projector_block_reset",
    }:
        raise ValueError(
            f"{jump_operator_design} jumps are assembled from local RDM data, "
            "not plaquette kinetic terms."
        )

    if jump_operator_design == "hamiltonian_outside_monitor_inside":
        monitor_terms_by_id = {int(term.term_id): term for term in monitor_kinetic_terms}
        jump_terms_by_id = {int(term.term_id): term for term in jump_kinetic_terms}
        all_jump_ids = sorted(set(monitor_terms_by_id) | set(jump_terms_by_id))

        jumps: list[Any] = []

        for plaquette_id in all_jump_ids:
            if plaquette_id in monitor_terms_by_id:
                kinetic = _local_term_matrix(
                    model=model,
                    build_result=build_result,
                    term=monitor_terms_by_id[plaquette_id],
                    builder=builder,
                    backend=backend,
                    local_term_cache=local_term_cache,
                )
                jumps.append(_left_multiply_sparse_csr(kinetic, monitor))
            else:
                kinetic_term = jump_terms_by_id[plaquette_id]
                local_hamiltonian = _build_local_kinetic_plus_potential(
                    model=model,
                    build_result=build_result,
                    kinetic_term=kinetic_term,
                    potential_terms_by_plaquette_id=potential_terms_by_plaquette_id,
                    builder=builder,
                    backend=backend,
                    local_term_cache=local_term_cache,
                )
                jumps.append(local_hamiltonian)

        return tuple(jumps)

    raise ValueError(f"Unknown jump_operator_design: {jump_operator_design!r}")


def _build_component_decomposition_jump_operators(
    *,
    model: Any,
    build_result: ModelBuildResult | None,
    component_jumps: tuple[Any, ...],
    jump_kinetic_terms: tuple[LocalTermDescriptor, ...],
    potential_terms_by_plaquette_id: dict[int, LocalTermDescriptor],
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    jump_operator_design: JumpOperatorDesign,
    local_term_cache: _LocalTermMatrixCache | None = None,
) -> tuple[Any, ...]:
    """Build jumps for frustration-free monitor decomposition.

    Component jumps already contain the dressed inside jumps

        J_{p,i} = K_p M_{R_i}.

    Depending on jump_operator_design, we may additionally add global
    outside/crossing jumps selected by jump_kinetic_terms.
    """
    if jump_operator_design == "kinetic_times_monitor":
        return component_jumps

    if jump_operator_design == "kinetic_outside_monitor_inside":
        outside_jumps = tuple(
            _local_term_matrix(
                model=model,
                build_result=build_result,
                term=kinetic_term,
                builder=builder,
                backend=backend,
                local_term_cache=local_term_cache,
            )
            for kinetic_term in jump_kinetic_terms
        )

        return component_jumps + outside_jumps

    if jump_operator_design in {
        "monitor_recycler",
        "local_rdm_parent_projector",
        "local_rdm_parent_projector_recycling",
        "local_rdm_parent_projector_block_reset",
    }:
        raise ValueError(
            f"{jump_operator_design} component jumps are assembled directly from " "local RDM data."
        )

    if jump_operator_design == "hamiltonian_outside_monitor_inside":
        outside_jumps = tuple(
            _build_local_kinetic_plus_potential(
                model=model,
                build_result=build_result,
                kinetic_term=kinetic_term,
                potential_terms_by_plaquette_id=potential_terms_by_plaquette_id,
                builder=builder,
                backend=backend,
                local_term_cache=local_term_cache,
            )
            for kinetic_term in jump_kinetic_terms
        )

        return component_jumps + outside_jumps

    raise ValueError(f"Unknown jump_operator_design: {jump_operator_design!r}")


def _build_local_kinetic_plus_potential(
    *,
    model: Any,
    build_result: ModelBuildResult | None,
    kinetic_term: LocalTermDescriptor,
    potential_terms_by_plaquette_id: dict[int, LocalTermDescriptor],
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    local_term_cache: _LocalTermMatrixCache | None = None,
):
    kinetic = _local_term_matrix(
        model=model,
        build_result=build_result,
        term=kinetic_term,
        builder=builder,
        backend=backend,
        local_term_cache=local_term_cache,
    )

    plaquette_id = int(kinetic_term.term_id)
    potential_term = potential_terms_by_plaquette_id.get(plaquette_id)

    if potential_term is None:
        return kinetic

    potential = _local_term_matrix(
        model=model,
        build_result=build_result,
        term=potential_term,
        builder=builder,
        backend=backend,
        local_term_cache=local_term_cache,
    )

    return (kinetic + potential).tocsr()


def _recycling_regions_from_construction_context(
    *,
    region: CageRegionSupport,
    monitor_components: tuple[Any, ...] | None = None,
) -> tuple[tuple[int, ...], ...]:
    if monitor_components is not None and len(monitor_components) > 0:
        return tuple(
            tuple(int(index) for index in component.support_variables)
            for component in monitor_components
        )

    return (tuple(sorted(int(index) for index in region.variable_index_set)),)


def _partition_plaquette_terms_by_region(
    terms: tuple[LocalTermDescriptor, ...],
    region_variables: frozenset[int] | None = None,
    *,
    variable_index_set: frozenset[int] | None = None,
) -> tuple[
    tuple[LocalTermDescriptor, ...],
    tuple[LocalTermDescriptor, ...],
    tuple[LocalTermDescriptor, ...],
]:
    if region_variables is None:
        if variable_index_set is None:
            raise TypeError("region_variables or variable_index_set is required")
        region_variables = variable_index_set
    elif variable_index_set is not None and region_variables != variable_index_set:
        raise ValueError("region_variables and variable_index_set disagree")

    inside: list[LocalTermDescriptor] = []
    outside: list[LocalTermDescriptor] = []
    crossing: list[LocalTermDescriptor] = []

    for term in terms:
        support = term.support_link_set

        if support <= region_variables:
            inside.append(term)
        elif support.isdisjoint(region_variables):
            outside.append(term)
        else:
            crossing.append(term)

    return tuple(inside), tuple(outside), tuple(crossing)


def _select_monitor_terms(
    *,
    inside_terms: tuple[LocalTermDescriptor, ...],
    outside_terms: tuple[LocalTermDescriptor, ...],
    crossing_terms: tuple[LocalTermDescriptor, ...],
    policy: MonitorPlaquettePolicy,
) -> tuple[LocalTermDescriptor, ...]:
    del outside_terms

    if policy == "strict_inside":
        return inside_terms

    if policy == "touching":
        return inside_terms + crossing_terms

    raise ValueError(f"Unknown monitor plaquette policy: {policy!r}")


def _select_jump_terms(
    *,
    inside_terms: tuple[LocalTermDescriptor, ...],
    outside_terms: tuple[LocalTermDescriptor, ...],
    crossing_terms: tuple[LocalTermDescriptor, ...],
    policy: JumpPlaquettePolicy,
) -> tuple[LocalTermDescriptor, ...]:
    if policy == "disjoint_outside":
        return outside_terms

    if policy == "crossing":
        return crossing_terms

    if policy == "outside_or_crossing":
        return outside_terms + crossing_terms

    if policy == "not_strictly_inside":
        return outside_terms + crossing_terms

    raise ValueError(f"Unknown jump plaquette policy: {policy!r}")


def _sum_local_terms(
    *,
    model: Any,
    build_result: ModelBuildResult,
    terms: tuple[LocalTermDescriptor, ...],
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    shape: tuple[int, int],
    local_term_cache: _LocalTermMatrixCache | None = None,
) -> Any:
    if len(terms) == 0:
        return sp.csr_array(shape, dtype=np.complex128)

    total = sp.csr_array(shape, dtype=np.complex128)

    for term in terms:
        total = total + _local_term_matrix(
            model=model,
            build_result=build_result,
            term=term,
            builder=builder,
            backend=backend,
            local_term_cache=local_term_cache,
        )

    return total.tocsr()


def _jump_residuals(
    *,
    state: NDArray[np.complex128],
    jumps: tuple[Any, ...],
) -> tuple[float, ...]:
    return tuple(float(np.linalg.norm(jump @ state)) for jump in jumps)


def _component_jump_residuals_from_monitor_state(
    *,
    monitor_state: NDArray[np.complex128],
    kinetic_terms: tuple[LocalTermDescriptor, ...],
    model: Any,
    build_result: ModelBuildResult,
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    local_term_cache: _LocalTermMatrixCache | None = None,
) -> tuple[float, ...]:
    """Return ``||K_p M_i psi||`` without materializing ``K_p M_i``."""

    return tuple(
        float(
            np.linalg.norm(
                _local_term_matrix(
                    model=model,
                    build_result=build_result,
                    term=kinetic_term,
                    builder=builder,
                    backend=backend,
                    local_term_cache=local_term_cache,
                )
                @ monitor_state
            )
        )
        for kinetic_term in kinetic_terms
    )


def _component_jump_residuals_from_components(
    components: tuple[ReducedIZMonitorComponent, ...],
) -> tuple[float, ...]:
    return tuple(
        float(residual) for component in components for residual in component.jump_residuals
    )


def _find_recycling_scan_result(
    recycling_build_result: LocalRecyclingBuildResult,
    *,
    variable_indices: tuple[int, ...],
):
    for scan_result in recycling_build_result.scan_results:
        if (
            tuple(int(index) for index in scan_result.reduced_density_matrix.variable_indices)
            == variable_indices
        ):
            return scan_result

    raise KeyError(f"No local recycling scan result for region {variable_indices}.")


def _local_vector_support_indices(
    vector: NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[int, ...]:
    return tuple(int(index) for index in np.flatnonzero(np.abs(vector) > tolerance))


def _monitor_recycler_region_specs(
    *,
    region: CageRegionSupport,
    monitor: Any,
    monitor_components: tuple[ReducedIZMonitorComponent, ...],
) -> tuple[tuple[tuple[int, ...], Any], ...]:
    if len(monitor_components) == 0:
        return (
            (
                tuple(sorted(int(index) for index in region.variable_index_set)),
                monitor,
            ),
        )

    return tuple(
        (
            tuple(int(index) for index in component.support_variables),
            component.monitor,
        )
        for component in monitor_components
    )


def _build_monitor_recycler_local_hamiltonian_closure_operators(
    *,
    model: Any,
    build_result: ModelBuildResult,
    regions: tuple[tuple[int, ...], ...],
    kinetic_terms: tuple[LocalTermDescriptor, ...],
    potential_terms: tuple[LocalTermDescriptor, ...],
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    shape: tuple[int, int],
    state: NDArray[np.complex128],
    hamiltonian_shift: MonitorRecyclerHamiltonianShift,
    hamiltonian_closure_source: MonitorRecyclerHamiltonianClosureSource,
    local_term_cache: _LocalTermMatrixCache | None = None,
) -> dict[tuple[int, ...], sp.csr_array]:
    """Build local Hamiltonian closure factors for monitor-recycler jumps.

    For a monitor component on support ``R_i``, this returns the sum of
    plaquette kinetic and potential terms selected by
    ``hamiltonian_closure_source``:

    * ``local_hamiltonian_terms``: terms strictly supported in ``R_i``;
    * ``boundary_hamiltonian_terms``: terms overlapping ``R_i`` but not
      strictly contained in it;
    * ``touching_hamiltonian_terms``: both of the above.

    The resulting closure jump is ``V_i P_i h`` (or powers thereof).

    Unlike the global ``P_i(H-E)`` closure, local closure is not guaranteed by
    global eigenstate-ness alone; downstream jump residuals therefore remain the
    authoritative dark-state check.
    """
    if hamiltonian_shift not in {"none", "local_expectation"}:
        raise ValueError(
            "local Hamiltonian closure supports hamiltonian_shift='none' or " "'local_expectation'."
        )

    kinetic_term_supports = _term_support_sets(kinetic_terms)
    potential_term_supports = _term_support_sets(potential_terms)
    identity = sp.identity(shape[0], format="csr", dtype=np.complex128)
    closure_operators: dict[tuple[int, ...], sp.csr_array] = {}

    for region in regions:
        region_key = tuple(int(index) for index in region)
        region_support = frozenset(region_key)
        local_kinetic_terms = _select_monitor_recycler_closure_terms(
            kinetic_term_supports,
            region_support,
            hamiltonian_closure_source=hamiltonian_closure_source,
        )
        local_potential_terms = _select_monitor_recycler_closure_terms(
            potential_term_supports,
            region_support,
            hamiltonian_closure_source=hamiltonian_closure_source,
        )
        local_hamiltonian = _sum_local_terms(
            model=model,
            build_result=build_result,
            terms=local_kinetic_terms + local_potential_terms,
            builder=builder,
            backend=backend,
            shape=shape,
            local_term_cache=local_term_cache,
        ).tocsr()

        if hamiltonian_shift == "local_expectation":
            local_expectation = complex(np.vdot(state, local_hamiltonian @ state))
            local_hamiltonian = (local_hamiltonian - local_expectation * identity).tocsr()

        closure_operators[region_key] = local_hamiltonian

    return closure_operators


def _build_local_rdm_parent_projector_jump_operators(
    *,
    basis_configs: NDArray[np.integer],
    state: NDArray[np.complex128],
    region_specs: tuple[tuple[tuple[int, ...], Any], ...],
    rdm_tolerance: float,
    rate: float = 1.0,
) -> tuple[sp.csr_array, ...]:
    """Build local parent-projector jumps from target local-RDM kernels.

    For each monitor component support ``R_i``, compute the local reduced
    density matrix ``rho_i`` of the target cage state and embed the local null
    projector

        Q_i = I_{R_i} - projector(support(rho_i))

    as a global constrained-basis jump.  The target is dark by construction,
    and ``ker(Q_i)`` preserves the full local-RDM support instead of applying a
    rank-one recycler.  This is a parent-Hamiltonian-style dissipator: it is
    usually less microscopic than a single matrix-unit recycler, but it is much
    less lossy and can remove the large common-kernel degeneracy seen in
    ``V_i P_i`` monitor-recycler jumps.
    """
    jumps_by_region: dict[tuple[int, ...], sp.csr_array] = {}
    ordered_regions: list[tuple[int, ...]] = []

    for region, _ in region_specs:
        region_key = tuple(int(index) for index in region)
        if region_key in jumps_by_region:
            continue

        reduced_density_matrix = local_reduced_density_matrix_from_state(
            basis_configs=basis_configs,
            state=state,
            variable_indices=region_key,
            tolerance=rdm_tolerance,
        )

        if reduced_density_matrix.nullity == 0:
            parent_projector = sp.csr_array(
                (basis_configs.shape[0], basis_configs.shape[0]),
                dtype=np.complex128,
            )
        else:
            local_null_projector = (
                reduced_density_matrix.null_basis @ reduced_density_matrix.null_basis.conj().T
            )
            parent_projector = embed_local_pattern_operator(
                basis_configs=basis_configs,
                variable_indices=region_key,
                local_patterns=reduced_density_matrix.local_patterns,
                local_operator=local_null_projector.astype(np.complex128, copy=False),
            ).tocsr()

        if rate != 1.0:
            parent_projector = (float(np.sqrt(rate)) * parent_projector).tocsr()

        jumps_by_region[region_key] = parent_projector
        ordered_regions.append(region_key)

    jumps = tuple(jumps_by_region[region] for region in ordered_regions)
    if not jumps:
        raise ValueError("local_rdm_parent_projector produced no jump operators.")

    return jumps


def _build_local_rdm_block_reset_jump_operators(
    *,
    basis_configs: NDArray[np.integer],
    state: NDArray[np.complex128],
    region_specs: tuple[tuple[tuple[int, ...], Any], ...],
    rdm_tolerance: float,
    rate: float = 1.0,
) -> tuple[sp.csr_array, ...]:
    """Build minimal local-RDM block-reset jumps for each monitor region.

    For a local target support ``S_i`` and null space ``N_i``, the block reset
    maps disjoint blocks of null vectors into the target support.  Each reset
    jump has local form ``sum_a |s_a><n_{b+a}|``.  Thus the number of reset
    jumps for a region is ``ceil(dim(N_i) / dim(S_i))``, the minimal number of
    local reset channels able to cover all null directions without leaving the
    target support.
    """
    jumps_by_region: dict[tuple[int, ...], list[sp.csr_array]] = {}
    ordered_regions: list[tuple[int, ...]] = []
    reset_scale = float(np.sqrt(rate))

    for region, _ in region_specs:
        region_key = tuple(int(index) for index in region)
        if region_key in jumps_by_region:
            continue

        reduced_density_matrix = local_reduced_density_matrix_from_state(
            basis_configs=basis_configs,
            state=state,
            variable_indices=region_key,
            tolerance=rdm_tolerance,
        )

        support_basis = reduced_density_matrix.support_basis
        null_basis = reduced_density_matrix.null_basis
        support_rank = int(support_basis.shape[1])
        nullity = int(null_basis.shape[1])

        region_jumps: list[sp.csr_array] = []
        if support_rank > 0 and nullity > 0 and rate > 0.0:
            for start in range(0, nullity, support_rank):
                block = null_basis[:, start : start + support_rank]
                block_rank = int(block.shape[1])
                local_operator = support_basis[:, :block_rank] @ block.conj().T
                reset_jump = embed_local_pattern_operator(
                    basis_configs=basis_configs,
                    variable_indices=region_key,
                    local_patterns=reduced_density_matrix.local_patterns,
                    local_operator=(reset_scale * local_operator).astype(
                        np.complex128,
                        copy=False,
                    ),
                ).tocsr()
                region_jumps.append(reset_jump)

        jumps_by_region[region_key] = region_jumps
        ordered_regions.append(region_key)

    jumps: list[sp.csr_array] = []
    for region_key in ordered_regions:
        jumps.extend(jumps_by_region[region_key])

    return tuple(jumps)


def _build_monitor_recycler_jump_operators(
    *,
    basis_configs: NDArray[np.integer],
    state: NDArray[np.complex128],
    hamiltonian: Any | None = None,
    local_closure_operators_by_region: dict[tuple[int, ...], Any] | None = None,
    region_specs: tuple[tuple[tuple[int, ...], Any], ...],
    source: RecyclingJumpSource,
    max_jumps_per_region: int,
    rdm_tolerance: float,
    dark_tolerance: float,
    inflow_tolerance: float,
    prefer_sparse: bool,
    two_pattern_tolerance: float,
    deduplicate_regions: bool = False,
    hamiltonian_closure_order: int = 0,
    hamiltonian_shift: MonitorRecyclerHamiltonianShift = "target_energy",
    hamiltonian_closure_source: MonitorRecyclerHamiltonianClosureSource = "global_hamiltonian",
) -> tuple[tuple[Any, ...], LocalRecyclingBuildResult, tuple[int, ...]]:
    if source == "none":
        raise ValueError(
            "jump_operator_design='monitor_recycler' requires recycling_jump_source "
            "to be 'local_rdm_two_pattern', 'local_rdm_rank_one', "
            "'local_rdm_null_basis', or 'local_rdm_block_reset'."
        )

    if hamiltonian_closure_order < 0:
        raise ValueError("hamiltonian_closure_order must be nonnegative.")
    if hamiltonian_closure_order > 0:
        if hamiltonian_closure_source == "global_hamiltonian" and hamiltonian is None:
            raise ValueError("global Hamiltonian closure requires a hamiltonian operator.")
        if (
            hamiltonian_closure_source
            in {
                "local_hamiltonian_terms",
                "boundary_hamiltonian_terms",
                "touching_hamiltonian_terms",
            }
            and local_closure_operators_by_region is None
        ):
            raise ValueError(
                "local Hamiltonian closure requires local_closure_operators_by_region."
            )

    shifted_hamiltonian = None
    if hamiltonian_closure_order > 0 and hamiltonian_closure_source == "global_hamiltonian":
        hamiltonian_csr = _as_csr_matrix(hamiltonian)
        if hamiltonian_shift == "target_energy":
            target_energy = complex(np.vdot(state, hamiltonian_csr @ state))
            shifted_hamiltonian = (
                hamiltonian_csr
                - target_energy
                * sp.identity(
                    hamiltonian_csr.shape[0],
                    format="csr",
                    dtype=np.complex128,
                )
            ).tocsr()
        elif hamiltonian_shift == "none":
            shifted_hamiltonian = hamiltonian_csr
        else:
            raise ValueError(f"Unknown hamiltonian_shift: {hamiltonian_shift!r}")

    regions = tuple(region for region, _ in region_specs)
    recycling_build_result = build_local_recycling_jumps_from_regions(
        basis_configs=basis_configs,
        target_state=state,
        regions=regions,
        source=source,
        max_jumps_per_region=max_jumps_per_region,
        deduplicate_regions=deduplicate_regions,
        rdm_tolerance=rdm_tolerance,
        dark_tolerance=dark_tolerance,
        inflow_tolerance=inflow_tolerance,
        prefer_sparse=prefer_sparse,
        two_pattern_tolerance=two_pattern_tolerance,
    )

    selections_by_region: dict[tuple[int, ...], list[Any]] = {}
    for selection in recycling_build_result.selections:
        region_key = tuple(int(index) for index in selection.candidate.variable_indices)
        selections_by_region.setdefault(region_key, []).append(selection)

    jumps: list[Any] = []
    jump_closure_orders: list[int] = []
    missing_regions: list[tuple[int, ...]] = []

    emitted_region_keys: set[tuple[int, ...]] = set()
    for region_key, component_monitor in region_specs:
        region_key = tuple(int(index) for index in region_key)
        if deduplicate_regions and region_key in emitted_region_keys:
            continue
        emitted_region_keys.add(region_key)

        selections = selections_by_region.get(region_key, [])
        if len(selections) == 0:
            missing_regions.append(region_key)
            continue

        monitor_for_order = _as_csr_matrix(component_monitor)
        monitors_by_order = [monitor_for_order]
        if hamiltonian_closure_order > 0:
            if hamiltonian_closure_source == "global_hamiltonian":
                if shifted_hamiltonian is None:
                    raise RuntimeError("shifted_hamiltonian was not initialized.")
                closure_factor = shifted_hamiltonian
            elif hamiltonian_closure_source in {
                "local_hamiltonian_terms",
                "boundary_hamiltonian_terms",
                "touching_hamiltonian_terms",
            }:
                if local_closure_operators_by_region is None:
                    raise RuntimeError("local_closure_operators_by_region was not initialized.")
                closure_factor = _as_csr_matrix(local_closure_operators_by_region[region_key])
            else:
                raise ValueError(
                    f"Unknown hamiltonian_closure_source: {hamiltonian_closure_source!r}"
                )

            for _ in range(hamiltonian_closure_order):
                monitor_for_order = (monitor_for_order @ closure_factor).tocsr()
                monitors_by_order.append(monitor_for_order)

        for selection in selections:
            recycler = selection.candidate.jump
            for closure_order, monitor_factor in enumerate(monitors_by_order):
                jumps.append(
                    _left_multiply_sparse_csr(
                        recycler,
                        monitor_factor,
                    )
                )
                jump_closure_orders.append(int(closure_order))

    if missing_regions:
        preview = ", ".join(str(region) for region in missing_regions[:4])
        raise ValueError(
            "Could not construct monitor-recycler jumps for "
            f"{len(missing_regions)} monitor region(s): {preview}. "
            "Try recycling_jump_source='local_rdm_rank_one', "
            "recycling_jump_source='local_rdm_null_basis', or "
            "recycling_jump_source='local_rdm_block_reset'."
        )

    return tuple(jumps), recycling_build_result, tuple(jump_closure_orders)


# Reduced-IZ report selection and support/decomposition grouping live in
# qlinks.caging.classification.  Keep private aliases here for compatibility
# with older tests and downstream imports that used the construction module.
_reduced_iz_reports_for_monitor = select_reduced_iz_monitor_reports
_support_key_from_mask = support_key_from_mask
_support_key_for_zero_report = support_key_for_zero_report
_group_reduced_iz_reports_by_exact_support = group_reduced_iz_reports_by_exact_support
_group_reduced_iz_reports_by_connected_support = group_reduced_iz_reports_by_connected_support
_group_reduced_iz_reports_for_monitor = group_reduced_iz_monitor_reports


@dataclass(slots=True)
class _ReducedIZAssemblyCache:
    """Cache basis-index lookups for reduced-IZ sparse assembly.

    Reduced-IZ monitor construction often builds many local operators with the
    same local support.  Scanning the full constrained basis for every report is
    the dominant cost in the Cage-Lindblad benchmark, so this cache stores the
    source basis indices and source->target row/column arrays for each local
    transition pattern.
    """

    basis_configs: NDArray[np.integer]
    config_to_index: dict[tuple[int, ...], int] | None = None
    _source_groups_by_mask: dict[
        tuple[int, ...],
        dict[tuple[int, ...], NDArray[np.int64]],
    ] = field(default_factory=dict, init=False, repr=False)
    _transition_indices_by_pattern: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
    ] = field(default_factory=dict, init=False, repr=False)
    _config_codes: NDArray[np.int64] | None = field(default=None, init=False, repr=False)
    _code_to_index: dict[int, int] | None = field(default=None, init=False, repr=False)
    _radix_weights: tuple[int, ...] = field(default=(), init=False, repr=False)
    _value_digits_by_variable: tuple[dict[int, int], ...] = field(
        default=(), init=False, repr=False
    )
    _integer_code_lookup_prepared: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.basis_configs = np.ascontiguousarray(self.basis_configs, dtype=np.int64)

    def _ensure_integer_code_lookup(self) -> None:
        if self._integer_code_lookup_prepared:
            return
        self._integer_code_lookup_prepared = True
        self._prepare_integer_code_lookup()

    def _config_to_index_lookup(self) -> dict[tuple[int, ...], int]:
        if self.config_to_index is None:
            self.config_to_index = {
                tuple(int(value) for value in config): index
                for index, config in enumerate(self.basis_configs)
            }

        return self.config_to_index

    def _prepare_integer_code_lookup(self) -> None:
        """Prepare a mixed-radix code lookup for fast target-state indexing.

        The generic fallback builds a full target configuration and looks it up
        as a tuple.  Reduced-IZ assembly repeats that lookup many times.  When
        the per-variable local alphabets fit in int64 mixed-radix codes, target
        indices can instead be found by applying a local code delta to the source
        basis codes.
        """

        if self.basis_configs.ndim != 2:
            return

        n_states, n_variables = self.basis_configs.shape
        if n_states == 0 or n_variables == 0:
            self._config_codes = np.zeros(n_states, dtype=np.int64)
            self._code_to_index = {}
            self._radix_weights = ()
            self._value_digits_by_variable = ()
            return

        max_code = int(np.iinfo(np.int64).max)
        multiplier = 1
        weights: list[int] = []
        value_maps: list[dict[int, int]] = []

        for variable_index in range(n_variables):
            values = tuple(int(value) for value in np.unique(self.basis_configs[:, variable_index]))
            base = len(values)

            if base == 0 or multiplier > max_code // max(base, 1):
                self._config_codes = None
                self._code_to_index = None
                self._radix_weights = ()
                self._value_digits_by_variable = ()
                return

            weights.append(multiplier)
            value_maps.append({value: digit for digit, value in enumerate(values)})
            multiplier *= base

        codes = np.zeros(n_states, dtype=np.int64)
        for variable_index, (weight, value_map) in enumerate(zip(weights, value_maps, strict=True)):
            digits = np.fromiter(
                (value_map[int(value)] for value in self.basis_configs[:, variable_index]),
                dtype=np.int64,
                count=n_states,
            )
            codes += digits * int(weight)

        code_to_index: dict[int, int] = {}
        for basis_index, code in enumerate(codes):
            code_key = int(code)
            if code_key in code_to_index:
                self._config_codes = None
                self._code_to_index = None
                self._radix_weights = ()
                self._value_digits_by_variable = ()
                return
            code_to_index[code_key] = int(basis_index)

        self._config_codes = codes
        self._code_to_index = code_to_index
        self._radix_weights = tuple(weights)
        self._value_digits_by_variable = tuple(value_maps)

    def source_indices(
        self,
        *,
        mask_key: tuple[int, ...],
        source_local: tuple[int, ...],
    ) -> NDArray[np.int64]:
        source_groups = self._source_groups_by_mask.get(mask_key)

        if source_groups is None:
            source_groups = self._build_source_groups(mask_key)
            self._source_groups_by_mask[mask_key] = source_groups

        return source_groups.get(
            _transition_pattern_key(source_local),
            np.empty(0, dtype=np.int64),
        )

    def transition_indices(
        self,
        *,
        local_mask: NDArray[np.bool_],
        source_local: tuple[int, ...] | NDArray[np.integer],
        target_local: tuple[int, ...] | NDArray[np.integer],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        return self.transition_indices_by_key(
            mask_key=_support_key_from_mask(local_mask),
            source_key=_transition_pattern_key(source_local),
            target_key=_transition_pattern_key(target_local),
        )

    def transition_indices_by_key(
        self,
        *,
        mask_key: tuple[int, ...],
        source_key: tuple[int, ...],
        target_key: tuple[int, ...],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        cache_key = (mask_key, source_key, target_key)

        cached = self._transition_indices_by_pattern.get(cache_key)
        if cached is not None:
            return cached

        source_indices = self.source_indices(
            mask_key=mask_key,
            source_local=source_key,
        )

        if len(source_indices) == 0:
            result = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            self._transition_indices_by_pattern[cache_key] = result
            return result

        encoded_result = self._transition_indices_from_integer_codes(
            mask_key=mask_key,
            source_key=source_key,
            target_key=target_key,
            source_indices=source_indices,
        )
        if encoded_result is not None:
            self._transition_indices_by_pattern[cache_key] = encoded_result
            return encoded_result

        target_configs = np.array(
            self.basis_configs[source_indices],
            copy=True,
        )
        target_configs[:, np.asarray(mask_key, dtype=np.int64)] = np.asarray(
            target_key,
            dtype=target_configs.dtype,
        )

        rows: list[int] = []
        cols: list[int] = []

        for column_index, target_config in zip(source_indices, target_configs, strict=True):
            row_index = self._config_to_index_lookup().get(tuple(int(x) for x in target_config))
            if row_index is None:
                continue

            rows.append(int(row_index))
            cols.append(int(column_index))

        result = (
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
        )
        self._transition_indices_by_pattern[cache_key] = result
        return result

    def _transition_indices_from_integer_codes(
        self,
        *,
        mask_key: tuple[int, ...],
        source_key: tuple[int, ...],
        target_key: tuple[int, ...],
        source_indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
        self._ensure_integer_code_lookup()

        if self._config_codes is None or self._code_to_index is None:
            return None

        delta = 0
        for variable_index, source_value, target_value in zip(
            mask_key,
            source_key,
            target_key,
            strict=True,
        ):
            value_map = self._value_digits_by_variable[int(variable_index)]
            source_digit = value_map.get(int(source_value))
            target_digit = value_map.get(int(target_value))

            if source_digit is None or target_digit is None:
                empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
                return empty

            weight = self._radix_weights[int(variable_index)]
            delta += (int(target_digit) - int(source_digit)) * int(weight)

        target_codes = self._config_codes[source_indices] + int(delta)

        rows: list[int] = []
        cols: list[int] = []
        code_to_index = self._code_to_index
        for target_code, column_index in zip(target_codes, source_indices, strict=True):
            row_index = code_to_index.get(int(target_code))
            if row_index is None:
                continue

            rows.append(int(row_index))
            cols.append(int(column_index))

        return (
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
        )

    def _build_source_groups(
        self,
        mask_key: tuple[int, ...],
    ) -> dict[tuple[int, ...], NDArray[np.int64]]:
        mask_indices = np.asarray(mask_key, dtype=np.int64)
        local_values = np.ascontiguousarray(self.basis_configs[:, mask_indices], dtype=np.int64)

        if local_values.ndim == 1:
            local_values = local_values.reshape(-1, 1)

        unique_values, inverse = np.unique(
            local_values,
            axis=0,
            return_inverse=True,
        )

        grouped: dict[tuple[int, ...], NDArray[np.int64]] = {}
        for group_index, source_local in enumerate(unique_values):
            grouped[_transition_pattern_key(source_local)] = np.flatnonzero(
                inverse == group_index
            ).astype(np.int64, copy=False)

        return grouped


def _reduced_iz_monitor_state_from_reports(
    *,
    reports: tuple[InterferenceZeroReport, ...],
    state: NDArray[np.complex128],
    use_collective_coefficients: bool,
) -> NDArray[np.complex128] | None:
    """Return ``M_reports |state>`` from cached classification actions if available."""

    if len(reports) == 0:
        return np.zeros_like(state, dtype=np.complex128)

    result = np.zeros_like(state, dtype=np.complex128)

    for zero_report in reports:
        action = np.asarray(zero_report.reduced_action_vector, dtype=np.complex128)
        if action.shape != state.shape:
            return None

        coefficient = _monitor_coefficient_for_zero_report(
            zero_report,
            use_collective_coefficients=use_collective_coefficients,
        )
        result = result + coefficient * action

    return result


def _build_reduced_iz_monitor_from_reports(
    *,
    reports: tuple[InterferenceZeroReport, ...],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int] | None,
    shape: tuple[int, int],
    use_collective_coefficients: bool,
    assembly_cache: _ReducedIZAssemblyCache | None = None,
) -> sp.csr_array:
    if len(reports) == 0:
        return sp.csr_array(shape, dtype=np.complex128)

    if assembly_cache is None:
        assembly_cache = _ReducedIZAssemblyCache(
            basis_configs=basis_configs,
            config_to_index=config_to_index,
        )

    transition_coefficients: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], complex
    ] = {}

    for zero_report in reports:
        coefficient = _monitor_coefficient_for_zero_report(
            zero_report,
            use_collective_coefficients=use_collective_coefficients,
        )
        mask_key = _support_key_from_mask(zero_report.local_mask)

        for transition in zero_report.local_transitions:
            source_key = _transition_pattern_key(transition.source_local)
            target_key = _transition_pattern_key(transition.target_local)
            key = (mask_key, source_key, target_key)
            transition_coefficients[key] = transition_coefficients.get(key, 0.0 + 0.0j) + (
                coefficient * complex(transition.matrix_element)
            )

    rows_parts: list[NDArray[np.int64]] = []
    cols_parts: list[NDArray[np.int64]] = []
    data_parts: list[NDArray[np.complex128]] = []

    for (mask_key, source_key, target_key), coefficient in transition_coefficients.items():
        if coefficient == 0.0:
            continue

        rows, cols = assembly_cache.transition_indices_by_key(
            mask_key=mask_key,
            source_key=source_key,
            target_key=target_key,
        )

        if len(rows) == 0:
            continue

        rows_parts.append(rows)
        cols_parts.append(cols)
        data_parts.append(
            np.full(
                len(rows),
                coefficient,
                dtype=np.complex128,
            )
        )

    if len(data_parts) == 0:
        return sp.csr_array(shape, dtype=np.complex128)

    return sp.csr_array(
        (
            np.concatenate(data_parts),
            (np.concatenate(rows_parts), np.concatenate(cols_parts)),
        ),
        shape=shape,
        dtype=np.complex128,
    ).tocsr()


def _build_reduced_iz_monitor(
    *,
    classification_report: CageClassificationReport,
    basis_configs: NDArray[np.integer],
    shape: tuple[int, int],
    model: Any | None = None,
    build_result: ModelBuildResult | None = None,
    potential_terms: tuple[LocalTermDescriptor, ...] = (),
    state: NDArray[np.complex128] | None = None,
    identity: Any | None = None,
    builder: HamiltonianBuilderName = "sparse",
    backend: SparseBackendName = "scipy",
    local_term_cache: _LocalTermMatrixCache | None = None,
    reduced_iz_monitor_content: ReducedIZMonitorContent = "offdiagonal_only",
    residual_tolerance: float = 1e-10,
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
    use_collective_coefficients: bool = True,
) -> tuple[sp.csr_array, tuple[InterferenceZeroReport, ...], complex | None]:
    selected_reports = _reduced_iz_reports_for_monitor(
        classification_report,
        include_q_empty=include_q_empty,
        include_closed_by_known_zeros=include_closed_by_known_zeros,
        include_projector_like=include_projector_like,
        include_collective_cancellation=include_collective_cancellation,
    )

    config_to_index = {
        tuple(int(value) for value in config): index for index, config in enumerate(basis_configs)
    }

    assembly_cache = _ReducedIZAssemblyCache(
        basis_configs=basis_configs,
        config_to_index=config_to_index,
    )

    monitor = _build_reduced_iz_monitor_from_reports(
        reports=selected_reports,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        shape=shape,
        use_collective_coefficients=use_collective_coefficients,
        assembly_cache=assembly_cache,
    )

    inferred_z_value: complex | None = None

    if reduced_iz_monitor_content == "offdiagonal_plus_potential":
        if model is None or build_result is None or state is None or identity is None:
            raise ValueError(
                "offdiagonal_plus_potential requires model, build_result, " "state, and identity."
            )

        monitor, inferred_z_value = _add_potential_to_monitor(
            monitor=monitor,
            model=model,
            build_result=build_result,
            potential_terms=potential_terms,
            state=state,
            identity=identity,
            builder=builder,
            backend=backend,
            shape=shape,
            residual_tolerance=residual_tolerance,
            label="full reduced-IZ monitor support",
            local_term_cache=local_term_cache,
        )

    elif reduced_iz_monitor_content != "offdiagonal_only":
        raise ValueError(f"Unknown reduced-IZ monitor content: " f"{reduced_iz_monitor_content!r}")

    return monitor.tocsr(), selected_reports, inferred_z_value


def _build_reduced_iz_monitor_components(
    *,
    classification_report: CageClassificationReport,
    basis_configs: NDArray[np.integer] | None,
    shape: tuple[int, int],
    decomposition: ReducedIZMonitorDecomposition,
    reduced_iz_monitor_content: ReducedIZMonitorContent,
    kinetic_terms: tuple[LocalTermDescriptor, ...],
    potential_terms: tuple[LocalTermDescriptor, ...],
    model: Any,
    build_result: ModelBuildResult,
    state: NDArray[np.complex128],
    identity: Any,
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    residual_tolerance: float,
    local_term_cache: _LocalTermMatrixCache | None = None,
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
    use_collective_coefficients: bool = True,
    compute_jump_residuals: bool = True,
    timing_collector: dict[str, float] | None = None,
) -> tuple[
    sp.csr_array,
    tuple[InterferenceZeroReport, ...],
    tuple[ReducedIZMonitorComponent, ...],
    tuple[Any, ...],
]:
    selected_reports = _reduced_iz_reports_for_monitor(
        classification_report,
        include_q_empty=include_q_empty,
        include_closed_by_known_zeros=include_closed_by_known_zeros,
        include_projector_like=include_projector_like,
        include_collective_cancellation=include_collective_cancellation,
    )

    # Keep sparse reduced-IZ matrix assembly lazy for the offdiagonal-only
    # component path.  The fast construction path only needs cached action
    # vectors stored in the classification report; basis-index maps are built
    # later only if a caller materializes a monitor matrix.
    config_to_index: dict[tuple[int, ...], int] | None = None
    assembly_cache: _ReducedIZAssemblyCache | None = None
    if reduced_iz_monitor_content == "offdiagonal_plus_potential":
        if basis_configs is None:
            basis_configs = basis_configs_from_build_result(build_result)
        config_to_index = {
            tuple(int(value) for value in config): index
            for index, config in enumerate(basis_configs)
        }
        assembly_cache = _ReducedIZAssemblyCache(
            basis_configs=basis_configs,
            config_to_index=config_to_index,
        )

    component_groups = classification_report.reduced_iz_component_groups(
        decomposition=decomposition,
        include_q_empty=include_q_empty,
        include_closed_by_known_zeros=include_closed_by_known_zeros,
        include_projector_like=include_projector_like,
        include_collective_cancellation=include_collective_cancellation,
        use_collective_coefficients=use_collective_coefficients,
    )
    report_groups = _report_groups_from_component_groups(
        selected_reports=selected_reports,
        component_groups=component_groups,
    )
    kinetic_term_supports = _term_support_sets(kinetic_terms)
    potential_term_supports = _term_support_sets(potential_terms)

    total_monitor_terms: list[Any] = []
    total_monitor_state = np.zeros(shape[0], dtype=np.complex128)
    component_specs: list[dict[str, Any]] = []
    jump_specs: list[tuple[LocalTermDescriptor, Any]] = []

    for component_group, report_group in zip(component_groups, report_groups, strict=True):
        component_id = int(component_group.component_id)
        component_stage_start = time.perf_counter()
        component_monitor_state = _state_action_from_component_group(
            component_group,
            state=state,
        )
        if component_monitor_state is None:
            component_monitor_state = _reduced_iz_monitor_state_from_reports(
                reports=report_group,
                state=state,
                use_collective_coefficients=use_collective_coefficients,
            )

        if reduced_iz_monitor_content == "offdiagonal_only":
            component_monitor = _LazyReducedIZMonitorOperator(
                reports=report_group,
                basis_configs=basis_configs,
                config_to_index=config_to_index,
                shape=shape,
                use_collective_coefficients=use_collective_coefficients,
                build_result=build_result,
                assembly_cache=assembly_cache,
                _state_action=component_monitor_state,
            )
        else:
            component_monitor = _build_reduced_iz_monitor_from_reports(
                reports=report_group,
                basis_configs=basis_configs,
                config_to_index=config_to_index,
                shape=shape,
                use_collective_coefficients=use_collective_coefficients,
                assembly_cache=assembly_cache,
            )
            if component_monitor_state is None:
                component_monitor_state = component_monitor @ state

        _record_construction_stage(
            timing_collector,
            "component_monitor_assembly",
            component_stage_start,
        )

        component_support = component_group.support_variables
        component_support_set = frozenset(component_support)

        component_kinetic_terms = _select_terms_inside_cached_supports(
            kinetic_term_supports,
            component_support_set,
        )

        component_potential_terms = _select_terms_inside_cached_supports(
            potential_term_supports,
            component_support_set,
        )

        component_support_plaquette_ids = _plaquette_ids_inside_cached_supports(
            kinetic_term_supports,
            component_support_set,
        )

        component_z_value: complex | None = None

        if reduced_iz_monitor_content == "offdiagonal_plus_potential":
            component_monitor, component_z_value = _add_potential_to_monitor(
                monitor=_as_csr_matrix(component_monitor),
                model=model,
                build_result=build_result,
                potential_terms=component_potential_terms,
                state=state,
                identity=identity,
                builder=builder,
                backend=backend,
                shape=shape,
                residual_tolerance=residual_tolerance,
                label=f"reduced-IZ monitor component {component_id}",
                local_term_cache=local_term_cache,
            )
            component_monitor_state = component_monitor @ state
        elif reduced_iz_monitor_content != "offdiagonal_only":
            raise ValueError(
                f"Unknown reduced-IZ monitor content: " f"{reduced_iz_monitor_content!r}"
            )

        if component_monitor_state is None:
            component_monitor_state = component_monitor @ state

        jump_indices: list[int] = []
        for kinetic_term in component_kinetic_terms:
            jump_indices.append(len(jump_specs))
            jump_specs.append((kinetic_term, component_monitor))

        component_specs.append(
            {
                "component_id": component_id,
                "report_group": report_group,
                "monitor": component_monitor,
                "monitor_state": component_monitor_state,
                "support": component_support,
                "support_plaquette_ids": component_support_plaquette_ids,
                "kinetic_terms": component_kinetic_terms,
                "potential_terms": component_potential_terms,
                "z_value": component_z_value,
                "jump_indices": tuple(jump_indices),
            }
        )

        component_stage_start = time.perf_counter()
        total_monitor_terms.append(component_monitor)
        total_monitor_state += component_monitor_state
        _record_construction_stage(
            timing_collector,
            "component_monitor_sum",
            component_stage_start,
        )

    component_stage_start = time.perf_counter()
    jumps = [
        _lazy_local_term_left_multiply_monitor(
            model=model,
            build_result=build_result,
            term=kinetic_term,
            builder=builder,
            backend=backend,
            local_term_cache=local_term_cache,
            monitor=component_monitor,
        )
        for kinetic_term, component_monitor in jump_specs
    ]
    _record_construction_stage(
        timing_collector,
        "component_jump_products",
        component_stage_start,
    )

    component_stage_start = time.perf_counter()
    components: list[ReducedIZMonitorComponent] = []
    for component_spec in component_specs:
        component_monitor = component_spec["monitor"]
        component_monitor_state = component_spec["monitor_state"]
        component_monitor_residual = float(np.linalg.norm(component_monitor_state))
        report_group = component_spec["report_group"]
        component_kinetic_terms = component_spec["kinetic_terms"]
        component_potential_terms = component_spec["potential_terms"]

        if compute_jump_residuals:
            if component_monitor_residual <= residual_tolerance:
                component_jump_residuals = tuple(0.0 for _ in component_kinetic_terms)
            else:
                component_jump_residuals = _component_jump_residuals_from_monitor_state(
                    monitor_state=component_monitor_state,
                    kinetic_terms=component_kinetic_terms,
                    model=model,
                    build_result=build_result,
                    builder=builder,
                    backend=backend,
                    local_term_cache=local_term_cache,
                )
        else:
            component_jump_residuals = ()

        components.append(
            ReducedIZMonitorComponent(
                component_id=int(component_spec["component_id"]),
                monitor=component_monitor,
                zero_indices=tuple(int(zero_report.zero_index) for zero_report in report_group),
                support_variables=component_spec["support"],
                support_plaquette_ids=component_spec["support_plaquette_ids"],
                monitor_plaquette_ids=tuple(int(term.term_id) for term in component_kinetic_terms),
                jump_plaquette_ids=tuple(int(term.term_id) for term in component_kinetic_terms),
                z_value=component_spec["z_value"],
                n_potential_terms=(
                    len(component_potential_terms)
                    if reduced_iz_monitor_content == "offdiagonal_plus_potential"
                    else 0
                ),
                monitor_residual=component_monitor_residual,
                jump_residuals=component_jump_residuals,
            )
        )

    _record_construction_stage(
        timing_collector,
        "component_diagnostics",
        component_stage_start,
    )

    total_monitor = _LazySparseSumOperator(
        terms=tuple(total_monitor_terms),
        shape=shape,
        _state_action=total_monitor_state,
    )

    return (
        total_monitor,
        selected_reports,
        tuple(components),
        tuple(jumps),
    )


def _union_support_for_zero_reports(
    reports: tuple[InterferenceZeroReport, ...],
) -> tuple[int, ...]:
    support: set[int] = set()

    for zero_report in reports:
        support.update(_support_key_for_zero_report(zero_report))

    return tuple(sorted(support))


def _plaquette_local_terms_by_operator_kind(
    model: Any,
) -> tuple[
    tuple[LocalTermDescriptor, ...],
    tuple[LocalTermDescriptor, ...],
    dict[int, LocalTermDescriptor],
]:
    """Return plaquette kinetic/potential descriptors from one model query.

    Calling ``local_term_descriptors`` once avoids rebuilding identical
    plaquette-support tuples separately for kinetic and potential terms.
    """
    terms = model.local_term_descriptors(term_kind="plaquette")
    kinetic_terms = tuple(term for term in terms if term.operator_kind == "kinetic")
    potential_terms = tuple(term for term in terms if term.operator_kind == "potential")
    potential_by_pid = {int(term.term_id): term for term in potential_terms}
    return kinetic_terms, potential_terms, potential_by_pid


def _term_support_sets(
    terms: tuple[LocalTermDescriptor, ...],
) -> tuple[tuple[LocalTermDescriptor, frozenset[int]], ...]:
    return tuple((term, term.support_link_set) for term in terms)


def _select_terms_inside_cached_supports(
    term_supports: tuple[tuple[LocalTermDescriptor, frozenset[int]], ...],
    variable_support: frozenset[int],
) -> tuple[LocalTermDescriptor, ...]:
    return tuple(term for term, support in term_supports if support <= variable_support)


def _select_monitor_recycler_closure_terms(
    term_supports: tuple[tuple[LocalTermDescriptor, frozenset[int]], ...],
    variable_support: frozenset[int],
    *,
    hamiltonian_closure_source: MonitorRecyclerHamiltonianClosureSource,
) -> tuple[LocalTermDescriptor, ...]:
    if hamiltonian_closure_source == "local_hamiltonian_terms":
        return tuple(term for term, support in term_supports if support <= variable_support)

    if hamiltonian_closure_source == "boundary_hamiltonian_terms":
        return tuple(
            term
            for term, support in term_supports
            if bool(support & variable_support) and not support <= variable_support
        )

    if hamiltonian_closure_source == "touching_hamiltonian_terms":
        return tuple(term for term, support in term_supports if bool(support & variable_support))

    raise ValueError(
        "Local monitor-recycler closure source must be one of "
        "'local_hamiltonian_terms', 'boundary_hamiltonian_terms', or "
        "'touching_hamiltonian_terms'."
    )


def _plaquette_ids_inside_cached_supports(
    term_supports: tuple[tuple[LocalTermDescriptor, frozenset[int]], ...],
    variable_support: frozenset[int],
) -> tuple[int, ...]:
    return tuple(
        int(term.term_id) for term, support in term_supports if support <= variable_support
    )


def _select_terms_inside_variable_support(
    terms: tuple[LocalTermDescriptor, ...],
    variable_support: frozenset[int],
) -> tuple[LocalTermDescriptor, ...]:
    return tuple(term for term in terms if term.support_link_set <= variable_support)


def _plaquette_ids_inside_variable_support(
    terms: tuple[LocalTermDescriptor, ...],
    variable_support: frozenset[int],
) -> tuple[int, ...]:
    return tuple(int(term.term_id) for term in terms if term.support_link_set <= variable_support)


def _plaquette_ids_touching_variable_support(
    terms: tuple[LocalTermDescriptor, ...],
    variable_support: frozenset[int],
) -> tuple[int, ...]:
    return tuple(
        int(term.term_id)
        for term in terms
        if not term.support_link_set.isdisjoint(variable_support)
    )


def _infer_sharp_potential_value(
    *,
    potential_matrix: Any,
    state: NDArray[np.complex128],
    residual_tolerance: float,
    label: str,
) -> complex:
    potential_state = potential_matrix @ state
    z_value = complex(np.vdot(state, potential_state))
    residual = float(np.linalg.norm(potential_state - z_value * state))

    if residual > residual_tolerance:
        raise ValueError(
            f"Could not infer a sharp potential value for {label}: "
            f"||V psi - z psi||={residual:.3e}."
        )

    return z_value


def _add_potential_to_monitor(
    *,
    monitor: Any,
    model: Any,
    build_result: ModelBuildResult,
    potential_terms: tuple[LocalTermDescriptor, ...],
    state: NDArray[np.complex128],
    identity: Any,
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    shape: tuple[int, int],
    residual_tolerance: float,
    label: str,
    local_term_cache: _LocalTermMatrixCache | None = None,
) -> tuple[Any, complex]:
    potential_matrix = _sum_local_terms(
        model=model,
        build_result=build_result,
        terms=potential_terms,
        builder=builder,
        backend=backend,
        shape=shape,
        local_term_cache=local_term_cache,
    )

    z_value = _infer_sharp_potential_value(
        potential_matrix=potential_matrix,
        state=state,
        residual_tolerance=residual_tolerance,
        label=label,
    )

    return (monitor + potential_matrix - z_value * identity).tocsr(), z_value


def _monitor_coefficient_for_zero_report(
    zero_report: InterferenceZeroReport,
    *,
    use_collective_coefficients: bool,
) -> complex:
    if (
        use_collective_coefficients
        and zero_report.probe_mechanism_label == "collective_cancellation"
    ):
        coefficient = complex(zero_report.collective_cancellation_coefficient)

        if coefficient == 0:
            raise ValueError(
                "Collective-cancellation zero report has zero stored "
                "coefficient. Cannot build reduced-IZ monitor with "
                "collective coefficients."
            )

        return coefficient

    return 1.0 + 0.0j


def _transition_pattern_key(values: tuple[int, ...] | NDArray[np.integer]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _group_local_transitions_by_source(
    transitions: tuple[LocalTransitionPattern, ...],
) -> dict[tuple[int, ...], tuple[LocalTransitionPattern, ...]]:
    grouped: dict[tuple[int, ...], list[LocalTransitionPattern]] = {}

    for transition in transitions:
        key = _transition_pattern_key(transition.source_local)
        grouped.setdefault(key, []).append(transition)

    return {key: tuple(group) for key, group in grouped.items()}


def _build_reduced_iz_operator_matrix(
    *,
    zero_report: InterferenceZeroReport,
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    shape: tuple[int, int],
    assembly_cache: _ReducedIZAssemblyCache | None = None,
) -> sp.csr_array:
    if assembly_cache is None:
        assembly_cache = _ReducedIZAssemblyCache(
            basis_configs=basis_configs,
            config_to_index=config_to_index,
        )

    rows_parts: list[NDArray[np.int64]] = []
    cols_parts: list[NDArray[np.int64]] = []
    data_parts: list[NDArray[np.complex128]] = []

    for transition in zero_report.local_transitions:
        rows, cols = assembly_cache.transition_indices(
            local_mask=zero_report.local_mask,
            source_local=transition.source_local,
            target_local=transition.target_local,
        )

        if len(rows) == 0:
            continue

        rows_parts.append(rows)
        cols_parts.append(cols)
        data_parts.append(
            np.full(
                len(rows),
                complex(transition.matrix_element),
                dtype=np.complex128,
            )
        )

    if len(data_parts) == 0:
        return sp.csr_array(shape, dtype=np.complex128)

    return sp.csr_array(
        (
            np.concatenate(data_parts),
            (np.concatenate(rows_parts), np.concatenate(cols_parts)),
        ),
        shape=shape,
        dtype=np.complex128,
    )


def _format_float(value: float) -> str:
    return f"{value:.3e}"


def _format_float_or_none(value: float | None) -> str:
    if value is None:
        return "not checked"

    return _format_float(float(value))


def _format_complex_or_none(value: complex | None) -> str:
    if value is None:
        return "None"

    value = complex(value)

    if abs(value.imag) < 1e-14:
        return f"{value.real:.12g}"

    return f"{value.real:.12g}{value.imag:+.12g}j"


def _status_for_residual(
    value: float | None,
    *,
    excellent: float = 1e-12,
    acceptable: float = 1e-8,
) -> str:
    if value is None:
        return "[dim]n/a[/dim]"

    if value <= excellent:
        return "[green]ok[/green]"

    if value <= acceptable:
        return "[yellow]warn[/yellow]"

    return "[red]large[/red]"


def _format_index_tuple(
    values: tuple[int, ...],
    *,
    max_items: int = 16,
) -> str:
    if len(values) == 0:
        return "∅"

    if len(values) <= max_items:
        return ", ".join(str(value) for value in values)

    head = ", ".join(str(value) for value in values[:max_items])
    return f"{head}, ... ({len(values)} total)"


def _format_tuple_of_pairs(values: tuple[tuple[int, int], ...]) -> str:
    if len(values) == 0:
        return "()"
    return "(" + ", ".join(f"{left}/{right}" for left, right in values) + ")"
