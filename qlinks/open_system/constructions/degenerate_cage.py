from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.basis import basis_configs_from_build_result
from qlinks.caging.search import CageRecord
from qlinks.models.base import ModelBuildResult
from qlinks.models.local_terms import LocalTermKind
from qlinks.open_system.backend import OpenSystemBackendName
from qlinks.open_system.constructions.cage import _local_terms_by_operator_kind
from qlinks.open_system.diagnostics import DarkManifoldDiagnostics, diagnose_dark_manifold
from qlinks.open_system.local_recycling import (
    LocalRecyclingBuildResult,
    LocalSubspaceSupportReport,
    RecyclingJumpSource,
    build_local_recycling_jumps_from_subspace_regions,
    local_subspace_support_report_from_recycling_build_result,
)
from qlinks.open_system.manifold_detectors import (
    DressedManifoldDarkDetectorReport,
    ManifoldDarkOperatorBasisReport,
    diagnose_dressed_manifold_dark_detectors,
    diagnose_manifold_dark_operator_basis,
)
from qlinks.open_system.operators import lindblad_rhs_density_matrix
from qlinks.open_system.solvers import LindbladProblem

LocalRegionSource = Literal["kinetic", "potential", "all"]


def _as_csr(operator: Any) -> sp.csr_array:
    if hasattr(operator, "tocsr"):
        return operator.tocsr()
    return sp.csr_array(operator)


def _orthonormalize_state_matrix(
    states: NDArray[np.complex128],
    *,
    dim: int,
    tolerance: float,
) -> NDArray[np.complex128]:
    matrix = np.asarray(states, dtype=np.complex128)

    if matrix.ndim == 1:
        if matrix.size != dim:
            raise ValueError("state vector has incompatible dimension.")
        matrix = matrix.reshape(dim, 1)
    elif matrix.ndim == 2:
        if matrix.shape[0] == dim:
            pass
        elif matrix.shape[1] == dim:
            matrix = matrix.T
        else:
            raise ValueError("states must have shape (dim, n_states) or (n_states, dim).")
    else:
        raise ValueError("states must be one- or two-dimensional.")

    if matrix.shape[1] == 0:
        raise ValueError("states must contain at least one state.")

    q, r = np.linalg.qr(matrix)
    rank = int(np.count_nonzero(np.abs(np.diag(r)) > tolerance))
    if rank == 0:
        raise ValueError("states have numerical rank zero.")

    return q[:, :rank].astype(np.complex128, copy=False)


def _state_matrix_from_records(
    records: Sequence[CageRecord],
    *,
    hilbert_size: int,
) -> NDArray[np.complex128]:
    if len(records) == 0:
        raise ValueError("records must contain at least one CageRecord.")

    matrix = np.zeros((hilbert_size, len(records)), dtype=np.complex128)
    for column_index, record in enumerate(records):
        if record.full_state is not None:
            state = np.asarray(record.full_state, dtype=np.complex128)
            if state.shape != (hilbert_size,):
                raise ValueError(
                    "record.full_state has incompatible dimension: "
                    f"{state.shape} != {(hilbert_size,)}."
                )
            matrix[:, column_index] = state
        else:
            support = np.asarray(record.support, dtype=np.int64)
            local_state = np.asarray(record.local_state, dtype=np.complex128)
            if support.ndim != 1 or local_state.shape != support.shape:
                raise ValueError("record support and local_state have incompatible shapes.")
            if np.any(support < 0) or np.any(support >= hilbert_size):
                raise ValueError("record support contains out-of-range basis indices.")
            matrix[support, column_index] = local_state

    return matrix


def _validate_record_signatures(
    records: Sequence[CageRecord],
) -> tuple[int, int] | None:
    if len(records) == 0:
        return None

    signature = tuple(int(value) for value in records[0].signature)
    for record in records[1:]:
        if tuple(int(value) for value in record.signature) != signature:
            raise ValueError(
                "all cage records must have the same signature when "
                "validate_record_signature=True."
            )

    return signature


def _local_regions_from_model_terms(
    *,
    model: Any,
    local_term_kind: LocalTermKind | None,
    region_source: LocalRegionSource,
) -> tuple[tuple[int, ...], ...]:
    kinetic_terms, potential_terms, _ = _local_terms_by_operator_kind(
        model,
        term_kind=local_term_kind,
    )

    if region_source == "kinetic":
        terms = kinetic_terms
    elif region_source == "potential":
        terms = potential_terms
    elif region_source == "all":
        terms = kinetic_terms + potential_terms
    else:
        raise ValueError("region_source must be 'kinetic', 'potential', or 'all'.")

    regions: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for term in terms:
        region = tuple(sorted(int(index) for index in term.support_variable_set))
        if len(region) == 0 or region in seen:
            continue
        seen.add(region)
        regions.append(region)

    if len(regions) == 0:
        raise ValueError(
            "Could not infer local regions from model local terms. "
            "Pass local_regions explicitly."
        )

    return tuple(regions)


def _normalize_local_regions(
    local_regions: Sequence[Sequence[int]],
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


def _manifold_projector(
    manifold_basis: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return manifold_basis @ manifold_basis.conj().T


def _hamiltonian_closure_residual(
    *,
    hamiltonian: Any,
    manifold_basis: NDArray[np.complex128],
) -> float:
    hamiltonian_matrix = _as_csr(hamiltonian)
    action = np.asarray(hamiltonian_matrix @ manifold_basis, dtype=np.complex128)
    projected_action = manifold_basis @ (manifold_basis.conj().T @ action)
    return float(np.linalg.norm(action - projected_action))


def _max_jump_residual(
    *,
    jumps: tuple[Any, ...],
    manifold_basis: NDArray[np.complex128],
) -> tuple[float, tuple[float, ...]]:
    residuals = tuple(float(np.linalg.norm(_as_csr(jump) @ manifold_basis)) for jump in jumps)
    return (max(residuals) if residuals else 0.0), residuals


def _inflow_norm(
    *,
    jumps: tuple[Any, ...],
    manifold_basis: NDArray[np.complex128],
) -> float:
    total = 0.0
    for jump in jumps:
        jump_matrix = _as_csr(jump)
        adjoint_action = np.asarray(jump_matrix.conj().T @ manifold_basis, dtype=np.complex128)
        total += float(np.linalg.norm(adjoint_action) ** 2)
    return float(np.sqrt(max(total, 0.0)))


def _manifold_density_matrix(
    manifold_basis: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    projector = _manifold_projector(manifold_basis)
    return projector / float(manifold_basis.shape[1])


def _format_float(value: float) -> str:
    return f"{value:.3e}"


@dataclass(frozen=True, slots=True)
class DegenerateCageLindbladConstruction:
    """Lindblad construction targeting a cage-state manifold in one sector.

    The target is the projector onto ``manifold_basis`` rather than a single
    vector.  Local reset jumps are built from the local RDM support of the
    normalized manifold projector, so every jump annihilates every vector in the
    target manifold.
    """

    manifold_basis: NDArray[np.complex128]
    jumps: tuple[Any, ...]
    local_regions: tuple[tuple[int, ...], ...]
    recycling_build_result: LocalRecyclingBuildResult
    open_system_backend: OpenSystemBackendName
    recycling_jump_source: RecyclingJumpSource
    record_signature: tuple[int, int] | None
    hamiltonian_closure_residual: float
    max_jump_residual: float
    jump_residuals: tuple[float, ...]
    inflow_norm: float
    liouvillian_residual: float | None = None

    @property
    def hilbert_dimension(self) -> int:
        return int(self.manifold_basis.shape[0])

    @property
    def manifold_dimension(self) -> int:
        return int(self.manifold_basis.shape[1])

    @property
    def n_jumps(self) -> int:
        return len(self.jumps)

    @property
    def target_density_matrix(self) -> NDArray[np.complex128]:
        return _manifold_density_matrix(self.manifold_basis)

    @property
    def local_subspace_support_report(self) -> LocalSubspaceSupportReport:
        """Explain local manifold support/nullity for each recycling region."""
        return local_subspace_support_report_from_recycling_build_result(
            self.recycling_build_result
        )

    def __rich__(self):
        return self.to_rich()

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "manifold_dimension": self.manifold_dimension,
            "record_signature": self.record_signature,
            "n_jumps": self.n_jumps,
            "n_regions": len(self.local_regions),
            "local_regions": self.local_regions,
            "recycling_jump_source": self.recycling_jump_source,
            "h_closure_residual": self.hamiltonian_closure_residual,
            "max_jump_residual": self.max_jump_residual,
            "jump_residuals": self.jump_residuals,
            "inflow_norm": self.inflow_norm,
            "liouvillian_residual": self.liouvillian_residual,
            "local_subspace_support": self.local_subspace_support_report.to_summary_dict(),
            "recycling_variable_indices": self.recycling_build_result.variable_indices,
            "recycling_alpha_beta_indices": self.recycling_build_result.alpha_beta_indices,
        }

    def to_rich(self, *, max_regions: int = 24):
        """Return a rich renderable summary of the degenerate construction."""
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DegenerateCageLindbladConstruction.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.hilbert_dimension))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("record signature", str(self.record_signature))
        overview.add_row("jumps", str(self.n_jumps))
        overview.add_row("regions", str(len(self.local_regions)))
        overview.add_row("recycling source", str(self.recycling_jump_source))

        checks = Table(title="Construction checks")
        checks.add_column("quantity", style="bold")
        checks.add_column("value", justify="right")
        checks.add_row("H closure residual", _format_float(self.hamiltonian_closure_residual))
        checks.add_row("max ||J_mu P_M||", _format_float(self.max_jump_residual))
        checks.add_row("inflow norm", _format_float(self.inflow_norm))
        checks.add_row(
            "||L(P_M/m)||",
            (
                "not checked"
                if self.liouvillian_residual is None
                else _format_float(float(self.liouvillian_residual))
            ),
        )

        return Panel(
            Group(
                overview,
                checks,
                self.local_subspace_support_report.to_rich(max_regions=max_regions),
            ),
            title=Text("Degenerate cage Lindblad construction", style="bold cyan"),
            border_style="cyan",
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

    def diagnose_manifold(
        self,
        *,
        hamiltonian: Any,
        backend: str | None = None,
        kernel_tolerance: float = 1e-10,
        liouvillian_zero_tolerance: float = 1e-9,
        check_liouvillian_spectrum: bool = True,
        max_liouvillian_dense_dimension: int = 4096,
        liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "auto",
        sparse_liouvillian_eigenvalue_count: int = 32,
    ) -> DarkManifoldDiagnostics:
        """Return manifold-aware dark/attractiveness diagnostics."""
        return diagnose_dark_manifold(
            hamiltonian=hamiltonian,
            jumps=self.jumps,
            target_states=self.manifold_basis,
            backend=self.open_system_backend if backend is None else backend,
            kernel_tolerance=kernel_tolerance,
            liouvillian_zero_tolerance=liouvillian_zero_tolerance,
            check_liouvillian_spectrum=check_liouvillian_spectrum,
            max_liouvillian_dense_dimension=max_liouvillian_dense_dimension,
            liouvillian_spectrum_method=liouvillian_spectrum_method,
            sparse_liouvillian_eigenvalue_count=sparse_liouvillian_eigenvalue_count,
        )

    def diagnose_dark_operator_basis(
        self,
        *,
        operators: tuple[Any, ...] | list[Any],
        operator_names: tuple[str, ...] | list[str] | None = None,
        tolerance: float = 1.0e-10,
        coefficient_tolerance: float = 1.0e-8,
        max_candidates: int | None = 16,
    ) -> ManifoldDarkOperatorBasisReport:
        """Find collective operator combinations ``D`` with ``D P_M = 0``.

        This is useful when every small region has full local RDM support, so no
        strictly local parent projector exists, but sums of local terms may still
        annihilate the manifold by interference/cancellation.
        """
        return diagnose_manifold_dark_operator_basis(
            states=self.manifold_basis,
            operators=operators,
            operator_names=operator_names,
            tolerance=tolerance,
            coefficient_tolerance=coefficient_tolerance,
            max_candidates=max_candidates,
        )

    def diagnose_dressed_dark_detectors(
        self,
        *,
        detector_operators: tuple[Any, ...] | list[Any],
        left_multipliers: tuple[Any, ...] | list[Any],
        detector_coefficients: NDArray[np.complex128] | None = None,
        dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
        detector_operator_names: tuple[str, ...] | list[str] | None = None,
        left_multiplier_names: tuple[str, ...] | list[str] | None = None,
        detector_names: tuple[str, ...] | list[str] | None = None,
        tolerance: float = 1.0e-10,
        dark_tolerance: float = 1.0e-10,
        inflow_tolerance: float = 1.0e-12,
        max_detectors: int | None = None,
        sort_by_inflow: bool = True,
    ) -> DressedManifoldDarkDetectorReport:
        """Test dressed jumps ``J = V D`` built from manifold-dark detectors.

        A nonzero inflow norm is a necessary condition for attraction into the
        target manifold.  This diagnostic does not by itself prove absence of
        invariant complement sectors; use ``diagnose_manifold`` on selected
        jumps for that stronger check.
        """
        return diagnose_dressed_manifold_dark_detectors(
            states=self.manifold_basis,
            detector_operators=detector_operators,
            detector_coefficients=detector_coefficients,
            dark_operator_report=dark_operator_report,
            left_multipliers=left_multipliers,
            detector_operator_names=detector_operator_names,
            left_multiplier_names=left_multiplier_names,
            detector_names=detector_names,
            tolerance=tolerance,
            dark_tolerance=dark_tolerance,
            inflow_tolerance=inflow_tolerance,
            max_detectors=max_detectors,
            sort_by_inflow=sort_by_inflow,
        )


def build_degenerate_cage_lindblad_construction(
    *,
    build_result: ModelBuildResult,
    records: Sequence[CageRecord] | None = None,
    states: NDArray[np.complex128] | None = None,
    model: Any | None = None,
    local_regions: Sequence[Sequence[int]] | None = None,
    local_term_kind: LocalTermKind | None = None,
    region_source: LocalRegionSource = "kinetic",
    recycling_jump_source: RecyclingJumpSource = "local_rdm_block_reset",
    max_jumps_per_region: int = 1,
    deduplicate_regions: bool = True,
    recycling_rdm_tolerance: float = 1e-10,
    recycling_dark_tolerance: float = 1e-10,
    recycling_inflow_tolerance: float = 1e-12,
    recycling_prefer_sparse: bool = True,
    recycling_two_pattern_tolerance: float = 1e-8,
    validate_record_signature: bool = True,
    open_system_backend: OpenSystemBackendName = "scipy",
    check_liouvillian: bool = True,
    residual_tolerance: float = 1e-10,
) -> DegenerateCageLindbladConstruction:
    """Build local reset jumps targeting a degenerate cage manifold.

    The supplied ``build_result`` is assumed to already represent the desired
    sector, for example one fixed QDM/QLM winding sector.  The target manifold
    can be supplied either as cage ``records`` or directly as a state matrix.
    If ``local_regions`` is omitted, they are inferred from the model's local
    kinetic-term supports.
    """
    dim = int(build_result.hamiltonian.shape[0])

    if records is None and states is None:
        raise ValueError("Either records or states must be provided.")
    if records is not None and states is not None:
        raise ValueError("Provide records or states, but not both.")

    record_signature = None
    if records is not None:
        if validate_record_signature:
            record_signature = _validate_record_signatures(records)
        states = _state_matrix_from_records(records, hilbert_size=dim)

    assert states is not None
    manifold_basis = _orthonormalize_state_matrix(
        np.asarray(states, dtype=np.complex128),
        dim=dim,
        tolerance=recycling_rdm_tolerance,
    )

    if local_regions is None:
        if model is None:
            raise ValueError(
                "model is required to infer local regions. Pass local_regions "
                "explicitly if no model is available."
            )
        regions = _local_regions_from_model_terms(
            model=model,
            local_term_kind=local_term_kind,
            region_source=region_source,
        )
    else:
        regions = _normalize_local_regions(local_regions)

    basis_configs = basis_configs_from_build_result(build_result)
    recycling_build_result = build_local_recycling_jumps_from_subspace_regions(
        basis_configs=basis_configs,
        states=manifold_basis,
        regions=regions,
        source=recycling_jump_source,
        max_jumps_per_region=max_jumps_per_region,
        deduplicate_regions=deduplicate_regions,
        rdm_tolerance=recycling_rdm_tolerance,
        dark_tolerance=recycling_dark_tolerance,
        inflow_tolerance=recycling_inflow_tolerance,
        prefer_sparse=recycling_prefer_sparse,
        two_pattern_tolerance=recycling_two_pattern_tolerance,
    )
    jumps = recycling_build_result.jumps

    h_closure_residual = _hamiltonian_closure_residual(
        hamiltonian=build_result.hamiltonian,
        manifold_basis=manifold_basis,
    )
    max_jump_residual, jump_residuals = _max_jump_residual(
        jumps=jumps,
        manifold_basis=manifold_basis,
    )
    inflow_norm = _inflow_norm(
        jumps=jumps,
        manifold_basis=manifold_basis,
    )

    if max_jump_residual > residual_tolerance:
        raise ValueError(
            "Degenerate cage jumps do not annihilate the target manifold: "
            f"max ||J P_M||_F={max_jump_residual:.3e}."
        )

    liouvillian_residual = None
    if check_liouvillian:
        rho = _manifold_density_matrix(manifold_basis)
        rhs = lindblad_rhs_density_matrix(
            rho,
            hamiltonian=build_result.hamiltonian,
            jumps=list(jumps),
            backend=open_system_backend,
        )
        liouvillian_residual = float(np.linalg.norm(rhs))

    return DegenerateCageLindbladConstruction(
        manifold_basis=manifold_basis,
        jumps=jumps,
        local_regions=regions,
        recycling_build_result=recycling_build_result,
        open_system_backend=open_system_backend,
        recycling_jump_source=recycling_jump_source,
        record_signature=record_signature,
        hamiltonian_closure_residual=h_closure_residual,
        max_jump_residual=max_jump_residual,
        jump_residuals=jump_residuals,
        inflow_norm=inflow_norm,
        liouvillian_residual=liouvillian_residual,
    )
