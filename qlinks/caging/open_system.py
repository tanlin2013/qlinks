from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.basis import basis_configs_from_build_result
from qlinks.caging.classification import (
    CageClassificationReport,
    InterferenceZeroReport,
)
from qlinks.caging.support import CageRegionSupport, extract_cage_region_support
from qlinks.models.base import (
    HamiltonianBuilderName,
    ModelBuildResult,
    SparseBackendName,
)
from qlinks.models.local_terms import LocalTermDescriptor
from qlinks.open_system import (
    LindbladProblem,
    LocalRecyclingBuildResult,
    OpenSystemBackendName,
    RecyclingJumpSource,
    build_local_recycling_jumps_from_regions,
    density_matrix_from_state,
    diagnose_absorbing_projector_symmetry,
    diagnose_dark_subspace,
    lindblad_rhs_density_matrix,
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
ReducedIZMonitorDecomposition = Literal[
    "single_sum",
    "exact_support",
    "connected_support",
]
ReducedIZMonitorContent = Literal[
    "offdiagonal_only",
    "offdiagonal_plus_potential",
]
JumpOperatorDesign = Literal[
    "kinetic_times_monitor",
    "kinetic_outside_monitor_inside",
    "hamiltonian_outside_monitor_inside",
]
JumpPlaquettePolicy = Literal[
    "disjoint_outside",
    "crossing",
    "outside_or_crossing",
    "not_strictly_inside",
]


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
    liouvillian_residual: float | None = None

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
            "monitor_residual": self.monitor_residual,
            "max_jump_residual": self.max_jump_residual,
            "liouvillian_residual": self.liouvillian_residual,
        }

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
    recycling_rdm_tolerance: float = 1e-10,
    recycling_dark_tolerance: float = 1e-10,
    recycling_inflow_tolerance: float = 1e-12,
    recycling_prefer_sparse: bool = True,
    recycling_two_pattern_tolerance: float = 1e-8,
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
    use_collective_coefficients: bool = True,
    check_liouvillian: bool = True,
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

    region = extract_cage_region_support(
        classification_report,
        include_collective_cancellation=include_collective_cancellation,
    )

    kinetic_terms = model.local_term_descriptors(
        operator_kind="kinetic",
        term_kind="plaquette",
    )
    potential_terms = model.local_term_descriptors(
        operator_kind="potential",
        term_kind="plaquette",
    )

    potential_by_pid = {int(term.term_id): term for term in potential_terms}

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

    dim = build_result.hamiltonian.shape[0]
    identity = sp.identity(dim, format="csr", dtype=np.complex128)
    shape = build_result.hamiltonian.shape

    monitor_components: tuple[ReducedIZMonitorComponent, ...] = ()
    component_jumps: tuple[Any, ...] | None = None

    if monitor_source == "local_hamiltonian_terms":
        kinetic_inside_matrix = _sum_local_terms(
            model=model,
            build_result=build_result,
            terms=monitor_kinetic_terms,
            builder=builder,
            backend=backend,
            shape=shape,
        )
        potential_inside_matrix = _sum_local_terms(
            model=model,
            build_result=build_result,
            terms=monitor_potential_terms,
            builder=builder,
            backend=backend,
            shape=shape,
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

            single_component_support = tuple(
                sorted(
                    {
                        index
                        for report in reduced_iz_monitor_reports
                        for index in _support_key_for_zero_report(report)
                    }
                )
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
                    support_variables=tuple(
                        sorted(
                            {
                                index
                                for report in reduced_iz_monitor_reports
                                for index in _support_key_for_zero_report(report)
                            }
                        )
                    ),
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
                residual_tolerance=residual_tolerance,
                include_q_empty=include_q_empty,
                include_closed_by_known_zeros=include_closed_by_known_zeros,
                include_projector_like=include_projector_like,
                include_collective_cancellation=include_collective_cancellation,
                use_collective_coefficients=use_collective_coefficients,
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

    if component_jumps is not None:
        jumps = _build_component_decomposition_jump_operators(
            model=model,
            build_result=build_result,
            component_jumps=component_jumps,
            jump_kinetic_terms=jump_kinetic_terms,
            potential_terms_by_plaquette_id=potential_by_pid,
            builder=builder,
            backend=backend,
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
            jump_operator_design=jump_operator_design,
        )

    recycling_build_result = None
    recycling_jumps: tuple[Any, ...] = ()
    recycling_regions = _recycling_regions_from_construction_context(
        region=region,
        monitor_components=monitor_components,
    )

    if recycling_jump_source != "none":
        basis_configs = basis_configs_from_build_result(build_result)

        recycling_build_result = build_local_recycling_jumps_from_regions(
            basis_configs=basis_configs,
            target_state=psi,
            regions=recycling_regions,
            source=recycling_jump_source,
            max_jumps_per_region=max_recycling_jumps_per_region,
            rdm_tolerance=recycling_rdm_tolerance,
            dark_tolerance=recycling_dark_tolerance,
            inflow_tolerance=recycling_inflow_tolerance,
            prefer_sparse=recycling_prefer_sparse,
            two_pattern_tolerance=recycling_two_pattern_tolerance,
        )

        recycling_jumps = recycling_build_result.jumps
        jumps = tuple(jumps) + tuple(recycling_jumps)

    monitor_residual = float(np.linalg.norm(monitor @ psi))
    jump_residuals = _jump_residuals(
        state=psi,
        jumps=jumps,
    )
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
        n_recycling_jumps = 0
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

    liouvillian_residual = None
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

    if monitor_residual > residual_tolerance:
        raise ValueError(
            "The inferred regional monitor does not annihilate the cage state: "
            f"||M_R psi||={monitor_residual:.3e}. "
            "This may mean z_value is wrong, R is incomplete, or the cage is "
            "not compatible with the current type-1 construction."
        )

    if max_jump_residual > residual_tolerance:
        raise ValueError(
            "The inferred jump operators do not annihilate the cage state: "
            f"max_p ||J_p psi||={max_jump_residual:.3e}."
        )

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
        monitor_residual=monitor_residual,
        max_jump_residual=max_jump_residual,
        jump_residuals=jump_residuals,
        liouvillian_residual=liouvillian_residual,
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
) -> tuple[Any, ...]:
    if jump_operator_design == "kinetic_times_monitor":
        return tuple(
            (
                model.build_local_term(
                    term,
                    build_result,
                    builder=builder,
                    backend=backend,
                ).tocsr()
                @ monitor
            ).tocsr()
            for term in jump_kinetic_terms
        )

    if jump_operator_design == "kinetic_outside_monitor_inside":
        monitor_term_ids = {int(term.term_id) for term in monitor_kinetic_terms}

        jump_terms_by_id = {int(term.term_id): term for term in jump_kinetic_terms}

        all_jump_ids = sorted(set(monitor_term_ids) | set(jump_terms_by_id))

        jumps: list[Any] = []

        for plaquette_id in all_jump_ids:
            if plaquette_id in monitor_term_ids:
                term = next(
                    term for term in monitor_kinetic_terms if int(term.term_id) == plaquette_id
                )
                kinetic = model.build_local_term(
                    term,
                    build_result,
                    builder=builder,
                    backend=backend,
                ).tocsr()
                jumps.append((kinetic @ monitor).tocsr())
            else:
                term = jump_terms_by_id[plaquette_id]
                kinetic = model.build_local_term(
                    term,
                    build_result,
                    builder=builder,
                    backend=backend,
                ).tocsr()
                jumps.append(kinetic)

        return tuple(jumps)

    if jump_operator_design == "hamiltonian_outside_monitor_inside":
        monitor_term_ids = {int(term.term_id) for term in monitor_kinetic_terms}

        jump_terms_by_id = {int(term.term_id): term for term in jump_kinetic_terms}

        all_jump_ids = sorted(set(monitor_term_ids) | set(jump_terms_by_id))

        jumps: list[Any] = []

        for plaquette_id in all_jump_ids:
            if plaquette_id in monitor_term_ids:
                term = next(
                    term for term in monitor_kinetic_terms if int(term.term_id) == plaquette_id
                )
                kinetic = model.build_local_term(
                    term,
                    build_result,
                    builder=builder,
                    backend=backend,
                ).tocsr()
                jumps.append((kinetic @ monitor).tocsr())
            else:
                kinetic_term = jump_terms_by_id[plaquette_id]
                local_hamiltonian = _build_local_kinetic_plus_potential(
                    model=model,
                    build_result=build_result,
                    kinetic_term=kinetic_term,
                    potential_terms_by_plaquette_id=potential_terms_by_plaquette_id,
                    builder=builder,
                    backend=backend,
                )
                jumps.append(local_hamiltonian)

        return tuple(jumps)

    raise ValueError(f"Unknown jump_operator_design: {jump_operator_design!r}")


def _build_component_decomposition_jump_operators(
    *,
    model: Any,
    build_result: ModelBuildResult,
    component_jumps: tuple[Any, ...],
    jump_kinetic_terms: tuple[LocalTermDescriptor, ...],
    potential_terms_by_plaquette_id: dict[int, LocalTermDescriptor],
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
    jump_operator_design: JumpOperatorDesign,
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
            model.build_local_term(
                kinetic_term,
                build_result,
                builder=builder,
                backend=backend,
            ).tocsr()
            for kinetic_term in jump_kinetic_terms
        )

        return component_jumps + outside_jumps

    if jump_operator_design == "hamiltonian_outside_monitor_inside":
        outside_jumps = tuple(
            _build_local_kinetic_plus_potential(
                model=model,
                build_result=build_result,
                kinetic_term=kinetic_term,
                potential_terms_by_plaquette_id=potential_terms_by_plaquette_id,
                builder=builder,
                backend=backend,
            )
            for kinetic_term in jump_kinetic_terms
        )

        return component_jumps + outside_jumps

    raise ValueError(f"Unknown jump_operator_design: {jump_operator_design!r}")


def _build_local_kinetic_plus_potential(
    *,
    model: Any,
    build_result: ModelBuildResult,
    kinetic_term: LocalTermDescriptor,
    potential_terms_by_plaquette_id: dict[int, LocalTermDescriptor],
    builder: HamiltonianBuilderName,
    backend: SparseBackendName,
):
    kinetic = model.build_local_term(
        kinetic_term,
        build_result,
        builder=builder,
        backend=backend,
    ).tocsr()

    plaquette_id = int(kinetic_term.term_id)
    potential_term = potential_terms_by_plaquette_id.get(plaquette_id)

    if potential_term is None:
        return kinetic

    potential = model.build_local_term(
        potential_term,
        build_result,
        builder=builder,
        backend=backend,
    ).tocsr()

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
    region_variables: frozenset[int],
) -> tuple[
    tuple[LocalTermDescriptor, ...],
    tuple[LocalTermDescriptor, ...],
    tuple[LocalTermDescriptor, ...],
]:
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
) -> Any:
    if len(terms) == 0:
        return sp.csr_array(shape, dtype=np.complex128)

    total = sp.csr_array(shape, dtype=np.complex128)

    for term in terms:
        matrix = model.build_local_term(
            term,
            build_result,
            builder=builder,
            backend=backend,
        )

        if matrix is not None:
            total = total + matrix.tocsr()

    return total.tocsr()


def _jump_residuals(
    *,
    state: NDArray[np.complex128],
    jumps: tuple[Any, ...],
) -> tuple[float, ...]:
    return tuple(float(np.linalg.norm(jump @ state)) for jump in jumps)


def _reduced_iz_reports_for_monitor(
    report: CageClassificationReport,
    *,
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
) -> tuple[InterferenceZeroReport, ...]:
    selected: list[InterferenceZeroReport] = []

    for zero_report in report.zero_reports:
        label = zero_report.probe_mechanism_label

        if label == "q_empty" and include_q_empty:
            selected.append(zero_report)
        elif label == "closed_by_known_zeros" and include_closed_by_known_zeros:
            selected.append(zero_report)
        elif label == "projector_like" and include_projector_like:
            selected.append(zero_report)
        elif label == "collective_cancellation" and include_collective_cancellation:
            selected.append(zero_report)
        elif label == "unexplained_leakage":
            continue

    return tuple(selected)


def _support_key_for_zero_report(
    zero_report: InterferenceZeroReport,
) -> tuple[int, ...]:
    return tuple(int(index) for index in np.flatnonzero(zero_report.local_mask))


def _group_reduced_iz_reports_by_exact_support(
    reports: tuple[InterferenceZeroReport, ...],
) -> tuple[tuple[InterferenceZeroReport, ...], ...]:
    grouped: dict[tuple[int, ...], list[InterferenceZeroReport]] = {}

    for zero_report in reports:
        key = _support_key_for_zero_report(zero_report)
        grouped.setdefault(key, []).append(zero_report)

    return tuple(tuple(group) for _key, group in sorted(grouped.items(), key=lambda item: item[0]))


def _group_reduced_iz_reports_by_connected_support(
    reports: tuple[InterferenceZeroReport, ...],
) -> tuple[tuple[InterferenceZeroReport, ...], ...]:
    if len(reports) == 0:
        return ()

    supports = [set(_support_key_for_zero_report(zero_report)) for zero_report in reports]

    visited: set[int] = set()
    groups: list[tuple[InterferenceZeroReport, ...]] = []

    for start_index in range(len(reports)):
        if start_index in visited:
            continue

        stack = [start_index]
        component_indices: list[int] = []
        visited.add(start_index)

        while stack:
            current_index = stack.pop()
            component_indices.append(current_index)
            current_support = supports[current_index]

            for candidate_index, candidate_support in enumerate(supports):
                if candidate_index in visited:
                    continue

                if not current_support.isdisjoint(candidate_support):
                    visited.add(candidate_index)
                    stack.append(candidate_index)

        groups.append(tuple(reports[index] for index in component_indices))

    return tuple(groups)


def _group_reduced_iz_reports_for_monitor(
    reports: tuple[InterferenceZeroReport, ...],
    *,
    decomposition: ReducedIZMonitorDecomposition,
) -> tuple[tuple[InterferenceZeroReport, ...], ...]:
    if decomposition == "single_sum":
        return (reports,) if reports else ()

    if decomposition == "exact_support":
        return _group_reduced_iz_reports_by_exact_support(reports)

    if decomposition == "connected_support":
        return _group_reduced_iz_reports_by_connected_support(reports)

    raise ValueError(f"Unknown reduced-IZ monitor decomposition: {decomposition!r}")


def _build_reduced_iz_monitor_from_reports(
    *,
    reports: tuple[InterferenceZeroReport, ...],
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    shape: tuple[int, int],
    use_collective_coefficients: bool,
) -> sp.csr_array:
    monitor = sp.csr_array(shape, dtype=np.complex128)

    for zero_report in reports:
        coefficient = _monitor_coefficient_for_zero_report(
            zero_report,
            use_collective_coefficients=use_collective_coefficients,
        )

        reduced_operator = _build_reduced_iz_operator_matrix(
            zero_report=zero_report,
            basis_configs=basis_configs,
            config_to_index=config_to_index,
            shape=shape,
        )

        monitor = monitor + coefficient * reduced_operator

    return monitor.tocsr()


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

    monitor = _build_reduced_iz_monitor_from_reports(
        reports=selected_reports,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        shape=shape,
        use_collective_coefficients=use_collective_coefficients,
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
        )

    elif reduced_iz_monitor_content != "offdiagonal_only":
        raise ValueError(f"Unknown reduced-IZ monitor content: " f"{reduced_iz_monitor_content!r}")

    return monitor.tocsr(), selected_reports, inferred_z_value


def _build_reduced_iz_monitor_components(
    *,
    classification_report: CageClassificationReport,
    basis_configs: NDArray[np.integer],
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
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
    use_collective_coefficients: bool = True,
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

    config_to_index = {
        tuple(int(value) for value in config): index for index, config in enumerate(basis_configs)
    }

    report_groups = _group_reduced_iz_reports_for_monitor(
        selected_reports,
        decomposition=decomposition,
    )

    total_monitor = sp.csr_array(shape, dtype=np.complex128)
    components: list[ReducedIZMonitorComponent] = []
    jumps: list[Any] = []

    for component_id, report_group in enumerate(report_groups):
        component_monitor = _build_reduced_iz_monitor_from_reports(
            reports=report_group,
            basis_configs=basis_configs,
            config_to_index=config_to_index,
            shape=shape,
            use_collective_coefficients=use_collective_coefficients,
        )

        component_support = _union_support_for_zero_reports(report_group)

        component_kinetic_terms = _select_terms_inside_variable_support(
            kinetic_terms,
            frozenset(component_support),
        )

        component_potential_terms = _select_terms_inside_variable_support(
            potential_terms,
            frozenset(component_support),
        )

        component_support_plaquette_ids = _plaquette_ids_inside_variable_support(
            kinetic_terms,
            frozenset(component_support),
        )

        component_z_value: complex | None = None

        if reduced_iz_monitor_content == "offdiagonal_plus_potential":
            component_monitor, component_z_value = _add_potential_to_monitor(
                monitor=component_monitor,
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
            )
        elif reduced_iz_monitor_content != "offdiagonal_only":
            raise ValueError(
                f"Unknown reduced-IZ monitor content: " f"{reduced_iz_monitor_content!r}"
            )

        component_jumps: list[Any] = []
        for kinetic_term in component_kinetic_terms:
            kinetic = model.build_local_term(
                kinetic_term,
                build_result,
                builder=builder,
                backend=backend,
            ).tocsr()

            component_jumps.append((kinetic @ component_monitor).tocsr())

        component_jump_residuals = _jump_residuals(
            state=state,
            jumps=tuple(component_jumps),
        )

        component_monitor_residual = float(np.linalg.norm(component_monitor @ state))

        components.append(
            ReducedIZMonitorComponent(
                component_id=component_id,
                monitor=component_monitor,
                zero_indices=tuple(int(zero_report.zero_index) for zero_report in report_group),
                support_variables=component_support,
                support_plaquette_ids=component_support_plaquette_ids,
                monitor_plaquette_ids=tuple(int(term.term_id) for term in component_kinetic_terms),
                jump_plaquette_ids=tuple(int(term.term_id) for term in component_kinetic_terms),
                z_value=component_z_value,
                n_potential_terms=(
                    len(component_potential_terms)
                    if reduced_iz_monitor_content == "offdiagonal_plus_potential"
                    else 0
                ),
                monitor_residual=component_monitor_residual,
                jump_residuals=component_jump_residuals,
            )
        )

        total_monitor = total_monitor + component_monitor
        jumps.extend(component_jumps)

    return (
        total_monitor.tocsr(),
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
) -> tuple[Any, complex]:
    potential_matrix = _sum_local_terms(
        model=model,
        build_result=build_result,
        terms=potential_terms,
        builder=builder,
        backend=backend,
        shape=shape,
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


def _build_reduced_iz_operator_matrix(
    *,
    zero_report: InterferenceZeroReport,
    basis_configs: NDArray[np.integer],
    config_to_index: dict[tuple[int, ...], int],
    shape: tuple[int, int],
) -> sp.csr_array:
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    local_mask = zero_report.local_mask

    for source_index, source_config in enumerate(basis_configs):
        source_local = tuple(int(x) for x in source_config[local_mask])

        for transition in zero_report.local_transitions:
            if tuple(int(x) for x in transition.source_local) != source_local:
                continue

            target_config = np.array(source_config, copy=True)
            target_config[local_mask] = np.asarray(
                transition.target_local,
                dtype=target_config.dtype,
            )

            target_index = config_to_index.get(tuple(int(x) for x in target_config))
            if target_index is None:
                continue

            rows.append(int(target_index))
            cols.append(int(source_index))
            data.append(complex(transition.matrix_element))

    return sp.csr_array(
        (
            np.asarray(data, dtype=np.complex128),
            (rows, cols),
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
