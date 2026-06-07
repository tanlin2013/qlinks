from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.caging.classification import (
    CageClassificationReport,
    InterferenceZeroReport,
    basis_configs_from_build_result,
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
    OpenSystemBackendName,
    density_matrix_from_state,
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

    open_system_backend: OpenSystemBackendName

    monitor_source: MonitorSource
    n_reduced_iz_monitor_terms: int
    reduced_iz_monitor_zero_indices: tuple[int, ...]

    jump_operator_design: JumpOperatorDesign
    monitor_plaquette_policy: MonitorPlaquettePolicy
    jump_plaquette_policy: JumpPlaquettePolicy

    monitor_plaquette_ids: tuple[int, ...]
    jump_plaquette_ids: tuple[int, ...]

    kinetic_terms_monitor: tuple[LocalTermDescriptor, ...]
    potential_terms_monitor: tuple[LocalTermDescriptor, ...]
    kinetic_terms_jump: tuple[LocalTermDescriptor, ...]

    monitor_residual: float
    max_jump_residual: float
    jump_residuals: tuple[float, ...]
    liouvillian_residual: float | None = None

    def __repr__(self) -> str:
        return (
            "CageLindbladConstruction("
            f"monitor_source={self.monitor_source!r}, "
            f"jump_operator_design={self.jump_operator_design!r}, "
            f"region_size={self.region.region_size}, "
            f"n_jumps={self.n_jumps}, "
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
            "n_monitor_plaquettes": len(self.monitor_plaquette_ids),
            "n_jump_plaquettes": len(self.jump_plaquette_ids),
            "n_jumps": self.n_jumps,
            "n_reduced_iz_monitor_terms": self.n_reduced_iz_monitor_terms,
            "z_value": self.z_value,
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
            "zero indices",
            _format_index_tuple(self.reduced_iz_monitor_zero_indices),
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

        return Panel(
            Group(
                overview,
                geometry,
                diagnostics,
                reduced_iz,
                plaquette_ids,
            ),
            title=title,
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
    monitor_plaquette_policy: MonitorPlaquettePolicy = "strict_inside",
    jump_plaquette_policy: JumpPlaquettePolicy = "outside_or_crossing",
    jump_operator_design: JumpOperatorDesign = "kinetic_outside_monitor_inside",
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

        monitor, reduced_iz_monitor_reports = _build_reduced_iz_monitor(
            classification_report=classification_report,
            basis_configs=basis_configs,
            shape=shape,
            include_q_empty=include_q_empty,
            include_closed_by_known_zeros=include_closed_by_known_zeros,
            include_projector_like=include_projector_like,
            include_collective_cancellation=include_collective_cancellation,
            use_collective_coefficients=use_collective_coefficients,
        )

        n_reduced_iz_monitor_terms = len(reduced_iz_monitor_reports)
        reduced_iz_monitor_zero_indices = tuple(
            int(report.zero_index) for report in reduced_iz_monitor_reports
        )

    else:
        raise ValueError(f"Unknown monitor_source: {monitor_source!r}")

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

    monitor_residual = float(np.linalg.norm(monitor @ psi))
    jump_residuals = _jump_residuals(
        state=psi,
        jumps=jumps,
    )
    max_jump_residual = max(jump_residuals) if jump_residuals else 0.0

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
        open_system_backend=open_system_backend,
        monitor_source=monitor_source,
        n_reduced_iz_monitor_terms=n_reduced_iz_monitor_terms,
        reduced_iz_monitor_zero_indices=reduced_iz_monitor_zero_indices,
        jump_operator_design=jump_operator_design,
        monitor_plaquette_policy=monitor_plaquette_policy,
        jump_plaquette_policy=jump_plaquette_policy,
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


def _build_reduced_iz_monitor(
    *,
    classification_report: CageClassificationReport,
    basis_configs: NDArray[np.integer],
    shape: tuple[int, int],
    include_q_empty: bool = True,
    include_closed_by_known_zeros: bool = True,
    include_projector_like: bool = True,
    include_collective_cancellation: bool = True,
    use_collective_coefficients: bool = True,
) -> tuple[sp.csr_array, tuple[InterferenceZeroReport, ...]]:
    selected_reports = _reduced_iz_reports_for_monitor(
        classification_report,
        include_q_empty=include_q_empty,
        include_closed_by_known_zeros=include_closed_by_known_zeros,
        include_projector_like=include_projector_like,
        include_collective_cancellation=include_collective_cancellation,
    )

    config_to_index = {
        tuple(int(x) for x in config): index for index, config in enumerate(basis_configs)
    }

    monitor = sp.csr_array(shape, dtype=np.complex128)

    for zero_report in selected_reports:
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

    return monitor.tocsr(), selected_reports


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
