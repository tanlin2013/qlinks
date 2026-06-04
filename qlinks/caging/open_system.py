from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.caging.classification import CageClassificationReport
from qlinks.caging.support import CageRegionSupport, extract_cage_region_support
from qlinks.models.base import (
    HamiltonianBuilderName,
    ModelBuildResult,
    SparseBackendName,
)
from qlinks.models.local_terms import LocalTermDescriptor
from qlinks.open_system import (
    build_liouvillian,
    dark_state_residual,
    liouvillian_residual_of_pure_state,
    verify_lindblad_final_state,
)

MonitorPlaquettePolicy = Literal[
    "strict_inside",
    "touching",
]
JumpPlaquettePolicy = Literal[
    "disjoint_outside",
    "crossing",
    "outside_or_crossing",
    "not_strictly_inside",
]
JumpOperatorDesign = Literal[
    "kinetic_times_monitor",
    "kinetic_outside_monitor_inside",
    "hamiltonian_outside_monitor_inside",
]


@dataclass(frozen=True, slots=True)
class CageLindbladConstruction:
    """Open-system construction associated with one cage state."""

    cage_state: NDArray[np.complex128]
    region: CageRegionSupport
    z_value: complex

    inside_plaquette_ids: tuple[int, ...]
    outside_plaquette_ids: tuple[int, ...]
    crossing_plaquette_ids: tuple[int, ...]

    monitor: Any
    jumps: tuple[Any, ...]
    n_jumps: int

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

    def build_liouvillian(self, hamiltonian: Any, *, fmt: str = "csc") -> Any:
        return build_liouvillian(
            hamiltonian,
            list(self.jumps),
            fmt=fmt,
        )

    def verify_final_state(
        self,
        rho,
        *,
        hamiltonian,
        atol: float = 1e-10,
    ):
        return verify_lindblad_final_state(
            rho,
            hamiltonian=hamiltonian,
            jumps=list(self.jumps),
            target_state=self.cage_state,
            atol=atol,
        )


def build_type1_cage_lindblad_construction(
    *,
    model: Any,
    build_result: ModelBuildResult,
    cage_state: NDArray[np.complex128],
    classification_report: CageClassificationReport,
    z_value: complex,
    builder: HamiltonianBuilderName = "sparse",
    backend: SparseBackendName = "scipy",
    monitor_plaquette_policy: MonitorPlaquettePolicy = "strict_inside",
    jump_plaquette_policy: JumpPlaquettePolicy = "outside_or_crossing",
    jump_operator_design: JumpOperatorDesign = "kinetic_outside_monitor_inside",
    include_collective_cancellation: bool = True,
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

    kinetic_inside_matrix = _sum_local_terms(
        model=model,
        build_result=build_result,
        terms=monitor_kinetic_terms,
        builder=builder,
        backend=backend,
        shape=(dim, dim),
    )

    potential_inside_matrix = _sum_local_terms(
        model=model,
        build_result=build_result,
        terms=monitor_potential_terms,
        builder=builder,
        backend=backend,
        shape=(dim, dim),
    )

    monitor = (
        kinetic_inside_matrix + potential_inside_matrix - complex(z_value) * identity
    ).tocsr()

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
    jump_residuals = tuple(dark_state_residual(psi, list(jumps)))
    max_jump_residual = max(jump_residuals) if jump_residuals else 0.0

    liouvillian_residual = None
    if check_liouvillian:
        liouvillian = build_liouvillian(
            build_result.hamiltonian,
            list(jumps),
            fmt="csc",
        )
        liouvillian_residual = liouvillian_residual_of_pure_state(
            psi,
            liouvillian,
        )

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
        z_value=complex(z_value),
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
                    potential_terms_by_plaquette_id=(potential_terms_by_plaquette_id),
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
