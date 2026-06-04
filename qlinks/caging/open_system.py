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


@dataclass(frozen=True, slots=True)
class CageLindbladConstruction:
    """Open-system construction associated with one cage state."""

    cage_state: NDArray[np.complex128]
    region: CageRegionSupport
    z_value: complex

    inside_plaquette_ids: tuple[int, ...]
    outside_plaquette_ids: tuple[int, ...]
    crossing_plaquette_ids: tuple[int, ...]

    monitor_plaquette_ids: tuple[int, ...]
    jump_plaquette_ids: tuple[int, ...]

    kinetic_terms_monitor: tuple[LocalTermDescriptor, ...]
    potential_terms_monitor: tuple[LocalTermDescriptor, ...]
    kinetic_terms_jump: tuple[LocalTermDescriptor, ...]

    monitor: Any
    jumps: tuple[Any, ...]

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

    jumps = tuple(
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
        monitor_residual=monitor_residual,
        max_jump_residual=max_jump_residual,
        jump_residuals=jump_residuals,
        liouvillian_residual=liouvillian_residual,
    )


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
