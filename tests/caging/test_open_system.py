from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.caging.open_system import (
    CageLindbladConstruction,
    _select_jump_terms,
    _select_monitor_terms,
)
from qlinks.models import LocalTermDescriptor
from qlinks.open_system import (
    LindbladEvolutionOptions,
    initial_density_matrix,
)


@dataclass(frozen=True)
class _FakeRegion:
    region_size: int = 0


def _fake_construction() -> CageLindbladConstruction:
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    zero = sp.csr_array((2, 2), dtype=np.complex128)

    return CageLindbladConstruction(
        cage_state=ket0,
        region=_FakeRegion(),  # type: ignore[arg-type]
        z_value=None,
        inside_plaquette_ids=(),
        outside_plaquette_ids=(),
        crossing_plaquette_ids=(),
        monitor=zero,
        jumps=(),
        n_jumps=0,
        open_system_backend="scipy",
        monitor_source="reduced_iz_operators",
        n_reduced_iz_monitor_terms=0,
        reduced_iz_monitor_zero_indices=(),
        jump_operator_design="kinetic_times_monitor",
        monitor_plaquette_policy="strict_inside",
        jump_plaquette_policy="outside_or_crossing",
        monitor_plaquette_ids=(),
        jump_plaquette_ids=(),
        kinetic_terms_monitor=(),
        potential_terms_monitor=(),
        kinetic_terms_jump=(),
        monitor_residual=0.0,
        max_jump_residual=0.0,
        jump_residuals=(),
        liouvillian_residual=0.0,
    )


def test_select_jump_terms_can_include_crossing_terms():
    inside = (
        LocalTermDescriptor(
            term_id=0,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(0, 1),
        ),
    )
    outside = ()
    crossing = (
        LocalTermDescriptor(
            term_id=1,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(1, 2),
        ),
    )

    selected = _select_jump_terms(
        inside_terms=inside,
        outside_terms=outside,
        crossing_terms=crossing,
        policy="outside_or_crossing",
    )

    assert tuple(term.term_id for term in selected) == (1,)


def test_select_monitor_terms_stays_strict_inside_by_default():
    inside = (
        LocalTermDescriptor(
            term_id=0,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(0, 1),
        ),
    )
    crossing = (
        LocalTermDescriptor(
            term_id=1,
            term_kind="plaquette",
            operator_kind="kinetic",
            support_links=(1, 2),
        ),
    )

    selected = _select_monitor_terms(
        inside_terms=inside,
        outside_terms=(),
        crossing_terms=crossing,
        policy="strict_inside",
    )

    assert tuple(term.term_id for term in selected) == (0,)


def test_cage_lindblad_construction_exposes_lindblad_problem():
    construction = _fake_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))

    problem = construction.to_lindblad_problem(hamiltonian=hamiltonian)

    assert problem.dim == 2
    assert len(problem.jumps) == construction.n_jumps


def test_cage_lindblad_construction_verify_final_state():
    construction = _fake_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))
    rho = np.outer(construction.cage_state, construction.cage_state.conj())

    result = construction.verify_final_state(
        rho,
        hamiltonian=hamiltonian,
    )

    assert result.density_matrix.fidelity_with_target == pytest.approx(1.0)
    assert result.lindblad_residual < 1e-12


def test_cage_lindblad_construction_evolve_uses_new_api():
    construction = _fake_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))

    rho0 = initial_density_matrix(2, kind="mixed", rng=0)
    times = np.linspace(0.0, 0.01, 3)

    result = construction.evolve(
        hamiltonian=hamiltonian,
        density_matrix_initial=rho0,
        times=times,
        options=LindbladEvolutionOptions(
            method="rk4_matrix",
            rk4_step_policy="adaptive",
        ),
    )

    assert len(result.density_matrices) == len(times)
