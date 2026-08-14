"""Black-box compatibility tests for the deprecated single-cage constructor result.

The deprecated implementation is intentionally not white-box tested. Its private helpers may
change or disappear during cleanup; this file protects only the small public compatibility
contract that remains useful to legacy notebooks.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.open_system import LindbladEvolutionOptions, initial_density_matrix
from qlinks.open_system.constructions.deprecated import CageLindbladConstruction


@dataclass(frozen=True)
class _FakeRegion:
    region_size: int = 0


def _minimal_construction() -> CageLindbladConstruction:
    ket0 = np.asarray([1.0, 0.0], dtype=np.complex128)
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
        n_component_jumps=0,
        n_global_jump_terms=0,
        open_system_backend="scipy",
        monitor_source="reduced_iz_operators",
        reduced_iz_monitor_decomposition="single_sum",
        reduced_iz_monitor_content="offdiagonal_only",
        n_reduced_iz_monitor_terms=0,
        reduced_iz_monitor_zero_indices=(),
        monitor_components=(),
        component_z_values=(),
        jump_operator_design="kinetic_times_monitor",
        monitor_plaquette_policy="strict_inside",
        jump_plaquette_policy="outside_or_crossing",
        monitor_plaquette_ids=(),
        jump_plaquette_ids=(),
        kinetic_terms_monitor=(),
        potential_terms_monitor=(),
        kinetic_terms_jump=(),
        recycling_jump_source="none",
        n_recycling_jumps=0,
        recycling_jump_variable_indices=(),
        recycling_jump_alpha_beta_indices=(),
        recycling_two_pattern_count=0,
        recycling_build_result=None,
        monitor_residual=0.0,
        max_jump_residual=0.0,
        jump_residuals=(),
    )


def test_deprecated_construction_exposes_compact_summary_and_problem() -> None:
    construction = _minimal_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))

    summary = construction.to_summary_dict()
    problem = construction.to_lindblad_problem(hamiltonian=hamiltonian)

    assert summary["region_size"] == 0
    assert summary["n_jumps"] == 0
    assert summary["monitor_source"] == "reduced_iz_operators"
    assert problem.dim == 2
    assert problem.jumps == ()


def test_deprecated_construction_verifies_its_target_state() -> None:
    construction = _minimal_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))
    target_density = np.outer(construction.cage_state, construction.cage_state.conj())

    verification = construction.verify_final_state(
        target_density,
        hamiltonian=hamiltonian,
    )

    assert verification.density_matrix.fidelity_with_target == pytest.approx(1.0)
    assert verification.lindblad_residual < 1.0e-12


def test_deprecated_construction_evolves_through_current_solver_api() -> None:
    construction = _minimal_construction()
    hamiltonian = sp.csr_array(np.diag([1.0, -1.0]).astype(np.complex128))
    initial = initial_density_matrix(2, kind="mixed", rng=0)
    times = np.linspace(0.0, 0.01, 3)

    result = construction.evolve(
        hamiltonian=hamiltonian,
        density_matrix_initial=initial,
        times=times,
        options=LindbladEvolutionOptions(
            method="rk4_matrix",
            rk4_step_policy="adaptive",
        ),
    )

    assert len(result.density_matrices) == len(times)
