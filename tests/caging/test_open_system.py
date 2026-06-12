from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.caging.open_system import (
    CageLindbladConstruction,
    _build_component_decomposition_jump_operators,
    _infer_sharp_potential_value,
    _LocalTermMatrixCache,
    _resolve_local_term_builder,
    _select_jump_terms,
    _select_monitor_terms,
)
from qlinks.encoded import BinaryEncodedBasis
from qlinks.models import LocalTermDescriptor
from qlinks.open_system import (
    LindbladEvolutionOptions,
    initial_density_matrix,
)
from qlinks.variables import LocalSpace, VariableLayout


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
        liouvillian_residual=0.0,
    )


def test_local_term_builder_uses_bitmask_for_encoded_build_result():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    encoded_basis = BinaryEncodedBasis.from_codes(layout, [0, 1, 2, 3])
    build_result = SimpleNamespace(basis=encoded_basis)

    assert _resolve_local_term_builder("sparse", build_result) == "bitmask"
    assert _resolve_local_term_builder("optimized", build_result) == "bitmask"


class _RecordingLocalTermModel:
    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    def build_local_term(self, term, build_result, *, builder, backend):
        del build_result, backend
        self.calls.append((int(term.term_id), str(builder)))
        return sp.csr_array(float(term.term_id) * np.eye(2, dtype=np.complex128))


def test_local_term_matrix_cache_reuses_terms_and_promotes_bitmask_builder():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    encoded_basis = BinaryEncodedBasis.from_codes(layout, [0, 1, 2, 3])
    build_result = SimpleNamespace(
        basis=encoded_basis,
        hamiltonian=sp.csr_array((2, 2), dtype=np.complex128),
    )
    model = _RecordingLocalTermModel()
    term = LocalTermDescriptor(
        term_id=3,
        term_kind="plaquette",
        operator_kind="kinetic",
        support_links=(0, 1),
    )
    cache = _LocalTermMatrixCache(
        model=model,
        build_result=build_result,
        builder="sparse",
        backend="scipy",
    )

    first = cache.get(term)
    second = cache.get(term)

    assert first is second
    assert model.calls == [(3, "bitmask")]
    np.testing.assert_allclose(first.toarray(), 3.0 * np.eye(2))


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


def test_infer_sharp_potential_value_accepts_eigenstate():
    state = np.array([1.0, 0.0], dtype=np.complex128)
    potential = sp.csr_array(np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.complex128))

    value = _infer_sharp_potential_value(
        potential_matrix=potential,
        state=state,
        residual_tolerance=1e-12,
        label="test",
    )

    assert value == pytest.approx(2.0)


def test_infer_sharp_potential_value_rejects_non_eigenstate():
    state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    potential = sp.csr_array(np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.complex128))

    with pytest.raises(ValueError, match="Could not infer"):
        _infer_sharp_potential_value(
            potential_matrix=potential,
            state=state,
            residual_tolerance=1e-12,
            label="test",
        )


def test_component_decomposition_kinetic_times_monitor_uses_only_component_jumps():
    component_jump_0 = sp.csr_array(np.eye(2, dtype=np.complex128))
    component_jump_1 = sp.csr_array(2.0 * np.eye(2, dtype=np.complex128))

    jumps = _build_component_decomposition_jump_operators(
        model=None,
        build_result=None,
        component_jumps=(component_jump_0, component_jump_1),
        jump_kinetic_terms=(),
        potential_terms_by_plaquette_id={},
        builder="sparse",
        backend="scipy",
        jump_operator_design="kinetic_times_monitor",
    )

    assert len(jumps) == 2
    np.testing.assert_allclose(jumps[0].toarray(), component_jump_0.toarray())
    np.testing.assert_allclose(jumps[1].toarray(), component_jump_1.toarray())


@dataclass(frozen=True)
class _FakeTerm:
    term_id: int

    @property
    def support_link_set(self):
        return frozenset()


class _FakeModel:
    def build_local_term(self, term, build_result, *, builder, backend):
        del build_result, builder, backend
        return sp.csr_array(
            np.array(
                [[float(term.term_id), 0.0], [0.0, float(term.term_id)]],
                dtype=np.complex128,
            )
        )


def test_component_decomposition_kinetic_outside_appends_bare_kinetic():
    component_jump = sp.csr_array(np.eye(2, dtype=np.complex128))
    model = _FakeModel()

    jumps = _build_component_decomposition_jump_operators(
        model=model,
        build_result=None,
        component_jumps=(component_jump,),
        jump_kinetic_terms=(_FakeTerm(3), _FakeTerm(5)),
        potential_terms_by_plaquette_id={},
        builder="sparse",
        backend="scipy",
        jump_operator_design="kinetic_outside_monitor_inside",
    )

    assert len(jumps) == 3
    np.testing.assert_allclose(jumps[0].toarray(), np.eye(2))
    np.testing.assert_allclose(jumps[1].toarray(), 3.0 * np.eye(2))
    np.testing.assert_allclose(jumps[2].toarray(), 5.0 * np.eye(2))


class _FakeHamiltonianModel:
    def build_local_term(self, term, build_result, *, builder, backend):
        del build_result, builder, backend

        if term.operator_kind == "kinetic":
            value = float(term.term_id)

        elif term.operator_kind == "potential":
            value = 10.0 * float(term.term_id)

        else:
            raise AssertionError(term.operator_kind)

        return sp.csr_array(value * np.eye(2, dtype=np.complex128))


def test_component_decomposition_hamiltonian_outside_appends_kinetic_plus_potential():
    component_jump = sp.csr_array(np.eye(2, dtype=np.complex128))
    model = _FakeHamiltonianModel()

    kinetic_term = LocalTermDescriptor(
        term_id=3,
        term_kind="plaquette",
        operator_kind="kinetic",
        support_links=(0, 1),
    )
    potential_term = LocalTermDescriptor(
        term_id=3,
        term_kind="plaquette",
        operator_kind="potential",
        support_links=(0, 1),
    )

    jumps = _build_component_decomposition_jump_operators(
        model=model,
        build_result=None,
        component_jumps=(component_jump,),
        jump_kinetic_terms=(kinetic_term,),
        potential_terms_by_plaquette_id={3: potential_term},
        builder="sparse",
        backend="scipy",
        jump_operator_design="hamiltonian_outside_monitor_inside",
    )

    assert len(jumps) == 2

    expected_outside = 33.0 * np.eye(2, dtype=np.complex128)
    np.testing.assert_allclose(jumps[1].toarray(), expected_outside)
