from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from qlinks.basis import basis_configs_from_build_result
from qlinks.caging import (
    CageRecord,
    CageState,
    CandidateSubgraph,
    local_term_operator_basis,
    scan_sga_ladder_basis_frequencies,
    sga_ladder_basis_diagnostic,
    sga_ladder_basis_diagnostic_from_cage_records,
    sga_operator_diagnostic,
)
from qlinks.models import LocalTermDescriptor, SpinOneXYChainModel
from qlinks.models.spin_one_xy import spin_one_xy_scar_tower_states
from qlinks.open_system.local_recycling import embed_local_pattern_operator


def _matrix_unit(target: int, source: int, size: int = 3):
    matrix = np.zeros((size, size), dtype=np.complex128)
    matrix[target, source] = 1.0
    return sp.csr_array(matrix)


def test_sga_operator_diagnostic_detects_single_frequency_ladder() -> None:
    hamiltonian = sp.diags([0.0, 1.0, 2.0], dtype=np.complex128, format="csr")
    states = np.eye(3, dtype=np.complex128)
    ladder = _matrix_unit(1, 0) + 2.0 * _matrix_unit(2, 1)

    diagnostic = sga_operator_diagnostic(
        hamiltonian=hamiltonian,
        states=states,
        operator=ladder,
        frequency=1.0,
        tolerance=1e-12,
    )

    assert diagnostic.is_sga_like
    assert diagnostic.closes_on_manifold
    assert diagnostic.has_single_frequency_action
    assert np.isclose(diagnostic.frequency, 1.0)
    assert diagnostic.n_transitions == 2
    summary = diagnostic.to_summary_dict()
    assert summary["is_sga_like"] is True


def test_sga_operator_diagnostic_flags_off_frequency_action() -> None:
    hamiltonian = sp.diags([0.0, 1.0, 2.0], dtype=np.complex128, format="csr")
    states = np.eye(3, dtype=np.complex128)
    mixed = _matrix_unit(1, 0) + _matrix_unit(2, 0)

    diagnostic = sga_operator_diagnostic(
        hamiltonian=hamiltonian,
        states=states,
        operator=mixed,
        frequency=1.0,
        tolerance=1e-12,
    )

    assert not diagnostic.has_single_frequency_action
    assert not diagnostic.is_sga_like
    assert diagnostic.off_frequency_action_norm > 0.0


def test_sga_ladder_basis_diagnostic_finds_frequency_subspace() -> None:
    hamiltonian = sp.diags([0.0, 1.0, 2.0], dtype=np.complex128, format="csr")
    states = np.eye(3, dtype=np.complex128)
    operators = (
        _matrix_unit(1, 0),
        _matrix_unit(2, 1),
        _matrix_unit(2, 0),
        _matrix_unit(0, 0),
    )

    diagnostic = sga_ladder_basis_diagnostic(
        hamiltonian=hamiltonian,
        states=states,
        operators=operators,
        operator_names=("E10", "E21", "E20", "E00"),
        frequency=1.0,
        tolerance=1e-12,
    )

    assert diagnostic.has_ladder_candidates
    assert diagnostic.ladder_nullity == 2
    assert diagnostic.constraint_rank == 2
    assert len(diagnostic.candidate_diagnostics) == 2
    assert all(candidate.is_sga_like for candidate in diagnostic.candidate_diagnostics)


def test_scan_sga_ladder_basis_frequencies_uses_energy_gaps() -> None:
    hamiltonian = sp.diags([0.0, 1.0, 3.0], dtype=np.complex128, format="csr")
    states = np.eye(3, dtype=np.complex128)
    operators = (
        _matrix_unit(1, 0),
        _matrix_unit(2, 1),
        _matrix_unit(2, 0),
    )

    diagnostics = scan_sga_ladder_basis_frequencies(
        hamiltonian=hamiltonian,
        states=states,
        operators=operators,
        max_ladder_candidates=4,
        tolerance=1e-12,
    )

    frequencies = {round(float(diagnostic.frequency.real), 8) for diagnostic in diagnostics}
    assert frequencies == {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0}
    by_frequency = {
        round(float(diagnostic.frequency.real), 8): diagnostic for diagnostic in diagnostics
    }
    assert by_frequency[1.0].ladder_nullity == 1
    assert by_frequency[2.0].ladder_nullity == 1
    assert by_frequency[3.0].ladder_nullity == 1


@dataclass(frozen=True)
class _FakeModel:
    def local_term_descriptors(self, *, operator_kind=None, term_kind=None):
        assert operator_kind == "kinetic"
        assert term_kind is None
        return (
            LocalTermDescriptor(
                term_id=0,
                term_kind="bond",
                operator_kind="kinetic",
                support_links=(),
                support_variables=(0, 1),
                label="K0",
            ),
            LocalTermDescriptor(
                term_id=1,
                term_kind="bond",
                operator_kind="kinetic",
                support_links=(),
                support_variables=(1, 2),
                label="K1",
            ),
        )

    def build_local_term(self, descriptor, build_result, *, builder, backend, on_missing):
        del build_result, builder, backend, on_missing
        return _matrix_unit(int(descriptor.term_id) + 1, int(descriptor.term_id))


def test_local_term_operator_basis_uses_model_local_term_interface() -> None:
    basis = local_term_operator_basis(
        model=_FakeModel(),
        build_result=object(),
        operator_kind="kinetic",
    )

    assert basis.operator_names == ("K0", "K1")
    assert len(basis.operators) == 2
    assert basis.descriptors[0].support_variable_set == frozenset({0, 1})
    assert basis.to_summary_dict()["n_operators"] == 2


def _spin_one_bimagnon_raising_basis(
    *,
    basis_configs: np.ndarray,
    model: SpinOneXYChainModel,
) -> tuple[tuple[sp.csr_array, ...], tuple[str, ...]]:
    local_patterns = ((-1,), (0,), (1,))
    # Spin-1 convention: (S^+)^2|-1> = 2|+1>.
    local_operator = np.zeros((3, 3), dtype=np.complex128)
    local_operator[2, 0] = 2.0

    operators: list[sp.csr_array] = []
    names: list[str] = []
    for site_id in model.lattice.site_ids:
        variable_index = int(model.layout.site_variable_index(int(site_id)))
        operators.append(
            embed_local_pattern_operator(
                basis_configs=basis_configs,
                variable_indices=(variable_index,),
                local_patterns=local_patterns,
                local_operator=local_operator,
            ).tocsr()
        )
        names.append(f"Sp2_{site_id}")

    return tuple(operators), tuple(names)


def _cage_records_from_state_matrix(
    *,
    hamiltonian: sp.csr_array,
    states: np.ndarray,
    tolerance: float = 1.0e-12,
) -> tuple[CageRecord, ...]:
    records: list[CageRecord] = []
    for state_index in range(states.shape[1]):
        full_state = np.asarray(states[:, state_index], dtype=np.complex128)
        support = np.flatnonzero(np.abs(full_state) > tolerance).astype(np.int64)
        energy = complex(np.vdot(full_state, hamiltonian @ full_state))
        records.append(
            CageRecord(
                cage_state=CageState(
                    energy=energy,
                    local_state=full_state[support],
                    support=support,
                    boundary_residual=0.0,
                    eigen_residual=0.0,
                    full_residual=0.0,
                ),
                signature=(0, state_index),
                candidate=CandidateSubgraph(support),
                full_state=full_state,
            )
        )
    return tuple(records)


def test_sga_ladder_basis_from_cage_records_finds_spin_one_xy_pi_bimagnon() -> None:
    # Use nonzero single-ion anisotropy to lift competing spin-1 XY scar towers;
    # the pi-bimagnon tower remains an exact SGA tower with spacing 2*h_z.
    model = SpinOneXYChainModel(
        length=4,
        boundary_condition="periodic",
        j_xy=1.0,
        h_z=1.0,
        d_z=0.7,
    )
    build_result = model.build(builder="sparse", basis_solver="dfs", sort_basis=True)
    basis_configs = basis_configs_from_build_result(build_result)
    states, labels = spin_one_xy_scar_tower_states(
        basis_configs=basis_configs,
        length=model.length,
    )
    assert labels == ("S_0", "S_1", "S_2", "S_3", "S_4")

    records = _cage_records_from_state_matrix(
        hamiltonian=build_result.hamiltonian.tocsr(),
        states=states,
    )
    operators, operator_names = _spin_one_bimagnon_raising_basis(
        basis_configs=basis_configs,
        model=model,
    )

    diagnostic = sga_ladder_basis_diagnostic_from_cage_records(
        hamiltonian=build_result.hamiltonian,
        records=records,
        operators=operators,
        operator_names=operator_names,
        frequency=2.0,
        tolerance=1.0e-10,
        max_ladder_candidates=None,
    )

    assert diagnostic.has_ladder_candidates
    assert diagnostic.ladder_nullity == 1
    assert len(diagnostic.candidate_diagnostics) == 1
    candidate = diagnostic.candidate_diagnostics[0]
    assert candidate.is_sga_like
    assert candidate.n_transitions == model.length
    assert candidate.relative_leakage_norm < 1.0e-10
    assert candidate.relative_off_frequency_norm < 1.0e-10
    assert candidate.relative_algebra_residual_norm < 1.0e-10

    coefficients = diagnostic.ladder_coefficients[:, 0]
    expected = np.asarray([(-1) ** int(site_id) for site_id in model.lattice.site_ids])
    overlaps = np.asarray(coefficients / expected, dtype=np.complex128)
    np.testing.assert_allclose(
        overlaps,
        np.full_like(overlaps, overlaps[0]),
        atol=1.0e-10,
    )
