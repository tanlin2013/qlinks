from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from qlinks.caging import (
    local_term_operator_basis,
    scan_sga_ladder_basis_frequencies,
    sga_ladder_basis_diagnostic,
    sga_operator_diagnostic,
)
from qlinks.models import LocalTermDescriptor


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
