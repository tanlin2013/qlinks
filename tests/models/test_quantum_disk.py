from __future__ import annotations

import numpy as np
import pytest

from qlinks.builders import is_hermitian_sparse
from qlinks.models import QuantumDiskModel, SquareQuantumDiskModel


def test_square_quantum_disk_alias_points_to_square_model() -> None:
    assert QuantumDiskModel is SquareQuantumDiskModel


def test_square_quantum_disk_cached_lattice_and_layout() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")

    assert model.lattice is model.lattice
    assert model.layout is model.layout
    assert model.make_lattice() is model.lattice
    assert model.make_layout() is model.layout
    assert model.disk_site_ids() == (0, 1, 2, 3)
    assert model.disk_coordinates() == {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def test_square_quantum_disk_basis_size_with_nearest_neighbor_blockade() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")

    basis = model.build_basis(sort=True)

    # Independent sets of the 2x2 square graph: empty, four singles, two diagonals.
    assert basis.n_states == 7
    constraints = model.make_constraints()
    assert len(constraints) == model.lattice.num_links
    assert all(
        all(constraint.is_satisfied(config) for constraint in constraints)
        for config in basis.states
    )


def test_square_quantum_disk_disk_number_sector_without_blockade() -> None:
    model = SquareQuantumDiskModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        hard_core_nearest_neighbor=False,
        disk_number=2,
    )

    basis = model.build_basis(sort=True)

    assert basis.n_states == 6
    assert {int(np.sum(config)) for config in basis.states} == {2}


def test_square_quantum_disk_hamiltonian_is_hermitian() -> None:
    model = SquareQuantumDiskModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        hop_families=("x_plus_y",),
        coup_kin=-1.0,
        chemical_potential=0.25,
    )

    result = model.build(on_missing="raise")

    assert is_hermitian_sparse(result.hamiltonian)
    assert result.kinetic is not None
    assert result.potential is not None


def test_square_quantum_disk_diagonal_sector_is_preserved_by_matching_hops() -> None:
    # One disk on the x+y=1 anti-diagonal.  The x_plus_y hopping family moves
    # disks along this anti-diagonal and should therefore stay in this sector.
    model = SquareQuantumDiskModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        hop_families=("x_plus_y",),
        x_plus_y_sums=(0, 1, 0, 0, 0),
    )

    basis = model.build_basis(sort=True)
    result = model.build(basis=basis, on_missing="raise")

    assert basis.n_states == 2
    assert {tuple(config) for config in basis.states} == {
        (0, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
    }
    assert result.hamiltonian.nnz == 2


def test_square_quantum_disk_x_minus_y_sector_is_preserved_by_matching_hops() -> None:
    model = SquareQuantumDiskModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        hop_families=("x_minus_y",),
        x_minus_y_sums=(0, 0, 1, 0, 0),
    )

    basis = model.build_basis(sort=True)
    result = model.build(basis=basis, on_missing="raise")

    assert basis.n_states == 3
    assert result.hamiltonian.nnz == 4


def test_square_quantum_disk_rejects_incompatible_diagonal_sector() -> None:
    model = SquareQuantumDiskModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        hop_families=("x_plus_y", "x_minus_y"),
        x_plus_y_sums=(0, 1, 0, 0, 0),
    )

    with pytest.raises(ValueError, match="x_plus_y_sums"):
        model.build_basis()


def test_square_quantum_disk_rejects_invalid_hop_families() -> None:
    with pytest.raises(ValueError, match="hop_families must contain"):
        SquareQuantumDiskModel(hop_families=())

    with pytest.raises(ValueError, match="hop_families entries"):
        SquareQuantumDiskModel(hop_families=("bad",))  # type: ignore[arg-type]


def test_square_quantum_disk_sparse_only_for_now() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")

    with pytest.raises(NotImplementedError, match="builder='sparse'"):
        model.make_kinetic_operators(builder="optimized")

    with pytest.raises(NotImplementedError, match="builder='sparse'"):
        model.make_potential_operators(builder="optimized")


def test_square_quantum_disk_potential_projectors_are_diagonal() -> None:
    model = SquareQuantumDiskModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        hop_families=("x_plus_y",),
        hard_core_nearest_neighbor=False,
        coup_kin=0.0,
        coup_pot=2.0,
        disk_number=1,
    )

    result = model.build(on_missing="raise", sort_basis=True)
    hamiltonian = result.hamiltonian.toarray()

    np.testing.assert_allclose(hamiltonian, np.diag(np.diag(hamiltonian)))
    # The two mobile one-disk states on the only x+y diagonal bond each receive
    # one directed-projector contribution.
    assert sorted(np.diag(hamiltonian).real.tolist()) == [0.0, 0.0, 2.0, 2.0]


def test_square_quantum_disk_local_term_descriptors() -> None:
    model = SquareQuantumDiskModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        hop_families=("x_plus_y",),
        coup_pot=0.5,
    )

    all_descriptors = model.local_term_descriptors()
    kinetic_descriptors = model.local_term_descriptors(operator_kind="kinetic")
    potential_descriptors = model.local_term_descriptors(operator_kind="potential")
    site_descriptors = model.local_term_descriptors(term_kind="site")

    assert len(all_descriptors) == 2
    assert len(kinetic_descriptors) == 1
    assert len(potential_descriptors) == 1
    assert site_descriptors == ()
    assert all_descriptors[0].support_sites == (1, 2)
    assert all_descriptors[0].support_links == ()
    assert all_descriptors[0].label == "T_x_plus_y_1_2"
    assert all_descriptors[1].label == "P_x_plus_y_1_2"


def test_square_quantum_disk_sector_label_helpers() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")

    allowed = model.allowed_sector_labels()
    nonempty = model.nonempty_sector_labels()

    assert allowed["disk_number"] == (0, 1, 2, 3, 4)
    assert allowed["x_plus_y_lines"] == (0, 1, 2)
    assert allowed["x_minus_y_lines"] == (-1, 0, 1)
    assert nonempty["disk_number"] == (0, 1, 2)


def test_square_quantum_disk_visualization_data_hook() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")
    config = np.asarray([1, 0, 0, 1], dtype=np.int64)

    data = model.basis_visualization_data(config)

    assert data["kind"] == "quantum_disk"
    assert data["site_occupations"] == {0: 1, 1: 0, 2: 0, 3: 1}
    assert data["site_coordinates"] == {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    assert data["boundary_condition"] == "open"


def test_square_quantum_disk_build_local_bond_terms() -> None:
    from tests.helpers.assertions import assert_sparse_allclose

    model = SquareQuantumDiskModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        hop_families=("x_plus_y",),
        hard_core_nearest_neighbor=False,
        coup_kin=-1.0,
        coup_pot=0.5,
        chemical_potential=0.0,
    )
    result = model.build(on_missing="raise", sort_basis=True)

    kinetic_descriptors = model.local_term_descriptors(operator_kind="kinetic", term_kind="bond")
    potential_descriptors = model.local_term_descriptors(
        operator_kind="potential",
        term_kind="bond",
    )

    assert len(kinetic_descriptors) == 1
    assert len(potential_descriptors) == 1
    assert kinetic_descriptors[0].support_sites == (1, 2)
    assert kinetic_descriptors[0].support_variables == (1, 2)

    local_kinetic = model.build_local_term(kinetic_descriptors[0], result, builder="sparse")
    local_potential = model.build_local_term(potential_descriptors[0], result, builder="sparse")

    assert result.kinetic is not None
    assert result.potential is not None
    assert_sparse_allclose(local_kinetic, result.kinetic)
    assert_sparse_allclose(local_potential, result.potential)
