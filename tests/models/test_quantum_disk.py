from __future__ import annotations

import numpy as np

from qlinks.constraints import SquareDiskDiagonalLineSumSector
from qlinks.models import SquareQuantumDiskModel


def test_square_quantum_disk_basis_respects_nearest_neighbor_blockade() -> None:
    model = SquareQuantumDiskModel(lx=3, ly=3, boundary_condition="open")
    basis = model.build_basis()
    constraints = model.make_constraints()

    assert basis.n_states > 0
    for config in basis.states:
        assert all(constraint.is_satisfied(config) for constraint in constraints)


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
    hamiltonian = result.hamiltonian

    diff = hamiltonian - hamiltonian.T.conj()
    assert diff.nnz == 0 or np.allclose(diff.data, 0.0)


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
    sector = SquareDiskDiagonalLineSumSector(
        layout=model.layout,
        lattice=model.lattice,
        family="x_plus_y",
        target=(0, 1, 0, 0, 0),
    )

    assert basis.n_states == 2
    assert all(sector.is_satisfied(config) for config in basis.states)
    assert result.hamiltonian.nnz == 2


def test_square_quantum_disk_rejects_incompatible_diagonal_sector() -> None:
    model = SquareQuantumDiskModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        hop_families=("x_plus_y", "x_minus_y"),
        x_plus_y_sums=(0, 1, 0, 0, 0),
    )

    try:
        model.build_basis()
    except ValueError as exc:
        assert "x_plus_y_sums" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected incompatible diagonal sector to be rejected.")


def test_square_quantum_disk_visualization_data_hook() -> None:
    model = SquareQuantumDiskModel(lx=2, ly=2, boundary_condition="open")
    config = np.asarray([1, 0, 0, 1], dtype=np.int64)

    data = model.basis_visualization_data(config)

    assert data["kind"] == "quantum_disk"
    assert data["site_occupations"] == {0: 1, 1: 0, 2: 0, 3: 1}
    assert data["site_coordinates"] == {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
