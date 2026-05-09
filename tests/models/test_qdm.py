import numpy as np

from qlinks.builders import is_hermitian_sparse
from qlinks.models import (
    HoneycombQDMModel,
    QDMModel,
    SquareQDMModel,
    TriangularQDMModel,
)


def test_square_qdm_single_plaquette_sparse() -> None:
    model = SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=2.0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    expected = np.array(
        [
            [2, -1],
            [-1, 2],
        ],
        dtype=np.complex128,
    )

    assert result.basis.n_states == 2
    assert result.kinetic is not None
    assert result.potential is not None
    np.testing.assert_allclose(result.hamiltonian.toarray(), expected)


def test_square_qdm_bitmask_matches_sparse() -> None:
    model = SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=2.0,
    )

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        bitmask_result.hamiltonian.toarray(),
    )


def test_square_qdm_electric_winding_4x4_known_count() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        required_count=1,
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states == 132


def test_square_qdm_total_4x4_known_count() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        required_count=1,
    )

    # No sector restriction.
    object.__setattr__(model, "winding_x", None)
    object.__setattr__(model, "winding_y", None)

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states == 272


def test_triangular_qdm_smoke() -> None:
    model = TriangularQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=0.0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0
    assert len(model.plaquette_ids()) > 0


def test_honeycomb_qdm_single_hexagon_sparse() -> None:
    model = HoneycombQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=1.0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    assert result.hamiltonian.shape == (
        result.basis.n_states,
        result.basis.n_states,
    )
    assert is_hermitian_sparse(result.hamiltonian)


def test_honeycomb_qdm_bitmask_matches_sparse() -> None:
    model = HoneycombQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=1.0,
    )

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        bitmask_result.hamiltonian.toarray(),
    )


def test_qdmmodel_factory_methods() -> None:
    tri = QDMModel.triangular(2, 2)
    honey = QDMModel.honeycomb(2, 2)

    assert isinstance(tri, TriangularQDMModel)
    assert isinstance(honey, HoneycombQDMModel)


def test_triangular_qdm_uses_only_rhombi():
    model = TriangularQDMModel(...)
    for pid in model.plaquette_ids():
        assert len(model.lattice.plaquette_links(pid)) == 4
        assert model.lattice.plaquettes[pid].kind.startswith("rhombus_")


def test_honeycomb_qdm_uses_only_hexagons():
    model = HoneycombQDMModel(...)
    for pid in model.plaquette_ids():
        assert len(model.lattice.plaquette_links(pid)) == 6
        assert model.lattice.plaquettes[pid].kind == "hexagon"

def test_triangular_qdm_z2_sector_builds() -> None:
    model = TriangularQDMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=1,
    )

    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states >= 0
