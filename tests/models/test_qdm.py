from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from qlinks.builders import is_hermitian_sparse
from qlinks.models import (
    HoneycombQDMModel,
    QDMModel,
    SquareQDMModel,
    TriangularQDMModel,
)
from tests.helpers.assertions import (
    assert_hermitian_sparse,
    assert_same_sparse_matrix,
    assert_sparse_allclose,
)


def test_square_qdm_single_plaquette_sparse(square_qdm_2x2_open) -> None:
    result = square_qdm_2x2_open.build(
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


def test_square_qdm_bitmask_matches_sparse(square_qdm_2x2_open) -> None:
    sparse_result = square_qdm_2x2_open.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    bitmask_result = square_qdm_2x2_open.build(
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
        coup_kin=-1.0,
        coup_pot=0.0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0
    assert len(model.plaquette_ids()) > 0


def test_honeycomb_qdm_single_hexagon_sparse(honeycomb_qdm_2x2_open) -> None:
    result = honeycomb_qdm_2x2_open.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    assert result.hamiltonian.shape == (
        result.basis.n_states,
        result.basis.n_states,
    )
    assert is_hermitian_sparse(result.hamiltonian)


def test_honeycomb_qdm_bitmask_matches_sparse(honeycomb_qdm_2x2_open) -> None:
    sparse_result = honeycomb_qdm_2x2_open.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    bitmask_result = honeycomb_qdm_2x2_open.build(
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


def test_honeycomb_qdm_winding_sector_builds() -> None:
    model = HoneycombQDMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        coup_kin=-1.0,
        coup_pot=0.0,
        required_count=1,
        winding_x=1,
        winding_y=1,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0

    sectors = model.make_sectors(model.layout)
    for state in basis.states:
        assert all(sector.is_satisfied(state) for sector in sectors)


@pytest.mark.parametrize(
    ("kinetic", "potential"),
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ],
)
def test_square_qdm_2x2_sparse_and_bitmask_match_in_electric_winding_sector(
    kinetic: float,
    potential: float,
) -> None:
    model = SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=kinetic,
        coup_pot=potential,
    )

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
        backend="scipy",
        sort_basis=False,
        on_missing="raise",
    )

    assert_same_sparse_matrix(bitmask_result.hamiltonian, sparse_result.hamiltonian)


@pytest.mark.parametrize("builder", ["sparse", "bitmask"])
def test_square_qdm_peierls_coup_kin_is_hermitian(builder: str) -> None:
    reference_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
    )

    coup_kin = {
        int(p): np.exp(0.2j * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=coup_kin,
        coup_pot=0.0,
    )

    result = model.build(builder=builder)

    assert_hermitian_sparse(result.kinetic)
    assert_hermitian_sparse(result.hamiltonian)


def test_square_qdm_sparse_and_bitmask_match_with_peierls_phases() -> None:
    reference_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
    )

    coup_kin = {
        int(p): np.exp(0.2j * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    sparse_result = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=coup_kin,
        coup_pot=0.0,
    ).build(builder="sparse")

    bitmask_result = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=coup_kin,
        coup_pot=0.0,
    ).build(builder="bitmask", sort_basis=False)

    assert_sparse_allclose(bitmask_result.kinetic, sparse_result.kinetic)


def test_square_qdm_nonempty_sector_labels_rectangular_pbc() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=2,
        boundary_condition="periodic",
        winding_convention="electric",
    )

    labels = model.nonempty_sector_labels(solver="dfs")

    assert set(labels) == {"winding"}
    assert all(len(pair) == 2 for pair in labels["winding"])

    allowed = model.allowed_sector_labels()
    allowed_pairs = set(product(allowed["winding_x"], allowed["winding_y"]))

    assert set(labels["winding"]) <= allowed_pairs
    assert len(labels["winding"]) > 0


def test_square_qdm_nonempty_sector_labels_are_buildable() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=2,
        boundary_condition="periodic",
        winding_convention="electric",
    )

    for winding_x, winding_y in model.nonempty_sector_labels()["winding"]:
        sector_model = replace(
            model,
            winding_x=winding_x,
            winding_y=winding_y,
        )
        assert sector_model.has_basis_state()


def test_model_build_basis_respects_max_states() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=2,
        boundary_condition="periodic",
        winding_convention="electric",
        winding_x=0,
        winding_y=0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=False,
        max_states=1,
    )

    assert basis.n_states <= 1


def test_model_has_basis_state_returns_bool() -> None:
    model = SquareQDMModel(
        lx=4,
        ly=2,
        boundary_condition="periodic",
        winding_convention="electric",
        winding_x=0,
        winding_y=0,
    )

    assert isinstance(model.has_basis_state(solver="dfs"), bool)
