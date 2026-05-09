import numpy as np

from qlinks.basis import Basis
from qlinks.builders import is_hermitian_sparse
from qlinks.encoded import binary_encoded_basis_from_flux_basis
from qlinks.models import (
    HoneycombQLMModel,
    QLMModel,
    SquareQLMModel,
    TriangularQLMModel,
)


def test_square_qlm_manual_single_plaquette_sparse() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=2.0,
        charges=0,
    )

    layout = model.layout

    basis = Basis.from_states(
        layout,
        np.array(
            [
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
            ],
            dtype=np.int64,
        ),
    )

    result = model.build(
        basis=basis,
        builder="sparse",
    )

    expected = np.array(
        [
            [2, -1],
            [-1, 2],
        ],
        dtype=np.complex128,
    )

    assert result.kinetic is not None
    assert result.potential is not None
    np.testing.assert_allclose(result.hamiltonian.toarray(), expected)
    assert is_hermitian_sparse(result.hamiltonian)


def test_square_qlm_bitmask_matches_sparse() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=2.0,
        charges=0,
    )

    layout = model.layout

    flux_basis = Basis.from_states(
        layout,
        np.array(
            [
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
            ],
            dtype=np.int64,
        ),
    )

    encoded_basis = binary_encoded_basis_from_flux_basis(flux_basis, sort=False)

    sparse_result = model.build(
        basis=flux_basis,
        builder="sparse",
    )

    bitmask_result = model.build(
        basis=encoded_basis,
        builder="bitmask",
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        bitmask_result.hamiltonian.toarray(),
    )


def test_square_qlm_optimized_matches_sparse_for_kinetic_only() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=0.0,
        charges=0,
    )

    layout = model.layout

    basis = Basis.from_states(
        layout,
        np.array(
            [
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
            ],
            dtype=np.int64,
        ),
    )

    sparse_result = model.build(
        basis=basis,
        builder="sparse",
    )

    optimized_result = model.build(
        basis=basis,
        builder="optimized",
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        optimized_result.hamiltonian.toarray(),
    )


def test_square_qlm_potential_rejects_optimized() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=1.0,
        charges=0,
    )

    layout = model.layout

    basis = Basis.from_states(
        layout,
        np.array(
            [
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
            ],
            dtype=np.int64,
        ),
    )

    import pytest

    with pytest.raises(NotImplementedError, match="potential"):
        model.build(
            basis=basis,
            builder="optimized",
        )


def test_square_qlm_electric_winding_4x4_known_count() -> None:
    model = SquareQLMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states == 990


def test_square_qlm_total_4x4_known_count() -> None:
    model = SquareQLMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
    )

    # No sector restriction.
    object.__setattr__(model, "winding_x", None)
    object.__setattr__(model, "winding_y", None)

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states == 2970


def test_square_qlm_basis_smoke() -> None:
    charges = np.array([-2, 0, 0, 2], dtype=np.int64)

    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=0.0,
        charges=charges,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 1


def test_triangular_qlm_smoke() -> None:
    model = TriangularQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=0.0,
        charges=0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0
    assert len(model.plaquette_ids()) > 0


def test_honeycomb_qlm_smoke() -> None:
    model = HoneycombQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        kinetic=-1.0,
        potential=0.0,
        charges=0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0
    assert len(model.plaquette_ids()) == 1


def test_qlmmodel_factory_methods() -> None:
    tri = QLMModel.triangular(2, 2)
    honey = QLMModel.honeycomb(2, 2)

    assert isinstance(tri, TriangularQLMModel)
    assert isinstance(honey, HoneycombQLMModel)
