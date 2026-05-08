import numpy as np

from qlinks.builders import is_hermitian_sparse
from qlinks.encoded import BinaryEncodedBasis
from qlinks.models import PXPModel


def test_pxp_chain_basis_size() -> None:
    model = PXPModel.chain(
        length=5,
        boundary_condition="open",
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    # Binary strings of length 5 with no adjacent 1s: F_7 = 13.
    assert basis.n_states == 13


def test_pxp_chain_periodic_basis_size() -> None:
    model = PXPModel.chain(
        length=5,
        boundary_condition="periodic",
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    # Circular binary strings of length 5 with no adjacent 1s: 11.
    assert basis.n_states == 11


def test_pxp_chain_sparse_hamiltonian() -> None:
    model = PXPModel.chain(
        length=4,
        boundary_condition="open",
        omega=1.0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    H = result.hamiltonian

    assert result.basis.n_states == 8
    assert H.shape == (8, 8)
    assert result.kinetic is not None
    assert result.potential is None
    assert is_hermitian_sparse(H)


def test_pxp_chain_optimized_matches_sparse() -> None:
    model = PXPModel.chain(
        length=4,
        boundary_condition="open",
        omega=1.0,
    )

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    optimized_result = model.build(
        basis=sparse_result.basis,
        builder="optimized",
        sort_basis=True,
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        optimized_result.hamiltonian.toarray(),
    )


def test_pxp_chain_bitmask_matches_sparse() -> None:
    model = PXPModel.chain(
        length=5,
        boundary_condition="open",
        omega=1.0,
    )

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
        sort_basis=True,
    )

    assert isinstance(bitmask_result.basis, BinaryEncodedBasis)

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        bitmask_result.hamiltonian.toarray(),
    )


def test_pxp_square_model_sparse() -> None:
    model = PXPModel.square(
        lx=2,
        ly=2,
        boundary_condition="open",
        omega=1.0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    H = result.hamiltonian

    assert result.basis.n_states > 0
    assert H.shape == (result.basis.n_states, result.basis.n_states)
    assert result.kinetic is not None
    assert result.potential is None
    assert is_hermitian_sparse(H)


def test_pxp_square_bitmask_matches_sparse() -> None:
    model = PXPModel.square(
        lx=2,
        ly=2,
        boundary_condition="open",
        omega=1.0,
    )

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
        sort_basis=True,
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        bitmask_result.hamiltonian.toarray(),
    )


def test_pxp_cached_lattice_and_layout() -> None:
    model = PXPModel.chain(
        length=4,
        boundary_condition="open",
    )

    assert model.lattice is model.lattice
    assert model.layout is model.layout
    assert model.make_lattice() is model.lattice
    assert model.make_layout() is model.layout
