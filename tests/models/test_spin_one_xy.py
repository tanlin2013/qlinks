import numpy as np
import pytest

from qlinks.models import SpinOneXYChainModel


def test_spin_one_xy_chain_basis_count_open() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert model.lattice.num_sites == 3
    assert basis.n_states == 3**3


def test_spin_one_xy_chain_basis_count_periodic() -> None:
    model = SpinOneXYChainModel(
        length=4,
        boundary_condition="periodic",
        j_xy=1.0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert model.lattice.num_sites == 4
    assert basis.n_states == 3**4


def test_spin_one_xy_chain_build_smoke() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(
        builder="sparse",
        basis_solver="dfs",
        sort_basis=True,
    )

    assert result.hamiltonian.shape == (3**3, 3**3)
    assert result.kinetic is not None
    assert result.potential is None
    assert result.kinetic.nnz > 0
    assert result.hamiltonian.nnz > 0


def test_spin_one_xy_chain_hamiltonian_is_hermitian() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    H = model.build_hamiltonian(builder="sparse")

    diff = H - H.conjugate().T

    assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-12


def test_spin_one_xy_chain_rejects_bitmask_builder_for_non_binary_basis() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    with pytest.raises(ValueError, match="BinaryEncodedBasis"):
        model.build(builder="bitmask")


def test_spin_one_xy_chain_rejects_optimized_builder() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    with pytest.raises(NotImplementedError, match="sparse"):
        model.build(builder="optimized")


def test_spin_one_xy_chain_two_site_matrix_elements() -> None:
    model = SpinOneXYChainModel(
        length=2,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(builder="sparse")
    basis = result.basis
    H = result.hamiltonian.tocsr()

    def idx(config: list[int]) -> int:
        key = np.asarray(config, dtype=np.int64).tobytes()
        return basis.index[key]

    # |0,0> connects to |1,-1> and |-1,1> with matrix element J_xy = 1.
    i = idx([0, 0])
    j1 = idx([1, -1])
    j2 = idx([-1, 1])

    assert np.isclose(H[j1, i], 1.0)
    assert np.isclose(H[j2, i], 1.0)

    # |-1,1> connects to |0,0> with matrix element 1.
    assert np.isclose(H[i, j2], 1.0)


def test_spin_one_xy_potential_diagonal_matches_h_and_d_terms():
    model = SpinOneXYChainModel(
        length=2,
        boundary_condition="open",
        j_xy=0.0,
        h_z=0.3,
        d_z=2.0,
    )

    result = model.build(
        builder="sparse",
        basis_solver="dfs",
        sort_basis=True,
        on_missing="raise",
    )

    assert result.potential is not None

    basis = result.basis.states
    diagonal = result.potential.diagonal()

    expected = np.array(
        [0.3 * np.sum(config) + 2.0 * np.sum(config * config) for config in basis],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(diagonal, expected)


def test_spin_one_xy_zero_h_and_d_has_no_potential_term_by_default():
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
        h_z=0.0,
        d_z=0.0,
    )

    result = model.build(
        builder="sparse",
        basis_solver="dfs",
        sort_basis=True,
        on_missing="raise",
    )

    assert result.kinetic is not None
    assert result.potential is None
