import numpy as np
import pytest

from qlinks.models import SpinOneXYChainModel
from tests.helpers.assertions import assert_sparse_allclose


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


def test_spin_one_xy_chain_optimized_builder_matches_sparse() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    sparse_result = model.build(builder="sparse")
    optimized_result = model.build(builder="optimized")

    np.testing.assert_array_equal(
        optimized_result.basis.states,
        sparse_result.basis.states,
    )
    assert_sparse_allclose(optimized_result.hamiltonian, sparse_result.hamiltonian)
    assert_sparse_allclose(optimized_result.kinetic, sparse_result.kinetic)
    assert optimized_result.potential is None


def test_spin_one_xy_chain_optimized_builder_matches_sparse_with_potential() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
        h_z=0.3,
        d_z=0.7,
    )

    sparse_result = model.build(builder="sparse")
    optimized_result = model.build(builder="optimized")

    np.testing.assert_array_equal(
        optimized_result.basis.states,
        sparse_result.basis.states,
    )
    assert optimized_result.potential is not None
    assert sparse_result.potential is not None
    assert_sparse_allclose(optimized_result.hamiltonian, sparse_result.hamiltonian)
    assert_sparse_allclose(optimized_result.kinetic, sparse_result.kinetic)
    assert_sparse_allclose(optimized_result.potential, sparse_result.potential)


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
        builder="optimized",
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


def test_spin_one_xy_local_term_descriptors_and_build_local_terms() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
        h_z=0.3,
        d_z=0.7,
    )
    result = model.build(builder="sparse", basis_solver="dfs", sort_basis=True)

    kinetic_descriptors = model.local_term_descriptors(operator_kind="kinetic", term_kind="bond")
    potential_descriptors = model.local_term_descriptors(
        operator_kind="potential",
        term_kind="site",
    )

    assert len(kinetic_descriptors) == 2
    assert len(potential_descriptors) == 6
    assert kinetic_descriptors[0].support_sites == (0, 1)
    assert kinetic_descriptors[0].support_variables == (0, 1)
    assert potential_descriptors[0].support_variables == (0,)

    local_kinetic = sum(
        (
            model.build_local_term(descriptor, result, builder="sparse")
            for descriptor in kinetic_descriptors
        ),
        0 * result.hamiltonian,
    )
    local_potential = sum(
        (
            model.build_local_term(descriptor, result, builder="sparse")
            for descriptor in potential_descriptors
        ),
        0 * result.hamiltonian,
    )

    assert result.kinetic is not None
    assert result.potential is not None
    assert_sparse_allclose(local_kinetic, result.kinetic)
    assert_sparse_allclose(local_potential, result.potential)


def test_spin_one_xy_total_sz_sector_filters_basis() -> None:
    model = SpinOneXYChainModel(
        length=4,
        boundary_condition="periodic",
        j_xy=1.0,
        total_sz=0,
    )

    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states == 19
    assert all(int(config.sum()) == 0 for config in basis.states)


def test_spin_one_xy_total_sz_sector_extreme_state() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        total_sz=-3,
    )

    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states == 1
    np.testing.assert_array_equal(basis.states[0], np.asarray([-1, -1, -1]))
