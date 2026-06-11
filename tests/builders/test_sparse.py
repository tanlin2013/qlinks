import numpy as np
import pytest

from qlinks.basis import Basis, BruteForceBasisSolver, DFSBasisSolver
from qlinks.builders import (
    SparseHamiltonianBuilder,
    build_sparse_hamiltonian,
    is_hermitian_sparse,
)
from qlinks.constraints import NearestNeighborBlockadeConstraint
from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.operators import (
    BinaryFlipOperator,
    ConstantDiagonalOperator,
    LocalSquareValueDiagonalOperator,
    LocalValueDiagonalOperator,
    OperatorAction,
    OperatorSum,
    PatternDiagonalOperator,
    PlaquettePatternOperator,
    PXPSpinFlipOperator,
    qdm_flippability_projectors,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_constant_diagonal_operator_builds_identity_term() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op = ConstantDiagonalOperator(layout=layout, coefficient=3.0)

    H = build_sparse_hamiltonian(basis, [op])

    expected = 3.0 * np.eye(4, dtype=np.complex128)

    np.testing.assert_allclose(H.toarray(), expected)
    assert is_hermitian_sparse(H)


def test_local_value_diagonal_operator() -> None:
    layout = VariableLayout.from_links(2, LocalSpace.spin_half_flux())

    states = np.array(
        [
            [-1, -1],
            [-1, +1],
            [+1, -1],
            [+1, +1],
        ],
        dtype=np.int64,
    )
    basis = Basis.from_states(layout, states)

    op = LocalValueDiagonalOperator(
        layout=layout,
        variable_index=1,
        coefficient=2.0,
    )

    H = build_sparse_hamiltonian(basis, [op])

    expected = np.diag([-2, +2, -2, +2]).astype(np.complex128)

    np.testing.assert_allclose(H.toarray(), expected)
    assert is_hermitian_sparse(H)


def test_binary_flip_operator_matrix() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    # Explicit ordering:
    #   0: 00
    #   1: 01
    #   2: 10
    #   3: 11
    basis = Basis.from_states(
        layout,
        np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.int64,
        ),
    )

    op = BinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0)

    H = build_sparse_hamiltonian(basis, [op])

    expected = np.zeros((4, 4), dtype=np.complex128)

    # 00 -> 10
    expected[2, 0] = 1

    # 01 -> 11
    expected[3, 1] = 1

    # 10 -> 00
    expected[0, 2] = 1

    # 11 -> 01
    expected[1, 3] = 1

    np.testing.assert_allclose(H.toarray(), expected)
    assert is_hermitian_sparse(H)


def test_operator_action_outside_basis_is_skipped_by_default() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    basis = Basis.from_states(
        layout,
        np.array(
            [
                [0, 0],
            ],
            dtype=np.int64,
        ),
    )

    op = BinaryFlipOperator(layout=layout, variable_index=0)

    result = SparseHamiltonianBuilder(on_missing="skip").build_with_stats(basis, [op])

    H = result.matrix

    assert H.shape == (1, 1)
    assert H.nnz == 0
    assert result.stats.n_missing_actions == 1


def test_operator_action_outside_basis_can_raise() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    basis = Basis.from_states(
        layout,
        np.array(
            [
                [0, 0],
            ],
            dtype=np.int64,
        ),
    )

    op = BinaryFlipOperator(layout=layout, variable_index=0)

    with pytest.raises(KeyError, match="outside the basis"):
        SparseHamiltonianBuilder(on_missing="raise").build(basis, [op])


def test_duplicate_actions_are_summed() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op1 = ConstantDiagonalOperator(layout=layout, coefficient=1.0)
    op2 = ConstantDiagonalOperator(layout=layout, coefficient=2.0)

    H = build_sparse_hamiltonian(basis, [op1, op2])

    expected = 3.0 * np.eye(4, dtype=np.complex128)

    np.testing.assert_allclose(H.toarray(), expected)


def test_operator_sum_can_be_used_as_one_operator() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op_sum = OperatorSum.from_terms(
        [
            ConstantDiagonalOperator(layout=layout, coefficient=1.0),
            BinaryFlipOperator(layout=layout, variable_index=0, coefficient=2.0),
        ]
    )

    H = build_sparse_hamiltonian(basis, [op_sum])

    expected = np.array(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(H.toarray(), expected)
    assert is_hermitian_sparse(H)


def test_builder_stats() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op = BinaryFlipOperator(layout=layout, variable_index=0)

    result = SparseHamiltonianBuilder().build_with_stats(basis, [op])

    assert result.stats.n_basis == 2
    assert result.stats.n_terms == 1
    assert result.stats.n_raw_actions == 2
    assert result.stats.n_kept_actions == 2
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 2


def test_empty_basis_builds_empty_matrix() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = Basis.empty(layout)

    op = ConstantDiagonalOperator(layout=layout, coefficient=1.0)

    result = SparseHamiltonianBuilder().build_with_stats(basis, [op])

    assert result.matrix.shape == (0, 0)
    assert result.matrix.nnz == 0
    assert result.stats.n_basis == 0


def test_pxp_chain_hamiltonian_length_3() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraints = NearestNeighborBlockadeConstraint.from_lattice(lattice, layout)
    basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    operators = [
        PXPSpinFlipOperator(
            layout=layout,
            lattice=lattice,
            site_id=i,
            coefficient=1.0,
        )
        for i in range(lattice.num_sites)
    ]

    H = build_sparse_hamiltonian(basis, operators)

    assert H.shape == (basis.n_states, basis.n_states)
    assert is_hermitian_sparse(H)

    # Length-3 open PXP basis size is F_5 = 5.
    assert basis.n_states == 5

    # Every nonzero off-diagonal element should correspond to one allowed spin flip.
    dense = H.toarray()

    assert np.allclose(np.diag(dense), 0.0)
    assert np.allclose(dense, dense.conjugate().T)


def test_qdm_single_plaquette_kinetic_operator() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    # Only keep the two flippable plaquette configurations.
    basis = Basis.from_states(
        layout,
        np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            dtype=np.int64,
        ),
    )

    op = PlaquettePatternOperator.qdm_flip(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=-1.0,
    )

    H = build_sparse_hamiltonian(basis, [op])

    expected = np.array(
        [
            [0, -1],
            [-1, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(H.toarray(), expected)
    assert is_hermitian_sparse(H)


def test_qdm_single_plaquette_potential_operator() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    basis = Basis.from_states(
        layout,
        np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ],
            dtype=np.int64,
        ),
    )

    projectors = qdm_flippability_projectors(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=2.5,
    )

    H = build_sparse_hamiltonian(basis, list(projectors))

    expected = np.diag([2.5, 2.5, 0.0]).astype(np.complex128)

    np.testing.assert_allclose(H.toarray(), expected)
    assert is_hermitian_sparse(H)


def test_non_hermitian_detection() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())

    basis = Basis.from_states(
        layout,
        np.array(
            [
                [0],
                [1],
            ],
            dtype=np.int64,
        ),
    )

    class RaisingOnly:
        def __init__(self, layout: VariableLayout) -> None:
            self.layout = layout
            self.name = "raising_only"

        def affected_variables(self):
            return np.array([0], dtype=np.int64)

        def apply(self, config):
            arr = np.asarray(config, dtype=np.int64)

            if arr[0] == 0:
                return (
                    OperatorAction(
                        coefficient=1.0,
                        config=np.array([1], dtype=np.int64),
                    ),
                )

            return ()

    H = build_sparse_hamiltonian(basis, [RaisingOnly(layout)])

    assert not is_hermitian_sparse(H)


def test_sparse_builder_backend_scipy_explicit() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op = BinaryFlipOperator(layout=layout, variable_index=0, coefficient=2.0)

    H = build_sparse_hamiltonian(
        basis,
        [op],
        backend="scipy",
    )

    expected = np.array(
        [
            [0, 2],
            [2, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(H.toarray(), expected)


def test_sparse_builder_backend_cupy() -> None:
    cp = pytest.importorskip("cupy")

    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op = BinaryFlipOperator(layout=layout, variable_index=0, coefficient=2.0)

    H = build_sparse_hamiltonian(
        basis,
        [op],
        backend="cupy",
        dtype=cp.complex128,
    )

    expected = np.array(
        [
            [0, 2],
            [2, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(cp.asnumpy(H.toarray()), expected)
    assert is_hermitian_sparse(H, backend="cupy")


def test_sparse_builder_uses_constant_diagonal_fast_path_stats() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op = ConstantDiagonalOperator(layout=layout, coefficient=3.0)
    result = SparseHamiltonianBuilder().build_with_stats(basis, [op])

    expected = 3.0 * np.eye(4, dtype=np.complex128)

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 4
    assert result.stats.n_kept_actions == 4
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 4


def test_sparse_builder_uses_local_square_diagonal_fast_path_stats() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.spin_one())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op = LocalSquareValueDiagonalOperator(
        layout=layout,
        variable_index=0,
        coefficient=2.0,
    )

    result = SparseHamiltonianBuilder().build_with_stats(basis, [op])

    expected_diag = [
        2.0 * int(config[0]) * int(config[0]) for config in basis.iter_states(copy=False)
    ]
    expected = np.diag(expected_diag).astype(np.complex128)

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == basis.n_states
    assert result.stats.n_kept_actions == 6
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 6


def test_sparse_builder_uses_pattern_diagonal_fast_path_stats() -> None:
    layout = VariableLayout.from_links(2, LocalSpace.binary())
    basis = Basis.from_states(
        layout,
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.int64,
        ),
    )

    op = PatternDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 1], dtype=np.int64),
        pattern=np.array([1, 0], dtype=np.int64),
        coefficient=2.5,
    )

    result = SparseHamiltonianBuilder().build_with_stats(basis, [op])

    expected = np.diag([2.5, 0.0, 0.0]).astype(np.complex128)

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 1
    assert result.stats.n_kept_actions == 1
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 1


def test_diagonal_fast_path_sums_multiple_diagonal_terms() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    operators = [
        ConstantDiagonalOperator(layout=layout, coefficient=1.0),
        LocalValueDiagonalOperator(
            layout=layout,
            variable_index=0,
            coefficient=2.0,
        ),
    ]

    result = SparseHamiltonianBuilder().build_with_stats(basis, operators)

    expected = np.diag([1.0, 3.0]).astype(np.complex128)

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 4
    assert result.stats.n_kept_actions == 2
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 2
