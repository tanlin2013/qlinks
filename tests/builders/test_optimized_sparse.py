import numpy as np
import pytest

from qlinks.basis import Basis, BruteForceBasisSolver, DFSBasisSolver
from qlinks.builders import (
    OptimizedSparseHamiltonianBuilder,
    SparseHamiltonianBuilder,
    build_optimized_sparse_hamiltonian,
    is_hermitian_sparse,
)
from qlinks.constraints import NearestNeighborBlockadeConstraint
from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.operators import (
    BinaryFlipOperator,
    LocalUpdateAction,
    PlaquettePatternOperator,
    PXPSpinFlipOperator,
    UpdateBinaryFlipOperator,
    UpdateOperatorSum,
    UpdatePlaquettePatternOperator,
    UpdatePXPSpinFlipOperator,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_optimized_binary_flip_matches_reference_builder() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    ref_op = BinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0)
    opt_op = UpdateBinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0)

    H_ref = SparseHamiltonianBuilder().build(basis, [ref_op])
    H_opt = OptimizedSparseHamiltonianBuilder().build(basis, [opt_op])

    np.testing.assert_allclose(H_opt.toarray(), H_ref.toarray())
    assert is_hermitian_sparse(H_opt)


def test_optimized_binary_flip_known_matrix() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op = UpdateBinaryFlipOperator(layout=layout, variable_index=0, coefficient=2.0)

    H = build_optimized_sparse_hamiltonian(basis, [op])

    expected = np.array(
        [
            [0, 2],
            [2, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(H.toarray(), expected)


def test_optimized_operator_sum() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op_sum = UpdateOperatorSum.from_terms(
        [
            UpdateBinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0),
            UpdateBinaryFlipOperator(layout=layout, variable_index=1, coefficient=1.0),
        ]
    )

    H = build_optimized_sparse_hamiltonian(basis, [op_sum])

    assert H.shape == (4, 4)
    assert H.nnz == 8
    assert is_hermitian_sparse(H)


def test_optimized_missing_action_skip() -> None:
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

    op = UpdateBinaryFlipOperator(layout=layout, variable_index=0)

    result = OptimizedSparseHamiltonianBuilder(on_missing="skip").build_with_stats(
        basis,
        [op],
    )

    assert result.matrix.shape == (1, 1)
    assert result.matrix.nnz == 0
    assert result.stats.n_missing_actions == 1
    assert result.stats.n_scratch_arrays == 1


def test_optimized_missing_action_raise() -> None:
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

    op = UpdateBinaryFlipOperator(layout=layout, variable_index=0)

    with pytest.raises(KeyError, match="outside the basis"):
        OptimizedSparseHamiltonianBuilder(on_missing="raise").build(basis, [op])


def test_optimized_pxp_matches_reference_builder() -> None:
    lattice = ChainLattice(5, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraints = NearestNeighborBlockadeConstraint.from_lattice(lattice, layout)
    basis = DFSBasisSolver(sort=True).solve(layout, constraints=constraints)

    ref_ops = [
        PXPSpinFlipOperator(
            layout=layout,
            lattice=lattice,
            site_id=i,
            coefficient=1.0,
        )
        for i in range(lattice.num_sites)
    ]

    opt_ops = [
        UpdatePXPSpinFlipOperator(
            layout=layout,
            lattice=lattice,
            site_id=i,
            coefficient=1.0,
        )
        for i in range(lattice.num_sites)
    ]

    H_ref = SparseHamiltonianBuilder().build(basis, ref_ops)
    H_opt = OptimizedSparseHamiltonianBuilder().build(basis, opt_ops)

    np.testing.assert_allclose(H_opt.toarray(), H_ref.toarray())
    assert is_hermitian_sparse(H_opt)


def test_optimized_qdm_single_plaquette_matches_reference_builder() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

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

    ref_op = PlaquettePatternOperator.qdm_flip(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=-1.0,
    )

    opt_op = UpdatePlaquettePatternOperator.qdm_flip(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=-1.0,
    )

    H_ref = SparseHamiltonianBuilder().build(basis, [ref_op])
    H_opt = OptimizedSparseHamiltonianBuilder().build(basis, [opt_op])

    expected = np.array(
        [
            [0, -1],
            [-1, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(H_ref.toarray(), expected)
    np.testing.assert_allclose(H_opt.toarray(), expected)
    assert is_hermitian_sparse(H_opt)


def test_optimized_builder_sums_duplicate_entries() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    op1 = UpdateBinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0)
    op2 = UpdateBinaryFlipOperator(layout=layout, variable_index=0, coefficient=2.0)

    H = OptimizedSparseHamiltonianBuilder().build(basis, [op1, op2])

    expected = np.array(
        [
            [0, 3],
            [3, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(H.toarray(), expected)


def test_optimized_builder_validates_updated_value() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)

    class BadUpdateOperator:
        def __init__(self, layout: VariableLayout) -> None:
            self.layout = layout
            self.name = "bad_update"

        def affected_variables(self):
            return np.array([0], dtype=np.int64)

        def apply_update(self, config):
            return (
                LocalUpdateAction(
                    coefficient=1.0,
                    variable_indices=np.array([0], dtype=np.int64),
                    new_values=np.array([2], dtype=np.int64),
                ),
            )

    with pytest.raises(ValueError, match="not allowed"):
        OptimizedSparseHamiltonianBuilder().build(basis, [BadUpdateOperator(layout)])


def test_optimized_empty_basis() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = Basis.empty(layout)

    op = UpdateBinaryFlipOperator(layout=layout, variable_index=0)

    result = OptimizedSparseHamiltonianBuilder().build_with_stats(basis, [op])

    assert result.matrix.shape == (0, 0)
    assert result.matrix.nnz == 0
    assert result.stats.n_basis == 0
    assert result.stats.n_scratch_arrays == 0
    