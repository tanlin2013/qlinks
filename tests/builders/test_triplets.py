import numpy as np
import pytest

from qlinks.basis import BruteForceBasisSolver
from qlinks.builders import OptimizedSparseHamiltonianBuilder, SparseHamiltonianBuilder
from qlinks.builders.triplets import SparseTripletBuffer
from qlinks.encoded import (
    BinaryEncodedBasis,
    BitmaskBinaryFlipOperator,
    BitmaskSparseHamiltonianBuilder,
)
from qlinks.operators import BinaryFlipOperator, UpdateBinaryFlipOperator
from qlinks.variables import LocalSpace, VariableLayout


def test_sparse_triplet_buffer_appends_and_materializes_arrays() -> None:
    triplets = SparseTripletBuffer()

    triplets.append(0, 1, 2.0)
    triplets.append(1, 0, 3.0)

    rows, cols = triplets.index_arrays()
    data = triplets.data_array(np.float64)

    assert triplets.size == 2
    assert rows.dtype == np.int64
    assert cols.dtype == np.int64
    assert data.dtype == np.float64
    np.testing.assert_array_equal(rows, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(cols, np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_allclose(data, np.asarray([2.0, 3.0]))


def test_sparse_triplet_buffer_detects_inconsistent_lists() -> None:
    triplets = SparseTripletBuffer(rows=[0], cols=[], data=[])

    with pytest.raises(ValueError, match="same length"):
        triplets.validate()


def test_generic_sparse_builder_uses_triplet_buffer() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)
    operator = BinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0)

    result = SparseHamiltonianBuilder().build_with_stats(basis, [operator])

    np.testing.assert_allclose(result.matrix.toarray(), np.asarray([[0, 1], [1, 0]]))
    assert result.stats.n_kept_actions == 2
    assert result.stats.nnz == 2


def test_optimized_sparse_builder_uses_triplet_buffer() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BruteForceBasisSolver(sort=True).solve(layout)
    operator = UpdateBinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0)

    result = OptimizedSparseHamiltonianBuilder().build_with_stats(basis, [operator])

    np.testing.assert_allclose(result.matrix.toarray(), np.asarray([[0, 1], [1, 0]]))
    assert result.stats.n_kept_actions == 2
    assert result.stats.nnz == 2


def test_bitmask_sparse_builder_uses_triplet_buffer() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BinaryEncodedBasis.full(layout)
    operator = BitmaskBinaryFlipOperator(layout=layout, variable_index=0, coefficient=1.0)

    result = BitmaskSparseHamiltonianBuilder().build_with_stats(basis, [operator])

    np.testing.assert_allclose(result.matrix.toarray(), np.asarray([[0, 1], [1, 0]]))
    assert result.stats.n_kept_actions == 2
    assert result.stats.nnz == 2
