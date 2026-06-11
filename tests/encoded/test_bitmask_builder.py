import numpy as np

from qlinks.encoded import (
    BinaryEncodedBasis,
    BitmaskAction,
    BitmaskBinaryFlipOperator,
    BitmaskConstantDiagonalOperator,
    BitmaskPatternDiagonalOperator,
    BitmaskPatternFlipOperator,
    BitmaskSparseHamiltonianBuilder,
    encode_binary_config,
)
from qlinks.variables import LocalSpace, VariableLayout


class SingleActionOnlyOperator:
    layout: VariableLayout
    name = "single_action_only"

    def __init__(self, layout: VariableLayout, coefficient: complex = 1.0) -> None:
        self.layout = layout
        self.coefficient = complex(coefficient)

    def affected_variables(self) -> np.ndarray:
        return np.asarray([0], dtype=np.int64)

    def single_action_code(self, code: int) -> tuple[complex, int] | None:
        return self.coefficient, int(code) ^ 1

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        raise AssertionError("builder should use single_action_code fast path")


def test_bitmask_builder_uses_constant_diagonal_fast_path_stats() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = BinaryEncodedBasis.full(layout)

    op = BitmaskConstantDiagonalOperator(layout=layout, coefficient=3.0)
    result = BitmaskSparseHamiltonianBuilder().build_with_stats(basis, [op])

    expected = 3.0 * np.eye(4, dtype=np.complex128)

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 4
    assert result.stats.n_kept_actions == 4
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 4


def test_bitmask_builder_uses_pattern_diagonal_fast_path_stats() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = BinaryEncodedBasis.from_configs(
        layout,
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.int64,
        ),
        sort=False,
    )

    op = BitmaskPatternDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 1], dtype=np.int64),
        pattern=np.array([1, 0], dtype=np.int64),
        coefficient=2.5,
    )

    result = BitmaskSparseHamiltonianBuilder().build_with_stats(basis, [op])

    expected = np.diag([2.5, 0.0, 0.0]).astype(np.complex128)

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 1
    assert result.stats.n_kept_actions == 1
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 1


def test_bitmask_diagonal_fast_path_sums_multiple_diagonal_terms() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BinaryEncodedBasis.full(layout)

    operators = [
        BitmaskConstantDiagonalOperator(layout=layout, coefficient=1.0),
        BitmaskPatternDiagonalOperator(
            layout=layout,
            variable_indices=np.array([0], dtype=np.int64),
            pattern=np.array([1], dtype=np.int64),
            coefficient=2.0,
        ),
    ]

    result = BitmaskSparseHamiltonianBuilder().build_with_stats(basis, operators)

    expected = np.diag([1.0, 3.0]).astype(np.complex128)

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 3
    assert result.stats.n_kept_actions == 2
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 2


def test_bitmask_diagonal_fast_path_keeps_offdiagonal_actions() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BinaryEncodedBasis.full(layout)

    operators = [
        BitmaskConstantDiagonalOperator(layout=layout, coefficient=1.0),
        BitmaskBinaryFlipOperator(layout=layout, variable_index=0, coefficient=-2.0),
    ]

    result = BitmaskSparseHamiltonianBuilder().build_with_stats(basis, operators)

    expected = np.array(
        [
            [1.0, -2.0],
            [-2.0, 1.0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 4
    assert result.stats.n_kept_actions == 4
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 4


def test_bitmask_diagonal_value_code_matches_apply_code() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    op = BitmaskPatternDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 1], dtype=np.int64),
        pattern=np.array([1, 0], dtype=np.int64),
        coefficient=3.0,
    )

    matching_code = encode_binary_config(np.array([1, 0], dtype=np.int64))
    nonmatching_code = encode_binary_config(np.array([0, 1], dtype=np.int64))

    assert op.diagonal_value_code(matching_code) == 3.0 + 0.0j
    assert op.diagonal_value_code(nonmatching_code) is None
    assert op.diagonal_value_code(matching_code) == op.apply_code(matching_code)[0].coefficient


def test_bitmask_builder_uses_single_action_fast_path_without_apply_code() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    basis = BinaryEncodedBasis.full(layout)

    op = SingleActionOnlyOperator(layout=layout, coefficient=-2.0)

    result = BitmaskSparseHamiltonianBuilder().build_with_stats(basis, [op])

    expected = np.array(
        [
            [0.0, -2.0],
            [-2.0, 0.0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(result.matrix.toarray(), expected)
    assert result.stats.n_raw_actions == 2
    assert result.stats.n_kept_actions == 2
    assert result.stats.n_missing_actions == 0
    assert result.stats.nnz == 2


def test_bitmask_pattern_flip_single_action_matches_apply_code() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    op = BitmaskPatternFlipOperator(
        layout=layout,
        variable_indices=np.asarray([0, 1], dtype=np.int64),
        initial_values=np.asarray([1, 0], dtype=np.int64),
        final_values=np.asarray([0, 1], dtype=np.int64),
        coefficient=3.0,
    )

    matching_code = encode_binary_config(np.asarray([1, 0], dtype=np.int64))
    nonmatching_code = encode_binary_config(np.asarray([0, 1], dtype=np.int64))

    assert op.single_action_code(nonmatching_code) is None

    single_action = op.single_action_code(matching_code)
    assert single_action is not None

    coefficient, new_code = single_action
    apply_action = op.apply_code(matching_code)[0]

    assert coefficient == apply_action.coefficient
    assert new_code == apply_action.code
