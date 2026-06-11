import numpy as np
import pytest

from qlinks.operators import (
    ConstantDiagonalOperator,
    LocalSquareValueDiagonalOperator,
    LocalSumDiagonalOperator,
    LocalValueDiagonalOperator,
    PatternDiagonalOperator,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_constant_diagonal_operator() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    op = ConstantDiagonalOperator(layout=layout, coefficient=3.0)

    actions = op.apply(np.array([0, 1]))

    assert len(actions) == 1
    assert actions[0].coefficient == 3.0 + 0j
    np.testing.assert_array_equal(actions[0].config, np.array([0, 1]))


def test_local_value_diagonal_operator() -> None:
    layout = VariableLayout.from_links(2, LocalSpace.spin_half_flux())
    op = LocalValueDiagonalOperator(layout=layout, variable_index=1, coefficient=2.0)

    actions = op.apply(np.array([-1, 1]))

    assert len(actions) == 1
    assert actions[0].coefficient == 2.0 + 0j


def test_local_sum_diagonal_operator() -> None:
    layout = VariableLayout.from_links(3, LocalSpace.spin_half_flux())

    op = LocalSumDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 1, 2]),
        weights=np.array([1, -1, 1]),
        coefficient=2.0,
    )

    actions = op.apply(np.array([-1, 1, 1]))

    # 2 * (-1 - 1 + 1) = -2
    assert actions[0].coefficient == -2.0 + 0j


def test_local_square_value_diagonal_value_matches_apply() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.spin_one())
    op = LocalSquareValueDiagonalOperator(
        layout=layout,
        variable_index=0,
        coefficient=3.0,
    )

    config = np.array([-1], dtype=np.int64)

    actions = op.apply(config)

    assert op.diagonal_value(config) == actions[0].coefficient
    np.testing.assert_array_equal(actions[0].config, config)


def test_pattern_diagonal_operator_matches() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = PatternDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 2]),
        pattern=np.array([1, 0]),
        coefficient=5.0,
    )

    actions = op.apply(np.array([1, 1, 0, 1]))

    assert len(actions) == 1
    assert actions[0].coefficient == 5.0 + 0j


def test_pattern_diagonal_operator_no_match() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = PatternDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 2]),
        pattern=np.array([1, 0]),
    )

    actions = op.apply(np.array([0, 1, 0, 1]))

    assert actions == ()


def test_pattern_diagonal_rejects_bad_value() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="not allowed"):
        PatternDiagonalOperator(
            layout=layout,
            variable_indices=np.array([0]),
            pattern=np.array([2]),
        )
