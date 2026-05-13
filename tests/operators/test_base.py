import numpy as np
import pytest

from qlinks.operators import (
    BaseLocalOperator,
    ConstantDiagonalOperator,
    OperatorAction,
    OperatorSum,
    combine_duplicate_actions,
)
from qlinks.variables import LocalSpace, VariableKind, VariableLayout, VariableSpec


def test_operator_action() -> None:
    action = OperatorAction(1.5, np.array([0, 1]))

    assert action.coefficient == 1.5 + 0j
    np.testing.assert_array_equal(action.config, np.array([0, 1]))


def test_operator_sum() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    op1 = ConstantDiagonalOperator(layout=layout, coefficient=1.0)
    op2 = ConstantDiagonalOperator(layout=layout, coefficient=2.0)

    op_sum = OperatorSum.from_terms([op1, op2])

    actions = op_sum.apply(np.array([0, 1]))

    assert len(actions) == 2
    assert actions[0].coefficient == 1.0 + 0j
    assert actions[1].coefficient == 2.0 + 0j


def test_combine_duplicate_actions() -> None:
    actions = [
        OperatorAction(1.0, np.array([0, 1])),
        OperatorAction(2.0, np.array([0, 1])),
        OperatorAction(3.0, np.array([1, 0])),
    ]

    combined = combine_duplicate_actions(actions)

    assert len(combined) == 2

    result = {tuple(action.config.tolist()): action.coefficient for action in combined}

    assert result[(0, 1)] == 3.0 + 0j
    assert result[(1, 0)] == 3.0 + 0j


def test_combine_duplicate_actions_drops_zero() -> None:
    actions = [
        OperatorAction(1.0, np.array([0, 1])),
        OperatorAction(-1.0, np.array([0, 1])),
    ]

    combined = combine_duplicate_actions(actions, atol=0.0)

    assert len(combined) == 0


# ------------------------------------------------------------------
# Tests for BaseLocalOperator helper methods that precompute variable indices
# and validate local-space values.
# ------------------------------------------------------------------


class DummyOperator(BaseLocalOperator):
    def __init__(self, layout: VariableLayout):
        self.layout = layout
        self.name = "dummy"


def _layout() -> VariableLayout:
    return VariableLayout(
        specs=(
            VariableSpec(
                kind=VariableKind.SITE,
                geometry_index=0,
                local_space=LocalSpace.binary(),
            ),
            VariableSpec(
                kind=VariableKind.LINK,
                geometry_index=0,
                local_space=LocalSpace.spin_half_flux(),
            ),
        )
    )


def test_base_operator_site_and_link_variable_helpers() -> None:
    layout = _layout()
    op = DummyOperator(layout)

    assert op._site_variable_index(0) == 0
    assert op._link_variable_index(0) == 1

    np.testing.assert_array_equal(
        op._site_variable_indices([0]),
        np.array([0], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        op._link_variable_indices([0]),
        np.array([1], dtype=np.int64),
    )


def test_base_operator_validate_local_spaces() -> None:
    layout = _layout()
    op = DummyOperator(layout)

    op._validate_local_space_values(0, {0, 1})
    op._validate_local_space_values(1, {-1, 1})

    with pytest.raises(ValueError, match="requires local-space values"):
        op._validate_local_space_values(0, {-1, 1})
