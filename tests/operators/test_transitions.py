import numpy as np
import pytest

from qlinks.operators import (
    BinaryFlipOperator,
    MultiNegationFlipOperator,
    NegationFlipOperator,
    SetVariablesOperator,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_set_variables_operator_matches() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = SetVariablesOperator(
        layout=layout,
        variable_indices=np.array([1, 3]),
        initial_values=np.array([0, 1]),
        final_values=np.array([1, 0]),
        coefficient=-2.0,
    )

    actions = op.apply(np.array([1, 0, 0, 1]))

    assert len(actions) == 1
    assert actions[0].coefficient == -2.0 + 0j
    np.testing.assert_array_equal(actions[0].config, np.array([1, 1, 0, 0]))


def test_set_variables_operator_no_match() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = SetVariablesOperator(
        layout=layout,
        variable_indices=np.array([1, 3]),
        initial_values=np.array([0, 1]),
        final_values=np.array([1, 0]),
    )

    actions = op.apply(np.array([1, 1, 0, 1]))

    assert actions == ()


def test_set_variables_rejects_bad_final_value() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="not allowed"):
        SetVariablesOperator(
            layout=layout,
            variable_indices=np.array([0]),
            initial_values=np.array([0]),
            final_values=np.array([2]),
        )


def test_binary_flip_operator() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    op = BinaryFlipOperator(layout=layout, variable_index=1, coefficient=3.0)

    actions = op.apply(np.array([0, 1, 0]))

    assert len(actions) == 1
    assert actions[0].coefficient == 3.0 + 0j
    np.testing.assert_array_equal(actions[0].config, np.array([0, 0, 0]))


def test_binary_flip_on_site_constructor() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    op = BinaryFlipOperator.on_site(layout, site_id=1)

    actions = op.apply(np.array([0, 0]))

    np.testing.assert_array_equal(actions[0].config, np.array([0, 1]))


def test_binary_flip_rejects_flux_space() -> None:
    layout = VariableLayout.from_links(1, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match="requires local values"):
        BinaryFlipOperator(layout=layout, variable_index=0)


def test_negation_flip_operator() -> None:
    layout = VariableLayout.from_links(2, LocalSpace.spin_half_flux())
    op = NegationFlipOperator(layout=layout, variable_index=0)

    actions = op.apply(np.array([-1, 1]))

    np.testing.assert_array_equal(actions[0].config, np.array([1, 1]))


def test_multi_negation_flip_operator() -> None:
    layout = VariableLayout.from_links(4, LocalSpace.spin_half_flux())

    op = MultiNegationFlipOperator(
        layout=layout,
        variable_indices=np.array([0, 2, 3]),
        coefficient=-1.0,
    )

    actions = op.apply(np.array([-1, 1, 1, -1]))

    assert actions[0].coefficient == -1.0 + 0j
    np.testing.assert_array_equal(actions[0].config, np.array([1, 1, -1, 1]))
