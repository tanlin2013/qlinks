from qlinks.qec import LocalErrorSet
from qlinks.variables import LocalSpace, VariableLayout


def test_local_error_set_from_layout_generates_single_variable_errors() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    errors = LocalErrorSet.from_layout(
        layout,
        include_value_diagonal=True,
        include_projectors=True,
        include_transitions=True,
    )

    # Per variable: one value operator, two projectors, two directed transitions.
    assert len(errors) == 2 * (1 + 2 + 2)
    assert all(error.weight == 1 for error in errors)
    assert errors.by_max_weight(1).errors == errors.errors


def test_local_error_set_from_layout_supports_higher_weight_generation() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    errors = LocalErrorSet.from_layout(
        layout,
        max_weight=2,
        include_value_diagonal=True,
        include_transitions=False,
    )

    assert errors.max_weight == 2
    assert len(errors.by_exact_weight(2)) == 1
    assert errors.by_exact_weight(2)[0].name == "prod_value_0__value_1"
