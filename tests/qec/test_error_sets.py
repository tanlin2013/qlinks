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


def test_local_error_set_rejects_higher_weight_generic_generation_for_now() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    try:
        LocalErrorSet.from_layout(layout, max_weight=2)
    except NotImplementedError as exc:
        assert "max_weight=1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected NotImplementedError.")
