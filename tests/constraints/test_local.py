import numpy as np
import pytest

from qlinks.constraints import (
    BoundedLocalCountConstraint,
    ConstraintCollection,
    FixedValueConstraint,
    LocalSumConstraint,
    TotalValueSector,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_constraint_collection_satisfied() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    collection = ConstraintCollection.from_sequences(
        constraints=[
            FixedValueConstraint.single(layout, variable_index=0, value=1),
        ],
        sectors=[
            TotalValueSector(layout=layout, variable_indices=np.array([0, 1, 2]), target=2),
        ],
    )

    config = np.array([1, 1, 0])

    assert collection.is_satisfied(config)
    assert collection.first_failure(config) is None


def test_constraint_collection_first_failure() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    collection = ConstraintCollection.from_sequences(
        constraints=[
            FixedValueConstraint.single(layout, variable_index=0, value=1),
        ],
        sectors=[
            TotalValueSector(layout=layout, target=2),
        ],
    )

    config = np.array([0, 1, 0])

    assert not collection.is_satisfied(config)

    failure = collection.first_failure(config)
    assert failure is not None
    assert failure.name == "fixed_value"


def test_bounded_local_count_exact_check_and_propagation() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    constraint = BoundedLocalCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([2, 0, 2, 1]),
        count=2,
    )

    np.testing.assert_array_equal(constraint.affected_variables(), np.array([2, 0, 1]))
    assert constraint.value(np.array([1, 1, 0])) == 2
    assert constraint.check(np.array([1, 1, 0])).satisfied
    assert not constraint.check(np.array([1, 0, 0])).satisfied

    propagation = constraint.propagate(
        np.array([1, 0, 0]),
        np.array([True, True, False]),
    )

    assert propagation.consistent
    assert propagation.forced_assignments == ((2, 1),)


def test_bounded_local_count_at_most_forces_zeros() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    constraint = BoundedLocalCountConstraint.at_most(
        layout=layout,
        variable_indices=np.array([0, 1, 2]),
        max_count=1,
    )

    propagation = constraint.propagate(
        np.array([1, 0, 0]),
        np.array([True, False, False]),
    )

    assert propagation.consistent
    assert propagation.forced_assignments == ((1, 0), (2, 0))
    assert constraint.partial_check(
        np.array([1, 0, 0]),
        np.array([True, False, False]),
    )


def test_bounded_local_count_detects_contradictions() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    constraint = BoundedLocalCountConstraint.exact(
        layout=layout,
        variable_indices=np.array([0, 1, 2]),
        count=2,
    )

    assert not constraint.propagate(
        np.array([1, 1, 1]),
        np.array([True, True, True]),
    ).consistent
    assert not constraint.propagate(
        np.array([1, 0, 0]),
        np.array([True, True, True]),
    ).consistent


def test_local_constraints_reject_invalid_construction() -> None:
    binary_layout = VariableLayout.from_sites(2, LocalSpace.binary())
    flux_layout = VariableLayout.from_links(2, LocalSpace.spin_half_flux())

    invalid_args = [
        {"variable_indices": np.array([[0]]), "min_count": 0, "max_count": 1},
        {"variable_indices": np.array([], dtype=np.int64), "min_count": 0, "max_count": 1},
        {"variable_indices": np.array([3]), "min_count": 0, "max_count": 1},
        {"variable_indices": np.array([0]), "min_count": -1, "max_count": 1},
        {"variable_indices": np.array([0]), "min_count": 0, "max_count": -1},
        {"variable_indices": np.array([0]), "min_count": 2, "max_count": 1},
        {"variable_indices": np.array([0]), "min_count": 2, "max_count": 2},
    ]

    for kwargs in invalid_args:
        with pytest.raises(ValueError):
            BoundedLocalCountConstraint(layout=binary_layout, **kwargs)

    with pytest.raises(ValueError, match="binary local spaces"):
        BoundedLocalCountConstraint.at_most(
            layout=flux_layout,
            variable_indices=np.array([0]),
            max_count=1,
        )


def test_fixed_value_constraint_check_and_propagation() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    constraint = FixedValueConstraint(
        layout=layout,
        variable_indices=np.array([0, 2]),
        values=np.array([1, 0]),
    )

    assert constraint.check(np.array([1, 1, 0])).satisfied
    assert not constraint.check(np.array([0, 1, 0])).satisfied

    propagation = constraint.propagate(
        np.array([0, 0, 0]),
        np.array([False, False, True]),
    )

    assert propagation.consistent
    assert propagation.forced_assignments == ((0, 1),)
    assert not constraint.propagate(
        np.array([1, 0, 1]),
        np.array([True, False, True]),
    ).consistent


def test_fixed_value_constraint_rejects_invalid_construction() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="variable_indices must"):
        FixedValueConstraint(layout, np.array([[0]]), np.array([1]))
    with pytest.raises(ValueError, match="values must"):
        FixedValueConstraint(layout, np.array([0]), np.array([[1]]))
    with pytest.raises(ValueError, match="same length"):
        FixedValueConstraint(layout, np.array([0, 1]), np.array([1]))
    with pytest.raises(ValueError, match="At least"):
        FixedValueConstraint(layout, np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def test_local_sum_constraint_check_and_validation() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    constraint = LocalSumConstraint(
        layout=layout,
        variable_indices=np.array([0, 2]),
        coefficients=np.array([2, -1]),
        target=1,
    )

    np.testing.assert_array_equal(constraint.affected_variables(), np.array([0, 2]))
    assert constraint.value(np.array([1, 0, 1])) == 1
    assert constraint.check(np.array([1, 0, 1])).satisfied
    assert not constraint.check(np.array([0, 1, 1])).satisfied

    with pytest.raises(ValueError, match="variable_indices must"):
        LocalSumConstraint(layout, np.array([[0]]), np.array([1]), target=1)
    with pytest.raises(ValueError, match="coefficients must"):
        LocalSumConstraint(layout, np.array([0]), np.array([[1]]), target=1)
    with pytest.raises(ValueError, match="same length"):
        LocalSumConstraint(layout, np.array([0, 1]), np.array([1]), target=1)
    with pytest.raises(ValueError, match="At least"):
        LocalSumConstraint(
            layout,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            target=1,
        )
