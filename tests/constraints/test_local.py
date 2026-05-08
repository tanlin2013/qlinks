import numpy as np

from qlinks.constraints import ConstraintCollection, FixedValueConstraint, TotalValueSector
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
    