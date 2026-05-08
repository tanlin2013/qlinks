import numpy as np

from qlinks.constraints import BaseConstraint, ConstraintResult, all_satisfied
from qlinks.variables import LocalSpace, VariableLayout


class DummyConstraint(BaseConstraint):
    def __init__(self, layout: VariableLayout, name: str = "dummy") -> None:
        self.layout = layout
        self.name = name


def test_constraint_result_bool() -> None:
    assert bool(ConstraintResult(True))
    assert not bool(ConstraintResult(False))


def test_base_constraint_accepts_valid_config() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    constraint = DummyConstraint(layout=layout)

    result = constraint.check(np.array([0, 1, 0]))

    assert result.satisfied
    assert constraint.is_satisfied(np.array([0, 1, 0]))


def test_all_satisfied() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    c1 = DummyConstraint(layout=layout, name="c1")
    c2 = DummyConstraint(layout=layout, name="c2")

    assert all_satisfied(np.array([0, 1]), constraints=[c1, c2])
