import numpy as np
import pytest

from qlinks.constraints import ParitySector, TotalValueSector
from qlinks.variables import LocalSpace, VariableLayout


def test_total_value_sector_all_variables() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    sector = TotalValueSector(layout=layout, target=2)

    assert sector.value(np.array([1, 0, 1, 0])) == 2
    assert sector.is_satisfied(np.array([1, 0, 1, 0]))
    assert not sector.is_satisfied(np.array([1, 1, 1, 0]))


def test_total_value_sector_selected_variables() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    sector = TotalValueSector(
        layout=layout,
        variable_indices=np.array([0, 2]),
        coefficients=np.array([1, 2]),
        target=3,
    )

    assert sector.value(np.array([1, 0, 1, 0])) == 3
    assert sector.is_satisfied(np.array([1, 0, 1, 0]))


def test_total_value_rejects_coeff_mismatch() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    with pytest.raises(ValueError, match="same length"):
        TotalValueSector(
            layout=layout,
            target=1,
            variable_indices=np.array([0, 1]),
            coefficients=np.array([1]),
        )


def test_parity_sector() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    even = ParitySector(layout=layout, target=0)
    odd = ParitySector(layout=layout, target=1)

    assert even.is_satisfied(np.array([1, 1, 0, 0]))
    assert odd.is_satisfied(np.array([1, 0, 0, 0]))


def test_parity_sector_selected_variables() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    sector = ParitySector(
        layout=layout,
        variable_indices=np.array([0, 2]),
        target=0,
    )

    assert sector.value(np.array([1, 1, 1, 0])) == 0
    assert sector.is_satisfied(np.array([1, 1, 1, 0]))


def test_parity_sector_rejects_bad_target() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="0 or 1"):
        ParitySector(layout=layout, target=2)
        