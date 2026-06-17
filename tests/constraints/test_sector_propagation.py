import numpy as np

from qlinks.constraints import (
    ParitySector,
    SquareQDMElectricWindingSector,
    TotalValueSector,
)
from qlinks.lattice import SquareLattice
from qlinks.variables import LocalSpace, VariableLayout


def test_total_value_sector_reports_affected_variables() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())
    sector = TotalValueSector(
        layout=layout,
        variable_indices=np.array([1, 3], dtype=np.int64),
        target=1,
    )

    np.testing.assert_array_equal(
        sector.affected_variables(),
        np.array([1, 3], dtype=np.int64),
    )


def test_total_value_sector_propagates_when_upper_bound_is_tight() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    sector = TotalValueSector(layout=layout, target=1)

    config = np.array([1, 0, 0], dtype=np.int64)
    assigned = np.array([True, False, False], dtype=bool)

    result = sector.propagate(config, assigned)

    assert result.consistent
    assert result.forced_assignments == ((1, 0), (2, 0))


def test_total_value_sector_propagates_when_lower_bound_is_tight() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    sector = TotalValueSector(layout=layout, target=2)

    config = np.array([0, 0, 0], dtype=np.int64)
    assigned = np.array([True, False, False], dtype=bool)

    result = sector.propagate(config, assigned)

    assert result.consistent
    assert result.forced_assignments == ((1, 1), (2, 1))


def test_total_value_sector_detects_impossible_partial_sum() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    sector = TotalValueSector(layout=layout, target=3)

    config = np.array([0, 0, 0], dtype=np.int64)
    assigned = np.array([True, False, False], dtype=bool)

    result = sector.propagate(config, assigned)

    assert not result.consistent


def test_parity_sector_forces_last_binary_variable() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    sector = ParitySector(layout=layout, target=1)

    config = np.array([1, 1, 0], dtype=np.int64)
    assigned = np.array([True, True, False], dtype=bool)

    result = sector.propagate(config, assigned)

    assert result.consistent
    assert result.forced_assignments == ((2, 1),)


def test_square_qdm_electric_winding_forces_last_cut_link() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    sector = SquareQDMElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=0,
    )

    cut_variables = sector.variable_indices
    signs = sector.signs
    assert cut_variables.size == 2

    config = np.zeros(layout.n_variables, dtype=np.int64)
    assigned = np.zeros(layout.n_variables, dtype=bool)

    first = int(cut_variables[0])
    second = int(cut_variables[1])

    # Make the first signed electric contribution +1.  At target zero, the
    # second and only remaining cut link must contribute -1.
    config[first] = 1 if int(signs[0]) == 1 else 0
    assigned[first] = True

    expected_second_value = 0 if int(signs[1]) == 1 else 1

    result = sector.propagate(config, assigned)

    assert result.consistent
    assert result.forced_assignments == ((second, expected_second_value),)
