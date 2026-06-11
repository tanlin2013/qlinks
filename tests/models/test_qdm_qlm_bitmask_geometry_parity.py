from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np
import pytest

from qlinks.models import (
    HoneycombQDMModel,
    HoneycombQLMModel,
    TriangularQDMModel,
    TriangularQLMModel,
)
from tests.helpers.assertions import (
    assert_optional_sparse_allclose,
    assert_same_binary_basis_order,
    assert_same_physical_flux_basis_order,
    assert_sparse_allclose,
)

BasisEncoding = Literal["binary", "flux_pm"]


@dataclass(frozen=True)
class BitmaskParityCase:
    name: str
    factory: Callable[[], object]
    basis_encoding: BasisEncoding


def _first_nonempty_triangular_qdm_pbc() -> TriangularQDMModel:
    seed = TriangularQDMModel(
        lx=4,
        ly=2,
        boundary_condition="periodic",
        coup_kin=-1.0,
        coup_pot=0.5,
    )
    labels = seed.nonempty_sector_labels()["z2_winding"]
    assert labels
    winding_a, winding_b = labels[0]
    return replace(seed, winding_a=winding_a, winding_b=winding_b)


def _first_nonempty_honeycomb_qdm_pbc() -> HoneycombQDMModel:
    seed = HoneycombQDMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        coup_kin=-1.0,
        coup_pot=0.5,
    )
    labels = seed.nonempty_sector_labels()["winding"]
    assert labels
    winding_x, winding_y = labels[0]
    return replace(seed, winding_x=winding_x, winding_y=winding_y)


def _first_nonempty_triangular_qlm_pbc() -> TriangularQLMModel:
    seed = TriangularQLMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        coup_kin=-1.0,
        coup_pot=0.5,
        charges=0,
    )
    labels = seed.nonempty_sector_labels()["z2_winding"]
    assert labels
    winding_a, winding_b = labels[0]
    return replace(seed, winding_a=winding_a, winding_b=winding_b)


def _first_nonempty_honeycomb_qlm_pbc() -> HoneycombQLMModel:
    seed = HoneycombQLMModel.from_staggered_background(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        coup_kin=-1.0,
        coup_pot=0.5,
    )
    labels = seed.nonempty_sector_labels()["winding"]
    assert labels
    winding_x, winding_y = labels[0]
    return replace(seed, winding_x=winding_x, winding_y=winding_y)


GEOMETRY_BITMASK_PARITY_CASES = [
    BitmaskParityCase(
        name="triangular_qdm_open",
        factory=lambda: TriangularQDMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.5,
        ),
        basis_encoding="binary",
    ),
    BitmaskParityCase(
        name="honeycomb_qdm_open",
        factory=lambda: HoneycombQDMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.5,
        ),
        basis_encoding="binary",
    ),
    BitmaskParityCase(
        name="triangular_qlm_open",
        factory=lambda: TriangularQLMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.5,
            charges=np.array([-2, 1, -1, 2], dtype=np.int64),
            charge_normalization="integer_flux",
        ),
        basis_encoding="flux_pm",
    ),
    BitmaskParityCase(
        name="honeycomb_qlm_open_nonempty",
        factory=lambda: HoneycombQLMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.5,
            charges=np.array(
                [-1, 3, -2, 2, -2, 2, -3, 1],
                dtype=np.int64,
            ),
            charge_normalization="integer_flux",
        ),
        basis_encoding="flux_pm",
    ),
    BitmaskParityCase(
        name="triangular_qdm_pbc_z2",
        factory=_first_nonempty_triangular_qdm_pbc,
        basis_encoding="binary",
    ),
    BitmaskParityCase(
        name="honeycomb_qdm_pbc_winding",
        factory=_first_nonempty_honeycomb_qdm_pbc,
        basis_encoding="binary",
    ),
    BitmaskParityCase(
        name="triangular_qlm_pbc_z2",
        factory=_first_nonempty_triangular_qlm_pbc,
        basis_encoding="flux_pm",
    ),
    BitmaskParityCase(
        name="honeycomb_qlm_pbc_staggered_winding",
        factory=_first_nonempty_honeycomb_qlm_pbc,
        basis_encoding="flux_pm",
    ),
]


@pytest.mark.parametrize(
    ("kinetic", "potential"),
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ],
)
def test_triangular_qdm_bitmask_matches_sparse_by_term(
    kinetic: float,
    potential: float,
) -> None:
    model = TriangularQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=kinetic,
        coup_pot=potential,
    )

    _assert_sparse_and_bitmask_builds_match(
        model,
        basis_encoding="binary",
        require_nonempty=True,
    )


@pytest.mark.parametrize("case", GEOMETRY_BITMASK_PARITY_CASES, ids=lambda case: case.name)
def test_triangular_honeycomb_qdm_qlm_bitmask_matches_sparse(
    case: BitmaskParityCase,
) -> None:
    model = case.factory()

    _assert_sparse_and_bitmask_builds_match(
        model,
        basis_encoding=case.basis_encoding,
        require_nonempty=True,
    )


def _assert_sparse_and_bitmask_builds_match(
    model,
    *,
    basis_encoding: BasisEncoding,
    require_nonempty: bool,
) -> None:
    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    if require_nonempty:
        assert sparse_result.basis.n_states > 0

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
        backend="scipy",
        sort_basis=False,
        on_missing="raise",
    )

    if basis_encoding == "binary":
        assert_same_binary_basis_order(sparse_result, bitmask_result)
    elif basis_encoding == "flux_pm":
        assert_same_physical_flux_basis_order(sparse_result, bitmask_result)
    else:
        raise ValueError(f"Unknown basis encoding: {basis_encoding!r}.")

    assert_sparse_allclose(bitmask_result.hamiltonian, sparse_result.hamiltonian)
    assert_optional_sparse_allclose(bitmask_result.kinetic, sparse_result.kinetic)
    assert_optional_sparse_allclose(bitmask_result.potential, sparse_result.potential)
