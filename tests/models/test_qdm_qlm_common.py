from dataclasses import dataclass
from typing import Callable

import pytest

from qlinks.models import (
    HoneycombQDMModel,
    HoneycombQLMModel,
    TriangularQDMModel,
    TriangularQLMModel,
)


@dataclass(frozen=True)
class ModelSmokeCase:
    name: str
    factory: Callable[[], object]
    expected_plaquette_count: int | None = None


MODEL_SMOKE_CASES = [
    ModelSmokeCase(
        name="triangular_qdm_open",
        factory=lambda: TriangularQDMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
        ),
    ),
    ModelSmokeCase(
        name="honeycomb_qdm_open",
        factory=lambda: HoneycombQDMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
        ),
    ),
    ModelSmokeCase(
        name="triangular_qlm_open",
        factory=lambda: TriangularQLMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
            charges=0,
        ),
    ),
    ModelSmokeCase(
        name="honeycomb_qlm_open",
        factory=lambda: HoneycombQLMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
            charges=0,
        ),
        expected_plaquette_count=1,
    ),
]


@pytest.mark.parametrize("case", MODEL_SMOKE_CASES, ids=lambda c: c.name)
def test_model_smoke(case: ModelSmokeCase) -> None:
    model = case.factory()
    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states >= 0

    if case.expected_plaquette_count is None:
        assert len(model.plaquette_ids()) > 0
    else:
        assert len(model.plaquette_ids()) == case.expected_plaquette_count
