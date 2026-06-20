from dataclasses import dataclass

import numpy as np

from qlinks.caging import (
    local_reduced_density_matrix_readout_from_state,
    reduced_iz_local_rdm_readouts_from_report,
)


@dataclass(frozen=True)
class _FakeReducedIZComponentGroup:
    component_id: int
    support_variables: tuple[int, ...]
    zero_indices: tuple[int, ...]


@dataclass(frozen=True)
class _FakeClassificationReport:
    groups: tuple[_FakeReducedIZComponentGroup, ...]

    def reduced_iz_component_groups(self, *, decomposition):
        assert decomposition == "exact_support"
        return self.groups


def test_local_reduced_density_matrix_readout_exposes_matrix_unit_terms() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    state = state / np.linalg.norm(state)

    readout = local_reduced_density_matrix_readout_from_state(
        basis_configs=basis_configs,
        state=state,
        variable_indices=(0,),
        tolerance=1e-12,
        max_matrix_unit_terms=None,
    )

    assert readout.variable_indices == (0,)
    assert readout.local_dim == 2
    assert readout.support_rank == 2
    assert readout.nullity == 0
    assert readout.n_matrix_unit_terms == 2
    assert not readout.matrix_unit_terms_truncated

    terms = {
        (term.target_pattern, term.source_pattern): term.coefficient
        for term in readout.matrix_unit_terms
    }
    assert np.isclose(terms[((0,), (0,))], 0.5)
    assert np.isclose(terms[((1,), (1,))], 0.5)


def test_reduced_iz_local_rdm_readouts_from_report_use_component_supports() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    state = state / np.linalg.norm(state)
    report = _FakeClassificationReport(
        groups=(
            _FakeReducedIZComponentGroup(
                component_id=3,
                support_variables=(0,),
                zero_indices=(4, 5),
            ),
            _FakeReducedIZComponentGroup(
                component_id=4,
                support_variables=(1,),
                zero_indices=(6,),
            ),
        )
    )

    readouts = reduced_iz_local_rdm_readouts_from_report(
        report,  # type: ignore[arg-type]
        basis_configs=basis_configs,
        state=state,
        decomposition="exact_support",
        tolerance=1e-12,
        max_matrix_unit_terms=1,
    )

    assert len(readouts) == 2
    assert readouts[0].component_index == 0
    assert readouts[0].component_id == 3
    assert readouts[0].zero_indices == (4, 5)
    assert readouts[0].variable_indices == (0,)
    assert readouts[0].n_matrix_unit_terms == 2
    assert readouts[0].matrix_unit_terms_truncated
    assert len(readouts[0].matrix_unit_terms) == 1
    assert readouts[1].variable_indices == (1,)
