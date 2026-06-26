import numpy as np

from qlinks.basis import full_basis_from_layout
from qlinks.operators import OperatorSum, SetVariablesOperator
from qlinks.qec import CodeSpace, ErrorOperator, search_projected_logical_operators
from qlinks.variables import LocalSpace, VariableLayout


def test_search_projected_logical_operators_finds_weight_two_logical_x() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = full_basis_from_layout(layout, sort=True)
    idx_00 = basis.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_11 = basis.require_index(np.asarray([1, 1], dtype=np.int64))
    code = CodeSpace.from_basis_indices(basis, [idx_00, idx_11])

    variables = np.asarray([0, 1], dtype=np.int64)
    logical_x = OperatorSum.from_terms(
        (
            SetVariablesOperator(
                layout=layout,
                variable_indices=variables,
                initial_values=np.asarray([0, 0], dtype=np.int64),
                final_values=np.asarray([1, 1], dtype=np.int64),
                name="00_to_11",
            ),
            SetVariablesOperator(
                layout=layout,
                variable_indices=variables,
                initial_values=np.asarray([1, 1], dtype=np.int64),
                final_values=np.asarray([0, 0], dtype=np.int64),
                name="11_to_00",
            ),
        ),
        name="logical_x_candidate",
    )

    report = search_projected_logical_operators(
        code,
        [
            ErrorOperator.from_operator(
                logical_x,
                name="logical_x_candidate",
                support_variables=(0, 1),
            )
        ],
    )
    candidate = report.best_candidate

    assert candidate is not None
    np.testing.assert_allclose(
        candidate.projected_matrix,
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    )
    assert candidate.traceless_spectral_norm > 0.9
    assert candidate.leakage_frobenius_norm < 1e-12
    assert report.low_leakage_candidates() == (candidate,)
