from dataclasses import dataclass

import numpy as np

from qlinks.basis import full_basis_from_layout
from qlinks.operators import ConstantDiagonalOperator, LocalValueDiagonalOperator
from qlinks.qec import (
    CodeSpace,
    LocalErrorSet,
    diagnose_cage_code_candidate,
    diagnose_local_indistinguishability,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class _Record:
    support: np.ndarray
    local_state: np.ndarray
    signature: tuple[int, int]


def _two_bit_repetition_code():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = full_basis_from_layout(layout, sort=True)
    idx_00 = basis.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_11 = basis.require_index(np.asarray([1, 1], dtype=np.int64))
    code = CodeSpace.from_basis_indices(basis, [idx_00, idx_11])
    return layout, basis, code, idx_00, idx_11


def test_local_error_set_from_layout_generates_weight_two_products() -> None:
    layout, _basis, _code, _idx_00, _idx_11 = _two_bit_repetition_code()

    errors = LocalErrorSet.from_layout(
        layout,
        max_weight=2,
        include_value_diagonal=True,
        include_projectors=False,
        include_transitions=False,
    )

    assert len(errors.by_exact_weight(1)) == 2
    assert len(errors.by_exact_weight(2)) == 1
    product_error = errors.by_exact_weight(2)[0]
    assert product_error.name == "prod_value_0__value_1"
    assert product_error.weight == 2


def test_diagnose_local_indistinguishability_reports_first_violating_weight() -> None:
    layout, _basis, code, _idx_00, _idx_11 = _two_bit_repetition_code()
    errors = LocalErrorSet.from_operators(
        [
            ConstantDiagonalOperator(layout=layout, coefficient=1.0, name="I"),
            LocalValueDiagonalOperator(layout=layout, variable_index=0, name="n0"),
        ],
        names=["I", "n0"],
    )

    report = diagnose_local_indistinguishability(
        code,
        errors,
        max_weight=1,
        tolerance=1e-12,
    )

    assert report.first_violating_weight == 1
    assert report.local_indistinguishability_weight == 0
    assert not report.passes_all_tested_weights
    assert report.worst_summary is not None
    assert report.worst_summary.dominant_failure == "distinguishability"


def test_diagnose_cage_code_candidate_accepts_cage_record_like_objects() -> None:
    layout, basis, _code, idx_00, idx_11 = _two_bit_repetition_code()
    records = [
        _Record(
            support=np.asarray([idx_00], dtype=np.int64),
            local_state=np.asarray([1.0], dtype=np.complex128),
            signature=(0, 2),
        ),
        _Record(
            support=np.asarray([idx_11], dtype=np.int64),
            local_state=np.asarray([1.0], dtype=np.complex128),
            signature=(0, 2),
        ),
    ]
    errors = LocalErrorSet.from_operators(
        [ConstantDiagonalOperator(layout=layout, coefficient=1.0, name="I")],
        names=["I"],
    )

    report = diagnose_cage_code_candidate(
        basis=basis,
        records=records,
        errors=errors,
        signature=(0, 2),
        max_weight=1,
        include_error_algebra=True,
    )

    assert report.code_dimension == 2
    assert report.record_count == 2
    assert report.signature == (0, 2)
    assert report.qec_candidate
    assert report.logical_operators is not None
    assert report.error_algebra is not None
    assert report.error_algebra.classification == "scalar_algebra_subspace_code"
