import numpy as np

from qlinks.basis import full_basis_from_layout
from qlinks.operators import ConstantDiagonalOperator, LocalValueDiagonalOperator
from qlinks.qec import (
    CageQECScanReport,
    CodeSpace,
    LocalErrorSet,
    diagnose_cage_code_candidate,
    diagnose_knill_laflamme,
    diagnose_local_indistinguishability,
    search_projected_logical_operators,
)
from qlinks.variables import LocalSpace, VariableLayout


def _simple_code_and_errors():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = full_basis_from_layout(layout, sort=True)
    idx_00 = basis.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_11 = basis.require_index(np.asarray([1, 1], dtype=np.int64))
    code = CodeSpace.from_basis_indices(basis, [idx_00, idx_11])
    errors = LocalErrorSet.from_operators(
        [
            ConstantDiagonalOperator(layout=layout, coefficient=1.0, name="I"),
            LocalValueDiagonalOperator(layout=layout, variable_index=0, name="n0"),
        ],
        names=["I", "n0"],
    )
    return basis, code, errors


def test_qec_reports_expose_summary_dicts_and_text() -> None:
    _basis, code, errors = _simple_code_and_errors()

    kl_report = diagnose_knill_laflamme(code, errors)
    logical_report = search_projected_logical_operators(code, errors)
    profile = diagnose_local_indistinguishability(code, errors, max_weight=1)

    assert code.to_summary_dict()["dimension"] == 2
    assert errors.to_summary_dict()["n_errors"] == 2
    assert kl_report.to_summary_dict()["passes_exact_kl"] is False
    assert logical_report.to_summary_dict()["n_candidates"] == 2
    assert profile.to_summary_dict()["first_violating_weight"] == 1

    assert "Code space" in code.format_summary()
    assert "Local error set" in errors.format_summary()
    assert "Knill-Laflamme report" in kl_report.format_summary()
    assert "Projected logical operators" in logical_report.format_summary()
    assert "Local indistinguishability profile" in profile.format_summary()


def test_qec_profile_and_scan_reports_expose_human_summaries() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Record:
        support: np.ndarray
        local_state: np.ndarray
        signature: tuple[int, int]

    basis, _code, errors = _simple_code_and_errors()
    records = [
        Record(
            support=np.asarray([0], dtype=np.int64),
            local_state=np.asarray([1.0], dtype=np.complex128),
            signature=(0, 2),
        ),
        Record(
            support=np.asarray([basis.n_states - 1], dtype=np.int64),
            local_state=np.asarray([1.0], dtype=np.complex128),
            signature=(0, 2),
        ),
    ]
    candidate = diagnose_cage_code_candidate(
        basis=basis,
        records=records,
        errors=errors,
        signature=(0, 2),
        max_weight=1,
    )
    scan = CageQECScanReport(candidate_reports=(candidate,))

    assert candidate.to_summary_dict()["signature"] == (0, 2)
    assert scan.to_summary_dict()["n_candidate_reports"] == 1
    assert "QEC code candidate" in candidate.format_summary()
    assert "Cage QEC scan" in scan.format_summary()


def test_qec_reports_are_rich_renderable() -> None:
    _basis, code, errors = _simple_code_and_errors()
    kl_report = diagnose_knill_laflamme(code, errors)
    logical_report = search_projected_logical_operators(code, errors)
    profile = diagnose_local_indistinguishability(code, errors, max_weight=1)

    for report in (code, errors, kl_report, logical_report, profile):
        assert report.to_rich() is not None
