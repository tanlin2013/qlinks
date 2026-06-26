import numpy as np

from qlinks.basis import full_basis_from_layout
from qlinks.operators import (
    BinaryFlipOperator,
    ConstantDiagonalOperator,
    LocalValueDiagonalOperator,
)
from qlinks.qec import CodeSpace, LocalErrorSet, diagnose_knill_laflamme
from qlinks.variables import LocalSpace, VariableLayout


def _repetition_code():
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = full_basis_from_layout(layout, sort=True)
    idx_00 = basis.require_index(np.asarray([0, 0], dtype=np.int64))
    idx_11 = basis.require_index(np.asarray([1, 1], dtype=np.int64))
    return layout, CodeSpace.from_basis_indices(basis, [idx_00, idx_11])


def test_knill_laflamme_passes_for_single_bit_flip_on_repetition_code() -> None:
    layout, code = _repetition_code()
    errors = LocalErrorSet.from_operators(
        [
            ConstantDiagonalOperator(layout=layout, coefficient=1.0, name="I"),
            BinaryFlipOperator(layout=layout, variable_index=0, name="X0"),
        ],
        names=["I", "X0"],
    )

    report = diagnose_knill_laflamme(code, errors)

    assert report.passes_exact_kl
    assert report.max_frobenius_residual < 1e-12
    assert report.worst_pair is not None
    assert all(pair.dominant_failure == "scalar" for pair in report.pair_reports)


def test_knill_laflamme_detects_local_distinguishability() -> None:
    layout, code = _repetition_code()
    errors = LocalErrorSet.from_operators(
        [
            ConstantDiagonalOperator(layout=layout, coefficient=1.0, name="I"),
            LocalValueDiagonalOperator(layout=layout, variable_index=0, name="n0"),
        ],
        names=["I", "n0"],
    )

    report = diagnose_knill_laflamme(code, errors, tolerance=1e-12)
    worst = report.worst_pair

    assert not report.passes_exact_kl
    assert worst is not None
    assert worst.names in {("I", "n0"), ("n0", "I"), ("n0", "n0")}
    assert worst.dominant_failure == "distinguishability"
    assert worst.diagonal_spread > 0.0


def test_knill_laflamme_reports_leakage_separately_from_kl_residual() -> None:
    layout, code = _repetition_code()
    errors = LocalErrorSet.from_operators(
        [BinaryFlipOperator(layout=layout, variable_index=0, name="X0")],
        names=["X0"],
    )

    report = diagnose_knill_laflamme(code, errors)
    image = report.worst_error_image

    assert report.passes_exact_kl
    assert image is not None
    assert image.leakage_frobenius_norm > 0.0
    assert np.isclose(image.relative_leakage_frobenius_norm, 1.0)
