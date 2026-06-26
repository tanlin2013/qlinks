import numpy as np

from qlinks.basis import full_basis_from_layout
from qlinks.qec import CodeSpace, LocalErrorSet, diagnose_projected_error_algebra
from qlinks.variables import LocalSpace, VariableLayout


def _full_code(dimension: int) -> CodeSpace:
    layout = VariableLayout.from_sites(dimension.bit_length() - 1, LocalSpace.binary())
    basis = full_basis_from_layout(layout, sort=True)
    assert basis.n_states == dimension
    return CodeSpace.from_basis_indices(basis, tuple(range(dimension)))


def test_projected_error_algebra_identifies_scalar_subspace_code() -> None:
    code = _full_code(2)
    errors = LocalErrorSet.from_operators(
        [np.eye(2, dtype=np.complex128)],
        names=["I"],
    )

    report = diagnose_projected_error_algebra(code, errors)

    assert report.algebra_dimension == 1
    assert report.commutant_dimension == 4
    assert report.center_dimension == 1
    assert report.classification == "scalar_algebra_subspace_code"
    assert "Projected error algebra" in report.format_summary()
    assert report.to_summary_dict()["has_nontrivial_commutant"] is True


def test_projected_error_algebra_identifies_full_logical_algebra() -> None:
    code = _full_code(2)
    identity = np.eye(2, dtype=np.complex128)
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    errors = LocalErrorSet.from_operators([identity, x, z], names=["I", "X", "Z"])

    report = diagnose_projected_error_algebra(code, errors)

    assert report.algebra_dimension == 4
    assert report.full_matrix_algebra_dimension == 4
    assert report.commutant_dimension == 1
    assert report.center_dimension == 1
    assert report.classification == "full_matrix_algebra_no_protected_subsystem"


def test_projected_error_algebra_finds_subsystem_candidate() -> None:
    code = _full_code(4)
    identity_2 = np.eye(2, dtype=np.complex128)
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    errors = LocalErrorSet.from_operators(
        [
            np.kron(identity_2, identity_2),
            np.kron(x, identity_2),
            np.kron(z, identity_2),
        ],
        names=["I", "X_gauge", "Z_gauge"],
    )

    report = diagnose_projected_error_algebra(code, errors)

    assert report.algebra_dimension == 4
    assert report.commutant_dimension == 4
    assert report.center_dimension == 1
    assert report.candidate_subsystem_dimension == 2
    assert report.classification == "subsystem_code_candidate"


def test_projected_error_algebra_separates_classical_blocks() -> None:
    code = _full_code(2)
    identity = np.eye(2, dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    errors = LocalErrorSet.from_operators([identity, z], names=["I", "Z"])

    report = diagnose_projected_error_algebra(code, errors)

    assert report.algebra_dimension == 2
    assert report.commutant_dimension == 2
    assert report.center_dimension == 2
    assert report.classification == "classical_block_structure"
