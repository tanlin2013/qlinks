import numpy as np

from qlinks.models import SpinOneXYChainModel
from qlinks.operators import as_basis_operator


def test_basis_operator_matvec_matches_sparse_matrix() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(builder="sparse")

    O = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(0)
    v = rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        O @ v,
        result.kinetic @ v,
        atol=1e-12,
    )


def test_basis_operator_rmatvec_matches_sparse_matrix() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(builder="sparse")

    O = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(1)
    v = rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        v @ O,
        v @ result.kinetic,
        atol=1e-12,
    )


def test_basis_operator_bilinear_form_matches_sparse_matrix() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(builder="sparse")

    O = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(2)
    v = rng.normal(size=result.basis.n_states)

    actual = v.T @ O @ v
    expected = v.T @ result.kinetic @ v

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
    )


def test_basis_operator_expectation_matches_sparse_matrix() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(builder="sparse")

    O = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(3)
    v = rng.normal(size=result.basis.n_states) + 1j * rng.normal(size=result.basis.n_states)

    actual = O.expectation(v)
    expected = v.conj() @ result.kinetic @ v

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-12,
    )


def test_basis_operator_transpose_matvec_matches_sparse_matrix() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(builder="sparse")

    O = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(4)
    v = rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        O.T @ v,
        result.kinetic.T @ v,
        atol=1e-12,
    )


def test_basis_operator_adjoint_matvec_matches_sparse_matrix() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0 + 0.3j,
    )

    result = model.build(builder="sparse")

    O = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(5)
    v = rng.normal(size=result.basis.n_states) + 1j * rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        O.H @ v,
        result.kinetic.conjugate().T @ v,
        atol=1e-12,
    )
