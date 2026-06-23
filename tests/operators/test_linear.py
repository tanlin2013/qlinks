import numpy as np
import pytest

from qlinks.models import SpinOneXYChainModel
from qlinks.operators import as_basis_operator


def test_basis_operator_matvec_matches_sparse_matrix() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )

    result = model.build(builder="sparse")

    Opt = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(0)
    v = rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        Opt @ v,
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

    Opt = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(1)
    v = rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        v @ Opt,
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

    Opt = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(2)
    v = rng.normal(size=result.basis.n_states)

    actual = v.T @ Opt @ v
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

    Opt = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(3)
    v = rng.normal(size=result.basis.n_states) + 1j * rng.normal(size=result.basis.n_states)

    actual = Opt.expectation(v)
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

    Opt = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(4)
    v = rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        Opt.T @ v,
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

    Opt = as_basis_operator(
        result.basis,
        result.kinetic_operators,
    )

    rng = np.random.default_rng(5)
    v = rng.normal(size=result.basis.n_states) + 1j * rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        Opt.H @ v,
        result.kinetic.conjugate().T @ v,
        atol=1e-12,
    )


def test_basis_operator_from_operator_and_matrix_products() -> None:
    model = SpinOneXYChainModel(
        length=3,
        boundary_condition="open",
        j_xy=1.0,
    )
    result = model.build(builder="sparse")
    operator = as_basis_operator(result.basis, result.kinetic_operators)

    vectors = np.eye(result.basis.n_states, 2, dtype=np.complex128)
    left_vectors = vectors.T.copy()

    np.testing.assert_allclose(operator @ vectors, result.kinetic @ vectors, atol=1e-12)
    np.testing.assert_allclose(left_vectors @ operator, left_vectors @ result.kinetic, atol=1e-12)
    np.testing.assert_allclose(operator.T @ vectors, result.kinetic.T @ vectors, atol=1e-12)
    np.testing.assert_allclose(
        left_vectors @ operator.H,
        left_vectors @ result.kinetic.conjugate().T,
        atol=1e-12,
    )


def test_basis_operator_rejects_wrong_operand_shapes() -> None:
    model = SpinOneXYChainModel(length=2, boundary_condition="open", j_xy=1.0)
    result = model.build(builder="sparse")
    operator = as_basis_operator(result.basis, result.kinetic_operators)

    with pytest.raises(ValueError, match="one-dimensional"):
        operator.matvec(np.zeros((result.basis.n_states, 1)))
    with pytest.raises(ValueError, match="vector length"):
        operator.matvec(np.zeros(result.basis.n_states + 1))
    with pytest.raises(ValueError, match="two-dimensional"):
        operator.matmat(np.zeros(result.basis.n_states))
    with pytest.raises(ValueError, match="matrix shape"):
        operator.matmat(np.zeros((result.basis.n_states + 1, 1)))
    with pytest.raises(ValueError, match="Right operand"):
        operator @ np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="Left operand"):
        np.zeros((1, 1, 1)) @ operator


def test_transposed_basis_operator_rejects_wrong_operand_shapes() -> None:
    model = SpinOneXYChainModel(length=2, boundary_condition="open", j_xy=1.0)
    result = model.build(builder="sparse")
    operator = as_basis_operator(result.basis, result.kinetic_operators).T

    with pytest.raises(ValueError, match="one-dimensional"):
        operator.matvec(np.zeros((result.basis.n_states, 1)))
    with pytest.raises(ValueError, match="vector length"):
        operator.matvec(np.zeros(result.basis.n_states + 1))
    with pytest.raises(ValueError, match="one-dimensional"):
        operator.rmatvec(np.zeros((1, result.basis.n_states)))
    with pytest.raises(ValueError, match="vector length"):
        operator.rmatvec(np.zeros(result.basis.n_states + 1))
    with pytest.raises(ValueError, match="Right operand"):
        operator @ np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="Left operand"):
        np.zeros((1, 1, 1)) @ operator


def test_basis_operator_expectation_without_conjugation() -> None:
    model = SpinOneXYChainModel(length=3, boundary_condition="open", j_xy=1.0 + 0.2j)
    result = model.build(builder="sparse")
    operator = as_basis_operator(result.basis, result.kinetic_operators)

    rng = np.random.default_rng(10)
    vector = rng.normal(size=result.basis.n_states) + 1j * rng.normal(size=result.basis.n_states)

    np.testing.assert_allclose(
        operator.expectation(vector, conjugate=False),
        vector.T @ result.kinetic @ vector,
        atol=1e-12,
    )


def test_transpose_and_adjoint_view_roundtrips() -> None:
    model = SpinOneXYChainModel(length=2, boundary_condition="open", j_xy=1.0 + 0.2j)
    result = model.build(builder="sparse")
    operator = as_basis_operator(result.basis, result.kinetic_operators)

    assert operator.T.T is operator
    assert operator.H.H is operator
    assert operator.H.T.conjugate is True
