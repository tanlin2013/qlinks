import numpy as np
import pytest

from qlinks.basis import full_basis_from_layout
from qlinks.qec import CodeSpace
from qlinks.variables import LocalSpace, VariableLayout


def _binary_basis(n_variables: int = 2):
    layout = VariableLayout.from_sites(n_variables, LocalSpace.binary())
    return full_basis_from_layout(layout, sort=True)


def test_code_space_from_basis_indices_builds_orthonormal_columns() -> None:
    basis = _binary_basis()
    code = CodeSpace.from_basis_indices(basis, [0, basis.n_states - 1])

    assert code.dimension == 2
    assert code.ambient_dimension == basis.n_states
    np.testing.assert_allclose(code.vectors.conj().T @ code.vectors, np.eye(2))
    np.testing.assert_allclose(code.projector @ code.projector, code.projector)


def test_code_space_from_vectors_rejects_rank_deficient_input_by_default() -> None:
    basis = _binary_basis()
    vector = np.zeros(basis.n_states, dtype=np.complex128)
    vector[0] = 1.0
    vectors = np.column_stack([vector, vector])

    with pytest.raises(ValueError, match="linearly dependent"):
        CodeSpace.from_vectors(basis, vectors)


def test_code_space_from_vectors_can_keep_span_of_rank_deficient_input() -> None:
    basis = _binary_basis()
    vector = np.zeros(basis.n_states, dtype=np.complex128)
    vector[0] = 1.0
    vectors = np.column_stack([vector, vector])

    code = CodeSpace.from_vectors(basis, vectors, allow_rank_deficient=True)

    assert code.dimension == 1
    np.testing.assert_allclose(code.vectors.conj().T @ code.vectors, np.ones((1, 1)))


def test_code_space_from_row_vectors_matches_column_constructor() -> None:
    basis = _binary_basis()
    rows = np.zeros((2, basis.n_states), dtype=np.complex128)
    rows[0, 0] = 1.0
    rows[1, basis.n_states - 1] = 1.0

    code = CodeSpace.from_row_vectors(basis, rows)

    assert code.dimension == 2
    np.testing.assert_allclose(np.abs(code.vectors), rows.T)
