import numpy as np
import pytest

from qlinks.caging.linear_independence import IndependentColumnSelector


def test_independent_column_selector_accepts_independent_vectors() -> None:
    selector = IndependentColumnSelector(tolerance=1e-12)

    assert selector.add(np.array([1.0, 0.0], dtype=np.complex128))
    assert selector.add(np.array([0.0, 1.0], dtype=np.complex128))
    assert selector.rank == 2

    np.testing.assert_allclose(selector.columns[0], [1.0, 0.0])
    np.testing.assert_allclose(selector.columns[1], [0.0, 1.0])


def test_independent_column_selector_rejects_dependent_vector() -> None:
    selector = IndependentColumnSelector(tolerance=1e-12)

    assert selector.add(np.array([1.0, 0.0], dtype=np.complex128))
    assert not selector.add(np.array([2.0, 0.0], dtype=np.complex128))
    assert selector.rank == 1


def test_independent_column_selector_rejects_nearly_dependent_vector() -> None:
    selector = IndependentColumnSelector(tolerance=1e-8)

    assert selector.add(np.array([1.0, 0.0], dtype=np.complex128))
    assert not selector.add(np.array([1.0, 1.0e-10], dtype=np.complex128))
    assert selector.rank == 1


def test_independent_column_selector_accepts_complex_independent_vector() -> None:
    selector = IndependentColumnSelector(tolerance=1e-12)

    assert selector.add(np.array([1.0, 1.0j], dtype=np.complex128))
    assert selector.add(np.array([1.0j, 1.0], dtype=np.complex128))
    assert selector.rank == 2


def test_independent_column_selector_rejects_non_vector_input() -> None:
    selector = IndependentColumnSelector(tolerance=1e-12)

    with pytest.raises(ValueError, match="one-dimensional"):
        selector.add(np.eye(2, dtype=np.complex128))
