from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tests.helpers.assertions import (
    assert_optional_sparse_allclose,
    assert_same_binary_basis_order,
    assert_same_physical_flux_basis_order,
)


@dataclass(frozen=True)
class DummyBasis:
    states: np.ndarray

    def to_array_basis(self) -> DummyBasis:
        return self


@dataclass(frozen=True)
class DummyBuildResult:
    basis: DummyBasis


def test_assert_same_physical_flux_basis_order_accepts_matching_basis() -> None:
    sparse_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [-1, 1],
                    [1, -1],
                ],
                dtype=np.int64,
            )
        )
    )
    bitmask_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [0, 1],
                    [1, 0],
                ],
                dtype=np.int64,
            )
        )
    )

    assert_same_physical_flux_basis_order(sparse_result, bitmask_result)


def test_assert_same_physical_flux_basis_order_rejects_mismatch() -> None:
    sparse_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [-1, 1],
                    [1, -1],
                ],
                dtype=np.int64,
            )
        )
    )
    bitmask_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=np.int64,
            )
        )
    )

    with pytest.raises(AssertionError):
        assert_same_physical_flux_basis_order(sparse_result, bitmask_result)


def test_assert_same_binary_basis_order_accepts_matching_basis() -> None:
    sparse_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [0, 1],
                    [1, 0],
                ],
                dtype=np.int64,
            )
        )
    )
    bitmask_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [0, 1],
                    [1, 0],
                ],
                dtype=np.int64,
            )
        )
    )

    assert_same_binary_basis_order(sparse_result, bitmask_result)


def test_assert_same_binary_basis_order_rejects_mismatch() -> None:
    sparse_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [0, 1],
                    [1, 0],
                ],
                dtype=np.int64,
            )
        )
    )
    bitmask_result = DummyBuildResult(
        basis=DummyBasis(
            states=np.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=np.int64,
            )
        )
    )

    with pytest.raises(AssertionError):
        assert_same_binary_basis_order(sparse_result, bitmask_result)


def test_assert_optional_sparse_allclose_accepts_both_none() -> None:
    assert_optional_sparse_allclose(None, None)


def test_assert_optional_sparse_allclose_rejects_one_none() -> None:
    with pytest.raises(AssertionError):
        assert_optional_sparse_allclose(None, object())
