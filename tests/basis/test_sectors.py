from dataclasses import dataclass

import numpy as np
import pytest

from qlinks.basis import (
    sector_mask_from_build_result,
    sector_mask_from_sectors,
)


class _FirstVariableEquals:
    def __init__(self, target: int):
        self.target = target

    def is_satisfied(self, config) -> bool:
        return int(config[0]) == self.target


class _SecondVariableEquals:
    def __init__(self, target: int):
        self.target = target

    def is_satisfied(self, config) -> bool:
        return int(config[1]) == self.target


@dataclass(frozen=True)
class _FakeBasis:
    states: np.ndarray


@dataclass(frozen=True)
class _FakeBuildResult:
    basis: _FakeBasis
    constraints: tuple[object, ...] = ()


def test_sector_mask_from_sectors_empty_selects_all() -> None:
    configs = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)

    mask = sector_mask_from_sectors(configs, sectors=())

    np.testing.assert_array_equal(mask, np.array([True, True, True]))


def test_sector_mask_from_sectors_combines_constraints() -> None:
    configs = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=np.int64,
    )

    mask = sector_mask_from_sectors(
        configs,
        sectors=(
            _FirstVariableEquals(1),
            _SecondVariableEquals(0),
        ),
    )

    np.testing.assert_array_equal(
        mask,
        np.array([False, False, True, False]),
    )


def test_sector_mask_from_build_result_uses_extra_sectors() -> None:
    configs = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=np.int64,
    )
    build_result = _FakeBuildResult(
        basis=_FakeBasis(states=configs),
    )

    mask = sector_mask_from_build_result(
        build_result,
        sectors=(_FirstVariableEquals(0),),
    )

    np.testing.assert_array_equal(
        mask,
        np.array([True, True, False, False]),
    )


def test_sector_mask_from_sectors_rejects_invalid_sector() -> None:
    configs = np.array([[0, 0]], dtype=np.int64)

    with pytest.raises(TypeError, match="is_satisfied"):
        sector_mask_from_sectors(configs, sectors=(object(),))
