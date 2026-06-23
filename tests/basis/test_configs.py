from dataclasses import dataclass

import numpy as np
import pytest

from qlinks.basis import (
    basis_configs_from_basis,
    basis_configs_from_build_result,
    decode_basis_configs_with_layout,
)


@dataclass(frozen=True)
class _FakeBasis:
    states: np.ndarray


@dataclass(frozen=True)
class _FakeArrayBasis:
    states: np.ndarray


@dataclass(frozen=True)
class _FakeEncodedBasis:
    states: np.ndarray

    def to_array_basis(self) -> _FakeArrayBasis:
        return _FakeArrayBasis(states=self.states)


@dataclass(frozen=True)
class _FakeBuildResult:
    basis: object


class _FakeLocalSpace:
    values = (0, 1)

    def contains(self, value: int) -> bool:
        return int(value) in self.values

    def decode(self, code: int) -> int:
        return self.values[int(code)]


class _FakeVariable:
    local_space = _FakeLocalSpace()


class _FakeLayout:
    def __init__(self, n_variables: int = 2):
        self.variables = tuple(_FakeVariable() for _ in range(n_variables))

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, index: int):
        return self.variables[index]

    def local_space(self, index: int):
        return self.variables[index].local_space

    def validate_batch(self, configs) -> None:
        configs_array = np.asarray(configs)

        if configs_array.ndim != 2:
            raise ValueError("configs must be two-dimensional.")

        if configs_array.shape[1] != len(self):
            raise ValueError("incompatible variable counts.")

        for column_index, variable in enumerate(self.variables):
            for value in configs_array[:, column_index]:
                if not variable.local_space.contains(int(value)):
                    raise ValueError(f"invalid value: {value}")


def test_basis_configs_from_basis_reads_states_attribute() -> None:
    states = np.array([[0, 1], [1, 0]], dtype=np.int64)
    basis = _FakeBasis(states=states)

    result = basis_configs_from_basis(basis)

    np.testing.assert_array_equal(result, states)


def test_basis_configs_from_basis_reads_encoded_basis() -> None:
    states = np.array([[0, 1], [1, 0]], dtype=np.int64)
    basis = _FakeEncodedBasis(states=states)

    result = basis_configs_from_basis(basis)

    np.testing.assert_array_equal(result, states)


def test_basis_configs_from_build_result_reads_basis_states() -> None:
    states = np.array([[0, 1], [1, 0]], dtype=np.int64)
    build_result = _FakeBuildResult(basis=_FakeBasis(states=states))

    result = basis_configs_from_build_result(build_result)

    np.testing.assert_array_equal(result, states)


def test_decode_basis_configs_with_layout() -> None:
    states = np.array([[0, 1], [1, 0]], dtype=np.int64)
    layout = _FakeLayout(n_variables=2)

    decoded = decode_basis_configs_with_layout(states, layout)

    np.testing.assert_array_equal(decoded, states)


def test_basis_configs_from_basis_rejects_missing_attribute() -> None:
    with pytest.raises(TypeError, match="Unsupported basis type"):
        basis_configs_from_basis(object())


def test_decode_basis_configs_with_layout_rejects_vector() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        decode_basis_configs_with_layout(
            np.array([0, 1], dtype=np.int64),
            _FakeLayout(),
        )


@dataclass(frozen=True)
class _FakeConfigsBasis:
    configs: np.ndarray


@dataclass(frozen=True)
class _FakeBadEncodedBasis:
    def to_array_basis(self) -> object:
        return object()


class _FakeFluxLocalSpace:
    values = np.array([-1, 1], dtype=np.int64)

    def contains(self, value: int) -> bool:
        return int(value) in (-1, 1)


class _FakeFluxVariable:
    local_space = _FakeFluxLocalSpace()


class _FakeFluxLayout(_FakeLayout):
    def __init__(self, n_variables: int = 2):
        self.variables = tuple(_FakeFluxVariable() for _ in range(n_variables))


def test_basis_configs_from_basis_reads_configs_attribute() -> None:
    configs = np.array([[0, 1], [1, 0]], dtype=np.int64)
    basis = _FakeConfigsBasis(configs=configs)

    result = basis_configs_from_basis(basis)

    np.testing.assert_array_equal(result, configs)


def test_basis_configs_from_basis_rejects_bad_encoded_basis() -> None:
    with pytest.raises(TypeError, match="Unsupported encoded basis type"):
        basis_configs_from_basis(_FakeBadEncodedBasis())


def test_basis_configs_from_basis_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="basis configurations"):
        basis_configs_from_basis(_FakeBasis(states=np.array([0, 1], dtype=np.int64)))


def test_basis_configs_from_build_result_without_layout_returns_basis_configs() -> None:
    states = np.array([[0, 1], [1, 0]], dtype=np.int64)
    build_result = _FakeBuildResult(basis=_FakeBasis(states=states))

    result = basis_configs_from_build_result(build_result)

    np.testing.assert_array_equal(result, states)


def test_decode_basis_configs_with_layout_decodes_integer_codes() -> None:
    codes = np.array([[0, 1], [1, 0]], dtype=np.int64)
    layout = _FakeFluxLayout(n_variables=2)

    decoded = decode_basis_configs_with_layout(codes, layout)

    np.testing.assert_array_equal(decoded, np.array([[-1, 1], [1, -1]], dtype=np.int64))


def test_decode_basis_configs_with_layout_rejects_incompatible_variable_count() -> None:
    with pytest.raises(ValueError, match="incompatible variable counts"):
        decode_basis_configs_with_layout(
            np.array([[0, 1, 0]], dtype=np.int64),
            _FakeLayout(n_variables=2),
        )


def test_decode_basis_configs_with_layout_rejects_invalid_codes() -> None:
    with pytest.raises(ValueError, match="neither valid physical values"):
        decode_basis_configs_with_layout(
            np.array([[0, 2]], dtype=np.int64),
            _FakeFluxLayout(n_variables=2),
        )
