import numpy as np
import pytest

from qlinks.basis import Basis
from qlinks.encoded import (
    BinaryEncodedBasis,
    bitmask_from_indices,
    decode_binary_code,
    encode_binary_config,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_encode_decode_binary_config() -> None:
    config = np.array([1, 0, 1, 1], dtype=np.int64)

    code = encode_binary_config(config)

    assert code == 13

    decoded = decode_binary_code(code, 4)

    np.testing.assert_array_equal(decoded, config)


def test_encode_rejects_non_binary_config() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        encode_binary_config(np.array([0, 2, 1]))


def test_bitmask_from_indices() -> None:
    mask = bitmask_from_indices([0, 2, 3])

    assert mask == 0b1101


def test_binary_encoded_basis_from_configs() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    configs = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )

    basis = BinaryEncodedBasis.from_configs(layout, configs)

    assert basis.n_states == 3
    assert basis.n_variables == 3

    assert basis.code(0) == 0
    assert basis.code(1) == 1
    assert basis.code(2) == 6

    assert basis.get_index(6) == 2
    assert basis.get_index(7) is None

    np.testing.assert_array_equal(basis.config(2), np.array([0, 1, 1]))


def test_binary_encoded_basis_from_array_basis() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    array_basis = Basis.from_states(
        layout,
        np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=np.int64,
        ),
    )

    encoded = BinaryEncodedBasis.from_basis(array_basis)

    assert encoded.n_states == 3
    assert encoded.code(0) == 0
    assert encoded.code(1) == 1
    assert encoded.code(2) == 2


def test_binary_encoded_basis_full() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    basis = BinaryEncodedBasis.full(layout)

    assert basis.n_states == 8
    assert basis.codes.tolist() == list(range(8))


def test_binary_encoded_basis_to_array_basis() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    encoded = BinaryEncodedBasis.from_codes(layout, [0, 5], sort=True)
    array_basis = encoded.to_array_basis()

    assert array_basis.n_states == 2
    np.testing.assert_array_equal(array_basis.states[0], np.array([0, 0, 0]))
    np.testing.assert_array_equal(array_basis.states[1], np.array([1, 0, 1]))


def test_reject_non_binary_layout() -> None:
    layout = VariableLayout.from_links(2, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match="requires every local space"):
        BinaryEncodedBasis.full(layout)


def test_reject_duplicate_codes() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="Duplicate"):
        BinaryEncodedBasis.from_codes(layout, [0, 0])
        