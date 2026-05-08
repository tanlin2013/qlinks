import numpy as np
import pytest

from qlinks.variables import (
    BitPackedBinaryEncoder,
    ConfigEncoder,
    LocalSpace,
    VariableLayout,
)


def test_config_encoder_roundtrip_binary() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())
    encoder = ConfigEncoder(layout)

    config = np.array([0, 1, 1, 0])
    key = encoder.encode(config)
    decoded = encoder.decode(key)

    np.testing.assert_array_equal(decoded, config)


def test_config_encoder_roundtrip_flux() -> None:
    layout = VariableLayout.from_links(4, LocalSpace.spin_half_flux())
    encoder = ConfigEncoder(layout)

    config = np.array([-1, 1, 1, -1])
    key = encoder.encode(config)
    decoded = encoder.decode(key)

    np.testing.assert_array_equal(decoded, config)


def test_config_encoder_rejects_invalid_config() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    encoder = ConfigEncoder(layout)

    with pytest.raises(ValueError, match="not allowed"):
        encoder.encode(np.array([0, 1, 2]))


def test_config_encoder_build_index() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    encoder = ConfigEncoder(layout)

    configs = [
        np.array([0, 0, 0]),
        np.array([0, 1, 0]),
        np.array([1, 0, 1]),
    ]

    index = encoder.build_index(configs)

    assert index[encoder.encode(np.array([0, 0, 0]))] == 0
    assert index[encoder.encode(np.array([0, 1, 0]))] == 1
    assert index[encoder.encode(np.array([1, 0, 1]))] == 2


def test_config_encoder_build_index_rejects_duplicates() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    encoder = ConfigEncoder(layout)

    configs = [
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
    ]

    with pytest.raises(ValueError, match="Duplicate configuration"):
        encoder.build_index(configs)


def test_bit_packed_binary_encoder_roundtrip() -> None:
    layout = VariableLayout.from_sites(10, LocalSpace.binary())
    encoder = BitPackedBinaryEncoder(layout)

    config = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    key = encoder.encode(config)
    decoded = encoder.decode(key)

    np.testing.assert_array_equal(decoded, config)


def test_bit_packed_binary_encoder_is_more_compact_than_int64_encoder() -> None:
    layout = VariableLayout.from_sites(64, LocalSpace.binary())

    config_encoder = ConfigEncoder(layout)
    packed_encoder = BitPackedBinaryEncoder(layout)

    config = np.ones(64, dtype=np.int64)

    normal_key = config_encoder.encode(config)
    packed_key = packed_encoder.encode(config)

    assert len(packed_key) < len(normal_key)


def test_bit_packed_binary_encoder_rejects_flux_space() -> None:
    layout = VariableLayout.from_links(3, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match="requires every local space"):
        BitPackedBinaryEncoder(layout)


def test_bit_packed_binary_encoder_rejects_invalid_config() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    encoder = BitPackedBinaryEncoder(layout)

    with pytest.raises(ValueError, match="not allowed"):
        encoder.encode(np.array([0, 1, 2]))
