from dataclasses import dataclass

import numpy as np
import pytest

from qlinks.variables import LocalSpace, VariableKind, VariableLayout, VariableSpec


@dataclass(frozen=True)
class DummyLattice:
    num_sites: int
    num_links: int


def test_site_layout() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    assert layout.n_variables == 4
    assert len(layout) == 4
    assert layout.shape == (4,)

    assert layout.site_variable_index(0) == 0
    assert layout.site_variable_index(3) == 3

    np.testing.assert_array_equal(layout.site_variable_indices(), np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(layout.link_variable_indices(), np.array([], dtype=np.int64))


def test_link_layout() -> None:
    layout = VariableLayout.from_links(3, LocalSpace.spin_half_flux())

    assert layout.n_variables == 3
    assert layout.link_variable_index(0) == 0
    assert layout.link_variable_index(2) == 2

    np.testing.assert_array_equal(layout.site_variable_indices(), np.array([], dtype=np.int64))
    np.testing.assert_array_equal(layout.link_variable_indices(), np.array([0, 1, 2]))


def test_site_and_link_layout_ordering() -> None:
    layout = VariableLayout.from_sites_and_links(
        num_sites=2,
        site_space=LocalSpace.binary(),
        num_links=3,
        link_space=LocalSpace.spin_half_flux(),
    )

    assert layout.n_variables == 5

    assert layout.site_variable_index(0) == 0
    assert layout.site_variable_index(1) == 1

    assert layout.link_variable_index(0) == 2
    assert layout.link_variable_index(1) == 3
    assert layout.link_variable_index(2) == 4


def test_layout_from_lattice_helpers() -> None:
    lattice = DummyLattice(num_sites=2, num_links=4)

    site_layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())
    link_layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())
    mixed_layout = VariableLayout.from_lattice_sites_and_links(
        lattice,
        site_space=LocalSpace.binary(),
        link_space=LocalSpace.spin_half_flux(),
    )

    assert site_layout.n_variables == 2
    assert link_layout.n_variables == 4
    assert mixed_layout.n_variables == 6


def test_duplicate_variable_rejected() -> None:
    space = LocalSpace.binary()

    with pytest.raises(ValueError, match="Duplicate variable"):
        VariableLayout(
            (
                VariableSpec(VariableKind.SITE, 0, space),
                VariableSpec(VariableKind.SITE, 0, space),
            )
        )


def test_invalid_counts_rejected() -> None:
    with pytest.raises(ValueError, match="num_sites must be positive"):
        VariableLayout.from_sites(0, LocalSpace.binary())

    with pytest.raises(ValueError, match="num_links must be positive"):
        VariableLayout.from_links(0, LocalSpace.binary())

    with pytest.raises(ValueError, match="At least one"):
        VariableLayout.from_sites_and_links(0, LocalSpace.binary(), 0, LocalSpace.binary())


def test_default_config() -> None:
    layout = VariableLayout.from_sites_and_links(
        num_sites=2,
        site_space=LocalSpace.binary(),
        num_links=2,
        link_space=LocalSpace.spin_half_flux(),
    )

    np.testing.assert_array_equal(layout.default_config(), np.array([0, 0, -1, -1]))


def test_validate_config_accepts_valid_config() -> None:
    layout = VariableLayout.from_sites_and_links(
        num_sites=2,
        site_space=LocalSpace.binary(),
        num_links=2,
        link_space=LocalSpace.spin_half_flux(),
    )

    layout.validate_config(np.array([0, 1, -1, 1]))


def test_validate_config_rejects_wrong_shape() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    with pytest.raises(ValueError, match="Expected config shape"):
        layout.validate_config(np.array([0, 1]))


def test_validate_config_rejects_invalid_value() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    with pytest.raises(ValueError, match="not allowed"):
        layout.validate_config(np.array([0, 1, 2]))


def test_validate_batch() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    configs = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
        ]
    )

    layout.validate_batch(configs)

    with pytest.raises(ValueError, match="two-dimensional"):
        layout.validate_batch(np.array([0, 1, 0]))

    with pytest.raises(ValueError, match="Expected configs with 3 variables"):
        layout.validate_batch(np.array([[0, 1]]))

    with pytest.raises(ValueError, match="outside local space"):
        layout.validate_batch(np.array([[0, 2, 1]]))


def test_as_metadata() -> None:
    layout = VariableLayout.from_sites_and_links(
        num_sites=1,
        site_space=LocalSpace.binary(),
        num_links=1,
        link_space=LocalSpace.spin_half_flux(),
    )

    metadata = layout.as_metadata()

    assert metadata["n_variables"] == 2
    assert metadata["variables"][0] == {
        "kind": "site",
        "geometry_index": 0,
        "values": [0, 1],
    }
    assert metadata["variables"][1] == {
        "kind": "link",
        "geometry_index": 0,
        "values": [-1, 1],
    }


def test_missing_variable_raises_key_error() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(KeyError):
        layout.link_variable_index(0)


def test_out_of_range_variable_index() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(IndexError):
        layout.spec(2)
