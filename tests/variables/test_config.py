import numpy as np
import pytest

from qlinks.variables import ConfigView, LocalSpace, VariableLayout


def test_config_view_site_access() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())
    cfg = ConfigView.from_array(layout, np.array([0, 1, 0]))

    assert cfg.site(0) == 0
    assert cfg.site(1) == 1
    assert cfg.site(2) == 0

    cfg.set_site(2, 1)
    assert cfg.site(2) == 1


def test_config_view_link_access() -> None:
    layout = VariableLayout.from_links(2, LocalSpace.spin_half_flux())
    cfg = ConfigView.from_array(layout, np.array([-1, 1]))

    assert cfg.link(0) == -1
    assert cfg.link(1) == 1

    cfg.set_link(0, 1)
    assert cfg.link(0) == 1


def test_config_view_mixed_access() -> None:
    layout = VariableLayout.from_sites_and_links(
        num_sites=2,
        site_space=LocalSpace.binary(),
        num_links=2,
        link_space=LocalSpace.spin_half_flux(),
    )

    cfg = ConfigView.from_array(layout, np.array([0, 1, -1, 1]))

    assert cfg.site(0) == 0
    assert cfg.site(1) == 1
    assert cfg.link(0) == -1
    assert cfg.link(1) == 1


def test_config_view_rejects_invalid_initial_array() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="not allowed"):
        ConfigView.from_array(layout, np.array([0, 2]))


def test_set_value_rejects_invalid_value() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    cfg = ConfigView.from_array(layout, np.array([0, 1]))

    with pytest.raises(ValueError, match="not allowed"):
        cfg.set_site(0, 2)


def test_copy_is_independent() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    cfg = ConfigView.from_array(layout, np.array([0, 1]))

    copied = cfg.copy()
    copied.set_site(0, 1)

    assert cfg.site(0) == 0
    assert copied.site(0) == 1


def test_flipped_binary_variable() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    cfg = ConfigView.from_array(layout, np.array([0, 1]))

    flipped0 = cfg.flipped(0)
    flipped1 = cfg.flipped(1)

    np.testing.assert_array_equal(flipped0.as_array(), np.array([1, 1]))
    np.testing.assert_array_equal(flipped1.as_array(), np.array([0, 0]))

    # Original is unchanged.
    np.testing.assert_array_equal(cfg.as_array(), np.array([0, 1]))


def test_flipped_rejects_non_binary_space() -> None:
    layout = VariableLayout.from_links(1, LocalSpace.spin_half_flux())
    cfg = ConfigView.from_array(layout, np.array([-1]))

    with pytest.raises(ValueError, match="binary"):
        cfg.flipped(0)


def test_negated_flux_variable() -> None:
    layout = VariableLayout.from_links(2, LocalSpace.spin_half_flux())
    cfg = ConfigView.from_array(layout, np.array([-1, 1]))

    negated0 = cfg.negated(0)
    negated1 = cfg.negated(1)

    np.testing.assert_array_equal(negated0.as_array(), np.array([1, 1]))
    np.testing.assert_array_equal(negated1.as_array(), np.array([-1, -1]))

    # Original is unchanged.
    np.testing.assert_array_equal(cfg.as_array(), np.array([-1, 1]))


def test_negated_rejects_if_negation_not_in_local_space() -> None:
    layout = VariableLayout.from_sites(1, LocalSpace.binary())
    cfg = ConfigView.from_array(layout, np.array([1]))

    with pytest.raises(ValueError, match="not allowed"):
        cfg.negated(0)


def test_as_array_copy_behavior() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    cfg = ConfigView.from_array(layout, np.array([0, 1]))

    copied = cfg.as_array(copy=True)
    copied[0] = 1
    assert cfg.site(0) == 0

    view = cfg.as_array(copy=False)
    view[0] = 1
    assert cfg.site(0) == 1
    