import pytest

from qlinks.lattice import Link, Plaquette, Site


def test_site_valid() -> None:
    site = Site(id=0, cell=(1, 2), sublattice=0, position=(1.0, 2.0))

    assert site.id == 0
    assert site.cell == (1, 2)
    assert site.position == (1.0, 2.0)


def test_site_rejects_negative_id() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Site(id=-1, cell=(0,))


def test_site_rejects_empty_cell() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Site(id=0, cell=())


def test_site_rejects_position_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        Site(id=0, cell=(0, 1), position=(0.0,))


def test_link_valid() -> None:
    link = Link(id=0, source=0, target=1, kind="x", wrap=False)

    assert link.id == 0
    assert link.source == 0
    assert link.target == 1
    assert link.kind == "x"
    assert not link.wrap


def test_link_rejects_self_link() -> None:
    with pytest.raises(ValueError, match="Self-links"):
        Link(id=0, source=1, target=1)


def test_link_rejects_negative_endpoint() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Link(id=0, source=-1, target=0)


def test_plaquette_valid() -> None:
    plaq = Plaquette(
        id=0,
        links=(0, 1, 2, 3),
        orientations=(1, 1, -1, -1),
        sites=(0, 1, 2, 3),
        kind="square",
    )

    assert plaq.id == 0
    assert plaq.links == (0, 1, 2, 3)
    assert plaq.orientations == (1, 1, -1, -1)
    assert plaq.kind == "square"


def test_plaquette_rejects_empty_links() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Plaquette(id=0, links=(), orientations=(), sites=(0, 1, 2))


def test_plaquette_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        Plaquette(id=0, links=(0, 1), orientations=(1,), sites=(0, 1, 2))


def test_plaquette_rejects_bad_orientation() -> None:
    with pytest.raises(ValueError, match="only contain"):
        Plaquette(id=0, links=(0, 1, 2), orientations=(1, 0, -1), sites=(0, 1, 2))


def test_plaquette_rejects_too_few_sites() -> None:
    with pytest.raises(ValueError, match="at least three"):
        Plaquette(id=0, links=(0, 1, 2), orientations=(1, 1, 1), sites=(0, 1))
