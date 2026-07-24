import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from qlinks.lattice import (
    HoneycombLattice,
    SquareLattice,
)
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
    LinkVisualStyle,
    basis_visual_style,
    plot_basis_config,
)

matplotlib.use("Agg")


def test_style_infers_site_label_fontsize_from_node_size() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    small = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=25.0),
    )

    large = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=400.0),
    )

    assert small._resolved_site_label_fontsize() < large._resolved_site_label_fontsize()


def test_style_explicit_site_label_fontsize_overrides_inference() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(
            node_size=25.0,
            site_label_fontsize=7.5,
        ),
    )

    assert visualizer._resolved_site_label_fontsize() == 7.5


def test_style_infers_arrow_parameters_from_node_size() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    small = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=25.0),
    )

    large = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(node_size=400.0),
    )

    assert small._resolved_arrow_mutation_scale() < large._resolved_arrow_mutation_scale()
    assert small._resolved_arrow_shrink_points() < large._resolved_arrow_shrink_points()


def test_style_explicit_arrow_parameters_override_inference() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        style=LinkVisualStyle(
            node_size=25.0,
            arrow_mutation_scale=9.0,
            arrow_shrink_points=0.0,
        ),
    )

    assert visualizer._resolved_arrow_mutation_scale() == 9.0
    assert visualizer._resolved_arrow_shrink_points() == 0.0


def test_honeycomb_site_label_includes_sublattice() -> None:
    lattice = HoneycombLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        site_label_style="sublattice_cell",
    )

    labels = {
        visualizer._format_site_label(site.id)
        for site in lattice.sites
        if tuple(site.cell) == (0, 0)
    }

    assert "A(0, 0)" in labels
    assert "B(0, 0)" in labels


def test_research_theme_preserves_legacy_style() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    visualizer = BasisConfigurationVisualizer(lattice=lattice)

    assert visualizer.theme == "research"
    assert visualizer.style == LinkVisualStyle()
    assert basis_visual_style("research") == LinkVisualStyle()


def test_paper_theme_resolves_publication_style() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        theme="paper",
    )

    assert visualizer.style == basis_visual_style("paper")
    assert visualizer.style != basis_visual_style("research")
    assert visualizer.style.node_color == "black"
    assert visualizer.style.occupied_width > visualizer.style.empty_width


def test_paper_theme_draws_hollow_lattice_sites() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    config = np.zeros(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()
    plot_basis_config(
        lattice=lattice,
        layout=layout,
        config=config,
        ax=ax,
        theme="paper",
        mode="dimers",
        with_plaquette_symbols=False,
        show=False,
    )

    node_collection = ax.collections[-1]
    assert tuple(node_collection.get_facecolors()[0]) == pytest.approx(to_rgba("white"))
    assert tuple(node_collection.get_edgecolors()[0]) == pytest.approx(to_rgba("black"))
    assert node_collection.get_linewidths()[0] == pytest.approx(0.9)

    plt.close(fig)


def test_explicit_style_overrides_paper_theme_style() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    custom_style = LinkVisualStyle(node_size=73.0, node_color="tab:green")

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        theme="paper",
        style=custom_style,
    )

    assert visualizer.style is custom_style


def test_invalid_basis_visualizer_theme_is_rejected() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    with pytest.raises(ValueError, match="research.*paper"):
        BasisConfigurationVisualizer(
            lattice=lattice,
            theme="presentation",  # type: ignore[arg-type]
        )


def test_paper_theme_omits_site_labels_by_default() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    config = np.zeros(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()
    plot_basis_config(
        lattice=lattice,
        layout=layout,
        config=config,
        ax=ax,
        theme="paper",
        mode="dimers",
        with_plaquette_symbols=False,
        show=False,
    )

    assert len(ax.texts) == 0
    plt.close(fig)


def test_research_theme_keeps_site_labels_by_default() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    config = np.zeros(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()
    plot_basis_config(
        lattice=lattice,
        layout=layout,
        config=config,
        ax=ax,
        theme="research",
        mode="dimers",
        with_plaquette_symbols=False,
        show=False,
    )

    assert len(ax.texts) >= lattice.num_sites
    plt.close(fig)
