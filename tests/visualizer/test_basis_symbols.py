import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qlinks.lattice import (
    HoneycombLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.visualizer import (
    BasisConfigurationVisualizer,
    plot_basis_grid,
)
from qlinks.visualizer.basis import _SQUARE_QLM_PLAQUETTE_SYMBOLS

matplotlib.use("Agg")


def test_square_qlm_symbols_draw_one_symbol_per_open_square_plaquette() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())
    config = np.array([1, -1, 1, -1], dtype=np.int64)

    fig, ax = plt.subplots()
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="square_qlm",
    )

    assert len(ax.texts) == lattice.num_plaquettes

    plt.close(fig)


def test_circulation_symbols_run_on_square() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    states = np.array(
        [
            [1, 1, 1, 1],
        ],
        dtype=np.int64,
    )

    fig, axes = plot_basis_grid(
        lattice=lattice,
        layout=layout,
        states=states,
        mode="arrows",
        plaquette_symbols="circulation",
        show=False,
    )

    assert axes.shape == (1, 1)

    plt.close(fig)


def test_square_2_by_2_positive_patch_draws_four_plaquette_visual_cells() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquettes = visualizer._draw_square_qlm_plaquette_primitives()

    visual_cells = {p.visual_cell for p in draw_plaquettes}

    assert {(0, 0), (1, 0), (0, 1), (1, 1)} <= visual_cells


def test_square_qlm_visual_symbol_key_accepts_circulating_open_plaquette() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.zeros(lattice.num_links, dtype=np.int64)
    plaquette = lattice.plaquettes[0]

    for link_id, orientation in zip(
        plaquette.links,
        plaquette.orientations,
        strict=True,
    ):
        config[int(link_id)] = int(orientation)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    assert visualizer._plaquette_key(values) in _SQUARE_QLM_PLAQUETTE_SYMBOLS


def test_square_qlm_visual_symbol_key_accepts_anticirculating_open_plaquette() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.zeros(lattice.num_links, dtype=np.int64)
    plaquette = lattice.plaquettes[0]

    for link_id, orientation in zip(
        plaquette.links,
        plaquette.orientations,
        strict=True,
    ):
        config[int(link_id)] = -int(orientation)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    assert visualizer._plaquette_key(values) in _SQUARE_QLM_PLAQUETTE_SYMBOLS


def test_square_qlm_visual_symbol_values_2x2_pbc_use_visual_cell() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.array([1, -1, -1, 1, 1, 1, -1, -1], dtype=np.int64)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    assert visualizer._plaquette_key(values) in _SQUARE_QLM_PLAQUETTE_SYMBOLS


def test_square_qlm_visual_symbol_key_order_bottom_left_right_top() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=None)

    config = np.array([1, -1, -1, 1, 1, 1, -1, -1], dtype=np.int64)

    values = visualizer._square_visual_qlm_symbol_link_values(
        config,
        visual_cell=(0, 0),
    )

    # bottom, left, right, top for visual cell (0, 0)
    assert values == [1, -1, 1, -1]


def test_honeycomb_drawn_plaquettes_carry_link_metadata() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    assert draw_plaquettes

    for draw_plaquette in draw_plaquettes:
        assert len(draw_plaquette.link_ids) == 6
        assert len(draw_plaquette.link_orientations) == 6
        assert all(isinstance(link_id, int) for link_id in draw_plaquette.link_ids)
        assert set(draw_plaquette.link_orientations) <= {-1, 1}


def test_triangular_circulation_primitives_skip_triangles() -> None:
    lattice = TriangularLattice(
        3,
        3,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    assert draw_plaquettes

    for draw_plaquette in draw_plaquettes:
        plaquette = lattice.plaquettes[draw_plaquette.plaquette_id]
        assert len(plaquette.links) == 4


def test_generic_circulation_primitives_are_closed_cycles() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    assert draw_plaquettes

    _nodes, draw_links = visualizer._draw_primitives()
    links_by_id = {}
    for draw_link in draw_links:
        links_by_id.setdefault(draw_link.link_id, []).append(draw_link)

    for draw_plaquette in draw_plaquettes:
        selected = visualizer._select_closed_visual_plaquette(
            [links_by_id[link_id] for link_id in draw_plaquette.link_ids],
            physical_link_ids=tuple(int(link_id) for link_id in draw_plaquette.link_ids),
            preferred_center=None,
        )

        assert selected is not None
        assert visualizer._draw_links_form_closed_cycle(selected)


def test_qdm_resonance_symbol_uses_diamond_markers() -> None:
    assert BasisConfigurationVisualizer._qdm_resonance_symbol([1, 0, 1, 0]) == (
        "◆",
        "blue",
    )
    assert BasisConfigurationVisualizer._qdm_resonance_symbol([0, 1, 0, 1]) == (
        "◇",
        "red",
    )
    assert BasisConfigurationVisualizer._qdm_resonance_symbol([1, 1, 0, 0]) is None
    assert BasisConfigurationVisualizer._qdm_resonance_symbol([0, 0, 0, 0]) is None


def test_triangular_qdm_canonical_visual_order_is_stable_for_horizontal_dimers() -> None:
    lattice = TriangularLattice(
        4,
        4,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = [
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 4
    ]

    assert draw_plaquettes

    for draw_plaquette in draw_plaquettes:
        # Canonical visual order should always contain the same physical links
        # as the abstract plaquette, only reordered.
        plaquette = lattice.plaquettes[draw_plaquette.plaquette_id]
        assert set(draw_plaquette.link_ids) == {int(link_id) for link_id in plaquette.links}


def test_qdm_circulation_style_draws_diamond_symbol_on_honeycomb() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 6
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)
    for i, link_id in enumerate(draw_plaquette.link_ids):
        config[int(link_id)] = 1 if i % 2 == 0 else 0

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    symbols = {text.get_text() for text in ax.texts}
    assert "◆" in symbols
    assert "↺" not in symbols
    assert "↻" not in symbols

    plt.close(fig)


def test_circulation_style_does_not_mark_binary_qdm_resonance() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 6
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)
    for i, link_id in enumerate(draw_plaquette.link_ids):
        config[int(link_id)] = 1 if i % 2 == 0 else 0

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="circulation",
    )

    assert not any(text.get_text() in {"◆", "◇"} for text in ax.texts)
    assert not any(text.get_text() in {"↺", "↻"} for text in ax.texts)

    plt.close(fig)


def test_resonance_style_marks_binary_qdm_resonance() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 6
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)
    for i, link_id in enumerate(draw_plaquette.link_ids):
        config[int(link_id)] = 1 if i % 2 == 0 else 0

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    symbols = {text.get_text() for text in ax.texts}
    assert symbols & {"◆", "◇"}
    assert "↺" not in symbols
    assert "↻" not in symbols

    plt.close(fig)


def test_qdm_one_vulnerable_link_points_to_blue_resonance() -> None:
    result = BasisConfigurationVisualizer._qdm_one_vulnerable_link([1, 0, 1, 1])

    assert result == (3, "skyblue")


def test_qdm_one_vulnerable_link_points_to_red_resonance() -> None:
    result = BasisConfigurationVisualizer._qdm_one_vulnerable_link([0, 1, 0, 0])

    assert result == (3, "salmon")


def test_qdm_one_vulnerable_link_skips_already_resonant() -> None:
    assert BasisConfigurationVisualizer._qdm_one_vulnerable_link([1, 0, 1, 0]) is None


def test_qdm_one_vulnerable_link_skips_multiple_or_none() -> None:
    assert BasisConfigurationVisualizer._qdm_one_vulnerable_link([1, 1, 0, 0]) is None


def test_flux_one_vulnerable_link_points_to_blue_circulation() -> None:
    result = BasisConfigurationVisualizer._flux_one_vulnerable_link(
        [1, 1, 1, -1],
        [1, 1, 1, 1],
    )

    assert result == (3, "skyblue")


def test_flux_one_vulnerable_link_points_to_red_circulation() -> None:
    result = BasisConfigurationVisualizer._flux_one_vulnerable_link(
        [-1, -1, -1, 1],
        [1, 1, 1, 1],
    )

    assert result == (3, "salmon")


def test_flux_one_vulnerable_link_skips_already_circulating() -> None:
    assert (
        BasisConfigurationVisualizer._flux_one_vulnerable_link(
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        )
        is None
    )


def test_resonance_style_draws_one_vulnerable_link_arrow() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 6
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)

    # Nearly resonant: 101011. Flipping the last link gives 101010.
    values = [1, 0, 1, 0, 1, 1]

    for link_id, value in zip(draw_plaquette.link_ids, values, strict=True):
        config[int(link_id)] = value

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    # Vulnerable-link arrow is a Matplotlib patch from annotate(... arrowprops).
    assert len(ax.patches) >= 1

    # It should not draw the full resonance diamond.
    assert not any(text.get_text() in {"◆", "◇"} for text in ax.texts)

    plt.close(fig)


def test_circulation_style_draws_one_vulnerable_link_arrow() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(lattice=lattice, layout=layout)

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 6
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    # Make the plaquette one sign flip away from positive circulation.
    for link_id, orientation in zip(
        draw_plaquette.link_ids,
        draw_plaquette.link_orientations,
        strict=True,
    ):
        config[int(link_id)] = int(orientation)

    vulnerable_link = int(draw_plaquette.link_ids[-1])
    config[vulnerable_link] *= -1

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="circulation",
    )

    # Link arrows plus vulnerable-link arrow are patches. We only need to know
    # the vulnerable arrow exists and no full circulation text was drawn.
    assert len(ax.patches) > lattice.num_links
    assert not any(text.get_text() in {"↺", "↻"} for text in ax.texts)

    plt.close(fig)


def test_generic_plaquette_primitives_exist_on_small_square_torus() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    assert {int(draw_plaquette.plaquette_id) for draw_plaquette in draw_plaquettes} == {
        int(plaquette.id) for plaquette in lattice.plaquettes if len(plaquette.links) == 4
    }


def test_generic_circulation_symbols_draw_on_small_square_torus() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 4
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    for link_id, orientation in zip(
        draw_plaquette.link_ids,
        draw_plaquette.link_orientations,
        strict=True,
    ):
        config[int(link_id)] = int(orientation)

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="circulation",
    )

    assert any(text.get_text() in {"↺", "↻"} for text in ax.texts)

    plt.close(fig)


def test_generic_resonance_symbols_draw_on_small_square_torus() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 4
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)

    for i, link_id in enumerate(draw_plaquette.link_ids):
        config[int(link_id)] = 1 if i % 2 == 0 else 0

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    assert any(text.get_text() in {"◆", "◇"} for text in ax.texts)

    plt.close(fig)


def test_honeycomb_resonance_symbols_draw_on_small_torus() -> None:
    lattice = HoneycombLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquettes = [
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 6
    ]

    assert draw_plaquettes

    draw_plaquette = draw_plaquettes[0]

    config = np.zeros(layout.n_variables, dtype=np.int64)
    for i, link_id in enumerate(draw_plaquette.link_ids):
        config[int(link_id)] = 1 if i % 2 == 0 else 0

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    assert any(text.get_text() in {"◆", "◇"} for text in ax.texts)

    plt.close(fig)


def test_generic_resonance_vulnerable_arrow_draws_on_small_square_torus() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 4
    )

    assert len(draw_plaquette.link_midpoints) == len(draw_plaquette.link_ids)

    config = np.zeros(layout.n_variables, dtype=np.int64)

    # One flip from 1010.
    values = [1, 0, 1, 1]

    for link_id, value in zip(draw_plaquette.link_ids, values, strict=True):
        config[int(link_id)] = value

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    # Vulnerable-link arrow is drawn as FancyArrowPatch.
    assert len(ax.patches) >= 1
    assert not any(text.get_text() in {"◆", "◇"} for text in ax.texts)

    plt.close(fig)


def test_generic_circulation_vulnerable_arrow_draws_on_small_square_torus() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 4
    )

    assert len(draw_plaquette.link_midpoints) == len(draw_plaquette.link_ids)

    config = np.ones(layout.n_variables, dtype=np.int64)

    # Start from positive circulation.
    for link_id, orientation in zip(
        draw_plaquette.link_ids,
        draw_plaquette.link_orientations,
        strict=True,
    ):
        config[int(link_id)] = int(orientation)

    # Break exactly one link.
    vulnerable_link = int(draw_plaquette.link_ids[-1])
    config[vulnerable_link] *= -1

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="circulation",
    )

    assert len(ax.patches) > lattice.num_links
    assert not any(text.get_text() in {"↺", "↻"} for text in ax.texts)

    plt.close(fig)


def test_honeycomb_vulnerable_resonance_arrow_draws_on_small_torus() -> None:
    lattice = HoneycombLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquettes = [
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 6
    ]

    assert draw_plaquettes

    draw_plaquette = draw_plaquettes[0]
    assert len(draw_plaquette.link_midpoints) == 6

    config = np.zeros(layout.n_variables, dtype=np.int64)

    # One flip from 101010.
    values = [1, 0, 1, 0, 1, 1]

    for link_id, value in zip(draw_plaquette.link_ids, values, strict=True):
        config[int(link_id)] = value

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    assert len(ax.patches) >= 1
    assert not any(text.get_text() in {"◆", "◇"} for text in ax.texts)

    plt.close(fig)


def test_generic_plaquette_primitives_have_midpoints_on_small_torus() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    assert draw_plaquettes

    for draw_plaquette in draw_plaquettes:
        assert len(draw_plaquette.link_midpoints) == len(draw_plaquette.link_ids)
        assert len(draw_plaquette.link_orientations) == len(draw_plaquette.link_ids)


def test_square_generic_plaquette_primitives_use_local_visual_links() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    for draw_plaquette in visualizer._draw_plaquette_primitives():
        x, y = draw_plaquette.visual_cell

        assert draw_plaquette.link_ids == (
            visualizer._square_visual_link_id(cell=(x, y), kind="x"),
            visualizer._square_visual_link_id(cell=(x + 1, y), kind="y"),
            visualizer._square_visual_link_id(cell=(x, y + 1), kind="x"),
            visualizer._square_visual_link_id(cell=(x, y), kind="y"),
        )
        assert draw_plaquette.link_orientations == (1, 1, -1, -1)
        assert len(draw_plaquette.link_midpoints) == 4


def test_small_square_torus_generic_plaquette_centers_are_in_positive_patch() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = [
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 4
    ]

    assert draw_plaquettes

    for draw_plaquette in draw_plaquettes:
        assert 0.0 <= float(draw_plaquette.center[1]) <= 2.0


def test_small_square_torus_generic_plaquette_primitives_keep_top_row() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = [
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if len(draw_plaquette.link_ids) == 4
    ]

    y_values = sorted(
        {round(float(draw_plaquette.center[1]), 6) for draw_plaquette in draw_plaquettes}
    )

    assert y_values == [0.5, 1.5]


def test_small_square_torus_generic_plaquette_primitives_draw_all_plaquettes() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    drawn_ids = {
        int(draw_plaquette.plaquette_id)
        for draw_plaquette in visualizer._draw_plaquette_primitives()
    }

    expected_ids = {
        int(plaquette.id) for plaquette in lattice.plaquettes if len(plaquette.links) == 4
    }

    assert drawn_ids == expected_ids


def test_small_square_torus_top_row_local_primitive_is_local() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    top = [
        draw_plaquette
        for draw_plaquette in draw_plaquettes
        if round(float(draw_plaquette.center[1]), 6) == 1.5
    ]

    assert top

    for draw_plaquette in top:
        assert len(draw_plaquette.link_ids) == 4
        assert len(set(draw_plaquette.link_ids)) == 4
        assert len(draw_plaquette.link_midpoints) == 4

        center = np.asarray(draw_plaquette.center, dtype=float)

        distances = [
            float(np.linalg.norm(np.asarray(midpoint, dtype=float) - center))
            for midpoint in draw_plaquette.link_midpoints
        ]

        assert max(distances) < 1.1


def test_small_square_torus_resonance_symbol_draws_on_top_row() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    top_plaquette = next(
        draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
        if round(float(draw_plaquette.center[1]), 6) == 1.5
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)

    for index, link_id in enumerate(top_plaquette.link_ids):
        config[int(link_id)] = 1 if index % 2 == 0 else 0

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
    )

    top_symbols = [
        text
        for text in ax.texts
        if text.get_text() in {"◆", "◇"}
        and float(getattr(text, "xy", text.get_position())[1]) > 1.0
    ]

    assert top_symbols

    plt.close(fig)


def test_small_square_torus_distinct_physical_plaquettes_have_distinct_centers() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    by_id = {
        int(draw_plaquette.plaquette_id): draw_plaquette
        for draw_plaquette in visualizer._draw_plaquette_primitives()
    }

    assert 2 in by_id
    assert 3 in by_id

    assert by_id[2].center != by_id[3].center


def test_square_generic_plaquette_primitives_use_positive_patch_cells() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_plaquettes = visualizer._draw_plaquette_primitives()

    centers = sorted(
        (round(float(p.center[0]), 6), round(float(p.center[1]), 6)) for p in draw_plaquettes
    )

    assert centers == [
        (0.5, 0.5),
        (0.5, 1.5),
        (1.5, 0.5),
        (1.5, 1.5),
        (2.5, 0.5),
        (2.5, 1.5),
        (3.5, 0.5),
        (3.5, 1.5),
    ]


def test_explicit_square_resonance_symbols_draw_all_requested_ids() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
        periodic_image_mode="positive_patch",
        collapse_duplicate_visual_links=True,
    )

    draw_nodes, draw_links = visualizer._draw_primitives()
    draw_plaquettes = visualizer._draw_plaquette_primitives()

    config = np.zeros(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer._plot_with_primitives(
        config,
        ax=ax,
        draw_nodes=draw_nodes,
        draw_links=draw_links,
        draw_plaquettes=draw_plaquettes,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=True,
        plaquette_symbol_style="resonance",
        plaquette_symbol_values={
            2: ("◆", "blue"),
            3: ("◆", "blue"),
            6: ("◆", "blue"),
            7: ("◆", "blue"),
        },
        title=None,
    )

    symbols = [text for text in ax.texts if text.get_text() == "◆"]

    assert len(symbols) == 4

    xy_positions = {
        tuple(float(v) for v in getattr(text, "xy", text.get_position())) for text in symbols
    }

    assert len(xy_positions) == 4

    plt.close(fig)


def test_square_qlm_symbol_table_is_used_by_circulation_style() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="arrows",
        plaquette_symbol_style="circulation",
        with_site_labels=False,
    )

    symbols = {text.get_text() for text in ax.texts}

    assert "◩" in symbols

    plt.close(fig)


def test_plot_can_overlay_link_ids() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_link_ids=True,
        with_plaquette_symbols=False,
    )

    link_id_texts = {text.get_text() for text in ax.texts}

    assert {str(link_id) for link_id in range(lattice.num_links)} <= link_id_texts

    plt.close(fig)


def test_plot_does_not_overlay_link_ids_by_default() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    visualizer = BasisConfigurationVisualizer(
        lattice=lattice,
        layout=layout,
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)

    fig, ax = plt.subplots()

    visualizer.plot(
        config,
        ax=ax,
        show=False,
        mode="dimers",
        with_site_labels=False,
        with_plaquette_symbols=False,
    )

    link_id_texts = {text.get_text() for text in ax.texts if text.get_text().isdigit()}

    assert not link_id_texts

    plt.close(fig)
