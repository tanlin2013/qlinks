from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.linalg import ishermitian

from qlinks.exceptions import (
    InvalidArgumentError,
    InvalidOperationError,
    LinkOverridingError,
)
from qlinks.lattice.component import Site, UnitVectors
from qlinks.lattice.spin_object import (
    Link,
    Spin,
    SpinConfigs,
    SpinOperator,
    SpinOperators,
)
from qlinks.lattice.square_lattice import (
    LatticeMultiStates,
    LatticeState,
    Plaquette,
    SquareLattice,
    Vertex,
)


class TestSquareLattice:
    def test_shape(self):
        assert SquareLattice(2, 2).shape == (2, 2)
        with pytest.raises(InvalidArgumentError):
            _ = SquareLattice(1, 2).shape

    @pytest.fixture(scope="function")
    def lattice(self):
        return SquareLattice(4, 4)

    def test_get_item(self, lattice):
        assert lattice[0, 0] == Site(0, 0)
        assert lattice[0, 4] == Site(0, 0)  # assume periodic b.c.
        assert lattice[0, 5] == Site(0, 1)
        assert lattice[Site(4, 0)] == Site(0, 0)
        assert lattice[Site(5, 0)] == Site(1, 0)
        assert lattice[-1, 0] == Site(3, 0)

    def test_iter(self):
        lattice = SquareLattice(2, 2)
        it = iter(lattice)
        assert next(it) == Site(0, 0)
        assert next(it) == Site(1, 0)
        assert next(it) == Site(0, 1)
        assert next(it) == Site(1, 1)

    def test_iter_links(self):
        lattice = SquareLattice(2, 2)
        it = lattice.iter_links()
        assert next(it) == Link(Site(0, 0), UnitVectors.rightward)
        assert next(it) == Link(Site(0, 0), UnitVectors.upward)
        assert next(it) == Link(Site(1, 0), UnitVectors.rightward)

    def test_iter_plaquettes(self, lattice):
        lattice = SquareLattice(2, 2)
        it = lattice.iter_plaquettes()
        assert next(it).site == Site(0, 0)
        assert next(it).site == Site(1, 0)
        assert next(it).site == Site(0, 1)

    @pytest.mark.parametrize("shape, expected", [((2, 2), 2**8), ((4, 2), 2**16)])
    def test_hilbert_dims(self, shape, expected):
        assert SquareLattice(*shape).hilbert_dims[0] == expected

    @pytest.mark.parametrize("site", [Site(0, 0), Site(3, 3)])
    @pytest.mark.parametrize(
        "unit_vector", [UnitVectors.upward, UnitVectors.rightward, UnitVectors.downward]
    )
    def test_get_link(self, lattice, site, unit_vector):
        link = lattice.get_link((site, unit_vector))
        if unit_vector.sign > 0:
            assert link.site == site
            assert link.unit_vector == unit_vector
        else:
            assert link.site == lattice[site + unit_vector]
            assert link.unit_vector == -1 * unit_vector
        assert link.operator == SpinOperators.I2
        assert link.state is None

    def test_get_vertex_links(self):
        lattice = SquareLattice(2, 2)
        links = lattice.get_vertex_links(Site(1, 1))
        for idx, unit_vector in enumerate(UnitVectors.iter_all_directions()):
            if unit_vector.sign < 1:
                assert links[idx].site == Site(1, 1) + unit_vector
                assert links[idx].unit_vector == -1 * unit_vector
            else:
                assert links[idx].site == Site(1, 1)
                assert links[idx].unit_vector == unit_vector

    def test_set_vertex_links(self, lattice):
        """
              ???      ???
              ???      ???
        ??????????????????o??????????????????o??????????????????
              ???      ???
              ???      ???
        """
        lattice.set_vertex_links(
            Site(0, 0), (SpinConfigs.down, SpinConfigs.down, SpinConfigs.up, SpinConfigs.up)
        )
        with pytest.raises(InvalidArgumentError):
            lattice.set_vertex_links(Site(1, 0), (SpinConfigs.up, SpinConfigs.down))
        with pytest.raises(LinkOverridingError):
            lattice.set_vertex_links(
                Site(1, 0), (SpinConfigs.up, SpinConfigs.down, SpinConfigs.down, SpinConfigs.down)
            )

    def test_reset_vertex_links(self, lattice):
        """
              ???
              ???
        ??????????????????o??????????????????
              ???
              ???
        """
        assert np.isnan(lattice.charge(Site(0, 0)))
        lattice.set_vertex_links(
            Site(0, 0), (SpinConfigs.down, SpinConfigs.down, SpinConfigs.up, SpinConfigs.up)
        )
        assert not np.isnan(lattice.charge(Site(0, 0)))
        lattice.reset_vertex_links(Site(0, 0))
        assert np.isnan(lattice.charge(Site(0, 0)))

    @pytest.fixture(scope="function")
    def preset_lattice(self):
        """
           ???      ???
           ???      ???
        ?????????o??????????????????o?????????
           ???      ???
           ???      ???
        ?????????o??????????????????o?????????
           ???      ???
           ???      ???
        """
        lattice = SquareLattice(2, 2)
        lattice.set_vertex_links(
            Site(0, 0), (SpinConfigs.down, SpinConfigs.up, SpinConfigs.down, SpinConfigs.up)
        )
        lattice.set_vertex_links(
            Site(1, 0), (SpinConfigs.down, SpinConfigs.down, SpinConfigs.up, SpinConfigs.up)
        )
        lattice.set_vertex_links(
            Site(0, 1), (SpinConfigs.up, SpinConfigs.up, SpinConfigs.down, SpinConfigs.down)
        )
        return lattice

    def test_charge(self, lattice, preset_lattice):
        assert np.isnan(lattice.charge(Site(0, 0)))
        assert preset_lattice.charge(Site(0, 0)) == 0
        assert preset_lattice.charge(Site(1, 0)) == 2
        assert preset_lattice.charge(Site(0, 1)) == -2
        assert preset_lattice.charge(Site(1, 1)) == 0

    def test_axial_flux(self, lattice, preset_lattice):
        with pytest.raises(ValueError):
            _ = lattice.axial_flux(0, axis=0)
        assert preset_lattice.axial_flux(0, axis=0) == 0.5 * -2
        assert preset_lattice.axial_flux(1, axis=0) == 0.5 * 2
        assert preset_lattice.axial_flux(0, axis=1) == 0.5 * 2
        assert preset_lattice.axial_flux(1, axis=1) == 0.5 * -2


class TestLatticeState:
    @pytest.fixture(scope="function")
    def state_data(self):
        """
           ???      ???           ???     ???            ???      ???           ???      ???
           ???      ???           ???     ???            ???      ???           ???      ???
        ?????????o??????????????????o?????????     ?????????o??????????????????o?????????     ?????????o??????????????????o?????????     ?????????o??????????????????o?????????
           ???      ???           ???     ???            ???      ???           ???      ???
           ???      ???           ???     ???            ???      ???           ???      ???
        ?????????o??????????????????o?????????     ?????????o??????????????????o?????????     ?????????o??????????????????o?????????     ?????????o??????????????????o?????????
           ???      ???           ???     ???            ???      ???           ???      ???
           ???      ???           ???     ???            ???      ???           ???      ???
        """
        return [
            (SpinConfigs.up, SpinConfigs.up),
            (SpinConfigs.up, SpinConfigs.down),
            (SpinConfigs.down, SpinConfigs.up),
            (SpinConfigs.down, SpinConfigs.down),
        ]

    @pytest.fixture(scope="function")
    def states(self, state_data):
        states = []
        for state in state_data:
            lattice = SquareLattice(2, 2)
            for link in lattice.iter_links():
                if link.unit_vector == UnitVectors.rightward:
                    lattice.set_link(link.index, state[0])
                elif link.unit_vector == UnitVectors.upward:
                    lattice.set_link(link.index, state[1])
            states.append(LatticeState(*lattice.shape, link_data=lattice.links))
        return states

    def test_comparison(self, states):
        assert states[0] > states[1]
        assert states[1] > states[2]
        assert states[2] > states[3]
        assert states[3] < states[0]
        assert states[3] != states[0]
        assert states[0] == deepcopy(states[0])

    def test_matrix_multiplication(self, states):
        assert states[0].T @ states[0] == 1
        assert states[1].T @ states[1] == 1
        assert states[2].T @ states[2] == 1
        assert states[3].T @ states[3] == 1
        assert states[1].T @ states[2] == 0
        assert states[2].T @ states[1] == 0
        assert states[0].T @ states[3] == 0

    def test_transpose(self, states):
        for state in states:
            for link in state.links.values():
                assert link.state.shape == (2, 1)
            for link in state.T.links.values():
                assert link.state.shape == (1, 2)


@pytest.fixture(scope="class")
def clockwise_state():
    """
       ???      ???
       ???      ???
    ?????????o??????????????????o?????????
       ???      ???
       ???      ???
    ?????????o??????????????????o?????????
       ???      ???
       ???      ???
    """
    lattice = SquareLattice(2, 2)
    lattice.set_vertex_links(
        Site(0, 0), (SpinConfigs.up, SpinConfigs.down, SpinConfigs.down, SpinConfigs.up)
    )
    lattice.set_vertex_links(
        Site(1, 0), (SpinConfigs.down, SpinConfigs.down, SpinConfigs.down, SpinConfigs.down)
    )
    lattice.set_vertex_links(
        Site(0, 1), (SpinConfigs.up, SpinConfigs.up, SpinConfigs.up, SpinConfigs.up)
    )
    lattice.set_vertex_links(
        Site(1, 1), (SpinConfigs.down, SpinConfigs.up, SpinConfigs.up, SpinConfigs.down)
    )
    return LatticeState(*lattice.shape, link_data=lattice.links)


@pytest.fixture(scope="class")
def anti_clockwise_state():
    """
       ???      ???
       ???      ???
    ?????????o??????????????????o?????????
       ???      ???
       ???      ???
    ?????????o??????????????????o?????????
       ???      ???
       ???      ???
    """
    lattice = SquareLattice(2, 2)
    lattice.set_vertex_links(
        Site(0, 0), (SpinConfigs.up, SpinConfigs.down, SpinConfigs.up, SpinConfigs.down)
    )
    lattice.set_vertex_links(
        Site(1, 0), (SpinConfigs.down, SpinConfigs.up, SpinConfigs.down, SpinConfigs.up)
    )
    lattice.set_vertex_links(
        Site(0, 1), (SpinConfigs.down, SpinConfigs.up, SpinConfigs.down, SpinConfigs.up)
    )
    lattice.set_vertex_links(
        Site(1, 1), (SpinConfigs.up, SpinConfigs.down, SpinConfigs.up, SpinConfigs.down)
    )
    return LatticeState(*lattice.shape, link_data=lattice.links)


@pytest.fixture(scope="class")
def zero_clock_state():
    """
       ???      ???
       ???   0  ???
    ?????????o??????????????????o?????????
       ???      ???
      0???      ???0
    ?????????o??????????????????o?????????
       ???   0  ???
       ???      ???
    """
    lattice = SquareLattice(2, 2)
    spin_zero = Spin([[0], [0]], dtype=float, read_only=True)
    lattice.set_vertex_links(Site(0, 0), (SpinConfigs.up, SpinConfigs.down, spin_zero, spin_zero))
    lattice.set_vertex_links(Site(1, 0), (SpinConfigs.down, spin_zero, SpinConfigs.down, spin_zero))
    lattice.set_vertex_links(Site(0, 1), (spin_zero, SpinConfigs.up, spin_zero, SpinConfigs.up))
    lattice.set_vertex_links(Site(1, 1), (spin_zero, spin_zero, SpinConfigs.up, SpinConfigs.down))
    return LatticeState(*lattice.shape, link_data=lattice.links)


class TestLatticeMultiStates:
    @pytest.fixture(scope="function")
    def lattice(self):
        return SquareLattice(2, 2)

    @pytest.fixture(scope="function")
    def plaquette(self, lattice):
        return Plaquette(lattice, Site(0, 0))

    @pytest.fixture(scope="function")
    def states(self, lattice, clockwise_state, anti_clockwise_state, zero_clock_state):
        return [
            LatticeMultiStates(*lattice.shape, states=[clockwise_state, clockwise_state]),
            LatticeMultiStates(*lattice.shape, states=[clockwise_state, anti_clockwise_state]),
            LatticeMultiStates(*lattice.shape, states=[anti_clockwise_state, clockwise_state]),
            LatticeMultiStates(*lattice.shape, states=[anti_clockwise_state, anti_clockwise_state]),
            LatticeMultiStates(*lattice.shape, states=[anti_clockwise_state, zero_clock_state]),
            LatticeMultiStates(*lattice.shape, states=[clockwise_state, zero_clock_state]),
        ]

    def test_matrix_multiplication(self, plaquette, states):
        assert plaquette @ states[0] == states[3]
        assert plaquette.conj() @ states[3] == states[0]
        assert plaquette @ states[1] == states[4]
        assert plaquette.conj() @ states[2] == states[5]
        assert plaquette.conj() * plaquette @ states[0] == states[0]

    def test_inner_product(self, plaquette, states):
        assert states[0].T @ states[3] == SpinOperators.O2
        assert states[1].T @ states[1] == SpinOperators.I2
        assert states[2].T @ states[1] == SpinOperator([[0, 1], [1, 0]])
        assert states[3].T @ states[1] == SpinOperator([[0, 1], [0, 1]])
        assert states[3].T @ plaquette @ states[0] == SpinOperator([[1, 1], [1, 1]])
        assert (states[3].T @ states[1]).dtype == np.float64
        with pytest.raises(ValueError):
            _ = states[1] @ states[1]


class TestPlaquette:
    @pytest.fixture(scope="function")
    def lattice(self):
        return SquareLattice(2, 2)

    @pytest.fixture(scope="function")
    def plaquette(self, lattice):
        return Plaquette(lattice, Site(0, 0))

    def test_iter(self, plaquette):
        it = iter(plaquette)
        assert next(it) == plaquette.link_d
        assert next(it) == plaquette.link_l
        assert next(it) == plaquette.link_r
        assert next(it) == plaquette.link_t

    def test_array(self, lattice, plaquette, clockwise_state, anti_clockwise_state):
        plaquette_arr = plaquette.toarray()
        assert np.allclose(plaquette_arr, np.triu(plaquette_arr), atol=1e-12)
        assert plaquette_arr.shape == lattice.hilbert_dims
        unique_value_counts = dict(zip(*np.unique(plaquette_arr.astype(float), return_counts=True)))
        assert tuple(unique_value_counts) == (0, 1)  # dict keys to tuple
        assert unique_value_counts[1] == 2 ** (lattice.num_links - 4)  # count non-zero elems
        assert np.allclose(
            plaquette_arr @ clockwise_state.toarray(), anti_clockwise_state.toarray(), atol=1e-12
        )
        assert np.allclose(
            plaquette.conj().toarray() @ anti_clockwise_state.toarray(),
            clockwise_state.toarray(),
            atol=1e-12,
        )
        plt.matshow(plaquette_arr)
        plt.colorbar()
        plt.show()

    def test_conj(self, plaquette):
        conj_plaquette_arr = plaquette.conj().toarray()
        assert np.allclose(conj_plaquette_arr, np.tril(conj_plaquette_arr), atol=1e-12)
        flipper = plaquette + plaquette.conj()
        assert ishermitian(flipper)
        assert np.all(
            np.linalg.eigvals(np.linalg.matrix_power(flipper, 2)) >= 0
        )  # positive semi-definite

    def test_addition(self, lattice, plaquette):
        assert isinstance(plaquette + plaquette.conj(), SpinOperator)
        with pytest.raises(InvalidOperationError):
            _ = Plaquette(lattice, Site(0, 0)) + Plaquette(lattice, Site(0, 1))

    def test_multiplication(self, plaquette, clockwise_state):
        opt = plaquette.conj() * plaquette
        assert opt.link_d.operator == SpinOperators.Sm @ SpinOperators.Sp
        assert opt.link_r.operator == SpinOperators.Sm @ SpinOperators.Sp
        assert opt.link_t.operator == SpinOperators.Sp @ SpinOperators.Sm
        assert opt.link_l.operator == SpinOperators.Sp @ SpinOperators.Sm
        opt2 = plaquette * plaquette.conj()
        assert opt2.link_d.operator == SpinOperators.Sp @ SpinOperators.Sm
        assert opt2.link_r.operator == SpinOperators.Sp @ SpinOperators.Sm
        assert opt2.link_t.operator == SpinOperators.Sm @ SpinOperators.Sp
        assert opt2.link_l.operator == SpinOperators.Sm @ SpinOperators.Sp
        for link in plaquette * plaquette:
            assert link.operator == SpinOperators.O2
        for link in plaquette.conj() * plaquette.conj():
            assert link.operator == SpinOperators.O2
        with pytest.raises(TypeError):
            _ = plaquette * clockwise_state

    def test_matrix_multiplication(self, lattice, plaquette, clockwise_state, anti_clockwise_state):
        """
        - Invalid state: 2 links become zero state.

           ???      ???
           ???   0  ???
        ?????????o??????????????????o?????????
           ???      ???
           ???      ???
        ?????????o??????????????????o?????????
           ???   0  ???
           ???      ???
        """
        assert plaquette @ clockwise_state == anti_clockwise_state
        assert plaquette.conj() @ anti_clockwise_state == clockwise_state
        assert clockwise_state.T @ plaquette @ clockwise_state == 0
        assert anti_clockwise_state.T @ plaquette @ anti_clockwise_state == 0
        assert anti_clockwise_state.T @ plaquette @ clockwise_state == 1
        invalid_state = Plaquette(lattice, Site(0, 1)) @ clockwise_state
        assert np.isnan(invalid_state.charge(Site(0, 0)))
        assert np.isnan(invalid_state.charge(Site(0, 1)))
        with pytest.raises(TypeError):
            _ = plaquette.conj() @ plaquette


class TestVertex:
    @pytest.fixture(scope="class")
    def lattice(self):
        return SquareLattice(2, 2)

    @pytest.fixture(scope="class")
    def vertex(self, lattice):
        return Vertex(lattice, Site(1, 1))
