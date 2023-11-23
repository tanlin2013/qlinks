from contextlib import nullcontext as does_not_raise

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import binom

from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.component import Site
from qlinks.solver.deep_first_search import DeepFirstSearch
from qlinks.symmetry.gauss_law import GaussLaw
from qlinks.visualizer.graph import Graph
from qlinks.visualizer.quiver import Quiver


class TestGaussLaw:
    @pytest.mark.parametrize(
        "charge_distri, expectation",
        [
            (np.array([[1, 0, 2], [-2, 0, -1], [1, 0, -1]]), does_not_raise()),
            (np.array([[3, 0], [0, -3]]), pytest.raises(InvalidArgumentError)),
        ],
    )
    def test_constructor(self, charge_distri, expectation):
        with expectation:
            gauss_law = GaussLaw(charge_distri)
            assert gauss_law.charge_distri[0, 0] == 1
            assert gauss_law.charge_distri[1, 0] == 0
            assert gauss_law.charge_distri[2, 0] == -1
            assert gauss_law.charge_distri[0, 1] == -2
            assert gauss_law.charge_distri[1, 1] == 0
            assert gauss_law.charge_distri[2, 1] == -1
            assert gauss_law.charge_distri[0, 2] == 1
            assert gauss_law.charge_distri[1, 2] == 0
            assert gauss_law.charge_distri[2, 2] == 2

    @pytest.mark.parametrize("charge_distri", [np.array([[1, 0], [0, -1]])])
    def test_get_item(self, charge_distri):
        gauss_law = GaussLaw(charge_distri)
        assert gauss_law[Site(0, 0)] == 0
        assert gauss_law[Site(1, 0)] == -1
        assert gauss_law[Site(0, 1)] == 1
        assert gauss_law[Site(0, 0)] == 0

    @pytest.mark.parametrize("charge", [-2, -1, 0, 1, 2])
    def test_possible_flows(self, charge: int):
        configs = GaussLaw.possible_flows(charge)
        assert len(configs) == binom(4, 2 - abs(charge))
        for config in configs:
            assert np.sum(config) / 2 == charge

    @pytest.mark.parametrize("charge", [-2, -1, 0, 1, 2])
    def test_possible_configs(self, charge: int):
        configs = GaussLaw.possible_configs(charge)
        assert len(configs) == binom(4, 2 - abs(charge))
        for config in configs:
            flow = (config - (config == 0)) * np.array([-1, -1, 1, 1])  # turn 0 to -1
            assert np.sum(flow) / 2 == charge

    @pytest.mark.parametrize("length_x, length_y", [(2, 2), (3, 3), (6, 8)])
    def test_random_charge_distri(self, length_x: int, length_y: int):
        sampled_vals = set()
        for _ in range(100):
            charges = GaussLaw.random_charge_distri(length_x, length_y)
            sampled_vals.update(set(charges.flatten()))
            assert np.all(np.isin(charges, [-2, -1, 0, 1, 2])), f"got {charges}"
            assert np.sum(charges) == 0
            assert charges.shape == (length_y, length_x)
        assert len(sampled_vals) == 5
        assert max(sampled_vals) == 2
        assert min(sampled_vals) == -2

    @pytest.mark.parametrize(
        "length_x, length_y, expectation",
        [
            (2, 2, does_not_raise()),
            (6, 8, does_not_raise()),
            (3, 3, pytest.raises(InvalidArgumentError)),
        ],
    )
    def test_staggered_charge_distri(self, length_x: int, length_y: int, expectation):
        with expectation:
            charges = GaussLaw.staggered_charge_distri(length_x, length_y)
            assert np.sum(charges) == 0
            assert np.all(np.unique(charges) == [-1, 1])
            assert charges.shape == (length_y, length_x)

    def test_next_empty_site(self):
        """
           │      │
           ▼      ▼
        ──►o◄─────o──►
           ▲      ▲
           │      │
        ──►o◄─────o──►
           │      │
           ▼      ▼
        """
        gauss_law = GaussLaw.from_zero_charge_distri(2, 2)
        assert gauss_law._next_empty_site() == Site(0, 0)
        gauss_law._lattice.links = np.array([0, 1, 1, -1, -1, 0, -1, -1])
        assert gauss_law._next_empty_site() == Site(1, 0)
        gauss_law._lattice.links = np.array([0, 1, 1, 1, -1, 0, -1, 0])
        assert gauss_law._next_empty_site() == Site(0, 1)

    @pytest.mark.parametrize("length_x, length_y", [(2, 2)])
    def test_extend_node(self, length_x: int, length_y: int):
        base_node = GaussLaw.from_zero_charge_distri(length_x, length_y)
        new_nodes = base_node.extend_node()
        assert len(set(new_nodes)) == len(new_nodes)
        first_site = base_node._next_empty_site()
        for new_node in new_nodes:
            new_empty_site = new_node._next_empty_site()
            assert new_empty_site > first_site
            assert not np.isnan(new_node._lattice.charge(first_site))
            assert np.isnan(new_node._lattice.charge(new_empty_site))

    @pytest.mark.parametrize(
        "charge_distri, expectation",
        [
            (np.zeros((2, 2)), does_not_raise()),
            ([[1, 0], [-1, 0]], does_not_raise()),
            ([[-2, 0], [0, 2]], does_not_raise()),
            ([[1, 1, -2], [-2, 0, 0], [0, 2, 0]], does_not_raise()),
            ([[1, 1, 1], [-2, 0, 0], [0, 2, 0]], pytest.raises(StopIteration)),
        ],
    )
    def test_search(self, charge_distri, expectation):
        gauss_law = GaussLaw(charge_distri)
        dfs = DeepFirstSearch(gauss_law)
        with expectation:
            filled_gauss_law, = dfs.solve(n_solution=1)
            assert isinstance(filled_gauss_law, GaussLaw)
            for site in filled_gauss_law._lattice:
                assert filled_gauss_law.charge(site) == filled_gauss_law[site]

    @pytest.mark.parametrize(
        "charge_distri, flux_sector, n_expected_solution, expectation",
        [
            (np.zeros((2, 2)), 18, does_not_raise()),
            ([[1, -1], [-1, 1]], 8, does_not_raise()),
            ([[1, 1, 1], [-2, 0, 0], [0, 2, 0]], None, pytest.raises(StopIteration)),
            # (np.zeros((4, 4)), 2970, does_not_raise()),  # 1 mins 30 secs
            (GaussLaw.staggered_charge_distri(4, 4), 272, does_not_raise()),  # 10 secs
        ],
    )
    def test_multi_solutions(self, charge_distri, flux_sector, n_expected_solution, expectation):
        gauss_law = GaussLaw(charge_distri, flux_sector)
        dfs = DeepFirstSearch(gauss_law, max_steps=np.iinfo(np.int32).max)
        with expectation:

    @pytest.fixture(
        scope="class",
        params=[
            [[-1, 0], [0, 1]],
            [[-2, 0], [0, 2]],
            [[1, 1, -2], [-2, 0, 0], [0, 2, 0]],
            np.zeros((4, 4)),
            GaussLaw.staggered_charge_distri(4, 4),
        ],
    )
    def snapshot(self, request):
        charge_distri = request.param
        width, length = np.asarray(charge_distri).shape
        snapshot = GaugeInvariantSnapshot(length, width, charge_distri)
        dfs = DeepFirstSearch(snapshot)
        filled_snapshot = dfs.search()
        return filled_snapshot

    def test_adjacency_matrix(self, snapshot):
        adj_mat = snapshot.adjacency_matrix
        assert np.all(np.sum(adj_mat, axis=0) + np.sum(adj_mat, axis=1) == 4), f"got {adj_mat}"
        assert adj_mat.dtype == np.int64
        assert np.all(adj_mat >= 0)

    def test_as_graph(self, snapshot):
        graph = snapshot.as_graph()
        assert len(graph.edges()) == snapshot.num_links
        for _, degree in graph.degree:
            assert degree == 4

    def test_plot(self, snapshot):
        g = Graph(snapshot)
        g.plot()
        plt.show()

        q = Quiver(snapshot)
        q.plot()
        plt.show()
            filled_gauss_laws = dfs.solve(n_solution=n_expected_solution+1)
            assert all(isinstance(s, GaussLaw) for s in filled_gauss_laws)
            assert len(filled_gauss_laws) == n_expected_solution
            assert dfs.frontier_is_empty
