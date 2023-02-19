from contextlib import nullcontext as does_not_raise

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import binom

from qlinks.exceptions import InvalidArgumentError
from qlinks.lattice.component import Site
from qlinks.solver.deep_first_search import DeepFirstSearch
from qlinks.symmetry.gauss_law import GaussLaw, GaugeInvariantSnapshot
from qlinks.visualizer.graph import Graph
from qlinks.visualizer.quiver import Quiver


class TestGaussLaw:
    @pytest.mark.parametrize(
        "charge_distri, expectation",
        [
            (np.array([[1, 0, 2], [-2, 0, -1], [1, 0, -1]]), does_not_raise()),
            (np.array([[3, 0], [0, -3]]), pytest.raises(InvalidArgumentError))
        ]
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
    def test_possible_flows(self, charge):
        assert len(GaussLaw.possible_flows(charge)) == binom(4, 2 - abs(charge))

    @pytest.mark.parametrize("charge", [-2, -1, 0, 1, 2])
    def test_possible_configs(self, charge):
        configs = GaussLaw.possible_configs(charge)
        assert len(configs) == binom(4, 2 - abs(charge))
        for config in configs:
            mag = list(map(lambda spin: spin.magnetization, config))
            assert np.sum(np.multiply(mag, [-1, -1, 1, 1])) == charge

    @pytest.mark.parametrize("length, width", [(2, 2), (3, 3), (6, 8)])
    def test_random_charge_distri(self, length, width):
        sampled_vals = set()
        for _ in range(100):
            charges = GaussLaw.random_charge_distri(length, width)
            sampled_vals.update(set(charges.flatten()))
            assert np.all(np.isin(charges, [-2, -1, 0, 1, 2])), f"got {charges}"
            assert np.sum(charges) == 0
            assert charges.shape == (width, length)
        assert len(sampled_vals) == 5
        assert max(sampled_vals) == 2
        assert min(sampled_vals) == -2

    @pytest.mark.parametrize(
        "length, width, expectation",
        [
            (2, 2, does_not_raise()),
            (6, 8, does_not_raise()),
            (3, 3, pytest.raises(InvalidArgumentError)),
        ],
    )
    def test_staggered_charge_distri(self, length, width, expectation):
        with expectation:
            charges = GaussLaw.staggered_charge_distri(length, width)
            assert np.sum(charges) == 0
            assert np.all(np.unique(charges) == [-1, 1])
            assert charges.shape == (width, length)


class TestGaugeInvariantSnapshot:
    @pytest.mark.parametrize("length, width", [(2, 2)])
    def test_extend_node(self, length, width):
        charge_distri = np.zeros((length, width))
        snapshot = GaugeInvariantSnapshot(length, width, charge_distri)
        new_nodes = snapshot.extend_node()
        assert len(set(new_nodes)) == len(new_nodes)
        extended_site = snapshot.find_first_empty_site()
        for node in new_nodes:
            new_empty_site = node.find_first_empty_site()
            assert new_empty_site > extended_site
            assert not np.isnan(node.charge(extended_site))
            assert np.isnan(node.charge(new_empty_site))

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
        width, length = np.asarray(charge_distri).shape
        snapshot = GaugeInvariantSnapshot(length, width, charge_distri)
        dfs = DeepFirstSearch(snapshot)
        with expectation:
            filled_snapshot = dfs.search(n_solution=1)
            assert isinstance(filled_snapshot, GaugeInvariantSnapshot)
            for site in filled_snapshot:
                assert filled_snapshot.charge(site) == filled_snapshot.gauss_law[site]

    @pytest.mark.parametrize(
        "charge_distri, n_expected_solution, expectation",
        [
            (np.zeros((2, 2)), 18, does_not_raise()),
            ([[1, -1], [-1, 1]], 8, does_not_raise()),
            ([[1, 1, 1], [-2, 0, 0], [0, 2, 0]], None, pytest.raises(StopIteration)),
            # (np.zeros((4, 4)), 2970, does_not_raise()),
        ],
    )
    def test_multi_solutions(self, charge_distri, n_expected_solution, expectation):
        width, length = np.asarray(charge_distri).shape
        snapshot = GaugeInvariantSnapshot(length, width, charge_distri)
        dfs = DeepFirstSearch(snapshot, max_steps=30000)
        with expectation:
            filled_snapshots = dfs.search(n_solution=3000)  # far more than all possibilities
            assert all(isinstance(s, GaugeInvariantSnapshot) for s in filled_snapshots)
            assert len(filled_snapshots) == n_expected_solution

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
