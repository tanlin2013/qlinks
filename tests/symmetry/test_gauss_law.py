from contextlib import nullcontext as does_not_raise
from dataclasses import astuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from scipy.special import binom

from qlinks.exceptions import InvalidArgumentError
from qlinks.solver.deep_first_search import DeepFirstSearch
from qlinks.symmetry.gauss_law import GaussLaw, SpinConfigSnapshot


class TestGaussLaw:
    @pytest.mark.parametrize("charge", [-2, -1, 0, 1, 2])
    def test_possible_flows(self, charge):
        assert len(GaussLaw(charge).possible_flows()) == binom(4, 2 - abs(charge))

    @pytest.mark.parametrize("charge", [-2, -1, 0, 1, 2])
    def test_possible_configs(self, charge):
        configs = GaussLaw(charge).possible_configs()
        for config in configs:
            mag = list(map(lambda spin: spin.magnetization, config))
            assert np.sum(np.multiply(mag, [-1, -1, 1, 1])) / 2 == charge

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
            (3, 3, pytest.raises(InvalidArgumentError))
        ]
    )
    def test_staggered_charge_distri(self, length, width, expectation):
        with expectation:
            charges = GaussLaw.staggered_charge_distri(length, width)
            assert np.sum(charges) == 0
            assert np.all(np.unique(charges) == [-1, 1])
            assert charges.shape == (width, length)


class TestSpinConfigSnapshot:
    @pytest.mark.parametrize("length, width", [(2, 2)])
    def test_extend_node(self, length, width):
        charge_distri = np.zeros((length, width))
        snapshot = SpinConfigSnapshot(length, width, charge_distri)
        for snap in snapshot.extend_node():
            print(snap.links)

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
        snapshot = SpinConfigSnapshot(length, width, charge_distri)
        dfs = DeepFirstSearch(snapshot)
        with expectation:
            filled_snapshot = dfs.search(n_solution=1)
            assert isinstance(filled_snapshot, SpinConfigSnapshot)
            for site in filled_snapshot:
                assert filled_snapshot.charge(site) == filled_snapshot.charge_distri[*astuple(site)]

    @pytest.mark.parametrize(
        "charge_distri, n_expected_solution, expectation",
        [
            (np.zeros((2, 2)), 18, does_not_raise()),
            ([[1, -1], [-1, 1]], 8, does_not_raise()),
            ([[1, 1, 1], [-2, 0, 0], [0, 2, 0]], None, pytest.raises(StopIteration))
        ]
    )
    def test_multi_solutions(self, charge_distri, n_expected_solution, expectation):
        width, length = np.asarray(charge_distri).shape
        snapshot = SpinConfigSnapshot(length, width, charge_distri)
        dfs = DeepFirstSearch(snapshot)
        with expectation:
            filled_snapshots = dfs.search(n_solution=1000)  # far more than all possibilities
            assert all(isinstance(s, SpinConfigSnapshot) for s in filled_snapshots)
            assert len(filled_snapshots) == n_expected_solution

    @pytest.fixture(
        scope="class",
        params=[[[-1, 0], [0, 1]], [[-2, 0], [0, 2]], [[1, 1, -2], [-2, 0, 0], [0, 2, 0]]],
    )
    def snapshot(self, request):
        charge_distri = request.param
        width, length = np.asarray(charge_distri).shape
        snapshot = SpinConfigSnapshot(length, width, charge_distri)
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

        pos = {idx: astuple(site) for idx, site in enumerate(snapshot)}
        labels = {idx: str(site) for idx, site in pos.items()}
        nx.draw(
            graph,
            pos,
            labels=labels,
            with_labels=True,
            node_color="tab:orange",
            node_size=2000,
            arrowstyle="simple",
            arrowsize=30,
            width=1,
            connectionstyle="arc3,rad=0.2",
            alpha=0.5,
        )

        # nx.draw_networkx_nodes(
        #     graph, pos, label=labels,
        #     node_color="tab:orange", node_size=2000, margins=(0.3, 0.3)
        # )
        # nx.draw_networkx_labels(
        #     graph, pos, labels=labels
        # )

        # nx.draw_networkx_edges(
        #     graph, pos, edgelist=,
        #     arrowstyle="simple", arrowsize=30, width=1
        # )
        # nx.draw_networkx_edges(
        #     graph, pos, edgelist=
        # )

        plt.show()