from contextlib import nullcontext as does_not_raise
from dataclasses import astuple

import numpy as np
import pytest
from scipy.special import binom

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
            ([[1, 1, 1], [-2, 0, 0], [0, 2, 0]], pytest.raises(StopIteration))
        ]
    )
    def test_search(self, charge_distri, expectation):
        width, length = np.asarray(charge_distri).shape
        snapshot = SpinConfigSnapshot(length, width, charge_distri)
        dfs = DeepFirstSearch(snapshot)
        with expectation:
            filled_snapshot = dfs.search()
            for site in filled_snapshot:
                assert filled_snapshot.charge(site) == filled_snapshot.charge_distri[*astuple(site)]

    @pytest.fixture(scope="class", params=[
        [[-1, 0], [0, 1]],
        [[-2, 0], [0, 2]],
        [[1, 1, -2], [-2, 0, 0], [0, 2, 0]]
    ])
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
