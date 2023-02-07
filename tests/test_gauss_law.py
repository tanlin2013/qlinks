from dataclasses import astuple

import numpy as np
import pytest

from qlinks.gauss_law import GaussLaw, SpinConfigSnapshot
from qlinks.solver.deep_first_search import DeepFirstSearch


class TestGaussLaw:
    @pytest.mark.parametrize("charge, n_config", [(-2, 1), (-1, 4), (0, 6), (1, 4), (2, 1)])
    def test_possible_flows(self, charge, n_config):
        assert len(GaussLaw(charge).possible_flows()) == n_config

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
        "charge_distri", [np.zeros((2, 2)), [[1, 0], [-1, 0]], [[-2, 0], [0, 2]]]
    )
    def test_search(self, charge_distri):
        width, length = np.asarray(charge_distri).shape
        snapshot = SpinConfigSnapshot(length, width, charge_distri)
        dfs = DeepFirstSearch(snapshot)
        filled_snapshot = dfs.search()
        for site in filled_snapshot:
            assert filled_snapshot.charge(site) == filled_snapshot.charge_distri[*astuple(site)]
