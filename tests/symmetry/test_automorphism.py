import networkx as nx
import pytest

from qlinks.symmetry.automorphism import Automorphism


class TestAutomorphism:
    @pytest.mark.parametrize(
        "graph, expected",
        [
            (nx.cycle_graph(4), {i: 2 for i in range(4)}),
            (nx.cycle_graph(6), {i: 2 for i in range(6)}),
            (nx.cycle_graph(8), {i: 2 for i in range(8)}),
            (
                nx.grid_graph((3, 3)),
                {
                    (0, 0): 2,
                    (0, 1): 3,
                    (0, 2): 2,
                    (1, 0): 3,
                    (1, 1): 4,
                    (1, 2): 3,
                    (2, 0): 2,
                    (2, 1): 3,
                    (2, 2): 2,
                },
            ),
        ],
    )
    def test_degree_series(self, graph, expected):
        aut = Automorphism(nx.to_numpy_array(graph))
        assert list(aut.degree_series.values()) == list(expected.values())

    @pytest.mark.parametrize(
        "graph, expected",
        [
            (nx.cycle_graph(4), {2: list(range(4))}),
            (nx.cycle_graph(6), {2: list(range(6))}),
            (nx.cycle_graph(8), {2: list(range(8))}),
            (nx.grid_graph((3, 3)), {2: [0, 2, 6, 8], 3: [1, 3, 5, 7], 4: [4]}),
        ],
    )
    def test_degree_partition(self, graph, expected):
        aut = Automorphism(nx.to_numpy_array(graph))
        assert aut.degree_partition == expected

    @pytest.mark.parametrize(
        "graph, expected",
        [
            (nx.cycle_graph(4), {0: "A", 1: "B", 2: "A", 3: "B"}),
            (nx.cycle_graph(6), {0: "A", 1: "B", 2: "A", 3: "B", 4: "A", 5: "B"}),
            (nx.cycle_graph(8), {0: "A", 1: "B", 2: "A", 3: "B", 4: "A", 5: "B", 6: "A", 7: "B"}),
            (
                nx.grid_graph((3, 3)),
                {
                    0: "A",
                    1: "B",
                    2: "A",
                    3: "B",
                    4: "A",
                    5: "B",
                    6: "A",
                    7: "B",
                    8: "A",
                },
            ),
        ],
    )
    def test_bipartition_series(self, graph, expected):
        assert nx.is_bipartite(graph)
        aut = Automorphism(nx.to_numpy_array(graph))
        assert aut.bipartition_series == expected

    @pytest.mark.parametrize(
        "graph, expected",
        [
            (nx.cycle_graph(4), {"A": [0, 2], "B": [1, 3]}),
            (nx.cycle_graph(6), {"A": [0, 2, 4], "B": [1, 3, 5]}),
            (nx.cycle_graph(8), {"A": [0, 2, 4, 6], "B": [1, 3, 5, 7]}),
            (
                nx.grid_graph((3, 3)),
                {"A": [0, 2, 4, 6, 8], "B": [1, 3, 5, 7]},
            ),
        ],
    )
    def test_bipartition(self, graph, expected):
        assert nx.is_bipartite(graph)
        aut = Automorphism(nx.to_numpy_array(graph))
        assert aut.bipartition == expected

    @pytest.mark.parametrize(
        "graph, expected",
        [
            (nx.cycle_graph(4), {(2, "A"): [0, 2], (2, "B"): [1, 3]}),
            (nx.cycle_graph(6), {(2, "A"): [0, 2, 4], (2, "B"): [1, 3, 5]}),
            (nx.cycle_graph(8), {(2, "A"): [0, 2, 4, 6], (2, "B"): [1, 3, 5, 7]}),
            (nx.grid_graph((3, 3)), {(2, "A"): [0, 2, 6, 8], (3, "B"): [1, 3, 5, 7], (4, "A"): [4]}),
        ]
    )
    def test_joint_partition(self, graph, expected):
        assert nx.is_bipartite(graph)
        aut = Automorphism(nx.to_numpy_array(graph))
        assert aut.joint_partition == expected
