import numpy as np
import pytest

from qlinks.caging import CandidateSubgraph


def test_candidate_subgraph_sorts_vertices() -> None:
    candidate = CandidateSubgraph(vertices=np.array([3, 1, 2]))

    np.testing.assert_array_equal(candidate.vertices, np.array([1, 2, 3]))
    assert candidate.size == 3


def test_candidate_subgraph_rejects_empty_vertices() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CandidateSubgraph(vertices=np.array([], dtype=np.int64))


def test_candidate_subgraph_rejects_duplicate_vertices() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        CandidateSubgraph(vertices=np.array([0, 1, 1]))


def test_candidate_subgraph_rejects_non_1d_vertices() -> None:
    with pytest.raises(ValueError, match="1D"):
        CandidateSubgraph(vertices=np.array([[0, 1], [2, 3]]))
