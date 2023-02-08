from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Deque, List, TypeVar

from qlinks import logger

AnyNode = TypeVar("AnyNode", bound="Node")


class Node(abc.ABC):
    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        ...

    @abc.abstractmethod
    def extend_node(self) -> List[AnyNode]:
        ...

    @abc.abstractmethod
    def is_the_solution(self) -> bool:
        ...


@dataclass
class DeepFirstSearch:
    """Deep first search (DFS) algorithm.

    Args:
        start_state: The initial state as the first :class:`Node`.
        max_steps: The maximum number of steps for deep first search. Default 1000.

    Attributes:
        frontier: A :class:`deque` of :class:`Node`.
        checked_nodes: A list of checked :class:`Node`.

    Examples:
        If :class:`Node` has been implemented, the algorithm can be launched through

        >>> init_state: AnyNode = Node()  # any child class of Node
        >>> dfs = DeepFirstSearch(init_state)
        >>> selected_state: AnyNode = dfs.search()

    Notes:
        For general purpose, users must implement all abstractive methods in :class:`Node`.

    References:
        https://python.plainenglish.io/solve-sudoku-using-depth-first-search-algorithm-dfs-in-python-2be3caa08ccd
    """

    start_state: AnyNode
    max_steps: int = 1000
    frontier: Deque[AnyNode] = field(default_factory=deque)
    checked_nodes: List[AnyNode] = field(default_factory=list)

    def __post_init__(self):
        self.frontier.append(self.start_state)

    def insert_to_frontier(self, node: AnyNode) -> None:
        self.frontier.appendleft(node)

    def remove_from_frontier(self) -> AnyNode:
        first_node = self.frontier.popleft()
        self.checked_nodes.append(first_node)
        return first_node

    def frontier_is_empty(self) -> bool:
        return True if len(self.frontier) == 0 else False

    def search(self) -> AnyNode:  # type: ignore[return]
        """Search for the solution.

        Returns:
            A :class:`Node` which fulfills :func:`~Node.is_the_solution`.

        Raises:
            StopIteration: if no solution can be found or the number of iteration
             exceeds `max_steps`.
        """
        for n_step in count(start=1):
            if self.frontier_is_empty() or n_step == self.max_steps:
                raise StopIteration(f"No Solution Found after {n_step} steps!")

            selected_node = self.remove_from_frontier()

            if selected_node.is_the_solution():
                logger.info(f"Solution Found in {n_step} steps.")
                logger.info(f"\n{selected_node}")
                return selected_node  # TODO: Do I need to find all possibilities, or just one?

            new_nodes = selected_node.extend_node()

            if len(new_nodes) > 0:
                for new_node in new_nodes:
                    if new_node not in self.frontier and new_node not in self.checked_nodes:
                        self.insert_to_frontier(new_node)
