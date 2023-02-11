from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Deque, List, Self, Set

from qlinks import logger


class Node(abc.ABC):
    @abc.abstractmethod
    def __hash__(self) -> int:
        ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        ...

    @abc.abstractmethod
    def extend_node(self) -> List[Self]:
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
        frontier: A :class:`deque` of :class:`Node` to be determined.
        checked_nodes: A :class:`set` of checked :class:`Node`.
        selected_nodes: A :class:`deque` of :class:`Node` representing candidate solutions.

    Examples:
        If :class:`Node` has been implemented, the algorithm can be launched through

        >>> init_state: Node = Node()  # replace with any derived child class from Node
        >>> dfs = DeepFirstSearch(init_state)
        >>> selected_state: Node = dfs.search()

    Notes:
        For general purpose, users must implement all abstractive methods in :class:`Node`.

    References:
        https://python.plainenglish.io/solve-sudoku-using-depth-first-search-algorithm-dfs-in-python-2be3caa08ccd
    """

    start_state: Node
    max_steps: int = 1000
    frontier: Deque[Node] = field(default_factory=deque)
    checked_nodes: Set[Node] = field(default_factory=set)
    selected_nodes: Deque[Node] = field(default_factory=deque)

    def __post_init__(self):
        self.frontier.append(self.start_state)

    def insert_to_frontier(self, node: Node) -> None:
        self.frontier.appendleft(node)

    def remove_from_frontier(self) -> Node:
        first_node = self.frontier.popleft()
        self.checked_nodes.add(first_node)
        return first_node

    def frontier_is_empty(self) -> bool:
        return True if len(self.frontier) == 0 else False

    def search(self, n_solution: int = 1) -> Node | List[Node]:  # type: ignore[return]
        """Search for the solutions.

        Args:
            n_solution: Desired number of solutions. Default 1. In practical usage, it's normally
            difficult to know all possible solutions in prior. In that scenario, `n_solution` can be
            just set as a large number (far larger than the configuration space), and the algorithm
            will automatically stop after it walked through the entire configuration space (or when
            it has exceeded `max_steps`).

        Returns:
            A :class:`Node` which fulfills :func:`~Node.is_the_solution`.

        Raises:
            StopIteration: if no solution can be found or the number of iteration
             exceeds `max_steps`.
        """
        for n_step in count(start=1):
            if self.frontier_is_empty() or n_step == self.max_steps:
                if len(self.selected_nodes) == 0:
                    raise StopIteration(f"No Solution Found after {n_step} steps!")
                logger.info(
                    f"No more new Solutions can be found, end up with "
                    f"{len(self.selected_nodes)} Solutions."
                )
                return list(self.selected_nodes)

            selected_node = self.remove_from_frontier()

            if selected_node.is_the_solution():
                self.selected_nodes.append(selected_node)
                logger.debug(f"A New solution is Found after {n_step} steps.")
                logger.debug(
                    f"Totally, we have Found {len(self.selected_nodes)} Solutions "
                    f"[{len(self.checked_nodes)} checked | {len(self.frontier)} unchecked]."
                )
                logger.debug(f"New Solution: \n{selected_node}")
                if n_solution == 1:
                    return selected_node
                elif len(self.selected_nodes) >= n_solution:
                    logger.info(f"Found {n_solution} Solutions as required in {n_step} steps.")
                    return list(self.selected_nodes)

            new_nodes: List[Node] = selected_node.extend_node()

            if len(new_nodes) > 0:
                for new_node in new_nodes:
                    if new_node not in self.frontier and new_node not in self.checked_nodes:
                        self.insert_to_frontier(new_node)
