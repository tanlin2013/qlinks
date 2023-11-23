from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Generic, List, Self, Set, TypeVar, Protocol

from qlinks import logger

AnyNode = TypeVar("AnyNode", bound="Node")


class Node(Protocol):
    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def extend_node(self) -> List[Self]:
        ...

    def is_the_solution(self) -> bool:
        ...


@dataclass(slots=True)
class DeepFirstSearch(Generic[AnyNode]):
    """Deep first search (DFS) algorithm.

    Args:
        start_state: The initial state as the first :class:`Node`.
        max_steps: The maximum number of steps for deep first search. Default 50000.

    Attributes:
        frontier: A :class:`deque` of :class:`Node` to be determined.
        checked_nodes: A :class:`set` of checked :class:`Node`.
        selected_nodes: A :class:`deque` of :class:`Node` representing candidate solutions.

    Examples:
        If :class:`Node` has been implemented, the algorithm can be launched through

        >>> init_state: AnyNode = Node()  # replace with any derived child class from Node
        >>> dfs = DeepFirstSearch(init_state)
        >>> selected_state: AnyNode = dfs.search()

    Notes:
        For general purpose, users must implement all abstractive methods in :class:`Node`.

    References:
        https://python.plainenglish.io/solve-sudoku-using-depth-first-search-algorithm-dfs-in-python-2be3caa08ccd
    """

    start_state: AnyNode
    max_steps: int = 50000
    frontier: Set[AnyNode] = field(default_factory=set)
    checked_nodes: Set[AnyNode] = field(default_factory=set)
    selected_nodes: List[AnyNode] = field(default_factory=list)

    def __post_init__(self):
        self.frontier.add(self.start_state)

    def insert_to_frontier(self, node: AnyNode) -> None:
        self.frontier.add(node)

    def remove_from_frontier(self) -> AnyNode:
        first_node = self.frontier.pop()
        self.checked_nodes.add(first_node)
        return first_node

    def frontier_is_empty(self) -> bool:
        return not bool(self.frontier)

    def _reach_stop_criteria(self, n_step: int) -> bool:
        if self.frontier_is_empty or n_step == self.max_steps:
            if len(self.selected_nodes) == 0:
                raise StopIteration(f"No Solution Found after {n_step} steps!")
            elif n_step == self.max_steps:
                logger.warning("Reach maximally allowed steps, search ends beforehand.")
            logger.info(
                f"No more new Solutions can be found, end up with "
                f"{len(self.selected_nodes)} Solutions in {n_step} steps."
            )
            return True
        return False

    def _diagnose_node(self) -> bool:
        """

        Returns:

        Notes:
            We didn't check if the new node is in `checked_nodes` before adding it to 'frontier',
            this is correct by assuming there is no loop in the tree.
        """
        selected_node = self._remove_from_frontier()

        if selected_node.is_the_solution():
            self.selected_nodes.append(selected_node)
            logger.debug(f"A New solution is Found after {len(self.checked_nodes)} steps.")
            logger.debug(
                f"Totally, we have Found {len(self.selected_nodes)} Solutions "
                f"[{len(self.checked_nodes)} checked | {len(self.frontier)} unchecked]."
            )
            logger.debug(f"New Solution: \n{selected_node}")
            return True

        new_nodes: List[AnyNode] = selected_node.extend_node()
        nodes_to_add = set(new_nodes) - self.frontier  # warn: no loop assumption
        self.frontier.update(nodes_to_add)
        return False

    def solve(self, n_solution: int = 1) -> List[AnyNode]:
        """Search for the solutions.

        Args:
            n_solution: Desired number of solutions. Default 1. In practical usage, it's normally
            difficult to know all possible solutions in prior. In that scenario, `n_solution` can be
            just set as a large number (far larger than the configuration space), and the algorithm
            will automatically stop after it walked through the entire configuration space (or when
            it has exceeded `max_steps`).

        Returns:
            A :class:`Node` or a list of :class:`Node` which fulfills :func:`~Node.is_the_solution`.

        Raises:
            StopIteration: if no solution can be found or the number of iteration
             exceeds `max_steps`.
        """
        for n_step in count(start=1):
            if self._reach_stop_criteria(n_step):
                return self.selected_nodes

            found_new = self._diagnose_node()
            if found_new and len(self.selected_nodes) >= n_solution:
                logger.info(f"Found {n_solution} Solution as required in {n_step} steps.")
                return self.selected_nodes
        return self.selected_nodes
