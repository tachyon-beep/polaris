"""
Preprocessing implementation for Contraction Hierarchies.

This module handles the preprocessing phase of the Contraction Hierarchies algorithm,
including node ordering and shortcut creation.
"""

import time
import logging
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from polaris.core.models import Edge
from polaris.core.exceptions import GraphOperationError
from .models import ContractionState, Shortcut
from .storage import ContractionStorage
from .utils import calculate_node_importance, is_shortcut_necessary, EPSILON

if TYPE_CHECKING:
    from polaris.core.graph import Graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ContractionPreprocessor:
    """
    Handles preprocessing for Contraction Hierarchies algorithm.

    Features:
    - Node importance calculation
    - Contraction ordering
    - Shortcut creation
    - Progress tracking
    """

    def __init__(self, graph: "Graph", storage: ContractionStorage):
        """
        Initialize preprocessor.

        Args:
            graph: Graph to preprocess
            storage: Storage manager for algorithm state
        """
        self.graph = graph
        self.storage = storage
        self._parallel_edges: Dict[Tuple[str, str], List["Edge"]] = {}
        self._build_parallel_edges_index()

    def _build_parallel_edges_index(self) -> None:
        """Build index of parallel edges between nodes."""
        self._parallel_edges.clear()
        for edge in self.graph.get_edges():
            key = (edge.from_entity, edge.to_entity)
            if key not in self._parallel_edges:
                self._parallel_edges[key] = []
            self._parallel_edges[key].append(edge)

    def preprocess(self, new_edge: Optional[Tuple[str, str]] = None) -> ContractionState:
        start_time = time.time()
        print("Starting preprocessing...")

        # Initialize structures
        nodes = list(self.graph.get_nodes())
        pq = []  # Priority queue for node importance
        level = 0
        affected_nodes = set()

        def update_node_importance(node: str) -> float:
            importance = calculate_node_importance(
                node,
                self.graph,
                self.storage.get_state().contracted_neighbors,
                self.storage.get_state().node_level,
                self._count_shortcuts(node),
            )
            logger.debug(f"Node {node} importance updated to {importance}")
            return importance

        if new_edge:
            from_node, to_node = new_edge
            self._invalidate_shortcuts_for_edge(from_node, to_node)
            self._update_shortcuts_for_new_edge(from_node, to_node)
            return self.storage.get_state()

        # Initialize contracted_neighbors explicitly
        state = self.storage.get_state()
        node_importance = {}
        for node in nodes:
            if node not in state.contracted_neighbors:
                state.contracted_neighbors[node] = set()
            if self.graph.has_edge(node, node):
                state.contracted_neighbors[node].add(node)
            importance = update_node_importance(node)
            heappush(pq, (importance, node))

        while pq:
            _, node = heappop(pq)
            if node in state.node_level:
                continue
            shortcuts = self._contract_node(node)
            self.storage.set_node_level(node, level)
            level += 1
            affected_nodes.clear()
            affected_nodes.update(u for u, _ in shortcuts)
            affected_nodes.update(v for _, v in shortcuts)
            affected_nodes.discard(node)
            for affected_node in affected_nodes:
                if affected_node not in state.node_level:
                    importance = update_node_importance(affected_node)
                    heappush(pq, (importance, affected_node))

        logger.debug("Final contracted_neighbors: %s", state.contracted_neighbors)
        if not state.contracted_neighbors:
            logger.error("Preprocessing failed: contracted_neighbors is empty")
        total_time = time.time() - start_time
        print(f"Preprocessing complete in {total_time:.1f}s")

        return state

    def _get_best_edge(self, from_node: str, to_node: str) -> Optional["Edge"]:
        """
        Get the edge with minimum weight between two nodes.

        Args:
            from_node: Source node
            to_node: Target node

        Returns:
            Edge with minimum weight, or None if no edge exists
        """
        best_edge = None
        key = (from_node, to_node)
        if key in self._parallel_edges:
            for edge in self._parallel_edges[key]:
                if not best_edge or edge.metadata.weight < best_edge.metadata.weight:
                    best_edge = edge

        shortcut = self.storage.get_shortcut(from_node, to_node)
        if shortcut and (
            not best_edge or shortcut.edge.metadata.weight < best_edge.metadata.weight
        ):
            best_edge = shortcut.edge

        return best_edge

    def _contract_node(self, node: str) -> List[Tuple[str, str]]:
        """Contract node and create necessary shortcuts."""
        shortcuts = []
        incoming = sorted(self.graph.get_neighbors(node, reverse=True))
        outgoing = sorted(self.graph.get_neighbors(node))

        # Access the global state
        global_state = self.storage.get_state()

        # Debug: Log initial state of contracted_neighbors for the node
        logger.debug(
            "Initial contracted_neighbors[%s]: %s",
            node,
            global_state.contracted_neighbors.get(node),
        )

        # Ensure contracted_neighbors exists and initialize if missing
        if node not in global_state.contracted_neighbors:
            global_state.contracted_neighbors[node] = set()

        # Update the contracted_neighbors for the current node
        neighbors_to_add = set(incoming + outgoing)
        global_state.contracted_neighbors[node].update(neighbors_to_add)

        # Debug: Log state after adding incoming and outgoing neighbors
        logger.debug(
            "After adding neighbors, contracted_neighbors[%s]: %s",
            node,
            global_state.contracted_neighbors[node],
        )

        # Ensure self-loops are accounted for in contracted_neighbors
        if self.graph.has_edge(node, node):  # Check if the node has a self-loop
            global_state.contracted_neighbors[node].add(node)

        # Debug: Log state after accounting for self-loops
        logger.debug(
            "After adding self-loops, contracted_neighbors[%s]: %s",
            node,
            global_state.contracted_neighbors[node],
        )

        # Create shortcuts between all pairs of incoming and outgoing neighbors
        for u in incoming:
            for v in outgoing:
                if u == v:  # Skip self-loops
                    continue

                lower_edge = self._get_best_edge(u, node)
                upper_edge = self._get_best_edge(node, v)
                if not lower_edge or not upper_edge:
                    continue

                shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                # Check if a better path exists bypassing the current node
                min_path_weight = float("inf")
                for intermediate in self.graph.get_nodes():
                    if intermediate not in {
                        node,
                        u,
                        v,
                    }:  # Avoid current node and direct connections
                        path1 = self._get_best_edge(u, intermediate)
                        path2 = self._get_best_edge(intermediate, v)
                        if path1 and path2:
                            path_weight = path1.metadata.weight + path2.metadata.weight
                            min_path_weight = min(min_path_weight, path_weight)

                # Only create a shortcut if it provides a better path
                if min_path_weight > shortcut_weight + EPSILON:
                    try:
                        shortcut = self._create_validated_shortcut(
                            u, v, node, lower_edge, upper_edge
                        )
                        self.storage.add_shortcut(shortcut)
                        shortcuts.append((u, v))
                    except GraphOperationError as e:
                        logger.error(
                            "Shortcut creation failed for %s -> %s via %s: %s", u, v, node, e
                        )

        # Debug: Log final state of contracted_neighbors before returning
        logger.debug(
            "Final contracted_neighbors[%s]: %s", node, global_state.contracted_neighbors[node]
        )

        # Return the list of newly created shortcuts
        return shortcuts

    def _validate_shortcut(self, shortcut: Shortcut) -> bool:
        """
        Validate shortcut consistency and correctness.

        Args:
            shortcut: Shortcut to validate

        Returns:
            True if valid, False otherwise
        """
        # Verify edge connectivity
        if (
            shortcut.edge.from_entity != shortcut.lower_edge.from_entity
            or shortcut.edge.to_entity != shortcut.upper_edge.to_entity
        ):
            return False

        # Verify via node
        if (
            shortcut.lower_edge.to_entity != shortcut.via_node
            or shortcut.upper_edge.from_entity != shortcut.via_node
        ):
            return False

        # Check weight consistency
        shortcut_weight = shortcut.edge.metadata.weight
        path_weight = shortcut.lower_edge.metadata.weight + shortcut.upper_edge.metadata.weight
        if abs(shortcut_weight - path_weight) > EPSILON:
            return False

        return True

    def _create_validated_shortcut(
        self, u: str, v: str, node: str, lower_edge: Edge, upper_edge: Edge
    ) -> Shortcut:
        """Create and validate shortcut edge."""
        shortcut = Shortcut.create(u, v, node, lower_edge, upper_edge)

        # Validate shortcut consistency
        if not self._validate_shortcut(shortcut):
            raise GraphOperationError(f"Invalid shortcut {u}->{v}")

        return shortcut

    def _update_shortcuts_for_new_edge(self, from_node: str, to_node: str) -> None:
        """
        Update shortcuts affected by the addition of a new edge.

        Args:
            from_node: Source node of the new edge.
            to_node: Target node of the new edge.
        """
        affected_shortcuts = set()

        # Check incoming and outgoing neighbors
        incoming = self.graph.get_neighbors(from_node, reverse=True)
        outgoing = self.graph.get_neighbors(to_node)

        for u in incoming:
            for v in outgoing:
                if u == v:
                    continue

                lower_edge = self._get_best_edge(u, from_node)
                upper_edge = self._get_best_edge(to_node, v)
                if not lower_edge or not upper_edge:
                    continue

                shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                # Check if the shortcut needs to be updated
                existing_shortcut = self.storage.get_shortcut(u, v)
                if (
                    existing_shortcut
                    and existing_shortcut.edge.metadata.weight <= shortcut_weight + EPSILON
                ):
                    continue

                # Create or update the shortcut
                try:
                    new_shortcut = self._create_validated_shortcut(
                        u, v, from_node, lower_edge, upper_edge
                    )
                    self.storage.add_shortcut(new_shortcut)
                    affected_shortcuts.add((u, v))
                except GraphOperationError as e:
                    logger.error("Failed to update shortcut for %s->%s: %s", u, v, e)

        logger.debug(
            "Updated shortcuts for new edge %s->%s: %s", from_node, to_node, affected_shortcuts
        )

    def _invalidate_shortcuts_for_edge(self, from_node: str, to_node: str) -> None:
        """
        Invalidate shortcuts affected by the addition of a new edge.

        Args:
            from_node: Source node of the edge.
            to_node: Target node of the edge.
        """
        affected_shortcuts = []
        for (u, v), _ in list(self.storage.shortcuts.items()):
            if from_node in [u, v] or to_node in [u, v]:
                del self.storage.shortcuts[(u, v)]
                affected_shortcuts.append((u, v))

        logger.debug(
            "Invalidated shortcuts due to edge %s->%s: %s", from_node, to_node, affected_shortcuts
        )

    def _count_shortcuts(self, node: str) -> int:
        """
        Count number of shortcuts needed when contracting node.

        Args:
            node: Node to analyze

        Returns:
            Number of shortcuts needed

        Raises:
            GraphOperationError: If shortcut counting fails
        """
        shortcuts = set()
        incoming = sorted(
            list(self.graph.get_neighbors(node, reverse=True))
        )  # Sort for determinism
        outgoing = sorted(list(self.graph.get_neighbors(node)))  # Sort for determinism

        for u in incoming:
            for v in outgoing:
                if u == v:
                    continue

                # Get best edges if they exist
                lower_edge = self._get_best_edge(u, node)
                upper_edge = self._get_best_edge(node, v)

                if not lower_edge or not upper_edge:
                    continue

                shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                # Check if there's already a better path
                existing_edge = self._get_best_edge(u, v)
                if existing_edge and existing_edge.metadata.weight <= shortcut_weight + EPSILON:
                    continue

                # Only count if shortcut would be necessary
                if is_shortcut_necessary(u, v, node, shortcut_weight, self.graph):
                    shortcuts.add((u, v))

        return len(shortcuts)
