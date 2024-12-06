"""
Preprocessing implementation for Contraction Hierarchies.

This module handles the preprocessing phase of the Contraction Hierarchies algorithm,
including node ordering and shortcut creation.
"""

import time
import logging
from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from polaris.core.exceptions import GraphOperationError
from .models import ContractionState, Shortcut
from .storage import ContractionStorage
from .utils import calculate_node_importance, is_shortcut_necessary

if TYPE_CHECKING:
    from polaris.core.graph import Graph
    from polaris.core.models import Edge

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

    def preprocess(self) -> ContractionState:
        """
        Preprocess graph to build contraction hierarchy.

        Returns:
            Preprocessed ContractionState

        Raises:
            GraphOperationError: If preprocessing fails
        """
        print("Starting preprocessing...")
        start_time = time.time()

        # Calculate initial node ordering
        nodes = list(self.graph.get_nodes())

        # Initialize contracted_neighbors for all nodes
        for node in nodes:
            if node not in self.storage.get_state().contracted_neighbors:
                self.storage.get_state().contracted_neighbors[node] = set()

        # Calculate initial importance for each node
        node_importance = {}
        for node in nodes:
            try:
                importance = calculate_node_importance(
                    node,
                    self.graph,
                    self.storage.get_state().contracted_neighbors,
                    self.storage.get_state().node_level,
                    self._count_shortcuts(node),
                )
                node_importance[node] = importance
            except Exception as e:
                raise GraphOperationError(
                    f"Failed to calculate importance for node {node}: {str(e)}"
                )

        pq = [(importance, node) for node, importance in node_importance.items()]

        level = 0
        total_nodes = len(nodes)
        last_progress = 0
        progress_interval = 5  # Show progress every 5%

        # Contract nodes in order of increasing importance
        while pq:
            self.storage.check_memory()
            _, node = heappop(pq)

            if node not in self.storage.get_state().node_level:
                try:
                    # Contract node
                    shortcuts = self._contract_node(node)
                    self.storage.set_node_level(node, level)

                    # Update contracted neighbors for all neighbors
                    incoming = self.graph.get_neighbors(node, reverse=True)
                    outgoing = self.graph.get_neighbors(node)
                    for neighbor in set(incoming) | set(outgoing):
                        if neighbor not in self.storage.get_state().contracted_neighbors:
                            self.storage.get_state().contracted_neighbors[neighbor] = set()
                        self.storage.get_state().contracted_neighbors[neighbor].add(node)

                    level += 1

                    # Show progress
                    progress = int((level * 100) / total_nodes)  # Convert to int for type safety
                    if progress - last_progress >= progress_interval:
                        elapsed = time.time() - start_time
                        remaining = (elapsed / level) * (total_nodes - level)
                        print(f"Preprocessing: {progress}% complete, ETA: {remaining:.1f}s")
                        last_progress = progress

                    # Update importance of affected nodes
                    affected = set()
                    for u, v in shortcuts:
                        affected.add(u)
                        affected.add(v)

                    for affected_node in affected:
                        if affected_node not in self.storage.get_state().node_level:
                            new_importance = calculate_node_importance(
                                affected_node,
                                self.graph,
                                self.storage.get_state().contracted_neighbors,
                                self.storage.get_state().node_level,
                                self._count_shortcuts(affected_node),
                            )
                            heappush(pq, (new_importance, affected_node))

                except Exception as e:
                    raise GraphOperationError(f"Failed to contract node {node}: {str(e)}")

        total_time = time.time() - start_time
        print(
            f"Preprocessing complete in {total_time:.1f}s, "
            f"created {len(self.storage.get_shortcuts())} shortcuts"
        )

        return self.storage.get_state()

    def _get_best_edge(self, from_node: str, to_node: str) -> Optional["Edge"]:
        """
        Get the edge with minimum weight between two nodes.

        Args:
            from_node: Source node
            to_node: Target node

        Returns:
            Edge with minimum weight, or None if no edge exists
        """
        min_weight = float("inf")
        best_edge = None

        # Check all parallel edges
        key = (from_node, to_node)
        if key in self._parallel_edges:
            for edge in self._parallel_edges[key]:
                if edge.metadata.weight < min_weight:
                    min_weight = edge.metadata.weight
                    best_edge = edge

        # Check existing shortcuts
        shortcut = self.storage.get_shortcut(from_node, to_node)
        if shortcut and shortcut.edge.metadata.weight < min_weight:
            min_weight = shortcut.edge.metadata.weight
            best_edge = shortcut.edge

        return best_edge

    def _contract_node(self, node: str) -> List[Tuple[str, str]]:
        """
        Contract a node and add necessary shortcuts.

        Args:
            node: Node to contract

        Returns:
            List of (u, v) pairs where shortcuts were added

        Raises:
            GraphOperationError: If contraction fails
        """
        shortcuts = []
        incoming = sorted(
            list(self.graph.get_neighbors(node, reverse=True))
        )  # Sort for determinism
        outgoing = sorted(list(self.graph.get_neighbors(node)))  # Sort for determinism

        logger.debug(f"Contracting node {node}")

        # Consider all pairs of incoming and outgoing edges
        for u in incoming:
            # Don't skip contracted nodes - we need to consider all paths
            for v in outgoing:
                if u == v:  # Skip self-loops
                    continue

                # Get best edges if they exist
                lower_edge = self._get_best_edge(u, node)
                upper_edge = self._get_best_edge(node, v)

                if not lower_edge or not upper_edge:
                    continue

                shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                # Check if there's already a better path
                existing_edge = self._get_best_edge(u, v)
                if existing_edge and existing_edge.metadata.weight <= shortcut_weight:
                    continue

                # Check if shortcut is necessary using witness search
                if is_shortcut_necessary(u, v, node, shortcut_weight, self.graph):
                    try:
                        # Create and store shortcut
                        shortcut = Shortcut.create(u, v, node, lower_edge, upper_edge)
                        self.storage.add_shortcut(shortcut)
                        shortcuts.append((u, v))
                    except Exception as e:
                        raise GraphOperationError(f"Failed to create shortcut: {str(e)}")

        return shortcuts

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
                if existing_edge and existing_edge.metadata.weight <= shortcut_weight:
                    continue

                # Only count if shortcut would be necessary
                if is_shortcut_necessary(u, v, node, shortcut_weight, self.graph):
                    shortcuts.add((u, v))

        return len(shortcuts)
