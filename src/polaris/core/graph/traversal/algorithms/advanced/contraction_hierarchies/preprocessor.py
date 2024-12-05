"""
Preprocessing implementation for Contraction Hierarchies.

This module handles the preprocessing phase of the Contraction Hierarchies algorithm,
including node ordering and shortcut creation.
"""

import time
from heapq import heappop, heappush
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

from polaris.core.models import Edge
from .models import ContractionState, Shortcut
from .storage import ContractionStorage
from .utils import calculate_node_importance, is_shortcut_necessary

if TYPE_CHECKING:
    from polaris.core.graph import Graph


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

    def preprocess(self) -> ContractionState:
        """
        Preprocess graph to build contraction hierarchy.

        Returns:
            Preprocessed ContractionState
        """
        print("Starting preprocessing...")
        start_time = time.time()

        # Calculate initial node ordering
        nodes = list(self.graph.get_nodes())
        node_importance = {
            node: calculate_node_importance(
                node,
                self.graph,
                self.storage.get_state().contracted_neighbors,
                self.storage.get_state().node_level,
                self._count_shortcuts(node),
            )
            for node in nodes
        }
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
                # Contract node
                shortcuts = self._contract_node(node)
                self.storage.set_node_level(node, level)
                level += 1

                # Debug: Print contracted node and shortcuts
                print(
                    f"Contracted node: {node}, Level: {level}, "
                    f"Shortcuts added: {len(shortcuts)}"
                )

                # Show progress
                progress = (level / total_nodes) * 100
                if progress - last_progress >= progress_interval:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / level) * (total_nodes - level)
                    print(f"Preprocessing: {progress:.1f}% complete, " f"ETA: {remaining:.1f}s")
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

        total_time = time.time() - start_time
        print(
            f"Preprocessing complete in {total_time:.1f}s, "
            f"created {len(self.storage.get_shortcuts())} shortcuts"
        )

        return self.storage.get_state()

    def _contract_node(self, node: str) -> List[Tuple[str, str]]:
        """
        Contract a node and add necessary shortcuts.

        Args:
            node: Node to contract

        Returns:
            List of (u, v) pairs where shortcuts were added
        """
        shortcuts = []
        incoming = sorted(
            list(self.graph.get_neighbors(node, reverse=True))
        )  # Sort for determinism
        outgoing = sorted(list(self.graph.get_neighbors(node)))  # Sort for determinism

        # Consider all pairs of incoming and outgoing edges
        for u in incoming:
            # Don't skip contracted nodes - we need to consider all paths
            for v in outgoing:
                if u == v:  # Skip self-loops
                    continue

                # Check if shortcut is necessary
                lower_edge = self.graph.get_edge(u, node)
                upper_edge = self.graph.get_edge(node, v)
                if not lower_edge or not upper_edge:
                    continue

                shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight

                # Check if shortcut is necessary using witness search
                if is_shortcut_necessary(u, v, node, shortcut_weight, self.graph):
                    # Create and store shortcut
                    shortcut = Shortcut.create(u, v, node, lower_edge, upper_edge)
                    self.storage.add_shortcut(shortcut)
                    shortcuts.append((u, v))

        return shortcuts

    def _count_shortcuts(self, node: str) -> int:
        """
        Count number of shortcuts needed when contracting node.

        Args:
            node: Node to analyze

        Returns:
            Number of shortcuts needed
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
                # Only count if shortcut would be necessary
                lower_edge = self.graph.get_edge(u, node)
                upper_edge = self.graph.get_edge(node, v)
                if lower_edge and upper_edge:
                    shortcut_weight = lower_edge.metadata.weight + upper_edge.metadata.weight
                    if is_shortcut_necessary(u, v, node, shortcut_weight, self.graph):
                        shortcuts.add((u, v))

        return len(shortcuts)
