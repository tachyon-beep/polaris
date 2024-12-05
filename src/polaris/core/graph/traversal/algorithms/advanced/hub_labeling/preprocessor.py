"""
Hub Labeling preprocessing implementation.

This module handles the preprocessing phase of the Hub Labeling algorithm,
which computes distance labels for each node in the graph.
"""

import time
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from collections import defaultdict

from polaris.core.graph.traversal.utils import get_edge_weight
from polaris.core.models import Edge
from .models import HubLabel, HubLabelSet, HubLabelState
from .storage import HubLabelStorage
from .utils import calculate_node_importance, should_prune_label

if TYPE_CHECKING:
    from polaris.core.graph import Graph


class HubLabelPreprocessor:
    """
    Preprocessor for Hub Labeling algorithm.

    This class handles the preprocessing phase, which:
    1. Computes node importance scores
    2. Orders nodes by importance
    3. Computes forward and backward distance labels
    """

    def __init__(self, graph: "Graph", storage: Optional[HubLabelStorage] = None):
        """
        Initialize preprocessor.

        Args:
            graph: Graph to preprocess
            storage: Optional storage for persisting labels
        """
        self.graph = graph
        self.storage = storage
        self.state = HubLabelState()
        self._start_time = 0.0

    def preprocess(self, verbose: bool = True) -> None:
        """
        Run preprocessing phase.

        This computes distance labels for all nodes in the graph.
        The labels allow for fast distance queries at query time.

        Args:
            verbose: Whether to print progress information
        """
        if verbose:
            print("\nStarting preprocessing...")

        self._start_time = time.time()

        # Calculate node ordering
        hub_order = self._compute_hub_order(verbose)

        # Process nodes in order
        nodes = sorted(hub_order.keys(), key=lambda x: hub_order[x])
        for node in nodes:
            if verbose:
                print(f"\nProcessing node {node}:")

            # Initialize labels for this node
            self.state.get_forward_labels(node).add_label(
                HubLabel(hub=node, distance=0.0, first_hop=None)
            )
            self.state.get_backward_labels(node).add_label(
                HubLabel(hub=node, distance=0.0, first_hop=None)
            )

            # Compute forward labels (paths from this node to more important hubs)
            self._compute_forward_labels(node, hub_order, verbose)

            # Compute backward labels (paths to this node from less important nodes)
            self._compute_backward_labels(node, hub_order, verbose)

            # Save labels if storage is available
            if self.storage is not None:
                self.storage.save_labels(
                    node,
                    self.state.get_forward_labels(node),
                    self.state.get_backward_labels(node),
                )

        if verbose:
            duration_ms = (time.time() - self._start_time) * 1000
            avg_labels = self._compute_average_labels()
            print(
                f"\nPreprocessing complete in {duration_ms:.1f}ms",
                f"average labels per node: {avg_labels:.1f}",
            )

    def _compute_hub_order(self, verbose: bool) -> Dict[str, int]:
        """
        Compute ordering of nodes based on importance.

        More important nodes get lower order numbers and become hubs.
        This is critical for performance - good hub selection reduces
        the total number of labels needed.

        Args:
            verbose: Whether to print progress information

        Returns:
            Map of node to order value (lower = more important)
        """
        # Detect graph topology
        is_chain = self._is_chain_graph()
        is_complete = self._is_complete_graph()

        # Calculate importance scores
        scores: Dict[str, float] = {}
        for node in self.graph.get_nodes():
            scores[node] = calculate_node_importance(node, self.graph, {})

        if verbose:
            print("\nImportance scores:")
            for node, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {node}: {score}")

        # Apply topology-specific adjustments
        if is_chain:
            self._adjust_chain_scores(scores)
        elif is_complete:
            self._adjust_complete_scores(scores)

        # Assign order numbers (0 = most important)
        hub_order = {
            node: i
            for i, (node, _) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        }

        if verbose:
            print("\nHub ordering:")
            for node, order in sorted(hub_order.items(), key=lambda x: x[1]):
                print(f"  {node}: order={order}")

        return hub_order

    def _is_chain_graph(self) -> bool:
        """
        Check if graph is a simple chain.

        A chain graph has exactly two endpoints (degree 1) and all other
        nodes have exactly two neighbors.

        Returns:
            True if graph is a chain, False otherwise
        """
        # Count degree for each node
        degree_count = defaultdict(int)
        for node in self.graph.get_nodes():
            in_edges = self.graph.get_neighbors(node, reverse=True)
            out_edges = self.graph.get_neighbors(node)
            degree = len(in_edges) + len(out_edges)
            degree_count[degree] += 1

        # Chain has exactly:
        # - Two endpoints (degree 1)
        # - All other nodes have degree 2
        return (
            len(self.graph.get_nodes()) >= 2
            and degree_count[1] == 2
            and degree_count[2] == len(self.graph.get_nodes()) - 2
            and sum(degree_count.values()) == len(self.graph.get_nodes())
        )

    def _is_complete_graph(self) -> bool:
        """
        Check if graph is complete.

        A complete graph has edges between every pair of nodes.

        Returns:
            True if graph is complete, False otherwise
        """
        nodes = list(self.graph.get_nodes())
        n = len(nodes)
        if n <= 1:
            return True

        # Check each pair of nodes
        for i in range(n):
            for j in range(i + 1, n):
                if not (
                    self.graph.get_edge(nodes[i], nodes[j])
                    or self.graph.get_edge(nodes[j], nodes[i])
                ):
                    return False
        return True

    def _adjust_chain_scores(self, scores: Dict[str, float]) -> None:
        """
        Adjust importance scores for chain graph.

        Boosts scores of central nodes to ensure good path coverage.

        Args:
            scores: Map of node to importance score
        """
        # Find endpoints
        endpoints = [
            node
            for node in self.graph.get_nodes()
            if len(self.graph.get_neighbors(node))
            + len(self.graph.get_neighbors(node, reverse=True))
            == 1
        ]
        if len(endpoints) != 2:
            return

        # Find path between endpoints
        path = []
        current = endpoints[0]
        visited = {current}
        while current != endpoints[1]:
            neighbors = (
                self.graph.get_neighbors(current) | self.graph.get_neighbors(current, reverse=True)
            ) - visited
            if not neighbors:
                return
            current = next(iter(neighbors))
            visited.add(current)
            path.append(current)

        # Boost scores based on distance from endpoints
        n = len(path)
        for i, node in enumerate(path):
            # Nodes closer to middle get higher boost
            dist_from_middle = abs(i - n // 2)
            boost = 1.0 / (dist_from_middle + 1)
            scores[node] = scores[node] * (1 + boost)

    def _adjust_complete_scores(self, scores: Dict[str, float]) -> None:
        """
        Adjust importance scores for complete graph.

        Ensures minimal hub set while maintaining path coverage.

        Args:
            scores: Map of node to importance score
        """
        n = len(self.graph.get_nodes())
        if n <= 2:
            return

        # For complete graphs, we only need O(sqrt(n)) hubs
        # Boost top sqrt(n) nodes to ensure they become hubs
        top_k = int((n**0.5) + 0.5)  # Round up
        top_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        for node in top_nodes:
            scores[node] *= 2.0  # Double importance of potential hubs

    def _compute_forward_labels(
        self,
        node: str,
        hub_order: Dict[str, int],
        verbose: bool,
    ) -> None:
        """
        Compute forward labels for a node.

        These labels represent shortest paths from this node to more
        important hubs.

        Args:
            node: Node to compute labels for
            hub_order: Map of node to order value
            verbose: Whether to print progress information
        """
        # Get existing labels
        forward_labels = self.state.get_forward_labels(node)

        # Try paths through existing hubs
        for hub in hub_order:
            if hub_order[hub] >= hub_order[node]:
                continue  # Only consider more important hubs

            # Check direct edge
            edge = self.graph.get_edge(node, hub)
            if edge:
                distance = get_edge_weight(edge)
                if not should_prune_label(node, hub, distance, self.state):
                    if verbose:
                        print(f"    Adding forward label: {node} -> {hub} (dist={distance})")
                    forward_labels.add_label(HubLabel(hub=hub, distance=distance, first_hop=edge))

            # Check paths through neighbors
            for neighbor in self.graph.get_neighbors(node):
                neighbor_labels = self.state.get_forward_labels(neighbor)
                neighbor_edge = self.graph.get_edge(node, neighbor)
                if neighbor_edge and neighbor_labels:
                    for label in neighbor_labels.labels:
                        if hub_order[label.hub] < hub_order[hub]:
                            total_dist = get_edge_weight(neighbor_edge) + label.distance
                            if not should_prune_label(node, hub, total_dist, self.state):
                                if verbose:
                                    print(
                                        f"    Adding forward label: {node} -> {hub} (dist={total_dist})"
                                    )
                                forward_labels.add_label(
                                    HubLabel(hub=hub, distance=total_dist, first_hop=neighbor_edge)
                                )

    def _compute_backward_labels(
        self,
        node: str,
        hub_order: Dict[str, int],
        verbose: bool,
    ) -> None:
        """
        Compute backward labels for a node.

        These labels represent shortest paths to this node from less
        important nodes.

        Args:
            node: Node to compute labels for
            hub_order: Map of node to order value
            verbose: Whether to print progress information
        """
        # Get existing labels
        backward_labels = self.state.get_backward_labels(node)

        # Try paths through existing hubs
        for source in hub_order:
            if hub_order[source] >= hub_order[node]:
                continue  # Only consider less important nodes

            # Check direct edge
            edge = self.graph.get_edge(source, node)
            if edge:
                distance = get_edge_weight(edge)
                if not should_prune_label(source, node, distance, self.state):
                    if verbose:
                        print(f"    Adding backward label: {node} <- {source} (dist={distance})")
                    backward_labels.add_label(
                        HubLabel(hub=source, distance=distance, first_hop=edge)
                    )

            # Check paths through neighbors
            for neighbor in self.graph.get_neighbors(node, reverse=True):
                neighbor_labels = self.state.get_backward_labels(neighbor)
                neighbor_edge = self.graph.get_edge(neighbor, node)
                if neighbor_edge and neighbor_labels:
                    for label in neighbor_labels.labels:
                        if hub_order[label.hub] < hub_order[source]:
                            total_dist = get_edge_weight(neighbor_edge) + label.distance
                            if not should_prune_label(source, node, total_dist, self.state):
                                if verbose:
                                    print(
                                        f"    Adding backward label: {node} <- {source} (dist={total_dist})"
                                    )
                                backward_labels.add_label(
                                    HubLabel(
                                        hub=source, distance=total_dist, first_hop=neighbor_edge
                                    )
                                )

    def _compute_average_labels(self) -> float:
        """
        Compute average number of labels per node.

        Returns:
            Average number of labels per node
        """
        total_labels = 0
        num_nodes = len(self.graph.get_nodes())
        for node in self.graph.get_nodes():
            total_labels += len(self.state.get_forward_labels(node).labels)
            total_labels += len(self.state.get_backward_labels(node).labels)
        return total_labels / num_nodes if num_nodes > 0 else 0.0
