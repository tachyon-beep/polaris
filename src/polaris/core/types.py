"""
Core type definitions and protocols.

This module provides type definitions and protocols used across the core graph
functionality to ensure type safety and clarity.
"""

from typing import Optional, Protocol, Set
from .models import Edge


class GraphProtocol(Protocol):
    """Protocol defining required graph operations."""

    def get_neighbors(self, node: str) -> Set[str]:
        """Get outgoing neighbors of a node."""
        ...

    def get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        """Get edge between two nodes if it exists."""
        ...

    def get_incoming_edges(self, node: str) -> Set[Edge]:
        """Get all incoming edges for a node."""
        ...

    def has_edge(self, from_node: str, to_node: str) -> bool:
        """Check if edge exists between nodes."""
        ...
