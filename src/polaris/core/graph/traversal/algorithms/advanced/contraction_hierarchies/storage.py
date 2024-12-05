"""
Storage management for Contraction Hierarchies algorithm.

This module handles memory management and data storage for the Contraction
Hierarchies implementation, ensuring efficient use of memory resources.
"""

from typing import Dict, Optional, Set, Tuple

from polaris.core.graph.traversal.utils import MemoryManager
from .models import ContractionState, Shortcut


class ContractionStorage:
    """
    Manages storage and memory for Contraction Hierarchies algorithm.

    Features:
    - Memory usage monitoring
    - Efficient data structure storage
    - State management
    """

    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        Initialize storage manager.

        Args:
            max_memory_mb: Optional memory limit in MB
        """
        self.memory_manager = MemoryManager(max_memory_mb)
        self._state = ContractionState()

    def check_memory(self) -> None:
        """Check if memory usage is within limits."""
        self.memory_manager.check_memory()

    def get_state(self) -> ContractionState:
        """
        Get current algorithm state.

        Returns:
            Current ContractionState
        """
        return self._state

    def add_shortcut(self, shortcut: Shortcut) -> None:
        """
        Add a shortcut to storage.

        Args:
            shortcut: Shortcut to store
        """
        self.check_memory()
        self._state.add_shortcut(shortcut)

    def get_shortcut(self, from_node: str, to_node: str) -> Optional[Shortcut]:
        """
        Get shortcut between nodes if it exists.

        Args:
            from_node: Source node
            to_node: Target node

        Returns:
            Shortcut if it exists, None otherwise
        """
        return self._state.get_shortcut(from_node, to_node)

    def set_node_level(self, node: str, level: int) -> None:
        """
        Set contraction level for a node.

        Args:
            node: Node ID
            level: Contraction level
        """
        self.check_memory()
        self._state.set_node_level(node, level)

    def get_node_level(self, node: str) -> int:
        """
        Get contraction level of a node.

        Args:
            node: Node ID

        Returns:
            Contraction level (0 if not contracted)
        """
        return self._state.get_node_level(node)

    def get_contracted_neighbors(self, node: str) -> Set[str]:
        """
        Get contracted neighbors of a node.

        Args:
            node: Node ID

        Returns:
            Set of contracted neighbor node IDs
        """
        return self._state.get_contracted_neighbors(node)

    def get_shortcuts(self) -> Dict[Tuple[str, str], Shortcut]:
        """
        Get all shortcuts.

        Returns:
            Dictionary mapping (from_node, to_node) to Shortcut
        """
        return self._state.shortcuts

    def clear(self) -> None:
        """Clear all stored data."""
        self._state = ContractionState()
