"""
Base plugin interfaces for storage implementations.

This package provides the core interfaces that all storage plugins must implement:
- StoragePlugin: Base interface for all storage plugins
- NodeStoragePlugin: Interface for node storage implementations
- EdgeStoragePlugin: Interface for edge storage implementations
"""

from .interfaces import EdgeStoragePlugin, NodeStoragePlugin, StoragePlugin

__all__ = [
    "StoragePlugin",
    "NodeStoragePlugin",
    "EdgeStoragePlugin",
]
