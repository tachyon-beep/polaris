"""Command Line Interface for the Knowledge Graph Management System.

This module provides a CLI for interacting with the knowledge graph system. It supports
operations such as adding nodes and edges, listing graph contents, and managing graph data
through JSON input files or direct JSON strings.

The CLI supports the following commands:
    - add: Add both nodes and edges to the graph
    - add-nodes: Add only nodes to the graph
    - add-edges: Add only edges to the graph
    - list: Display all nodes and edges in the graph

JSON input can be provided either as a direct string or as a file path prefixed with '@'.
When using file paths, both absolute paths and paths relative to the current directory
are supported.

Example Usage:
    python -m polaris cli add @data/graph_data.json
    python -m polaris cli add-nodes '{"name": "example", "entity_type": "CONCEPT"}'
    python -m polaris cli list
"""

import argparse

# Standard library imports
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from polaris.core.enums import EntityType, RelationType
from polaris.core.models import Edge, EdgeMetadata, Node, NodeMetadata, NodeMetrics
from polaris.infrastructure.storage.service import StorageService


def get_data_dir() -> str:
    """Get the absolute path to the data directory.

    The data directory is located at 'data' relative to the project root.

    Returns:
        str: Absolute path to the data directory.
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent.parent / "data"
    return str(data_dir)


async def setup_storage() -> StorageService:
    """Initialize and configure the storage service.

    Sets up the storage service with the JSON plugin and initializes it
    with the appropriate data directory.

    Returns:
        StorageService: Initialized storage service instance.
    """
    data_dir = get_data_dir()
    storage_service = StorageService(storage_dir=data_dir)
    await storage_service.initialize()
    return storage_service


def parse_json_input(json_str: str) -> dict:
    """Parse JSON input from either a string or file.

    Args:
        json_str (str): Either a JSON string or a file path prefixed with '@'.
                       For file paths, both absolute paths and paths relative to
                       the current directory are supported.

    Returns:
        dict: Parsed JSON data.

    Raises:
        ValueError: If the JSON is invalid or the specified file is not found.
    """
    if json_str.startswith("@"):
        # Handle file path input - strip '@' and any 'data/' prefix
        file_path = json_str[1:]

        # Resolve the file path - handle both absolute and relative paths
        if not os.path.isabs(file_path):
            if file_path.startswith("data/"):
                # Paths starting with 'data/' are relative to project root
                file_path = os.path.join(Path(__file__).parent.parent.parent.parent, file_path)
            else:
                # Other paths are relative to current working directory
                file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            return json.load(f)

    # Handle direct JSON string input
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")


async def create_nodes(storage: StorageService, nodes_data: List[Dict[str, Any]]) -> List[Node]:
    """Create and store multiple nodes from JSON data.

    Args:
        storage (StorageService): The storage service instance.
        nodes_data (List[Dict[str, Any]]): List of node data dictionaries.

    Returns:
        List[Node]: List of created Node instances.

    Raises:
        ValueError: If node creation fails due to invalid data.
    """
    nodes = []
    for node_data in nodes_data:
        try:
            # Convert string entity_type to enum
            node_data["entity_type"] = EntityType(node_data["entity_type"])

            # Initialize node metrics
            metrics = NodeMetrics()

            # Create metadata with default values where not provided
            metadata = NodeMetadata(
                created_at=datetime.now(),
                last_modified=datetime.now(),
                version=1,
                author=node_data.get("metadata", {}).get("author", "system"),
                source=node_data.get("metadata", {}).get("source", "cli"),
                metrics=metrics,
                custom_attributes=node_data.get("metadata", {}).get("custom_attributes", {}),
            )

            node_data["metadata"] = metadata
            node = Node(**node_data)
            node = await storage.create_node(node)
            nodes.append(node)

        except (KeyError, ValueError) as e:
            raise ValueError(f"Error creating node: {e}")
    return nodes


async def create_edges(storage: StorageService, edges_data: List[Dict[str, Any]]) -> List[Edge]:
    """Create and store multiple edges from JSON data.

    Args:
        storage (StorageService): The storage service instance.
        edges_data (List[Dict[str, Any]]): List of edge data dictionaries.

    Returns:
        List[Edge]: List of created Edge instances.

    Raises:
        ValueError: If edge creation fails due to invalid data.
    """
    edges = []
    for edge_data in edges_data:
        try:
            # Convert string relation_type to enum
            edge_data["relation_type"] = RelationType(edge_data["relation_type"])

            # Create metadata with default values where not provided
            metadata = EdgeMetadata(
                created_at=datetime.now(),
                last_modified=datetime.now(),
                confidence=edge_data.get("metadata", {}).get("confidence", 1.0),
                source=edge_data.get("metadata", {}).get("source", "cli"),
                bidirectional=edge_data.get("metadata", {}).get("bidirectional", False),
                weight=edge_data.get("metadata", {}).get("weight", 1.0),
                custom_attributes=edge_data.get("metadata", {}).get("custom_attributes", {}),
            )

            edge_data["metadata"] = metadata

            # Ensure required impact_score is present
            if "impact_score" not in edge_data:
                edge_data["impact_score"] = 1.0

            edge = Edge(**edge_data)
            edge = await storage.create_edge(edge)
            edges.append(edge)

        except (KeyError, ValueError) as e:
            raise ValueError(f"Error creating edge: {e}")
    return edges


async def list_all(storage: StorageService) -> None:
    """Display all nodes and edges currently in the graph.

    Args:
        storage (StorageService): The storage service instance.
    """
    nodes = await storage.list_nodes()
    edges = await storage.list_edges()

    print("\nNodes:")
    for node in nodes:
        print(f"- {node.name} ({node.entity_type.value})")
        print(f"  Observations: {node.observations}")
        print(f"  Metadata: {node.metadata}")
        print(f"  Attributes: {node.attributes}")

    print("\nEdges:")
    for edge in edges:
        print(f"- {edge.from_entity} -> {edge.to_entity} ({edge.relation_type.value})")
        print(f"  Metadata: {edge.metadata}")
        print(f"  Attributes: {edge.attributes}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(description="Knowledge Graph CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add command for both nodes and edges
    add = subparsers.add_parser("add", help="Add nodes and edges to the graph")
    add.add_argument("data", help="JSON string or @filename containing nodes and edges data")

    # Add command for nodes only
    add_nodes = subparsers.add_parser("add-nodes", help="Add nodes to the graph")
    add_nodes.add_argument("nodes", help="JSON string or @filename containing node data")

    # Add command for edges only
    add_edges = subparsers.add_parser("add-edges", help="Add edges to the graph")
    add_edges.add_argument("edges", help="JSON string or @filename containing edge data")

    # List command
    subparsers.add_parser("list", help="List all nodes and edges")

    return parser


async def main() -> None:
    """Main entry point for the CLI application.

    Handles command-line argument parsing and executes the appropriate
    commands for managing the knowledge graph.
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    storage = await setup_storage()

    try:
        if args.command == "add":
            # Handle combined nodes and edges data
            data = parse_json_input(args.data)
            nodes_data = data.get("nodes", [])
            edges_data = data.get("edges", [])

            # Create nodes first, then edges
            nodes = await create_nodes(storage, nodes_data)
            print(f"Created {len(nodes)} nodes:")
            for node in nodes:
                print(f"- {node.name} ({node.entity_type.value})")

            edges = await create_edges(storage, edges_data)
            print(f"\nCreated {len(edges)} edges:")
            for edge in edges:
                print(f"- {edge.from_entity} -> {edge.to_entity} ({edge.relation_type.value})")

        elif args.command == "add-nodes":
            # Handle nodes-only data
            nodes_data = parse_json_input(args.nodes)
            if not isinstance(nodes_data, list):
                nodes_data = [nodes_data]
            nodes = await create_nodes(storage, nodes_data)
            print(f"Created {len(nodes)} nodes:")
            for node in nodes:
                print(f"- {node.name} ({node.entity_type.value})")

        elif args.command == "add-edges":
            # Handle edges-only data
            edges_data = parse_json_input(args.edges)
            if not isinstance(edges_data, list):
                edges_data = [edges_data]
            edges = await create_edges(storage, edges_data)
            print(f"Created {len(edges)} edges:")
            for edge in edges:
                print(f"- {edge.from_entity} -> {edge.to_entity} ({edge.relation_type.value})")

        elif args.command == "list":
            await list_all(storage)

    finally:
        # Ensure proper cleanup of storage resources
        await storage.cleanup()


if __name__ == "__main__":
    # Ensure data directory exists before running
    os.makedirs(get_data_dir(), exist_ok=True)
    asyncio.run(main())
