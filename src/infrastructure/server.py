"""
MCP server implementation for knowledge graph operations.

This module implements a server using the MCP (Machine Control Protocol) to provide
a standardized interface for knowledge graph operations. It handles node and edge
management through a set of tools and resources that can be accessed remotely.

The server provides capabilities for:
- Creating, reading, updating and deleting nodes
- Creating, reading, updating and deleting edges
- Batch operations for improved performance
- Resource management and access
"""

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from ..core.enums import EntityType, RelationType
from ..core.exceptions import InvalidOperationError
from ..core.models import Edge, Node

logger = logging.getLogger(__name__)

# Create server instance
server = Server("knowledge-graph")


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """
    List available knowledge graph tools.

    This function defines and returns the set of tools available for knowledge graph
    operations. Each tool includes a name, description, and JSON schema defining its
    input parameters.

    Returns:
        List[types.Tool]: List of available tools with their specifications
    """
    return [
        types.Tool(
            name="create_nodes",
            description="Create multiple nodes in batch",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "entity_type": {
                                    "type": "string",
                                    "enum": [e.value for e in EntityType],
                                },
                                "observations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "metadata": {"type": "object"},
                                "attributes": {"type": "object"},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "name",
                                "entity_type",
                                "observations",
                                "metadata",
                            ],
                        },
                    }
                },
                "required": ["nodes"],
            },
        ),
        types.Tool(
            name="create_edges",
            description="Create multiple edges in batch",
            inputSchema={
                "type": "object",
                "properties": {
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from_entity": {"type": "string"},
                                "to_entity": {"type": "string"},
                                "relation_type": {
                                    "type": "string",
                                    "enum": [r.value for r in RelationType],
                                },
                                "metadata": {"type": "object"},
                                "attributes": {"type": "object"},
                            },
                            "required": [
                                "from_entity",
                                "to_entity",
                                "relation_type",
                                "metadata",
                            ],
                        },
                    }
                },
                "required": ["edges"],
            },
        ),
        types.Tool(
            name="get_nodes",
            description="Retrieve multiple nodes by their IDs",
            inputSchema={
                "type": "object",
                "properties": {"node_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["node_ids"],
            },
        ),
        types.Tool(
            name="get_edges",
            description="Retrieve multiple edges by their IDs",
            inputSchema={
                "type": "object",
                "properties": {"edge_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["edge_ids"],
            },
        ),
        # Add other tools for update_nodes, update_edges, delete_nodes, delete_edges
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Any
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool calls for knowledge graph operations.

    This function processes incoming tool calls, executing the requested operation
    and returning the results. It supports operations for node and edge
    management.

    Args:
        name: The name of the tool to execute
        arguments: The arguments for the tool execution

    Returns:
        List[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
            The results of the tool execution

    Raises:
        InvalidOperationError: If tool execution fails or tool is unknown
    """
    try:
        result = None

        if name == "create_nodes":
            nodes = [Node(**node_data) for node_data in arguments["nodes"]]
            # Implementation for creating nodes
            result = [asdict(node) for node in nodes]

        elif name == "create_edges":
            edges = [Edge(**edge_data) for edge_data in arguments["edges"]]
            # Implementation for creating edges
            result = [asdict(edge) for edge in edges]

        elif name == "get_nodes":
            # Implementation for retrieving nodes
            result = []  # Replace with actual node retrieval

        elif name == "get_edges":
            # Implementation for retrieving edges
            result = []  # Replace with actual edge retrieval

        else:
            raise ValueError(f"Unknown tool: {name}")

        return [types.TextContent(type="text", text=str(result))]

    except Exception as e:
        logger.error(f"Tool call failed: {name}, error: {str(e)}")
        raise InvalidOperationError(f"Failed to execute tool {name}: {str(e)}")


@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """
    List available knowledge graph resources.

    This function returns a list of resources that can be accessed through the server.
    Each resource represents a collection of nodes or edges in the knowledge graph.

    Returns:
        List[types.Resource]: List of available resources with their specifications
    """
    return [
        types.Resource(
            uri=types.AnyUrl("kg://nodes"),
            name="Knowledge Graph Nodes",
            mimeType="application/json",
            description="Collection of all nodes in the knowledge graph",
        ),
        types.Resource(
            uri=types.AnyUrl("kg://edges"),
            name="Knowledge Graph Edges",
            mimeType="application/json",
            description="Collection of all edges in the knowledge graph",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: types.AnyUrl) -> str:
    """
    Read knowledge graph resources.

    This function handles requests to read resources from the knowledge graph,
    such as collections of nodes or edges.

    Args:
        uri: The URI of the resource to read

    Returns:
        str: The string representation of the requested resource

    Raises:
        InvalidOperationError: If resource reading fails or resource is unknown
    """
    try:
        if str(uri) == "kg://nodes":
            # Implementation for reading all nodes
            nodes = []  # Replace with actual node retrieval
            return str(nodes)

        elif str(uri) == "kg://edges":
            # Implementation for reading all edges
            edges = []  # Replace with actual edge retrieval
            return str(edges)

        raise ValueError(f"Unknown resource: {uri}")

    except Exception as e:
        logger.error(f"Resource read failed: {uri}, error: {str(e)}")
        raise InvalidOperationError(f"Failed to read resource {uri}: {str(e)}")


async def run():
    """
    Run the knowledge graph server.

    This function initializes and runs the MCP server with the configured capabilities.
    It sets up the server with standard I/O streams for communication and handles
    the server lifecycle.
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="knowledge-graph",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(run())
