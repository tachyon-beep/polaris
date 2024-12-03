# Polaris

[![Build Status](https://github.com/tachyon-beep/Polaris/workflows/CI/badge.svg)](https://github.com/tachyon-beep/Polaris/actions)
[![Coverage Status](https://coveralls.io/repos/github/tachyon-beep/polaris/badge.svg?branch=main)](https://coveralls.io/github/tachyon-beep/polaris?branch=main)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=tachyon-beep_polaris&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=tachyon-beep_polaris)

## Overview

`Polaris` is a Python project that connects large language models (LLMs) to an MCP (Model-Controlled Processes) knowledge graph. It specializes in system engineering, game design, OS intelligence analysis, and domain-specific insights. This library allows you to create, query, and analyze entities and relations, facilitating advanced data management and analytical workflows.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
  - [Core Package](#core-package)
  - [Infrastructure Layer](#infrastructure-layer)
  - [Repository Layer](#repository-layer)
  - [Search Module](#search-module)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Entity Management**: Define entities with rich metadata, attributes, and metrics
- **Relation Management**: Establish bidirectional and weighted relationships between entities
- **Domain-Specific Queries**: Filter entities and relations by domain, confidence level, and time range
- **Analysis Tools**: Perform domain-level analysis to derive metrics and insights
- **Semantic Search**: Advanced search capabilities with vector embeddings and similarity matching
- **Data Integrity**: Comprehensive validation and integrity checking
- **Caching**: Efficient LRU caching with time-based expiration
- **Event Processing**: Robust event handling system with persistence

## Architecture

The system is organized into four main components:

1. **Core Package**: Foundation layer with graph operations and domain models
2. **Infrastructure Layer**: Technical services for persistence, caching, and events
3. **Repository Layer**: Data access abstraction using the repository pattern
4. **Search Module**: Comprehensive search capabilities including semantic search

## Requirements

- Python 3.12+
- Required packages:
  - `asyncio` for async operations
  - `dataclasses` for data structures
  - `networkx` for graph algorithms
  - `pandas` for data processing
  - `mcp.client` for MCP integration
  - `jsonschema` for validation
  - `aiofiles` for async I/O

Install dependencies:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tachyon-beep/Polaris.git
   cd Polaris
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the example script:
   ```bash
   python main.py
   ```

## Usage

### Basic Setup

```python
from mcp.client import Client
import mcp.client.stdio

async with mcp.client.stdio.stdio_client() as (read_stream, write_stream):
    client = Client()
    await client.connect(read_stream, write_stream)
    kg_client = Polaris(client)
```

### Entity Operations

```python
# Create a code entity
await kg_client.create_code_entity(
    name="data_processor",
    code_type="class",
    source_code="class DataProcessor:\n    def process(self):\n        pass",
    dependencies=["pandas", "numpy"]
)

# Create a game system
await kg_client.create_game_system(
    name="resource_system",
    mechanics=["gathering", "crafting"],
    resources=["wood", "stone"],
    balance_metrics={"gathering_rate": 0.5}
)
```

### Search Operations

```python
# Semantic search
query = SemanticQuery(
    query_text="Find documents about machine learning",
    similarity_threshold=0.7,
    semantic_operators=["expand_synonyms"]
)
results = await search_engine.search(query)

# Graph search
graph_query = GraphQuery(
    entity_types=["person", "organization"],
    max_depth=3,
    traversal_strategy="breadth_first"
)
graph_results = await search_engine.search(graph_query)
```

## Components

### Core Package

The core package provides fundamental functionality for working with knowledge graphs:

- **Graph Operations**:

  - Efficient adjacency list representation
  - Fast neighbor lookups
  - Comprehensive traversal algorithms
  - Path finding and analysis
  - Component identification
  - Metric calculations

- **Domain Models**:
  - Entity and relation models
  - Rich metadata support
  - Quality metrics integration
  - Type-based classification

### Infrastructure Layer

The infrastructure layer provides essential services:

- **Storage System**:

  - Pluggable storage backends (JSON, SQLite)
  - Entity and relation storage
  - Backup and restore functionality
  - Thread-safe operations

- **Cache System**:

  - LRU caching with TTL
  - Thread-safe operations
  - Automatic cleanup
  - Custom serialization

- **Event System**:
  - Async event processing
  - Event persistence
  - Dead letter queue
  - Schema validation

### Repository Layer

The repository layer implements the repository pattern:

- **Base Repository**:

  - Generic CRUD operations
  - Caching support
  - Validation framework
  - Transaction support

- **Specialized Repositories**:
  - Entity repository
  - Relation repository
  - Custom validation
  - Dependency management

### Search Module

The search module provides comprehensive search capabilities:

- **Search Types**:

  - Basic text search
  - Semantic search with embeddings
  - Graph-based search
  - Faceted search

- **Query Features**:
  - Complex filtering
  - Custom sorting
  - Pagination
  - Result aggregation

## Best Practices

1. **Entity Management**

   - Use meaningful entity names
   - Provide comprehensive metadata
   - Maintain consistent entity types

2. **Performance Optimization**

   - Leverage caching appropriately
   - Use batch operations when possible
   - Implement proper indexing
   - Monitor resource usage

3. **Error Handling**
   - Implement comprehensive error catching
   - Provide meaningful error messages
   - Use appropriate recovery strategies

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Follow code style guidelines (black)
4. Add comprehensive tests
5. Update documentation
6. Submit a pull request with a detailed description

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or feedback, please contact:

- Email: john@foundryside.dev
- GitHub: [tachyon-beep](https://github.com/tachyon-beep)
