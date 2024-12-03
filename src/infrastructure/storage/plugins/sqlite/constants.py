"""
Constants for SQLite storage plugin.

This module defines constants used by the SQLite storage implementation:
- File names and paths
- SQL schema definitions
- Common SQL queries
"""

# Database file name
STORAGEDB = "storage.db"

# SQL Schema Definitions

NODE_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    name TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    observations TEXT NOT NULL,
    attributes TEXT NOT NULL,
    metadata TEXT NOT NULL,
    dependencies TEXT NOT NULL,
    documentation TEXT,
    code_reference TEXT,
    data_schema TEXT,
    metrics TEXT,
    validation_rules TEXT,
    examples TEXT
)
"""

EDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    attributes TEXT NOT NULL,
    metadata TEXT NOT NULL,
    context TEXT,
    impact_score REAL,
    validation_status TEXT,
    PRIMARY KEY (from_node, to_node, relation_type)
)
"""

# Common SQL Queries

# Node queries
SELECT_NODE_BY_NAME = "SELECT * FROM nodes WHERE name = ?"
SELECT_ONE_BY_NAME = "SELECT 1 FROM nodes WHERE name = ?"
INSERT_NODE = """
INSERT INTO nodes (
    name, entity_type, observations, attributes,
    metadata, dependencies, documentation,
    code_reference, data_schema, metrics,
    validation_rules, examples
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
UPDATE_NODE = """
UPDATE nodes
SET entity_type = ?, observations = ?, attributes = ?,
    metadata = ?, dependencies = ?, documentation = ?,
    code_reference = ?, data_schema = ?, metrics = ?,
    validation_rules = ?, examples = ?
WHERE name = ?
"""
DELETE_NODE = "DELETE FROM nodes WHERE name = ?"

# Edge queries
SELECT_EDGE = """
SELECT * FROM edges
WHERE from_node = ? AND to_node = ? AND relation_type = ?
"""
SELECT_ONE_EDGE = """
SELECT 1 FROM edges
WHERE from_node = ? AND to_node = ? AND relation_type = ?
"""
INSERT_EDGE = """
INSERT INTO edges (
    from_node, to_node, relation_type,
    attributes, metadata, context,
    impact_score, validation_status
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""
UPDATE_EDGE = """
UPDATE edges
SET attributes = ?, metadata = ?, context = ?,
    impact_score = ?, validation_status = ?
WHERE from_node = ? AND to_node = ? AND relation_type = ?
"""
DELETE_EDGE = """
DELETE FROM edges
WHERE from_node = ? AND to_node = ? AND relation_type = ?
"""
