"""
Constants and schema definitions for event store.

This module defines:
- SQL schemas for event store tables
- Common SQL queries
- Configuration constants
"""

# Schema Definitions

CREATE_TABLES_SQL = """
-- Events table stores all published events
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    data TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT,
    batch_id TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Dead letter queue for failed events
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    error_message TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    last_retry TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(id)
);

-- Event batches for bulk processing
CREATE TABLE IF NOT EXISTS event_batches (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    event_count INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

-- Event schemas for validation
CREATE TABLE IF NOT EXISTS event_schemas (
    event_type TEXT PRIMARY KEY,
    schema TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# Common SQL Queries

# Event queries
INSERT_EVENT = """
INSERT INTO events (event_type, data, timestamp, metadata, batch_id)
VALUES (?, ?, ?, ?, ?)
"""

SELECT_EVENTS_BASE = """
SELECT * FROM events WHERE 1=1
"""

# Dead letter queue queries
INSERT_DLQ = """
INSERT INTO dead_letter_queue (event_id, error_message)
VALUES (?, ?)
"""

SELECT_DLQ_EVENTS = """
SELECT e.*, d.retry_count
FROM events e
JOIN dead_letter_queue d ON e.id = d.event_id
WHERE d.retry_count < ?
"""

UPDATE_DLQ_RETRY = """
UPDATE dead_letter_queue
SET retry_count = retry_count + 1,
    last_retry = CURRENT_TIMESTAMP
WHERE event_id = ?
"""

# Batch queries
INSERT_BATCH = """
INSERT INTO event_batches (id, status, event_count)
VALUES (?, 'PENDING', 0)
"""

UPDATE_BATCH_COMPLETE = """
UPDATE event_batches
SET status = 'COMPLETED',
    completed_at = CURRENT_TIMESTAMP
WHERE id = ?
"""

# Schema queries
UPSERT_SCHEMA = """
INSERT OR REPLACE INTO event_schemas (event_type, schema, updated_at)
VALUES (?, ?, CURRENT_TIMESTAMP)
"""

SELECT_SCHEMA = """
SELECT schema FROM event_schemas WHERE event_type = ?
"""

# Configuration Constants
DEFAULT_BATCH_SIZE = 100
MAX_RETRY_ATTEMPTS = 3
DEFAULT_DB_PATH = "events.db"
