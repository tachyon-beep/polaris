"""
Event store implementation.

This module provides a SQLite-based event store that supports:
- Event persistence and retrieval
- Event batch processing
- Dead letter queue for failed events
- Event schema validation
- Event replay functionality
"""

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .constants import (
    CREATE_TABLES_SQL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DB_PATH,
    INSERT_BATCH,
    INSERT_DLQ,
    INSERT_EVENT,
    MAX_RETRY_ATTEMPTS,
    SELECT_DLQ_EVENTS,
    SELECT_EVENTS_BASE,
    SELECT_SCHEMA,
    UPDATE_BATCH_COMPLETE,
    UPDATE_DLQ_RETRY,
    UPSERT_SCHEMA,
)
from .models import Event, EventType

logger = logging.getLogger(__name__)


class EventStore:
    """
    Persistent storage for event bus events.

    This class provides a SQLite-based implementation of an event store,
    supporting event persistence, batching, dead letter queue, and schema
    validation.

    Attributes:
        db_path (str): Path to SQLite database file
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize event store with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """
        Initialize database tables.

        Creates all required tables if they don't exist:
        - events: Main event storage
        - dead_letter_queue: Failed events queue
        - event_batches: Batch processing tracking
        - event_schemas: Event validation schemas
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(CREATE_TABLES_SQL)
            conn.commit()

    def store_event(self, event: Event, batch_id: Optional[str] = None) -> int:
        """
        Store an event in the database.

        Args:
            event: Event to store
            batch_id: Optional batch ID for bulk processing

        Returns:
            ID of stored event

        Raises:
            sqlite3.Error: If database operation fails
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                INSERT_EVENT,
                (
                    event.type.name,
                    json.dumps(event.data),
                    event.timestamp.isoformat(),
                    json.dumps(event.metadata) if event.metadata else None,
                    batch_id,
                ),
            )
            conn.commit()
            event_id = cursor.lastrowid
            if event_id is None:
                raise sqlite3.Error("Failed to get last inserted row ID")
            return event_id

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """
        Retrieve events matching specified criteria.

        Args:
            event_type: Optional event type filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            batch_id: Optional batch ID filter
            limit: Optional limit on number of events

        Returns:
            List of matching events
        """
        query = SELECT_EVENTS_BASE
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.name)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            Event(
                type=EventType[row["event_type"]],
                data=json.loads(row["data"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def move_to_dead_letter_queue(self, event_id: int, error_message: str) -> None:
        """
        Move a failed event to the dead letter queue.

        Args:
            event_id: ID of failed event
            error_message: Error message describing failure
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(INSERT_DLQ, (event_id, error_message))
            conn.commit()

    def retry_dead_letter_queue(self, max_retries: int = MAX_RETRY_ATTEMPTS) -> List[Event]:
        """
        Retry processing events in the dead letter queue.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            List of events to retry
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get failed events that haven't exceeded retry limit
            cursor.execute(SELECT_DLQ_EVENTS, (max_retries,))
            rows = cursor.fetchall()

            # Update retry counts
            retry_ids = [row["id"] for row in rows]
            if retry_ids:
                placeholders = ",".join("?" * len(retry_ids))
                cursor.execute(
                    f"""
                    UPDATE dead_letter_queue
                    SET retry_count = retry_count + 1,
                        last_retry = CURRENT_TIMESTAMP
                    WHERE event_id IN ({placeholders})
                    """,
                    retry_ids,
                )
                conn.commit()

        return [
            Event(
                type=EventType[row["event_type"]],
                data=json.loads(row["data"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    def create_batch(self, batch_id: str) -> None:
        """
        Create a new event batch.

        Args:
            batch_id: Unique identifier for the batch
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(INSERT_BATCH, (batch_id,))
            conn.commit()

    def complete_batch(self, batch_id: str) -> None:
        """
        Mark an event batch as completed.

        Args:
            batch_id: ID of batch to complete
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(UPDATE_BATCH_COMPLETE, (batch_id,))
            conn.commit()

    def store_event_schema(self, event_type: EventType, schema: Dict[str, Any]) -> None:
        """
        Store validation schema for an event type.

        Args:
            event_type: Event type to store schema for
            schema: JSON Schema definition
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                UPSERT_SCHEMA,
                (event_type.name, json.dumps(schema)),
            )
            conn.commit()

    def get_event_schema(self, event_type: EventType) -> Optional[Dict[str, Any]]:
        """
        Retrieve validation schema for an event type.

        Args:
            event_type: Event type to get schema for

        Returns:
            JSON Schema definition if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(SELECT_SCHEMA, (event_type.name,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
