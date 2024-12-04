"""
Event bus system for knowledge graph events with persistence, batching, and validation.

This module implements a robust event bus system for handling knowledge graph events.
Key features include:
- Asynchronous event processing
- Event persistence and dead letter queue
- Event batching for improved performance
- Schema validation for event data
- Event replay capabilities
- Thread-safe operations
- Comprehensive error handling and logging

The event bus supports various types of events related to node and edge
operations, as well as system events like cache operations and error reporting.
It provides both a core EventBus implementation and a simplified client interface
for common operations.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate

from ..core.exceptions import ValidationError
from .storage.plugins.event_store import EventStore
from .storage.plugins.event_store.models import Event, EventType

logger = logging.getLogger(__name__)


class BatchConfig:
    """
    Configuration for event batching.

    This class defines parameters for controlling event batching behavior:
    - Maximum batch size before processing
    - Timeout period for batch processing
    - Maximum retry attempts for failed events

    Attributes:
        batch_size: Maximum number of events in a batch
        batch_timeout: Time in seconds before processing an incomplete batch
        max_retries: Maximum number of retry attempts for failed events
    """

    def __init__(
        self,
        batch_size: int = 100,
        batch_timeout: float = 1.0,
        max_retries: int = 3,
    ):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_retries = max_retries


class EventBus:
    """
    Asynchronous event bus implementation with persistence, batching, and validation.

    This class provides the core event bus functionality, including:
    - Event publication and subscription
    - Event persistence and retrieval
    - Batch processing of events
    - Schema validation
    - Event replay capabilities
    - Dead letter queue management

    The event bus ensures reliable event delivery and processing while
    maintaining system performance through batching and asynchronous operations.
    """

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        batch_config: Optional[BatchConfig] = None,
    ):
        """
        Initialize the event bus.

        Args:
            event_store: Optional custom event store implementation
            batch_config: Optional custom batch configuration
        """
        self.subscribers: Dict[EventType, List[Callable[[Event], Any]]] = {
            event_type: [] for event_type in EventType
        }
        self.event_store = event_store or EventStore()
        self.batch_config = batch_config or BatchConfig()

        # Batching state
        self.current_batch: Dict[str, List[Event]] = {}
        self.batch_timers: Dict[str, asyncio.Task] = {}

        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Configure event bus logging.

        Sets up file-based logging for the event bus with timestamp,
        log level, and formatted messages.
        """
        handler = logging.FileHandler("event_bus.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    async def publish(self, event: Event, batch: bool = True, validate_schema: bool = True) -> None:
        """
        Publish an event to all subscribers.

        This method handles the complete event publication process:
        1. Optional schema validation
        2. Event persistence
        3. Batching (if enabled)
        4. Delivery to subscribers

        Args:
            event: Event to publish
            batch: Whether to batch the event for processing
            validate_schema: Whether to validate event schema

        Raises:
            ValidationError: If event fails schema validation
            Exception: If event publication fails
        """
        logger.debug(f"Publishing event: {event.type.value}")

        # Initialize event_id before try block
        event_id = None

        try:
            # Validate event schema if enabled
            if validate_schema:
                await self._validate_event_schema(event)

            # Store event
            event_id = self.event_store.store_event(event)

            if batch:
                await self._handle_batched_event(event)
            else:
                await self._process_event(event)

        except Exception as e:
            logger.error(f"Error publishing event: {str(e)}")
            if event_id is not None:
                self.event_store.move_to_dead_letter_queue(event_id, str(e))
            raise

    async def _validate_event_schema(self, event: Event) -> None:
        """
        Validate event against its schema.

        Validates the event data against a registered JSON schema for its type.

        Args:
            event: Event to validate

        Raises:
            ValidationError: If event fails schema validation
        """
        schema = self.event_store.get_event_schema(event.type)
        if schema:
            try:
                validate(instance=event.data, schema=schema)
            except JsonSchemaError as e:
                raise ValidationError(f"Event schema validation failed: {str(e)}")

    async def _handle_batched_event(self, event: Event) -> None:
        """
        Handle event batching logic.

        Manages the addition of events to batches and triggers batch processing
        when size thresholds are met.

        Args:
            event: Event to batch
        """
        batch_key = event.type.value

        # Create new batch if needed
        if batch_key not in self.current_batch:
            self.current_batch[batch_key] = []
            self._start_batch_timer(batch_key)

        self.current_batch[batch_key].append(event)

        # Process batch if size threshold reached
        if len(self.current_batch[batch_key]) >= self.batch_config.batch_size:
            await self._process_batch(batch_key)

    def _start_batch_timer(self, batch_key: str) -> None:
        """
        Start timer for batch processing.

        Creates an async timer that will trigger batch processing after
        the configured timeout period.

        Args:
            batch_key: Key identifying the batch
        """
        if batch_key in self.batch_timers:
            self.batch_timers[batch_key].cancel()

        async def _timer():
            await asyncio.sleep(self.batch_config.batch_timeout)
            await self._process_batch(batch_key)

        self.batch_timers[batch_key] = asyncio.create_task(_timer())

    async def _process_batch(self, batch_key: str) -> None:
        """
        Process a batch of events.

        Handles the processing of a complete batch of events, including:
        1. Batch creation in event store
        2. Processing of individual events
        3. Batch completion
        4. Cleanup of batch state

        Args:
            batch_key: Key identifying the batch

        Raises:
            Exception: If batch processing fails
        """
        if batch_key not in self.current_batch:
            return

        events = self.current_batch[batch_key]
        if not events:
            return

        # Create batch ID and mark batch start
        batch_id = str(uuid.uuid4())
        self.event_store.create_batch(batch_id)

        try:
            # Process all events in batch
            for event in events:
                await self._process_event(event, batch_id)

            # Mark batch complete
            self.event_store.complete_batch(batch_id)

        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {str(e)}")
            raise
        finally:
            # Clear batch state
            del self.current_batch[batch_key]
            if batch_key in self.batch_timers:
                self.batch_timers[batch_key].cancel()
                del self.batch_timers[batch_key]

    async def _process_event(self, event: Event, _batch_id: Optional[str] = None) -> None:
        """
        Process a single event.

        Delivers the event to all registered subscribers, handling both
        synchronous and asynchronous subscribers appropriately.

        Args:
            event: Event to process
            _batch_id: Optional batch ID for the event (unused but required for batch tracking)
        """
        if event.type not in self.subscribers:
            logger.warning(f"No subscribers for event type: {event.type.value}")
            return

        coroutines = []
        for subscriber in self.subscribers[event.type]:
            try:
                # Wrap synchronous handlers in asyncio.to_thread
                if asyncio.iscoroutinefunction(subscriber):
                    coroutines.append(subscriber(event))
                else:
                    coroutines.append(asyncio.to_thread(subscriber, event))
            except Exception as e:
                logger.error(f"Error calling subscriber for {event.type.value}: {str(e)}")

        if coroutines:
            # Wait for all subscribers to process the event
            await asyncio.gather(*coroutines, return_exceptions=True)

    async def retry_failed_events(self) -> None:
        """
        Retry processing events from the dead letter queue.

        Attempts to reprocess events that previously failed, up to the
        configured maximum retry limit.
        """
        events = self.event_store.retry_dead_letter_queue(max_retries=self.batch_config.max_retries)

        for event in events:
            try:
                await self._process_event(event)
            except Exception as e:
                logger.error(f"Error retrying event: {str(e)}")

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Any],
        replay_history: bool = False,
    ) -> None:
        """
        Subscribe to an event type.

        Registers a handler function to receive events of a specific type.
        Optionally replays historical events to the new subscriber.

        Args:
            event_type: Type of event to subscribe to
            handler: Event handler function
            replay_history: Whether to replay historical events
        """
        logger.debug(f"Adding subscriber for: {event_type.value}")
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

        # Replay historical events if requested
        if replay_history:
            events = self.event_store.get_events(event_type=event_type)
            for historical_event in events:
                asyncio.create_task(self._replay_event(historical_event, handler))

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> bool:
        """
        Unsubscribe from an event type.

        Removes a handler function from the subscribers list for a specific event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Event handler function to remove

        Returns:
            True if handler was removed, False otherwise
        """
        logger.debug(f"Removing subscriber for: {event_type.value}")
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            return True
        return False

    async def _replay_event(self, event: Event, handler: Callable[[Event], Any]) -> None:
        """
        Replay a historical event to a subscriber.

        Delivers a historical event to a specific handler, handling both
        synchronous and asynchronous handlers appropriately.

        Args:
            event: Historical event to replay
            handler: Handler to receive the event
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                await asyncio.to_thread(handler, event)
        except Exception as e:
            logger.error(f"Error replaying event {event.type.value}: {str(e)}")

    def get_subscriber_count(self, event_type: EventType) -> int:
        """
        Get the number of subscribers for an event type.

        Args:
            event_type: Event type to check

        Returns:
            Number of subscribers for the event type
        """
        return len(self.subscribers.get(event_type, []))

    def clear_subscribers(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear all subscribers for a specific event type or all event types.

        Args:
            event_type: Optional event type to clear subscribers for.
                       If None, clears all subscribers.
        """
        if event_type:
            self.subscribers[event_type] = []
        else:
            self.subscribers = {event_type: [] for event_type in EventType}

    def register_event_schema(self, event_type: EventType, schema: Dict[str, Any]) -> None:
        """
        Register a JSON schema for event validation.

        Args:
            event_type: Event type to register schema for
            schema: JSON Schema definition for validating event data
        """
        self.event_store.store_event_schema(event_type, schema)


class EventBusClient:
    """
    Client interface for interacting with the event bus.

    This class provides a simplified interface for common event bus operations,
    abstracting away the complexity of direct event bus interaction. It offers
    convenience methods for publishing different types of events with appropriate
    defaults and error handling.
    """

    def __init__(self, event_bus: EventBus):
        """
        Initialize the event bus client.

        Args:
            event_bus: EventBus instance to interact with
        """
        self.event_bus = event_bus

    async def publish_node_event(
        self,
        operation: str,
        node_id: str,
        data: Dict[str, Any],
        batch: bool = True,
    ) -> None:
        """
        Publish a node-related event.

        Args:
            operation: Operation type (created, updated, deleted)
            node_id: ID of the node
            data: Additional event data
            batch: Whether to batch the event

        Raises:
            ValueError: If operation type is invalid
        """
        event_type_map = {
            "created": EventType.NODE_CREATED,
            "updated": EventType.NODE_UPDATED,
            "deleted": EventType.NODE_DELETED,
        }

        event_type = event_type_map.get(operation.lower())
        if not event_type:
            raise ValueError(f"Invalid operation: {operation}")

        event = Event(
            type=event_type,
            data={"node_id": node_id, **data},
            timestamp=datetime.now(),
            metadata={"operation": operation},
        )
        await self.event_bus.publish(event, batch=batch)

    async def publish_edge_event(
        self,
        operation: str,
        edge_id: str,
        data: Dict[str, Any],
        batch: bool = True,
    ) -> None:
        """
        Publish an edge-related event.

        Args:
            operation: Operation type (created, updated, deleted)
            edge_id: ID of the edge
            data: Additional event data
            batch: Whether to batch the event

        Raises:
            ValueError: If operation type is invalid
        """
        event_type_map = {
            "created": EventType.EDGE_CREATED,
            "updated": EventType.EDGE_UPDATED,
            "deleted": EventType.EDGE_DELETED,
        }

        event_type = event_type_map.get(operation.lower())
        if not event_type:
            raise ValueError(f"Invalid operation: {operation}")

        event = Event(
            type=event_type,
            data={"edge_id": edge_id, **data},
            timestamp=datetime.now(),
            metadata={"operation": operation},
        )
        await self.event_bus.publish(event, batch=batch)

    async def publish_error_event(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        batch: bool = False,
    ) -> None:
        """
        Publish an error event.

        Args:
            error_type: Type of error
            message: Error message
            details: Optional error details
            batch: Whether to batch the event (defaults to False for faster processing)
        """
        event = Event(
            type=EventType.SCHEMA_UPDATED,  # Using SCHEMA_UPDATED as a generic error event type
            data={
                "error_type": error_type,
                "message": message,
                "details": details or {},
            },
            timestamp=datetime.now(),
        )
        # Error events are not batched by default for faster processing
        await self.event_bus.publish(event, batch=batch)
