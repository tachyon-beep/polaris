"""
Custom exceptions for the knowledge graph system.

This module defines the hierarchy of custom exceptions used throughout the system
to handle various error conditions in a structured and meaningful way. Each exception
type corresponds to a specific category of errors that may occur during system operation.
"""


class ValidationError(Exception):
    """
    Raised when data validation fails.

    This exception is raised when input data fails to meet the required validation
    criteria, such as schema validation, data type checks, or business rule validation.

    Examples:
        * Invalid node attributes
        * Malformed relationship data
        * Schema validation failures
    """

    def __str__(self) -> str:
        """Format validation error message."""
        return f"Validation Error: {super().__str__()}"


class StorageError(Exception):
    """
    Raised when storage operations fail.

    This exception is raised when operations involving data persistence encounter
    errors, such as database connection issues, file system errors, or storage
    constraint violations.

    Examples:
        * Database connection failures
        * File system access errors
        * Storage quota exceeded
    """


class CacheError(Exception):
    """
    Raised when cache operations fail.

    This exception is raised when operations involving the cache system encounter
    errors, such as serialization failures, capacity issues, or cache consistency
    problems.

    Examples:
        * Serialization/deserialization failures
        * Cache capacity exceeded
        * Cache consistency violations
        * Thread synchronization issues
    """


class QueryError(Exception):
    """
    Raised when query operations fail.

    This exception is raised when errors occur during query execution, such as
    invalid query parameters, malformed query syntax, or query execution timeouts.

    Examples:
        * Invalid query parameters
        * Malformed query syntax
        * Query timeout
    """


class GraphOperationError(Exception):
    """
    Raised when graph operations fail.

    This exception is raised when operations on the knowledge graph structure
    encounter errors, such as invalid node/edge operations or graph integrity
    violations.

    Examples:
        * Invalid node operations
        * Edge creation failures
        * Graph integrity violations
    """

    def __str__(self) -> str:
        """Format graph operation error message."""
        return f"Graph Operation Error: {super().__str__()}"


class EventError(Exception):
    """
    Raised when event operations fail.

    This exception is raised when event processing encounters errors, such as
    event publication failures, subscription errors, or event handling issues.

    Examples:
        * Event publication failures
        * Subscription errors
        * Event handler exceptions
    """


class ConfigurationError(Exception):
    """
    Raised when configuration is invalid.

    This exception is raised when system configuration issues are detected,
    such as missing required settings, invalid configuration values, or
    configuration conflicts.

    Examples:
        * Missing required settings
        * Invalid configuration values
        * Configuration conflicts
    """


class AuthenticationError(Exception):
    """
    Raised when authentication fails.

    This exception is raised when user authentication encounters errors,
    such as invalid credentials, expired tokens, or authentication service
    failures.

    Examples:
        * Invalid credentials
        * Expired authentication tokens
        * Authentication service failures
    """


class AuthorizationError(Exception):
    """
    Raised when authorization fails.

    This exception is raised when a user lacks the necessary permissions
    to perform an operation, or when access control checks fail.

    Examples:
        * Insufficient permissions
        * Role-based access control violations
        * Resource access restrictions
    """


class ResourceNotFoundError(Exception):
    """
    Raised when a requested resource is not found.

    This exception is raised when attempting to access or operate on a
    resource that does not exist in the system.

    Examples:
        * Node not found
        * Edge not found
        * Configuration not found
    """


class NodeNotFoundError(ResourceNotFoundError):
    """
    Raised when a requested node is not found.

    This exception is a specialized version of ResourceNotFoundError specifically
    for node-related operations where the requested node does not exist.

    Examples:
        * Node lookup by non-existent ID
        * Node update for missing node
        * Node deletion for non-existent node
    """


class EdgeNotFoundError(ResourceNotFoundError):
    """
    Raised when a requested edge is not found.

    This exception is a specialized version of ResourceNotFoundError specifically
    for edge-related operations where the requested edge does not exist.

    Examples:
        * Edge lookup by non-existent ID
        * Edge update for missing edge
        * Edge deletion for non-existent edge
    """


class DuplicateResourceError(Exception):
    """
    Raised when attempting to create a duplicate resource.

    This exception is raised when attempting to create a resource that
    already exists in the system, violating uniqueness constraints.

    Examples:
        * Duplicate node creation
        * Duplicate edge creation
        * Unique constraint violations
    """


class InvalidOperationError(Exception):
    """
    Raised when an operation is invalid in the current context.

    This exception is raised when attempting to perform an operation that
    is not valid given the current state or context of the system.

    Examples:
        * Invalid state transitions
        * Unsupported operations
        * Context-specific violations
    """


class RateLimitError(Exception):
    """
    Raised when rate limits are exceeded.

    This exception is raised when operation frequency exceeds defined
    rate limits, protecting system resources and ensuring fair usage.

    Examples:
        * API rate limit exceeded
        * Query frequency limits
        * Resource utilization thresholds
    """


class ServiceUnavailableError(Exception):
    """
    Raised when a required service is unavailable.

    This exception is raised when a dependent service or system component
    is unavailable, preventing normal operation.

    Examples:
        * External service downtime
        * System maintenance
        * Resource exhaustion
    """
