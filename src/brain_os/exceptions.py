"""Custom exception hierarchy for Ira.

All Ira-specific exceptions inherit from :class:`IraError` so callers
can catch the entire family with a single ``except IraError`` clause
while still being able to handle specific sub-types.
"""

from __future__ import annotations


class IraError(Exception):
    """Base exception for all Ira-specific errors."""


class LLMError(IraError):
    """An LLM API call failed after retries."""


class ToolExecutionError(IraError):
    """A ReAct tool raised an unexpected error during execution."""


class ConfigurationError(IraError):
    """A required configuration value is missing or invalid."""


class DatabaseError(IraError):
    """A database operation (CRM, SQLite, Neo4j) failed."""


class IngestionError(IraError):
    """Document ingestion or indexing failed."""


class PathTraversalError(IraError):
    """A file path resolved outside the allowed root directory."""
