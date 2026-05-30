"""Custom exception hierarchy for Brain OS.

All Brain OS-specific exceptions inherit from :class:`BrainOSError` so callers
can catch the entire family with a single ``except BrainOSError`` clause
while still being able to handle specific sub-types.
"""

from __future__ import annotations


class BrainOSError(Exception):
    """Base exception for all Brain OS-specific errors."""


class LLMError(BrainOSError):
    """An LLM API call failed after retries."""


class ToolExecutionError(BrainOSError):
    """A ReAct tool raised an unexpected error during execution."""


class ConfigurationError(BrainOSError):
    """A required configuration value is missing or invalid."""


class DatabaseError(BrainOSError):
    """A database operation (CRM, SQLite, Neo4j) failed."""


class IngestionError(BrainOSError):
    """Document ingestion or indexing failed."""


class PathTraversalError(BrainOSError):
    """A file path resolved outside the allowed root directory."""
