"""Embedded memory for Rust and Python, with an optional simple agent."""
from ._native import __version__
from .api import AsyncKnowledgeBase, KnowledgeBase, import_legacy
from .errors import (ClosedError, ConflictError, EmbedCallbackError, InvalidVectorError,
                     LockedError, NotFoundError, PMemoryError, SchemaVersionError,
                     StaleRevisionError, StorageError, ValidationError)

__all__ = ["KnowledgeBase", "AsyncKnowledgeBase", "import_legacy", "__version__",
           "PMemoryError", "ValidationError", "NotFoundError", "ConflictError",
           "LockedError", "ClosedError", "InvalidVectorError", "StaleRevisionError",
           "StorageError", "SchemaVersionError", "EmbedCallbackError"]
