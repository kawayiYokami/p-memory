"""Stable error categories shared with the Rust core."""
from __future__ import annotations


class PMemoryError(Exception):
    code = "unknown"


class ValidationError(PMemoryError, ValueError):
    code = "validation"


class NotFoundError(PMemoryError, LookupError):
    code = "not_found"


class ConflictError(PMemoryError):
    code = "conflict"


class LockedError(PMemoryError):
    code = "locked"


class ClosedError(PMemoryError):
    code = "closed"


class SchemaVersionError(PMemoryError):
    code = "schema_version"


class InvalidVectorError(PMemoryError, ValueError):
    code = "invalid_vector"


class StaleRevisionError(PMemoryError):
    code = "stale_revision"


class IndexError(PMemoryError):
    code = "index"


class StorageError(PMemoryError):
    code = "storage"


class IOError(PMemoryError):
    code = "io"


class EmbedCallbackError(Exception):
    """Raised inside a host embedder to declare the failure category.

    The core does not parse error text: it reads `kind` and decides whether to
    halve the batch and retry (`too_large`), back off and retry (`rate_limited`),
    or stop and degrade to text (`other`).
    """
    code = "embed_callback"

    def __init__(self, message: str, *, kind: str = "other"):
        super().__init__(message)
        self.kind = kind

    @classmethod
    def too_large(cls, message: str) -> EmbedCallbackError:
        return cls(message, kind="too_large")

    @classmethod
    def rate_limited(cls, message: str) -> EmbedCallbackError:
        return cls(message, kind="rate_limited")

    @classmethod
    def other(cls, message: str) -> EmbedCallbackError:
        return cls(message, kind="other")


_ERRORS = {cls.code: cls for cls in (
    ValidationError, NotFoundError, ConflictError, LockedError, ClosedError,
    SchemaVersionError, InvalidVectorError, StaleRevisionError, IndexError,
    StorageError, IOError,
)}


def from_native(error: dict) -> PMemoryError:
    return _ERRORS.get(error.get("code"), PMemoryError)(error.get("message", "native error"))
