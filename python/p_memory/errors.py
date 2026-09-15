"""Stable error categories shared with the Rust core."""


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


_ERRORS = {cls.code: cls for cls in (
    ValidationError, NotFoundError, ConflictError, LockedError, ClosedError,
    SchemaVersionError, InvalidVectorError, StaleRevisionError, IndexError,
    StorageError, IOError,
)}


def from_native(error: dict) -> PMemoryError:
    return _ERRORS.get(error.get("code"), PMemoryError)(error.get("message", "native error"))
