"""Structured error categories shared across serving layers."""

import asyncio
from enum import StrEnum


class ErrorCategory(StrEnum):
    """Stable, low-cardinality error categories for logs and API details."""

    BACKEND_UNHEALTHY = "backend_unhealthy"
    BACKEND_LOAD_FAILED = "backend_load_failed"
    BACKEND_GENERATION_FAILED = "backend_generation_failed"
    MEMORY_PRESSURE = "memory_pressure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    OVERLOADED = "overloaded"
    UNKNOWN = "unknown"


class OrchardError(RuntimeError):
    """Base class for categorized runtime failures."""

    category = ErrorCategory.UNKNOWN


class BackendLoadError(OrchardError):
    """Raised when a backend cannot load its model."""

    category = ErrorCategory.BACKEND_LOAD_FAILED


class BackendUnhealthyError(OrchardError):
    """Raised when a backend refuses work because it is not ready."""

    category = ErrorCategory.BACKEND_UNHEALTHY


class BackendGenerationError(OrchardError):
    """Raised when backend generation fails before normal completion."""

    category = ErrorCategory.BACKEND_GENERATION_FAILED


class MemoryPressureError(OrchardError):
    """Raised when a backend simulates or detects memory pressure."""

    category = ErrorCategory.MEMORY_PRESSURE


def error_category(exc: BaseException) -> ErrorCategory:
    """Return a bounded error category for logging, metrics, and API details."""

    if isinstance(exc, OrchardError):
        return exc.category
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, asyncio.CancelledError):
        return ErrorCategory.CANCELLED
    return ErrorCategory.UNKNOWN
