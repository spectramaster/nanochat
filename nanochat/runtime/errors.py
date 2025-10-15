"""Custom runtime exceptions used across NanoChat."""

from __future__ import annotations


class NanoChatError(Exception):
    """Base class for runtime errors."""


class InferenceTimeout(NanoChatError):
    """Raised when inference exceeds the configured timeout."""


class InvalidRequest(NanoChatError):
    """Raised when inputs do not satisfy the runtime contract."""


class EngineInitializationError(NanoChatError):
    """Raised when the engine fails to initialize required resources."""
