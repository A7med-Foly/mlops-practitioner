"""Structured JSON logging configuration for prodml.

Provides a custom JSONFormatter that formats every log record into valid JSON
carrying timestamp, level, logger, message, and correlation_id (via contextvars).
"""

from contextvars import ContextVar, Token
from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any

# ContextVar storing the current request/trace correlation ID
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Retrieve the correlation ID for the current execution context."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str | None) -> Token:
    """Set the correlation ID in the current execution context.

    Args:
        correlation_id: Unique string identifier (e.g. uuid4) or None.

    Returns:
        Token that can be used to reset the context variable.
    """
    return correlation_id_ctx.set(correlation_id)


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs JSON objects.

    Every log line contains:
    - timestamp: ISO 8601 UTC timestamp string
    - level: Log level name (e.g. DEBUG, INFO, WARNING, ERROR)
    - logger: Name of the logger emitting the record
    - message: Formatted message string
    - correlation_id: Contextual trace/request ID (or None if unbound)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        # Include exception trace if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger with the structured JSONFormatter on stdout.

    Args:
        level: Minimum log level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to prevent duplicate lines
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    stream_handler.setLevel(numeric_level)

    root_logger.addHandler(stream_handler)
