"""
SatQuery AI — Structured Logging

Provides a pre-configured logger factory that attaches agent context
and outputs structured JSON-style log messages.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


_LOG_FORMAT = (
    "[%(asctime)s] %(levelname)-8s | %(name)-28s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track whether the root SatQuery logger has been configured
_configured = False


def _configure_root_logger(level: str = "INFO") -> None:
    """One-time setup of the root 'satquery' logger."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("satquery")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Prevent propagation to the global root logger
    root.propagate = False
    _configured = True


def get_logger(agent_name: str, level: str = "INFO") -> logging.Logger:
    """Return a child logger under the 'satquery' namespace.

    Args:
        agent_name: Name of the agent or module (used as logger suffix).
        level: Log level string (INFO, DEBUG, WARNING, etc.).

    Returns:
        A configured logging.Logger instance.
    """
    _configure_root_logger(level)
    return logging.getLogger(f"satquery.{agent_name}")


def log_agent_event(
    logger: logging.Logger,
    event: str,
    *,
    agent: str = "",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a structured log entry for an agent lifecycle event.

    Args:
        logger: The logger instance to use.
        event: Short event label (e.g., 'start', 'complete', 'error').
        agent: Agent class name.
        session_id: Current session/trace ID.
        extra: Additional key-value pairs to include.
    """
    parts = [f"event={event}"]
    if agent:
        parts.append(f"agent={agent}")
    if session_id:
        parts.append(f"session={session_id}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))
