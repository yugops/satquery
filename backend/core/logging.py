"""
SatQuery AI — Structured Audit Logging
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_LOGGER: logging.Logger | None = None


def get_logger(name: str = "SatQueryAI") -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [SatQuery] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGER = logger
    return logger


def log_agent_event(agent_name: str, status: str, detail: str = "") -> None:
    logger = get_logger()
    msg = f"🛰️ [{agent_name}] -> {status}"
    if detail:
        msg += f" | {detail}"
    logger.info(msg)
