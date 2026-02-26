# =============================================================================
# src/utils/logging_setup.py
# Consistent logging configuration for the project.
# =============================================================================
"""
Logging Setup
=============
Call ``setup_logging()`` once at application entry points (CLI, notebooks)
to configure a uniform log format.
"""

from __future__ import annotations

import logging
import sys

from src.config import get_settings


def setup_logging(level: str | None = None) -> None:
    """
    Configure the root logger with a consistent format.

    Parameters
    ----------
    level : str, optional
        Override the log level from settings.
    """
    level = level or get_settings().log_level

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
