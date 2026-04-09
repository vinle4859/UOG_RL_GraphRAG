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
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import get_settings


def setup_logging(level: str | None = None, log_file: Path | str | None = None) -> None:
    """
    Configure the root logger with a consistent format.

    Parameters
    ----------
    level : str, optional
        Override the log level from settings.
    log_file : Path, optional
        Write logs to this file in addition to stdout.  If not given,
        defaults to ``results/logs/graphrag_<date>.log``.  Pass ``False``
        to disable file logging entirely.
    """
    level = level or get_settings().log_level
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    # File handler — always on unless caller explicitly passes log_file=False
    if log_file is not False:
        if log_file is None:
            log_dir = Path("results/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"graphrag_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def log_stage(logger: logging.Logger, title: str) -> None:
    """Emit a readable stage banner for long-running workflows."""
    line = "=" * 18
    logger.info("%s %s %s", line, title, line)
