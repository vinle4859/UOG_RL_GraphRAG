# =============================================================================
# src/utils/io_helpers.py
# File I/O and path helpers.
# =============================================================================
"""
I/O Helpers
===========
Small utilities for reading/writing JSON, ensuring directories exist, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: Any, path: Path | str, indent: int = 2) -> None:
    """Write *data* to a JSON file, creating parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.debug("Wrote JSON to %s", path)


def read_json(path: Path | str) -> Any:
    """Read and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
