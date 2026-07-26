from __future__ import annotations

import os
import tempfile
from pathlib import Path


def default_runtime_dir() -> Path:
    """Return a user-writable directory for traces, eval output and local artifacts."""
    configured = os.getenv("CODEGRAPH_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / "codegraph-agent"


def resolve_runtime_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return default_runtime_dir() / path
