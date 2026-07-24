"""Stage 7: Session persistence to JSONL.

Each session is one ``<key>.jsonl`` file under
``~/.mybot/sessions/`` (mirroring nananobot's
``~/.nanobot/workspace/sessions/``). Writes are atomic via
``temp + fsync + os.replace``; reads skip and warn on corrupt lines
(matching ``SessionManager._repair``).

Two entry points: :func:`save_messages` and :func:`load_messages`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIRNAME = ".mybot"
SESSIONS_DIRNAME = "sessions"
SESSION_SUFFIX = ".jsonl"


def _sanitize_key(key: str) -> str:
    """Map arbitrary session keys to safe filename components."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return cleaned or "default"


def sessions_dir() -> Path:
    """Return ``~/.mybot/sessions/``, creating it if necessary."""
    path = Path.home() / DATA_DIRNAME / SESSIONS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_path(key: str) -> Path:
    return sessions_dir() / f"{_sanitize_key(key)}{SESSION_SUFFIX}"


def save_messages(key: str, messages: Iterable[dict[str, Any]]) -> Path:
    """Atomically write ``messages`` to the session's JSONL file."""
    path = session_path(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_messages(key: str) -> list[dict[str, Any]]:
    """Return messages for ``key``, or ``[]`` if no session exists.

    Corrupt lines are skipped with a warning (mirrors nananobot's
    :meth:`SessionManager._repair`).
    """
    path = session_path(key)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        logger.warning(
            "Skipped %d corrupt line(s) loading session %r", skipped, key
        )
    return out