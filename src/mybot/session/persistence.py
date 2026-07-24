"""Stage 7 + 7.1: Session persistence to JSONL.

Each session is one ``<key>.jsonl`` file under
``~/.mybot/sessions/`` (mirroring nananobot's
``~/.nanobot/workspace/sessions/``). Writes are atomic via
``temp + fsync + os.replace``; reads skip and warn on corrupt lines
(matching ``SessionManager._repair``).

Stage 7.1: the first JSONL line is a metadata record of the form
``{"_type": "metadata", ...}`` mirroring
``nanobot/agent/memory.py:616``. :func:`save_messages` accepts an
optional :class:`~mybot.session.metadata.SessionMeta`; :func:`load_messages`
returns ``(messages, meta_or_None)`` so callers can introspect
title / created_at / updated_at.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mybot.session.metadata import (
    SessionMeta,
    is_metadata_record,
    meta_from_record,
    new_meta,
    record_from_meta,
)

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


def save_messages(
    key: str,
    messages: Iterable[dict[str, Any]],
    meta: SessionMeta | None = None,
) -> Path:
    """Atomically write ``messages`` (and optional ``meta``) to the session file.

    File layout::

        {"_type": "metadata", ...}      ← only if meta is provided
        {"role": "system", ...}
        {"role": "user", ...}
        ...
    """
    path = session_path(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if meta is not None:
            f.write(json.dumps(record_from_meta(meta), ensure_ascii=False) + "\n")
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_messages(key: str) -> tuple[list[dict[str, Any]], SessionMeta | None]:
    """Return ``(messages, meta)`` for ``key``. Missing file → ``([], None)``.

    The first line of an existing file may be a metadata record (Stage 7.1+)
    or a regular message (Stage 7 files written before this change). Both
    shapes are accepted; the metadata line, if present, is consumed and not
    returned in the messages list.
    """
    path = session_path(key)
    if not path.exists():
        return [], None
    out: list[dict[str, Any]] = []
    meta: SessionMeta | None = None
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if meta is None and is_metadata_record(rec):
                meta = meta_from_record(rec)
                continue
            out.append(rec)
    if skipped:
        logger.warning(
            "Skipped %d corrupt line(s) loading session %r", skipped, key
        )
    return out, meta


def load_or_init(key: str) -> tuple[list[dict[str, Any]], SessionMeta]:
    """Like :func:`load_messages` but returns a fresh meta when missing
    or when the on-disk file pre-dates metadata (Stage 7 era)."""
    messages, meta = load_messages(key)
    if meta is None:
        meta = new_meta()
    return messages, meta