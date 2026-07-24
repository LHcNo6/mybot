"""Stage 7.1: Session metadata.

A :class:`SessionMeta` records lifecycle information about a session:
title, created_at, updated_at, and (in 7.2) the consolidation cursor
and last summary. The metadata is persisted as the first JSONL line
using the ``_type=metadata`` sentinel that nananobot uses in
``nanobot/agent/memory.py:616``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

META_TYPE_KEY = "_type"
META_TYPE_VALUE = "metadata"


@dataclass
class SessionMeta:
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_consolidated: int = 0
    last_summary: str | None = None


def new_meta() -> SessionMeta:
    """Return a fresh metadata block with current timestamps."""
    now = datetime.now().isoformat(timespec="seconds")
    return SessionMeta(created_at=now, updated_at=now)


def touch(meta: SessionMeta) -> None:
    """Bump ``updated_at`` to now (mutates in place)."""
    meta.updated_at = datetime.now().isoformat(timespec="seconds")


def is_metadata_record(record: dict[str, Any]) -> bool:
    return record.get(META_TYPE_KEY) == META_TYPE_VALUE


def record_from_meta(meta: SessionMeta) -> dict[str, Any]:
    return {META_TYPE_KEY: META_TYPE_VALUE, **asdict(meta)}


def meta_from_record(record: dict[str, Any]) -> SessionMeta:
    return SessionMeta(
        title=record.get("title"),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
        last_consolidated=record.get("last_consolidated", 0),
        last_summary=record.get("last_summary"),
    )