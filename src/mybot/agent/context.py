"""Stage 6.1: Sliding-window compaction aligned to user-turn boundaries.

Keeps the system message plus the last ``max_user_turns`` user turns,
preserving the matching assistant / tool messages that follow each kept
user turn. After tail-truncation the front of the retained slice is
walked forward until it lands on a ``user`` role, so we never emit an
orphan ``tool`` result or tool-bearing ``assistant`` message whose
partner was dropped.

This aligns with nananobot's ``SessionManager.retain_recent_legal_suffix``
— the same "legal boundary" idea, expressed in the simplest possible form.

Stage 6.2: also returns the dropped body slice so the caller can hand
it to :mod:`mybot.agent.summarize` for LLM-based summarization.

Stage 6.3: adds :func:`estimate_tokens`, a character-based heuristic
that lets callers trigger compaction on a token budget instead of a
fixed turn count.
"""

from __future__ import annotations

import json
from typing import Any


def compact_messages(
    messages: list[dict[str, Any]],
    max_user_turns: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(kept, dropped)`` — compacted copy plus the body slice that was removed.

    Invariants:

    - ``messages[0]`` (if it is a system message) is always preserved in ``kept``.
    - ``kept`` contains at most ``max_user_turns`` user messages.
    - ``kept`` starts on a user turn (no leading orphan tool results
      or tool-bearing assistant messages).
    - ``dropped`` contains the body messages that were removed, in
      original order, excluding the system message.
    - When nothing needed dropping, ``dropped`` is ``[]`` and ``kept``
      equals the input slice.
    """
    if not messages:
        return list(messages), []

    system: dict[str, Any] | None = None
    body: list[dict[str, Any]]
    if messages[0].get("role") == "system":
        system = messages[0]
        body = list(messages[1:])
    else:
        body = list(messages)

    kept_from_tail: list[dict[str, Any]] = []
    user_turns = 0
    for msg in reversed(body):
        kept_from_tail.append(msg)
        if msg.get("role") == "user":
            user_turns += 1
            if user_turns >= max_user_turns:
                break

    kept_body = list(reversed(kept_from_tail))
    kept_body_len = len(kept_body)

    # Strip any leading non-user messages (orphaned tool / mid-tool-call
    # assistant). Mirrors nananobot's find_legal_message_start.
    while kept_body and kept_body[0].get("role") != "user":
        kept_body.pop(0)
        kept_body_len -= 1

    dropped_body = body[: len(body) - kept_body_len]

    kept_full = ([system, *kept_body] if system is not None else kept_body)
    return kept_full, dropped_body


def compact_for_budget(
    messages: list[dict[str, Any]],
    token_budget: int,
    min_user_turns: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compact until ``estimate_tokens(kept) <= token_budget``.

    Linear-scans :func:`compact_messages` with progressively smaller
    ``max_user_turns`` until the kept slice fits the budget. Falls back
    to ``min_user_turns`` when the budget is impossibly tight.

    Used by Stage 6.3: token-budget-driven compaction trigger.
    """
    if estimate_tokens(messages) <= token_budget:
        return list(messages), []
    user_count = sum(1 for m in messages if m.get("role") == "user")
    for target in range(user_count, min_user_turns - 1, -1):
        kept, dropped = compact_messages(messages, max_user_turns=target)
        if estimate_tokens(kept) <= token_budget:
            return kept, dropped
    return compact_messages(messages, max_user_turns=min_user_turns)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Heuristic token estimate: ~4 chars per token plus per-message overhead.

    Mirrors the spirit of nananobot's
    ``nanobot/utils/helpers.py:estimate_message_tokens`` — a character
    count divided by four, plus a small per-message overhead to account
    for the role label and JSON framing. No tokenizer dependency.
    """
    total = 0
    for msg in messages:
        total += 4  # role label + structural overhead
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(json.dumps(part, ensure_ascii=False)) // 4
        for tc in msg.get("tool_calls") or []:
            total += len(json.dumps(tc, ensure_ascii=False)) // 4
        if msg.get("tool_call_id"):
            total += 4
    return total