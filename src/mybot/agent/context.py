"""Stage 6.1: Sliding-window compaction aligned to user-turn boundaries.

Keeps the system message plus the last ``max_user_turns`` user turns,
preserving the matching assistant / tool messages that follow each kept
user turn. After tail-truncation the front of the retained slice is
walked forward until it lands on a ``user`` role, so we never emit an
orphan ``tool`` result or tool-bearing ``assistant`` message whose
partner was dropped.

This aligns with nananobot's ``SessionManager.retain_recent_legal_suffix``
— the same "legal boundary" idea, expressed in the simplest possible form.
"""

from __future__ import annotations

from typing import Any


def compact_messages(
    messages: list[dict[str, Any]],
    max_user_turns: int = 10,
) -> list[dict[str, Any]]:
    """Return a compacted copy of ``messages``.

    Invariants:

    - ``messages[0]`` (if it is a system message) is always preserved.
    - The returned slice contains at most ``max_user_turns`` user messages.
    - The retained slice starts on a user turn (no leading orphan tool
      results or tool-bearing assistant messages).
    - When ``messages`` has fewer user turns than ``max_user_turns``
      nothing is dropped (no-op for short histories).
    """
    if not messages:
        return list(messages)

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

    kept = list(reversed(kept_from_tail))

    # Strip any leading non-user messages (orphaned tool / mid-tool-call
    # assistant). Mirrors nananobot's find_legal_message_start.
    while kept and kept[0].get("role") != "user":
        kept.pop(0)

    return ([system, *kept] if system is not None else kept)