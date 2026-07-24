"""Stage 6.2 / 7.2: LLM-based summarization of compacted message slices.

When :func:`compact_messages` decides to drop earlier messages, this
module asks the LLM for a short text summary of what was dropped and
hands it back so the caller can prepend it to the kept messages.

Mirrors the spirit of nananobot's
``Consolidator._consolidate_replay_overflow`` — find a boundary, ask
the model to summarize the chunk — but in the simplest possible form:
no token estimation, no async locking.

Stage 7.2 adds ``prev_summary`` support: when the caller has already
summarized earlier slices in this session, pass that text here and the
LLM is asked to merge it with the newly-dropped turns instead of
re-reading the previously summarized content.
"""

from __future__ import annotations

from typing import Any

MIN_USER_TURNS = 2
_SUMMARY_MAX_OUTPUT_TOKENS = 300


async def summarize_dropped(
    provider: Any,
    dropped: list[dict[str, Any]],
    *,
    model: str | None = None,
    min_user_turns: int = MIN_USER_TURNS,
    prev_summary: str | None = None,
) -> str | None:
    """If ``dropped`` covers at least ``min_user_turns`` user turns,
    return a short LLM-generated summary; otherwise return ``None``.

    When ``prev_summary`` is given, the prompt asks the LLM to merge it
    with the new ``dropped`` content into an updated combined summary
    instead of summarizing from scratch. This is the incremental
    compaction hook (Stage 7.2).

    The function is intentionally defensive: any provider failure or
    empty summary degrades to ``None`` (caller can fall back to plain
    truncation without advancing the cursor).
    """
    user_count = sum(1 for m in dropped if m.get("role") == "user")
    if user_count < min_user_turns:
        return None

    if prev_summary:
        prompt = (
            "You have an earlier summary of a prior part of this conversation:\n"
            f"<earlier>\n{prev_summary}\n</earlier>\n\n"
            "Merge in these new turns (preserve every earlier fact, decision, "
            "and tool result; only ADD or REFINE — never DROP):\n"
            f"<new>\n{_format_transcript(dropped)}\n</new>\n\n"
            "Produce an updated combined summary in 2-3 sentences."
        )
    else:
        prompt = (
            "Summarize the following conversation excerpt in 2-3 sentences. "
            "Preserve concrete facts, decisions, tool calls, and any results "
            "the user might want to reference later. "
            "Be terse; do not editorialize.\n\n"
            + _format_transcript(dropped)
        )
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=_SUMMARY_MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
    except Exception:
        return None

    text = (response.content or "").strip()
    return text or None


def _format_transcript(messages: list[dict[str, Any]]) -> str:
    """Render a message list as a simple plain-text transcript."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            lines.append(f"User: {msg.get('content', '')}")
        elif role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                content = (content + " " if content else "") + f"[called {', '.join(names)}]"
            if content:
                lines.append(f"Assistant: {content}")
        elif role == "tool":
            lines.append(
                f"Tool ({msg.get('name', '?')}): {msg.get('content', '')}"
            )
    return "\n".join(lines)