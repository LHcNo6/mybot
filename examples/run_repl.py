"""Stage 5 / 6 / 8.3: Multi-turn REPL demo with compaction + summarization.

A minimal interactive loop around :class:`AgentRunner`. Each turn:

1. Read a line from stdin.
2. Append it to the running ``messages`` history.
3. Run :func:`compact_messages` to truncate old turns and possibly
   call :func:`summarize_dropped` on what was dropped, prepending the
   summary as a system message so the model can reference it.
4. Run the runner; stream the assistant reply to stdout.
5. Replace ``messages`` with the runner's returned history and repeat.

Run with::

    python -m examples.run_repl

Commands inside the REPL:
    /exit, /quit   leave the loop
    <empty line>   skip this turn
    anything else  sent to the model as a user message

Stage 6.1 added sliding-window truncation. Stage 6.2 added LLM
summarization of the dropped portion. Stage 8.3 added TTL-based
auto-compaction: if the session has been idle ≥ MYBOT_TTL_MINUTES
(default 30), the next turn triggers compaction + summarization
before processing the new input (mirroring nananobot's
``AutoCompactor.check_expired`` flow, polling-style).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mybot.agent import (
    AgentRunSpec,
    AgentRunner,
    compact_for_budget,
    compact_messages,
    estimate_tokens,
    summarize_dropped,
)
from mybot.providers import OpenAICompatProvider
from mybot.session import SessionMeta, load_or_init, save_messages, touch
from mybot.tools import EchoTool, GetTimeTool, ReverseTool, ToolRegistry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them whenever the user asks for a reversal, an echo, or the time."
)
EXIT_COMMANDS = {"/exit", "/quit"}
MAX_USER_TURNS = 12
DEFAULT_TOKEN_BUDGET = 8000
DEFAULT_TTL_MINUTES = 30
SUMMARY_PREFIX = "[Earlier conversation summary]\n"
SESSION_KEY = os.environ.get("MYBOT_SESSION_KEY", "default")
TOKEN_BUDGET = int(os.environ.get("MYBOT_TOKEN_BUDGET", str(DEFAULT_TOKEN_BUDGET)))
TTL_MINUTES = int(os.environ.get("MYBOT_TTL_MINUTES", str(DEFAULT_TTL_MINUTES)))


async def stream_printer(delta: str) -> None:
    print(delta, end="", flush=True)


async def stream_end(**_: object) -> None:
    print()


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(ReverseTool())
    registry.register(GetTimeTool())
    return registry


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    runner_logger = logging.getLogger("mybot.agent")
    runner_logger.setLevel(logging.INFO)
    runner_logger.addHandler(handler)
    runner_logger.propagate = False


def configure_stdout() -> None:
    """Allow emoji / non-ASCII characters to print on Windows consoles
    (cp936 / cp1252) by reconfiguring stdout to UTF-8 with replacement."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def age_minutes(meta: SessionMeta) -> float:
    """How long since meta.updated_at, in minutes. 0 if no timestamp yet."""
    if not meta.updated_at:
        return 0.0
    last = datetime.fromisoformat(meta.updated_at)
    return (datetime.now() - last).total_seconds() / 60


async def compact_and_summarize(
    provider: Any,
    messages: list[dict],
    meta: SessionMeta,
    *,
    max_user_turns: int = MAX_USER_TURNS,
    token_budget: int = TOKEN_BUDGET,
) -> tuple[list[dict], str | None]:
    """Run compact + (optional) summary. Updates ``meta`` in place.

    Returns ``(new_messages, summary_text_or_none)``. ``new_messages`` is
    always the post-compaction slice; summary (if generated) has been
    inserted at index 1 and recorded in ``meta.last_summary``.
    """
    pre_tokens = estimate_tokens(messages)
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    over_budget = pre_tokens > token_budget
    over_turns = user_turns > max_user_turns

    kept: list[dict]
    dropped: list[dict]
    if over_budget:
        kept, dropped = compact_for_budget(messages, token_budget)
    elif over_turns:
        kept, dropped = compact_messages(messages, max_user_turns=max_user_turns)
    else:
        kept, dropped = list(messages), []

    # Skip already-summarized messages at the head of `dropped` (Stage 7.2
    # fix: count summaries by content prefix, not by an unreliable cursor).
    summary_count = sum(
        1 for m in dropped
        if m.get("role") == "system"
        and (m.get("content") or "").startswith(SUMMARY_PREFIX)
    )
    unconsolidated_dropped = dropped[summary_count:]

    summary_text: str | None = None
    if unconsolidated_dropped:
        summary_text = await summarize_dropped(
            provider,
            unconsolidated_dropped,
            model=provider.get_default_model(),
            prev_summary=meta.last_summary,
        )
        if summary_text:
            kept.insert(
                1,
                {"role": "system", "content": SUMMARY_PREFIX + summary_text},
            )
            meta.last_summary = summary_text
            meta.last_consolidated = len(kept)
    return kept, summary_text


async def maybe_auto_compact(
    provider: Any,
    messages: list[dict],
    meta: SessionMeta,
) -> tuple[list[dict], str | None]:
    """If ``meta.updated_at`` is older than ``TTL_MINUTES``, run compaction
    before the next user input. Returns ``(messages, summary_or_none)`` —
    the summary is ``None`` if no compaction was needed.
    """
    age = age_minutes(meta)
    if age < TTL_MINUTES:
        return messages, None
    print(
        f"[auto-compact] idle {age:.1f}min >= {TTL_MINUTES}min, "
        f"compacting {len(messages)} messages..."
    )
    new_messages, summary_text = await compact_and_summarize(provider, messages, meta)
    print(f"[auto-compact] done, {len(new_messages)} messages after compact")
    return new_messages, summary_text


async def main() -> None:
    configure_logging()
    configure_stdout()
    registry = build_registry()

    saved, meta = load_or_init(SESSION_KEY)
    messages: list[dict] = (
        saved if saved else [{"role": "system", "content": SYSTEM_PROMPT}]
    )

    async with OpenAICompatProvider() as provider:
        print(
            f"mybot REPL ready (model={provider.get_default_model()}, "
            f"session={SESSION_KEY}, "
            f"resumed={bool(saved)}, "
            f"created={meta.created_at}, "
            f"ttl={TTL_MINUTES}min, "
            f"budget={TOKEN_BUDGET}t). Type /exit to quit.\n"
        )
        while True:
            # Stage 8.3: TTL-based auto-compaction (polling-style).
            messages, _ttl_summary = await maybe_auto_compact(provider, messages, meta)
            if messages != saved:
                # Auto-compact produced new messages; persist them.
                touch(meta)
                save_messages(SESSION_KEY, messages, meta)

            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not user_input:
                continue
            if user_input in EXIT_COMMANDS:
                return

            messages.append({"role": "user", "content": user_input})

            kept, summary_text = await compact_and_summarize(provider, messages, meta)
            messages = kept

            spec = AgentRunSpec(
                messages=messages,
                tools=registry,
                provider=provider,
                on_stream=stream_printer,
                on_stream_end=stream_end,
            )
            result = await AgentRunner().run(spec)
            messages = result.messages
            touch(meta)
            save_messages(SESSION_KEY, messages, meta)

            tail = f" [summary={len(summary_text or '')}c]" if summary_text else ""
            post_tokens = estimate_tokens(messages)
            pre_over_budget = estimate_tokens(messages) > TOKEN_BUDGET
            print(
                f"\n[msgs={len(messages)}, "
                f"tokens~{post_tokens}, "
                f"tools_used={result.tools_used}, "
                f"stop_reason={result.stop_reason}, "
                f"cursor={meta.last_consolidated}{tail}]\n"
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # asyncio.run() can't be called from a running event loop; ignore.
        if "running event loop" not in str(e):
            raise