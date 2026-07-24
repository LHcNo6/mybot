"""Stage 5 / 6: Multi-turn REPL demo with compaction + summarization.

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
summarization of the dropped portion so the model can still recall
older decisions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from mybot.agent import (
    AgentRunSpec,
    AgentRunner,
    compact_messages,
    summarize_dropped,
)
from mybot.providers import OpenAICompatProvider
from mybot.session import load_messages, save_messages
from mybot.tools import EchoTool, GetTimeTool, ReverseTool, ToolRegistry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them whenever the user asks for a reversal, an echo, or the time."
)
EXIT_COMMANDS = {"/exit", "/quit"}
MAX_USER_TURNS = 12
SUMMARY_PREFIX = "[Earlier conversation summary]\n"
SESSION_KEY = os.environ.get("MYBOT_SESSION_KEY", "default")


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


async def main() -> None:
    configure_logging()
    configure_stdout()
    registry = build_registry()

    saved = load_messages(SESSION_KEY)
    messages: list[dict] = (
        saved if saved else [{"role": "system", "content": SYSTEM_PROMPT}]
    )

    async with OpenAICompatProvider() as provider:
        print(
            f"mybot REPL ready (model={provider.get_default_model()}, "
            f"session={SESSION_KEY}, "
            f"resumed={bool(saved)}). Type /exit to quit.\n"
        )
        while True:
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

            kept, dropped = compact_messages(messages, max_user_turns=MAX_USER_TURNS)
            summary_text: str | None = None
            if dropped:
                summary_text = await summarize_dropped(
                    provider, dropped, model=provider.get_default_model()
                )
                if summary_text:
                    kept.insert(
                        1,
                        {"role": "system", "content": SUMMARY_PREFIX + summary_text},
                    )
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
            save_messages(SESSION_KEY, messages)

            tail = f" [summary={len(summary_text or '')}c]" if summary_text else ""
            print(
                f"\n[msgs={len(messages)}, "
                f"tools_used={result.tools_used}, "
                f"stop_reason={result.stop_reason}{tail}]\n"
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # asyncio.run() can't be called from a running event loop; ignore.
        if "running event loop" not in str(e):
            raise