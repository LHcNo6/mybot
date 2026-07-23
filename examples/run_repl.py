"""Stage 5: Multi-turn REPL demo.

A minimal interactive loop around :class:`AgentRunner`. Each turn:

1. Read a line from stdin.
2. Append it to the running ``messages`` history.
3. Run the runner; stream the assistant reply to stdout.
4. Replace ``messages`` with the runner's returned history and repeat.

Run with::

    python -m examples.run_repl

Commands inside the REPL:
    /exit, /quit   leave the loop
    <empty line>   skip this turn
    anything else  sent to the model as a user message

After a handful of turns the request payload will grow linearly with
history length. Stage 6 will introduce message compaction.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from mybot.agent import AgentRunSpec, AgentRunner
from mybot.providers import OpenAICompatProvider
from mybot.tools import EchoTool, GetTimeTool, ReverseTool, ToolRegistry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them whenever the user asks for a reversal, an echo, or the time."
)
EXIT_COMMANDS = {"/exit", "/quit"}


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


async def main() -> None:
    configure_logging()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    registry = build_registry()

    async with OpenAICompatProvider() as provider:
        print(f"mybot REPL ready (model={provider.get_default_model()}). "
              f"Type /exit to quit.\n")
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
            spec = AgentRunSpec(
                messages=messages,
                tools=registry,
                provider=provider,
                on_stream=stream_printer,
                on_stream_end=stream_end,
            )
            result = await AgentRunner().run(spec)
            messages = result.messages

            print(
                f"\n[turns={len(messages) - 1} messages, "
                f"tools_used={result.tools_used}, "
                f"stop_reason={result.stop_reason}]\n"
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # asyncio.run() can't be called from a running event loop; ignore.
        if "running event loop" not in str(e):
            raise