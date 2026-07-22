"""End-to-end demo using a real OpenAI-compatible API.

Set ``MYBOT_API_KEY`` (and optionally ``MYBOT_BASE_URL`` / ``MYBOT_MODEL``)
before running. The script will then exercise the full AgentRunner loop
against a real model::

    set MYBOT_API_KEY=sk-...
    set MYBOT_BASE_URL=https://api.deepseek.com/v1   (optional)
    set MYBOT_MODEL=deepseek-chat                    (optional)
    python -m examples.run_real_api

Without an API key the script just prints a friendly reminder.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from mybot.agent import AgentRunSpec, AgentRunner
from mybot.providers import OpenAICompatProvider
from mybot.tools import EchoTool, GetTimeTool, ReverseTool, ToolRegistry


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


SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them whenever the user asks for a reversal, an echo, or the time."
)


async def main() -> None:
    if not os.environ.get("MYBOT_API_KEY"):
        print(
            "MYBOT_API_KEY is not set. Export it (and optionally MYBOT_BASE_URL, "
            "MYBOT_MODEL) then re-run this script.",
            file=sys.stderr,
        )
        return

    runner_logger = logging.getLogger("mybot.agent")
    runner_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    runner_logger.addHandler(handler)
    runner_logger.propagate = False

    user_query = "Please reverse the string 'mybot' and tell me the current time."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    async with OpenAICompatProvider() as provider:
        spec = AgentRunSpec(
            messages=messages,
            tools=build_registry(),
            provider=provider,
            on_stream=stream_printer,
            on_stream_end=stream_end,
        )
        result = await AgentRunner().run(spec)

    print()
    print(f"final: {result.final_content!r}")
    print(f"tools_used: {result.tools_used}")
    print(f"stop_reason: {result.stop_reason}")


if __name__ == "__main__":
    asyncio.run(main())