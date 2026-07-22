"""End-to-end demo: scripted mock provider + sample tools + AgentRunner.

Run with::

    python -m examples.run_mock

This script exercises every piece of Stage 2 + Stage 2.5 without needing
any API keys or external services. It prints what happens at each step
so the loop is easy to follow.
"""

from __future__ import annotations

import asyncio
import logging

from mybot.agent import AgentRunSpec, AgentRunner
from mybot.providers import MockScriptedProvider, MockScriptedStep
from mybot.tools import EchoTool, GetTimeTool, ReverseTool, ToolRegistry

logger = logging.getLogger(__name__)


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


def scripted_demo() -> None:
    """A deterministic, hand-written script of LLM turns."""
    logger.info("=== Scripted demo ===")
    provider = MockScriptedProvider(
        steps=[
            MockScriptedStep(
                tool_calls=[{"name": "reverse", "arguments": {"text": "hello"}}],
            ),
            MockScriptedStep(content="The reversed string is: olleh"),
        ]
    )
    spec = AgentRunSpec(
        messages=[{"role": "user", "content": "Reverse 'hello' please."}],
        tools=build_registry(),
        provider=provider,
        on_stream=stream_printer,
        on_stream_end=stream_end,
    )
    result = asyncio.run(AgentRunner().run(spec))
    print(f"\nfinal: {result.final_content!r}")
    print(f"tools_used: {result.tools_used}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"turns: {provider.call_count}")


def tool_chain_demo() -> None:
    """Mock makes two tool calls in a row, then answers."""
    logger.info("=== Tool-chain demo ===")
    provider = MockScriptedProvider(
        steps=[
            MockScriptedStep(
                tool_calls=[{"name": "echo", "arguments": {"text": "first"}}],
            ),
            MockScriptedStep(
                tool_calls=[{"name": "get_time", "arguments": {}}],
            ),
            MockScriptedStep(content="Done: got echo and the current time."),
        ]
    )
    spec = AgentRunSpec(
        messages=[{"role": "user", "content": "Echo and tell me the time."}],
        tools=build_registry(),
        provider=provider,
        on_stream=stream_printer,
        on_stream_end=stream_end,
    )
    result = asyncio.run(AgentRunner().run(spec))
    print(f"\nfinal: {result.final_content!r}")
    print(f"tools_used: {result.tools_used}")
    print(f"turns: {provider.call_count}")


def main() -> None:
    logger_handler = logging.StreamHandler()
    logger_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    runner_logger = logging.getLogger("mybot.agent")
    runner_logger.setLevel(logging.INFO)
    runner_logger.addHandler(logger_handler)
    runner_logger.propagate = False

    scripted_demo()
    print()
    tool_chain_demo()


if __name__ == "__main__":
    main()