"""Stage 8.1: Hook system demo.

Demonstrates how to attach a :class:`HookManager` to an
:class:`AgentRunSpec` to observe LLM calls. The demo:

1. Times every LLM call (pre → post diff of ``time.monotonic()``).
2. Prints token usage after each call (``response.usage``).
3. Registers an intentionally-broken hook to prove that one bad
   hook does not break the agent loop.

Run with::

    python -m examples.run_with_hooks

A subset of agent turns will trigger ``echo`` / ``reverse`` / ``get_time``
so the metric prints a mix of tool-using and plain replies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from mybot.agent import AgentRunSpec, AgentRunner, HookManager
from mybot.providers import OpenAICompatProvider
from mybot.tools import EchoTool, GetTimeTool, ReverseTool, ToolRegistry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them whenever the user asks for a reversal, an echo, or the time."
)


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
    handler.setFormatter(logging.Formatter("[hook-warn] %(message)s"))
    hook_logger = logging.getLogger("mybot.agent.hooks")
    hook_logger.setLevel(logging.WARNING)
    hook_logger.addHandler(handler)
    hook_logger.propagate = False


def build_hooks() -> HookManager:
    """Return a HookManager wired with timing + token tracking + a
    demonstrably broken hook (proves resilience)."""
    hooks = HookManager()
    state: dict[str, float] = {}

    @hooks.on("pre_llm_call")
    def start_timer(**_: object) -> None:
        state["start"] = time.monotonic()

    @hooks.on("post_llm_call")
    def report(response, **_: object) -> None:
        elapsed = time.monotonic() - state.pop("start", time.monotonic())
        usage = response.usage or {}
        prompt = usage.get("prompt_tokens", "?")
        completion = usage.get("completion_tokens", "?")
        total = usage.get("total_tokens", "?")
        tool_count = len(response.tool_calls or [])
        print(
            f"[metric] llm_call elapsed={elapsed:.2f}s "
            f"tokens prompt={prompt} completion={completion} total={total} "
            f"tool_calls={tool_count}"
        )

    @hooks.on("pre_tool_call")
    def log_tool_call(name: str, arguments: object, **_: object) -> None:
        print(f"[tool] → {name}({arguments})")

    @hooks.on("post_tool_call")
    def log_tool_result(name: str, result: object, **_: object) -> None:
        snippet = str(result)[:80].replace("\n", " ")
        print(f"[tool] ← {name} → {snippet}")

    @hooks.on("post_llm_call")
    def broken_hook(**_: object) -> None:
        # Intentionally raise to demonstrate resilience: this should NOT
        # break the agent loop. The next hook (if any) should still fire.
        raise RuntimeError("simulated hook bug for resilience demo")

    return hooks


async def main() -> None:
    configure_logging()
    if not __import__("os").environ.get("MYBOT_API_KEY"):
        print(
            "MYBOT_API_KEY is not set. Fill it into .env (see .env.example) "
            "then re-run this script.",
            file=__import__("sys").stderr,
        )
        return

    hooks = build_hooks()
    print(f"HookManager registered {sum(len(v) for v in hooks._hooks.values())} hooks.\n")

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    registry = build_registry()

    inputs = [
        "please reverse the string 'mybot'",
        "now tell me the current time",
        "echo the word hello",
    ]

    async with OpenAICompatProvider() as provider:
        for inp in inputs:
            print(f"> {inp}")
            messages.append({"role": "user", "content": inp})
            spec = AgentRunSpec(
                messages=messages,
                tools=registry,
                provider=provider,
                on_stream=stream_printer,
                on_stream_end=stream_end,
                hooks=hooks,
            )
            result = await AgentRunner().run(spec)
            messages = result.messages
            print(f"\n[done] stop_reason={result.stop_reason}, "
                  f"tools_used={result.tools_used}\n")


if __name__ == "__main__":
    asyncio.run(main())