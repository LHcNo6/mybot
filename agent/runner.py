"""Stage 0: Minimal LLM call — nothing else.

This is the absolute simplest agent runner. It takes messages + a provider,
calls the LLM, and returns the response. No tools, no hooks, no context
governance, no streaming — just a single LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRunResult:
    """The bare-minimum result of a single LLM call."""

    final_content: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "completed"


class AgentRunner:
    """AgentRunner — Stage 0: only a single LLM call, nothing more."""

    async def run(
        self,
        provider: Any,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Make a single LLM call and return the response."""
        response = await provider.chat_with_retry(
            messages=messages,
            model=model,
            **kwargs,
        )

        messages = list(messages)
        if response.content:
            messages.append({"role": "assistant", "content": response.content})

        return AgentRunResult(
            final_content=response.content,
            messages=messages,
            stop_reason=response.finish_reason or "completed",
        )
