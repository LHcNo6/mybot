"""Stage 1: LLM call with streaming support.

Adds ``on_stream`` and ``on_stream_end`` callbacks. When ``on_stream`` is
provided, the runner uses ``chat_stream_with_retry`` instead of
``chat_with_retry``, forwarding each text delta to the callback.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRunResult:
    """The result of an agent run."""

    final_content: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "completed"


class AgentRunner:
    """AgentRunner — Stage 1: single LLM call with optional streaming."""

    async def run(
        self,
        provider: Any,
        messages: list[dict[str, Any]],
        model: str | None = None,
        *,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Make a single LLM call with optional streaming."""
        wants_streaming = on_stream is not None

        if wants_streaming:
            response = await provider.chat_stream_with_retry(
                messages=messages,
                model=model,
                on_content_delta=on_stream,
                **kwargs,
            )
        else:
            response = await provider.chat_with_retry(
                messages=messages,
                model=model,
                **kwargs,
            )

        messages = list(messages)
        if response.content:
            messages.append({"role": "assistant", "content": response.content})

        if wants_streaming and on_stream_end is not None:
            await on_stream_end(resuming=False)

        return AgentRunResult(
            final_content=response.content,
            messages=messages,
            stop_reason=response.finish_reason or "completed",
        )
