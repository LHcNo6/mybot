"""Stage 2: LLM call + tool execution loop.

Adds ``AgentRunSpec`` and an iteration loop. When the LLM returns tool
calls, the runner executes them, appends results, and loops back to the
LLM until the model produces a final text response or max iterations is
reached.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mybot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
    from mybot.tools.registry import ToolRegistry

HOOK_PRE_LLM_CALL = "pre_llm_call"
HOOK_POST_LLM_CALL = "post_llm_call"
HOOK_PRE_TOOL_CALL = "pre_tool_call"
HOOK_POST_TOOL_CALL = "post_tool_call"

logger = logging.getLogger(__name__)


@dataclass
class AgentRunSpec:
    """Configuration for a single agent run."""

    messages: list[dict[str, Any]]
    tools: Any = None                     # ToolRegistry with get_definitions() + execute()
    max_iterations: int = 10
    model: str | None = None
    provider: Any = None                   # LLMProvider; can also pass to run()
    on_stream: Callable[[str], Awaitable[None]] | None = None
    on_stream_end: Callable[..., Awaitable[None]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # forwarded to provider
    hooks: Any = None                     # HookManager (Stage 8.1) — emits pre/post LLM call


@dataclass
class AgentRunResult:
    """Outcome of an agent run."""

    final_content: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str = "completed"


class AgentRunner:
    """AgentRunner — Stage 2: iterative LLM + tool execution loop."""

    async def run(
        self,
        spec: AgentRunSpec,
        provider: Any | None = None,
    ) -> AgentRunResult:
        """Run the LLM loop, executing tools until a final text response."""
        provider = provider or spec.provider
        messages = list(spec.messages)
        tools_used: list[str] = []
        stop_reason = "completed"

        has_tools = spec.tools is not None
        wants_streaming = spec.on_stream is not None

        for iteration in range(spec.max_iterations):
            kwargs: dict[str, Any] = dict(spec.extra)
            tools_defs = spec.tools.get_definitions() if has_tools else None

            if spec.hooks is not None:
                await spec.hooks.emit(
                    HOOK_PRE_LLM_CALL,
                    messages=messages,
                    iteration=iteration,
                    max_iterations=spec.max_iterations,
                )

            response = await self._request_model(
                provider, messages, tools_defs,
                on_stream=spec.on_stream if wants_streaming else None,
                model=spec.model,
                **kwargs,
            )

            if spec.hooks is not None:
                await spec.hooks.emit(
                    HOOK_POST_LLM_CALL,
                    response=response,
                    iteration=iteration,
                    max_iterations=spec.max_iterations,
                )

            if response.should_execute_tools and tools_defs:
                messages.append(self._build_assistant_message(response))
                results = await self._execute_tools(
                    spec.tools,
                    response.tool_calls,
                    hooks=spec.hooks,
                    iteration=iteration,
                )
                for tc, result in zip(response.tool_calls, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    })
                    tools_used.append(tc.name)
                continue

            if wants_streaming and spec.on_stream_end is not None:
                await spec.on_stream_end(resuming=False)

            final = response.content or ""
            messages.append({"role": "assistant", "content": final})
            return AgentRunResult(
                final_content=final,
                messages=messages,
                tools_used=tools_used,
                stop_reason=stop_reason,
            )

        stop_reason = "max_iterations"
        return AgentRunResult(
            final_content=None,
            messages=messages,
            tools_used=tools_used,
            stop_reason=stop_reason,
        )

    async def _request_model(
        self,
        provider: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if on_stream is not None:
            return await provider.chat_stream_with_retry(
                messages=messages,
                tools=tools,
                model=model,
                on_content_delta=on_stream,
                **kwargs,
            )
        return await provider.chat_with_retry(
            messages=messages,
            tools=tools,
            model=model,
            **kwargs,
        )

    async def _execute_tools(
        self,
        tools: Any,
        tool_calls: list[Any],
        *,
        hooks: Any = None,
        iteration: int = 0,
    ) -> list[Any]:
        results: list[Any] = []
        for tc in tool_calls:
            logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
            if hooks is not None:
                await hooks.emit(
                    HOOK_PRE_TOOL_CALL,
                    name=tc.name,
                    arguments=tc.arguments,
                    iteration=iteration,
                )
            result = await tools.execute(tc.name, tc.arguments)
            if hooks is not None:
                await hooks.emit(
                    HOOK_POST_TOOL_CALL,
                    name=tc.name,
                    arguments=tc.arguments,
                    result=result,
                    iteration=iteration,
                )
            results.append(result)
        return results

    @staticmethod
    def _build_assistant_message(response: Any) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        if response.tool_calls:
            msg["tool_calls"] = [tc.to_openai_tool_call() for tc in response.tool_calls]
        return msg
