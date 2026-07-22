"""Base LLM provider interface."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: Any
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def has_valid_name(self) -> bool:
        return isinstance(self.name, str) and bool(self.name)

    def to_openai_tool_call(self) -> dict[str, Any]:
        arguments = (
            self.arguments
            if isinstance(self.arguments, str)
            else json.dumps(self.arguments, ensure_ascii=False)
        )
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": arguments,
            },
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None
    thinking_blocks: list[dict] | None = None

    error_status_code: int | None = None
    error_kind: str | None = None
    error_type: str | None = None
    error_code: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def should_execute_tools(self) -> bool:
        """Execute tools only when finish_reason is a tool-capable stop."""
        if not self.has_tool_calls:
            return False
        return self.finish_reason in ("tool_calls", "function_call", "stop")


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation settings."""

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """Base class for LLM providers."""

    generation: GenerationSettings = GenerationSettings()

    _RETRY_DELAYS = (1, 2, 4)

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Single LLM call."""
        ...

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Streaming LLM call. Subclasses may override for true streaming."""
        return await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = None,
        temperature: object = None,
        reasoning_effort: object = None,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Call chat() with retry on transient failures."""
        if max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is None:
            reasoning_effort = self.generation.reasoning_effort

        last_error: Exception | None = None
        for delay in self._RETRY_DELAYS:
            try:
                return await self.chat(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
            except Exception as e:
                last_error = e
                if on_retry_wait:
                    await on_retry_wait(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = None,
        temperature: object = None,
        reasoning_effort: object = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_stream_recover: Callable[[], Awaitable[None]] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Call chat_stream() with retry on transient failures."""
        if max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is None:
            reasoning_effort = self.generation.reasoning_effort

        last_error: Exception | None = None
        for delay in self._RETRY_DELAYS:
            try:
                return await self.chat_stream(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                    on_content_delta=on_content_delta,
                    on_thinking_delta=on_thinking_delta,
                    on_tool_call_delta=on_tool_call_delta,
                )
            except Exception as e:
                last_error = e
                if on_retry_wait:
                    await on_retry_wait(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]

    def get_default_model(self) -> str | None:
        """Return the default model name, if known."""
        return None
