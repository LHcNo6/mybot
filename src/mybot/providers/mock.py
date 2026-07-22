"""Mock LLM provider for offline testing.

Two flavors:

- ``MockScriptedProvider``: returns a pre-configured sequence of responses.
  Useful for writing deterministic end-to-end tests of the runner loop.
- ``MockKeywordProvider``: inspects the last user message and decides
  whether to emit a tool call (when a known tool name is mentioned) or a
  plain text reply. Mimics how a real LLM might "decide" to call a tool.

Neither flavor makes any network calls.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mybot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest


@dataclass
class MockScriptedStep:
    """One scripted turn of the mock LLM."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None

    def finish(self) -> str:
        if self.finish_reason:
            return self.finish_reason
        return "tool_calls" if self.tool_calls else "stop"


class MockScriptedProvider(LLMProvider):
    """Returns a pre-recorded sequence of responses.

    After the script is exhausted, the provider keeps returning a short
    fallback text so the runner can exit cleanly.
    """

    def __init__(
        self,
        steps: list[MockScriptedStep] | None = None,
        *,
        fallback: str = "[mock] Script exhausted.",
    ) -> None:
        self.generation = GenerationSettings()
        self.steps: list[MockScriptedStep] = list(steps or [])
        self.fallback = fallback
        self.call_count = 0
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_tools: list[dict[str, Any]] | None = None

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
        self.last_messages = messages
        self.last_tools = tools
        if self.call_count < len(self.steps):
            step = self.steps[self.call_count]
        else:
            step = MockScriptedStep(content=self.fallback)
        self.call_count += 1
        return self._build_response(step)

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
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        if response.content and on_content_delta:
            for chunk in _chunk_text(response.content):
                await on_content_delta(chunk)
        return response

    def _build_response(self, step: MockScriptedStep) -> LLMResponse:
        tool_calls: list[ToolCallRequest] = []
        for idx, raw in enumerate(step.tool_calls):
            tool_calls.append(
                ToolCallRequest(
                    id=raw.get("id") or f"call_{self.call_count}_{idx}",
                    name=raw["name"],
                    arguments=raw.get("arguments", {}),
                )
            )
        return LLMResponse(
            content=step.content,
            tool_calls=tool_calls,
            finish_reason=step.finish(),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )


@dataclass
class KeywordRule:
    """A simple keyword-based rule for ``MockKeywordProvider``."""

    match: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    extract: str | None = None


class MockKeywordProvider(LLMProvider):
    """A lightweight "smart" mock that picks tool calls based on keywords.

    Rules are checked in order. The first rule whose ``match`` substring is
    found in the latest user message wins and triggers a tool call. If
    ``extract`` is set, that named regex group becomes the value of
    ``arguments[extract]``. If no rule matches, a fixed reply is returned.
    """

    def __init__(
        self,
        rules: list[KeywordRule] | None = None,
        *,
        default_reply: str = "我不太明白，需要我用工具吗？",
    ) -> None:
        import re

        self.generation = GenerationSettings()
        self.default_reply = default_reply
        self._rules: list[tuple[re.Pattern[str], KeywordRule]] = []
        for rule in rules or []:
            self._rules.append((re.compile(rule.match), rule))
        self.call_count = 0
        self.last_user_text: str | None = None

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
        self.call_count += 1
        text = _last_user_text(messages)
        self.last_user_text = text

        for pattern, rule in self._rules:
            match = pattern.search(text) if text else None
            if match:
                arguments = dict(rule.arguments)
                if rule.extract and match.groupdict().get(rule.extract):
                    arguments[rule.extract] = match.group(rule.extract)
                tool_call = ToolCallRequest(
                    id=f"call_{self.call_count}",
                    name=rule.tool_name,
                    arguments=arguments,
                )
                return LLMResponse(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                )

        return LLMResponse(
            content=self.default_reply,
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

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
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        if response.content and on_content_delta:
            for chunk in _chunk_text(response.content):
                await on_content_delta(chunk)
        return response


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _chunk_text(text: str, size: int = 4) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]