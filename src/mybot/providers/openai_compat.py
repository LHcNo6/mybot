"""OpenAI-compatible provider.

Works against any service that speaks the OpenAI Chat Completions API:
- api.openai.com
- DeepSeek, Moonshot, Zhipu, Ollama, vLLM, etc.

Configuration via environment variables (or constructor args):

- ``MYBOT_API_KEY`` (required): bearer token.
- ``MYBOT_BASE_URL`` (optional): API root, default ``https://api.openai.com/v1``.
- ``MYBOT_MODEL`` (optional): default model, default ``gpt-4o-mini``.

Streaming is implemented as an SSE consumer over httpx's streaming
response. Tool calls are parsed from the standard ``tool_calls`` field
on the assistant message.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from mybot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest


class OpenAICompatProvider(LLMProvider):
    """Minimal OpenAI Chat Completions client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.generation = GenerationSettings()
        self.api_key = api_key or os.environ.get("MYBOT_API_KEY", "")
        self.base_url = (base_url or os.environ.get("MYBOT_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("MYBOT_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatProvider:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def get_default_model(self) -> str | None:
        return self.model

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
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
        )
        response = await self._client.post(
            self._url("/chat/completions"),
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._parse_response(response.json())

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
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tool_choice=tool_choice,
            stream=True,
        )
        aggregated = _StreamAggregator()
        async with self._client.stream(
            "POST",
            self._url("/chat/completions"),
            json=payload,
            headers=self._headers(),
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (event.get("choices") or [{}])[0].get("delta") or {}
                if on_content_delta and delta.get("content"):
                    await on_content_delta(delta["content"])
                aggregated.feed(event)
        return aggregated.finalize()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"

    def _build_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        tool_choice: str | dict[str, Any] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        choice = data["choices"][0]
        message = choice.get("message", {})
        tool_calls: list[ToolCallRequest] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function", {})
            arguments = fn.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            tool_calls.append(
                ToolCallRequest(
                    id=raw.get("id") or "",
                    name=fn.get("name", ""),
                    arguments=arguments,
                )
            )
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
        )


class _StreamAggregator:
    """Accumulate streaming SSE events into a single ``LLMResponse``."""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason = "stop"
        self.usage: dict[str, Any] = {}

    def feed(self, event: dict[str, Any]) -> None:
        choice = (event.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if "content" in delta and delta["content"] is not None:
            self.content_parts.append(delta["content"])
        for raw in delta.get("tool_calls") or []:
            idx = raw.get("index", 0)
            slot = self.tool_calls.setdefault(
                idx,
                {"id": raw.get("id", ""), "name": "", "arguments": ""},
            )
            if raw.get("id"):
                slot["id"] = raw["id"]
            fn = raw.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if "arguments" in fn and fn["arguments"] is not None:
                slot["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        if event.get("usage"):
            self.usage = event["usage"]

    def finalize(self) -> LLMResponse:
        tool_calls: list[ToolCallRequest] = []
        for idx in sorted(self.tool_calls):
            slot = self.tool_calls[idx]
            arguments = slot["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            tool_calls.append(
                ToolCallRequest(
                    id=slot["id"] or f"call_{idx}",
                    name=slot["name"],
                    arguments=arguments,
                )
            )
        return LLMResponse(
            content="".join(self.content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=self.finish_reason or "stop",
            usage=self.usage,
        )