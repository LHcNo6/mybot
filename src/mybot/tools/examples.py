"""A few simple tools for offline testing.

These are intentionally trivial so the AgentRunner loop can be exercised
without any external service.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from mybot.tools.base import Tool, ToolResult


class EchoTool(Tool):
    """Echoes back whatever it is given."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the input text back to the caller."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo."},
            },
            "required": ["text"],
        }

    async def execute(self, *, text: str) -> ToolResult:
        return ToolResult(f"echo: {text}")


class ReverseTool(Tool):
    """Reverses a string."""

    @property
    def name(self) -> str:
        return "reverse"

    @property
    def description(self) -> str:
        return "Reverse the input string."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to reverse."},
            },
            "required": ["text"],
        }

    async def execute(self, *, text: str) -> ToolResult:
        return ToolResult(text[::-1])


class GetTimeTool(Tool):
    """Returns the current local time."""

    @property
    def name(self) -> str:
        return "get_time"

    @property
    def description(self) -> str:
        return "Return the current local time as an ISO 8601 string."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self) -> ToolResult:
        await asyncio.sleep(0)
        return ToolResult(datetime.now().isoformat(timespec="seconds"))