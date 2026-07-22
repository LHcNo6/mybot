"""Tool registry for dynamic tool management."""

from __future__ import annotations

import json
from typing import Any

from mybot.tools.base import Tool, ToolResult


def is_tool_error_result(name: str, result: Any) -> bool:
    return isinstance(result, ToolResult) and result.is_error


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def prepare_call(
        self,
        name: str,
        params: Any,
    ) -> tuple[Tool | None, Any, str | None]:
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self.tool_names)
            return None, params, (
                f"Error: Tool '{name}' not found. Available: {available}"
            )
        params = self._coerce_params(params)
        if not isinstance(params, dict):
            return tool, params, (
                f"Error: Tool '{name}' parameters must be a JSON object, got {type(params).__name__}."
            )
        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, (
                f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            )
        return tool, cast_params, None

    def _coerce_params(self, params: Any) -> Any:
        if params is None:
            return {}
        if not isinstance(params, str):
            return params
        stripped = params.strip()
        if not stripped or not stripped.startswith(("{", "[")):
            return params
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return params

    async def execute(self, name: str, params: Any) -> Any:
        hint = "\n\n[Analyze the error above and try a different approach.]"
        try:
            tool, resolved_params, error = self.prepare_call(name, params)
        except Exception as e:
            return ToolResult.error(f"Error preparing tool '{name}': {e}" + hint)
        if error:
            return ToolResult.error(error + hint)
        try:
            assert tool is not None
            result = await tool.execute(**resolved_params)
            if is_tool_error_result(name, result):
                return ToolResult.error(str(result) + hint)
            return result
        except Exception as e:
            return ToolResult.error(f"Error executing {name}: {e}" + hint)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
