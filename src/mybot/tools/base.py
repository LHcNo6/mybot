"""Base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolResult(str):
    """String-compatible tool output with structured status."""

    is_error: bool = False

    def __new__(cls, content: str, *, is_error: bool = False) -> ToolResult:
        obj = str.__new__(cls, content)
        obj.is_error = is_error
        return obj

    @classmethod
    def error(cls, content: str) -> ToolResult:
        return cls(content, is_error=True)


class Tool(ABC):
    """Agent capability: base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @property
    def concurrency_safe(self) -> bool:
        return False

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Safe schema-driven casts before validation. Override for custom type coercion."""
        return params

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate against JSON Schema; empty list means valid."""
        errors: list[str] = []
        schema = self.parameters or {}
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in params:
                errors.append(f"missing required parameter '{key}'")
        for key, value in params.items():
            if key not in props:
                continue
            prop = props[key]
            expected = prop.get("type")
            if expected == "string" and not isinstance(value, str):
                errors.append(f"'{key}' should be string")
            elif expected == "integer" and not isinstance(value, int):
                errors.append(f"'{key}' should be integer")
            elif expected == "boolean" and not isinstance(value, bool):
                errors.append(f"'{key}' should be boolean")
            elif expected == "number" and not isinstance(value, (int, float)):
                errors.append(f"'{key}' should be number")
        return errors

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        ...

    def to_schema(self) -> dict[str, Any]:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @staticmethod
    def error(content: str) -> ToolResult:
        return ToolResult.error(content)
