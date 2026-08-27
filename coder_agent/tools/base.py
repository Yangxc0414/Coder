"""Tool base protocol and result dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Unified result wrapper for tool execution.

    Using a dataclass instead of raw strings makes error handling
    consistent across all tools and the ReAct loop.
    """

    output: str = ""
    error: str | None = None
    returncode: int = 0

    @property
    def success(self) -> bool:
        return self.error is None

    def to_message_content(self) -> str:
        """Format as a string for the tool result message."""
        if self.error:
            return f"(error) {self.error}"
        return self.output or "(no output)"


class Tool(ABC):
    """Abstract base class for all agent tools.

    Every tool must implement name, description, parameters (JSON Schema),
    and execute(). This uniform interface lets the ReAct loop dispatch
    any tool without knowing its internals.

    Design inspired by:
    - smolagents Tool基类 (tools.py)
    - SWE-agent Agent-Computer Interface
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name — must match the function name in tool calling schema."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description — injected into the model's tool schema."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema defining the tool's arguments.

        Example:
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        }
        """
        ...

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given arguments.

        - On success: return ToolResult(output="...")
        - On failure: return ToolResult(error="description of error")
        Do NOT raise exceptions — let the caller handle errors uniformly.
        """
        ...

    def to_tool_def(self) -> dict:
        """Convert this tool to OpenAI function-calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
