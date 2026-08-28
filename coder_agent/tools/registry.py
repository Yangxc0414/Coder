"""Tool registry — manages registration, discovery, and schema generation."""

from __future__ import annotations

from typing import Any

from .base import Tool


class ToolRegistry:
    """Central registry for all available tools.

    Provides:
    - register(): Add a new tool
    - get(): Retrieve a tool by name
    - list_tools(): Get OpenAI function-calling schemas for all tools
    - list_names(): Get all registered tool names (for validation)

    Inspired by:
    - smolagents tool dict pattern
    - Goose extension system
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if name already exists."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(
                f"Unknown tool: '{name}'. "
                f"Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all tools."""
        return [tool.to_tool_def() for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def create_default_registry(workspace, mode) -> ToolRegistry:
    """Create a registry with standard tools for the given workspace and mode.

    Usage:
        from coder_agent.tools.registry import create_default_registry
        registry = create_default_registry("/path/to/project", AgentMode.FULL)
    """
    from pathlib import Path

    workspace_path = Path(workspace).resolve()
    dry_run = (mode.value == "dry-run")

    from coder_agent.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
    from coder_agent.tools.search import SearchTextTool
    from coder_agent.tools.shell import RunCommandTool

    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace_path))
    registry.register(WriteFileTool(workspace_path, dry_run=dry_run))
    registry.register(ListFilesTool(workspace_path))
    registry.register(SearchTextTool(workspace_path))
    registry.register(RunCommandTool(workspace_path))
    return registry
