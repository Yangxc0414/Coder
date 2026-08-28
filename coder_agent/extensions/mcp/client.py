"""Simplified MCP (Model Context Protocol) client.

This is a simplified implementation for demonstration and educational purposes.
Full MCP would require subprocess management, JSON-RPC, stdio transport, etc.

Supported "servers" in this simplified version:
- Built-in mock tools (for testing/demo)
- Local command-based tools
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class McpServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class McpTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str
    parameters: dict[str, Any]
    server_name: str
    executor: Callable[[dict[str, Any]], dict[str, str]] | None = None

    def to_tool_def(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": f"mcp_{self.server_name}_{self.name}",
                "description": f"[MCP: {self.server_name}] {self.description}",
                "parameters": self.parameters,
            },
        }


class McpClient:
    """Simplified MCP client.

    In a full implementation, this would:
    1. Start MCP servers as subprocesses
    2. Communicate via JSON-RPC over stdio/SSE
    3. Discover tools via initialize + tools/list
    4. Execute tools via tools/call

    For now, we support built-in mock tools and command-based tools.
    """

    def __init__(self) -> None:
        self._servers: dict[str, McpServerConfig] = {}
        self._tools: dict[str, McpTool] = {}

    def register_server(self, config: McpServerConfig) -> None:
        """Register an MCP server configuration."""
        self._servers[config.name] = config

    def add_tool(self, tool: McpTool) -> None:
        """Add an MCP tool."""
        key = f"{tool.server_name}_{tool.name}"
        self._tools[key] = tool

    def list_tools(self) -> list[McpTool]:
        """List all available MCP tools."""
        return list(self._tools.values())

    def execute_tool(self, server_name: str, tool_name: str, args: dict) -> dict:
        """Execute an MCP tool.

        Returns a dict with 'output' and 'error' keys.
        """
        key = f"{server_name}_{tool_name}"
        tool = self._tools.get(key)
        if not tool:
            return {"output": "", "error": f"Tool not found: {key}"}

        if tool.executor:
            return tool.executor(args)

        return {"output": f"Executed {tool_name} with args: {json.dumps(args)}"}

    def get_tool_defs(self) -> list[dict]:
        """Get OpenAI function-calling schema for all tools."""
        return [t.to_tool_def() for t in self.list_tools()]

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists."""
        return tool_name in self._tools


# ── Built-in mock MCP tools for testing ─────────────────────

def create_mock_mcp_client() -> McpClient:
    """Create a mock MCP client with sample tools."""
    client = McpClient()

    def git_log(args: dict) -> dict:
        try:
            limit = args.get("limit", 10)
            result = subprocess.run(
                ["git", "log", "--oneline", f"-{limit}"],
                capture_output=True, text=True, timeout=10
            )
            return {"output": result.stdout, "error": result.stderr}
        except Exception as e:
            return {"output": "", "error": str(e)}

    def git_diff(args: dict) -> dict:
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True, text=True, timeout=10
            )
            return {"output": result.stdout, "error": result.stderr}
        except Exception as e:
            return {"output": "", "error": str(e)}

    def system_info(args: dict) -> dict:
        import platform
        import sys
        return {
            "output": (
                f"Python {sys.version.split()[0]}\n"
                f"Platform: {platform.platform()}\n"
                f"CPU: {platform.processor() or 'unknown'}"
            ),
        }

    client.add_tool(McpTool(
        name="git_log",
        description="Show recent git commit history",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of commits", "default": 10},
            },
        },
        server_name="git",
        executor=git_log,
    ))

    client.add_tool(McpTool(
        name="git_diff",
        description="Show uncommitted changes",
        parameters={"type": "object", "properties": {}},
        server_name="git",
        executor=git_diff,
    ))

    client.add_tool(McpTool(
        name="system_info",
        description="Get system information",
        parameters={"type": "object", "properties": {}},
        server_name="system",
        executor=system_info,
    ))

    return client
