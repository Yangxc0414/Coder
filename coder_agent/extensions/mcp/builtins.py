"""Built-in MCP tools for the agent."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable

from ..base import McpTool


def create_builtin_mcp_tools(workspace) -> list[McpTool]:
    """Create built-in MCP tools (git, system info, etc.)."""

    def git_log(args: dict) -> dict[str, str]:
        try:
            limit = args.get("limit", 10)
            result = subprocess.run(
                ["git", "log", "--oneline", f"-{limit}"],
                capture_output=True, text=True, timeout=30, cwd=str(workspace),
            )
            return {"output": result.stdout, "error": result.stderr or ""}
        except Exception as e:
            return {"output": "", "error": str(e)}

    def git_status(args: dict) -> dict[str, str]:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, timeout=30, cwd=str(workspace),
            )
            return {"output": result.stdout, "error": result.stderr or ""}
        except Exception as e:
            return {"output": "", "error": str(e)}

    def git_diff(args: dict) -> dict[str, str]:
        try:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True, text=True, timeout=30, cwd=str(workspace),
            )
            return {"output": result.stdout, "error": result.stderr or ""}
        except Exception as e:
            return {"output": "", "error": str(e)}

    def system_info(args: dict) -> dict[str, str]:
        import platform
        import sys
        return {
            "output": (
                f"Python {sys.version.split()[0]}\n"
                f"Platform: {platform.platform()}\n"
                f"Working dir: {workspace}"
            ),
        }

    def read_trace(args: dict) -> dict[str, str]:
        """Read the agent's trace log."""
        trace_path = args.get("path", "")
        if not trace_path:
            return {"output": "", "error": "Trace path required"}
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"output": content[:5000]}
        except Exception as e:
            return {"output": "", "error": str(e)}

    return [
        McpTool(
            name="git_log",
            description="Show recent git commit history",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of commits to show", "default": 10},
                },
            },
            server_name="git",
            executor=git_log,
        ),
        McpTool(
            name="git_status",
            description="Show git status (modified, staged, untracked files)",
            parameters={"type": "object", "properties": {}},
            server_name="git",
            executor=git_status,
        ),
        McpTool(
            name="git_diff",
            description="Show uncommitted changes as diff",
            parameters={"type": "object", "properties": {}},
            server_name="git",
            executor=git_diff,
        ),
        McpTool(
            name="system_info",
            description="Get system and Python environment information",
            parameters={"type": "object", "properties": {}},
            server_name="system",
            executor=system_info,
        ),
        McpTool(
            name="read_trace",
            description="Read the agent execution trace log (JSONL format)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the trace JSONL file"},
                },
                "required": ["path"],
            },
            server_name="debug",
            executor=read_trace,
        ),
    ]
