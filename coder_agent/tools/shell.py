"""Shell command execution tool with safety controls."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import Tool, ToolResult


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Execute a shell command in the workspace and return stdout/stderr. "
        "Use for running tests, building code, installing dependencies, etc. "
        "Commands run in a sandboxed environment (no API keys exposed)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds (default: 60)",
                "default": 60,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory relative to workspace (default: root)",
                "default": ".",
            },
        },
        "required": ["command"],
    }

    # Dangerous command patterns — hard-blocked at code level
    # Inspired by Cline's tool-policies.ts whitelist/denylist approach
    DANGEROUS_PATTERNS = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",
        "> /",
        "sudo ",
        "sudo@",
    ]

    MAX_OUTPUT_CHARS = 8000

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    def execute(self, args: dict[str, str]) -> ToolResult:
        command = args["command"]
        timeout = int(args.get("timeout", 60))
        cwd_str = args.get("cwd", ".")

        # Safety check: dangerous command patterns
        if self._is_dangerous(command):
            return ToolResult(
                error=f"Dangerous command blocked by policy: {command[:100]}"
            )

        try:
            cwd = (self.workspace / cwd_str).resolve()
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=str(cwd),
                env=self._safe_env(),
            )
            output = self._truncate(result.stdout)
            stderr = self._truncate(result.stderr) if result.returncode != 0 else ""
            return ToolResult(
                output=output,
                error=stderr if stderr else None,
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                error=f"Command timed out after {timeout}s: {command[:200]}"
            )
        except Exception as e:
            return ToolResult(error=f"Command execution failed: {e}")

    def _is_dangerous(self, command: str) -> bool:
        cmd_lower = command.lower()
        return any(p.lower() in cmd_lower for p in self.DANGEROUS_PATTERNS)

    @staticmethod
    def _truncate(text: str, max_len: int = MAX_OUTPUT_CHARS) -> str:
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}\n... [truncated, total {len(text)} chars]"

    @staticmethod
    def _safe_env() -> dict[str, str]:
        """Return a sanitized environment — no API keys or secrets."""
        safe_keys = {"PATH", "HOME", "USER", "LANG", "TERM", "PYTHONPATH"}
        return {k: v for k, v in os.environ.items() if k in safe_keys}
