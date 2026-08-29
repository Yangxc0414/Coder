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
        "For BLOCKING services (dev servers like http.server) set background=true "
        "— the command returns immediately with a PID and log path; read the log "
        "file later to check output. Note: on Windows use 'python', not 'python3'. "
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
            "background": {
                "type": "boolean",
                "description": (
                    "Run without blocking (for dev servers / watchers). "
                    "Returns PID + log file path immediately (default: false)"
                ),
                "default": False,
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
    # timeout 钳制：模型传 timeout=999999 会让主循环无限卡住
    # （实测 sleep 命令完全阻塞调度）
    MAX_TIMEOUT_SECONDS = 300

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    def execute(self, args: dict[str, str]) -> ToolResult:
        command = args["command"]
        timeout = int(args.get("timeout", 60))
        cwd_str = args.get("cwd", ".")
        background = bool(args.get("background", False))

        # Safety check: dangerous command patterns
        if self._is_dangerous(command):
            return ToolResult(
                error=f"Dangerous command blocked by policy: {command[:100]}"
            )

        try:
            cwd = (self.workspace / cwd_str).resolve()
            # cwd 逃逸校验（与 read_file 的 commonpath 同源）：Path 拼接
            # 绝对路径会直接替换工作区——实测 agent 可借此在外部目录
            # （甚至用户主目录）执行命令，击穿工作区隔离。
            import os as _os
            try:
                inside = _os.path.commonpath(
                    [str(self.workspace), str(cwd)]) == str(self.workspace)
            except ValueError:
                inside = False  # 跨盘符无法比较 → 视为逃逸
            if not inside:
                return ToolResult(error=f"cwd escapes workspace: {cwd_str[:100]}")

            if background:
                return self._run_background(command, cwd)

            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=str(cwd),
                env=self._safe_env(),
                encoding="utf-8",
                errors="replace",
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
                error=(
                    f"Command timed out after {timeout}s: {command[:200]}. "
                    "If this is a long-running service (server/watcher), "
                    "re-run it with background=true instead."
                )
            )
        except Exception as e:
            return ToolResult(error=f"Command execution failed: {e}")

    BG_LOG_DIR = ".coder_bg"

    def _run_background(self, command: str, cwd: Path) -> ToolResult:
        """Launch a long-running command without blocking.

        Real-run finding: the agent needed a live dev server for page
        verification, but blocking `http.server` only ever died at the
        tool timeout. Background mode returns immediately; output goes to
        a log file the agent can inspect with read_file.
        """
        import time

        log_dir = self.workspace / self.BG_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"bg_{time.strftime('%H%M%S')}_{os.getpid()}.log"

        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(cwd),
                    env=self._safe_env(),
                )
            return ToolResult(
                output=(
                    f"已在后台启动 (PID {proc.pid}): {command[:100]}\n"
                    f"日志文件: {self.BG_LOG_DIR}/{log_path.name}\n"
                    f"稍后用 read_file 查看该日志确认服务状态；"
                    f"停止服务请用系统命令结束该 PID。"
                ),
                returncode=0,
            )
        except Exception as e:
            return ToolResult(error=f"Background launch failed: {e}")

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
        """Block secrets from leaking into subprocesses, keep OS essentials.

        Design note: a strict whitelist (PATH/HOME/...) breaks Python on
        Windows — without SYSTEMROOT the interpreter dies at startup with
        "_Py_HashRandomization_Init: failed to get random numbers", which
        silently killed every python command in a real agent run (13
        consecutive failures). Invert the strategy: pass everything EXCEPT
        variables whose name looks like a credential.
        """
        import re

        secret_pattern = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
        return {k: v for k, v in os.environ.items() if not secret_pattern.search(k)}
